# Real-time Auralization Implementation Plan

## Overview
Implement robust real-time auralization by convolving dynamically generated impulse responses with dry audio signals using the Web Audio API's ConvolverNode. This builds upon the existing foundation in audio-processor.ts to create a complete real-time spatial audio system.

## Current System Analysis

### Existing Foundation
- **ConvolverNode usage**: Already implemented in audio-processor.ts
- **Impulse response generation**: Basic IR generation from ray tracing
- **Web Audio API integration**: Core audio processing pipeline established
- **Spatial audio processing**: Basic spatial audio calculations

### Current Limitations
- **Static impulse responses**: IRs not updated in real-time
- **Limited audio sources**: Single source processing
- **No dynamic updates**: Room changes don't update auralization
- **Performance bottlenecks**: IR generation blocks audio processing

### Goals for Real-time Auralization
- **Dynamic IR updates**: Real-time impulse response regeneration
- **Multiple audio sources**: Support for multiple simultaneous sources
- **Low-latency processing**: Minimal delay between changes and audio output
- **Smooth transitions**: Seamless IR updates without audio artifacts

## Technical Architecture

### Real-time Processing Pipeline
```
Audio Input → IR Generation → Convolution → Spatial Processing → Audio Output
     ↓              ↓              ↓              ↓              ↓
  Dry Signal → Dynamic IR → ConvolverNode → HRTF/Panning → Speakers/Headphones
```

### Update Triggers
- **Listener movement**: Position or orientation changes
- **Source movement**: Audio source position changes  
- **Room modifications**: Material or geometry changes
- **Environmental changes**: Temperature, humidity updates

## Task Breakdown

### 1. Implement Dynamic Impulse Response System
**Goal:** Create system for real-time impulse response generation and updates.

#### 1.1 Create IR Update Manager
- **Description:** Coordinate impulse response updates based on scene changes
- **Implementation Details:**
  ```typescript
  class IRUpdateManager {
    private updateQueue: IRUpdateRequest[] = [];
    private isProcessing: boolean = false;
    private updateThreshold: number = 0.1; // Minimum change to trigger update
    
    public requestIRUpdate(source: AudioSource, listener: Listener, priority: UpdatePriority): void {
      const request = new IRUpdateRequest(source, listener, priority, Date.now());
      
      // Check if update is necessary
      if (this.shouldUpdate(request)) {
        this.addToQueue(request);
        this.processQueue();
      }
    }
    
    private shouldUpdate(request: IRUpdateRequest): boolean {
      // Check position changes, room changes, etc.
      const lastUpdate = this.getLastUpdate(request.source.id);
      return this.calculateChangeSignificance(request, lastUpdate) > this.updateThreshold;
    }
    
    private async processQueue(): Promise<void> {
      if (this.isProcessing) return;
      
      this.isProcessing = true;
      while (this.updateQueue.length > 0) {
        const request = this.updateQueue.shift()!;
        await this.processIRUpdate(request);
      }
      this.isProcessing = false;
    }
    
    private async processIRUpdate(request: IRUpdateRequest): Promise<void>
  }
  
  interface IRUpdateRequest {
    source: AudioSource;
    listener: Listener;
    priority: UpdatePriority;
    timestamp: number;
    changeType: ChangeType;
  }
  
  enum UpdatePriority { LOW, MEDIUM, HIGH, CRITICAL }
  enum ChangeType { POSITION, ORIENTATION, ROOM, MATERIAL, ENVIRONMENT }
  ```
- **Features Required:**
  - Change detection and significance calculation
  - Priority-based update queue
  - Asynchronous processing to avoid blocking
  - Update throttling to prevent overload
- **Files:** `src/audio/ir-update-manager.ts` (new)

#### 1.2 Implement Incremental IR Updates
- **Description:** Optimize IR generation by updating only changed portions
- **Implementation Details:**
  ```typescript
  class IncrementalIRGenerator {
    private cachedIRs: Map<string, CachedIR> = new Map();
    
    public generateIncrementalIR(source: AudioSource, listener: Listener, 
                                changes: SceneChange[]): IncrementalIRResult {
      const cacheKey = this.generateCacheKey(source, listener);
      const cachedIR = this.cachedIRs.get(cacheKey);
      
      if (!cachedIR) {
        // Generate full IR for new configuration
        return this.generateFullIR(source, listener);
      }
      
      // Determine which parts need updating
      const updateRegions = this.analyzeChanges(changes, cachedIR);
      
      if (updateRegions.requiresFullUpdate) {
        return this.generateFullIR(source, listener);
      }
      
      // Update only changed regions
      return this.updateIRRegions(cachedIR, updateRegions);
    }
    
    private analyzeChanges(changes: SceneChange[], cachedIR: CachedIR): UpdateRegions
    private updateIRRegions(cachedIR: CachedIR, regions: UpdateRegions): IncrementalIRResult
    private generateFullIR(source: AudioSource, listener: Listener): IncrementalIRResult
  }
  
  interface CachedIR {
    impulseResponse: Float32Array[];
    metadata: IRMetadata;
    lastUpdate: number;
    dependencies: SceneDependency[];
  }
  
  interface UpdateRegions {
    earlyReflections: boolean;
    lateReverberation: boolean;
    directSound: boolean;
    requiresFullUpdate: boolean;
    affectedTimeRange: [number, number];
  }
  ```
- **Features Required:**
  - Intelligent change analysis
  - Partial IR updates
  - Dependency tracking
  - Cache management
- **Files:** `src/audio/incremental-ir-generator.ts` (new)

#### 1.3 Create IR Interpolation System
- **Description:** Smooth transitions between different impulse responses
- **Implementation Details:**
  ```typescript
  class IRInterpolator {
    private interpolationTime: number = 0.1; // 100ms interpolation
    
    public createInterpolatedIR(fromIR: Float32Array[], toIR: Float32Array[], 
                               progress: number): Float32Array[] {
      const interpolatedIR = [
        new Float32Array(fromIR[0].length),
        new Float32Array(fromIR[1].length)
      ];
      
      // Apply different interpolation strategies for different time regions
      this.interpolateEarlyReflections(fromIR, toIR, interpolatedIR, progress);
      this.interpolateLateReverberation(fromIR, toIR, interpolatedIR, progress);
      
      return interpolatedIR;
    }
    
    private interpolateEarlyReflections(from: Float32Array[], to: Float32Array[], 
                                      result: Float32Array[], progress: number): void {
      // Linear interpolation for early reflections
      const earlyLength = Math.floor(0.08 * 44100); // First 80ms
      
      for (let channel = 0; channel < 2; channel++) {
        for (let i = 0; i < earlyLength; i++) {
          result[channel][i] = from[channel][i] * (1 - progress) + to[channel][i] * progress;
        }
      }
    }
    
    private interpolateLateReverberation(from: Float32Array[], to: Float32Array[], 
                                       result: Float32Array[], progress: number): void {
      // Crossfade for late reverberation to avoid artifacts
      const earlyLength = Math.floor(0.08 * 44100);
      const crossfadeLength = Math.floor(0.05 * 44100); // 50ms crossfade
      
      for (let channel = 0; channel < 2; channel++) {
        for (let i = earlyLength; i < result[channel].length; i++) {
          const crossfadeProgress = Math.min(1, (i - earlyLength) / crossfadeLength);
          const weight = this.applyCrossfadeWindow(crossfadeProgress, progress);
          result[channel][i] = from[channel][i] * (1 - weight) + to[channel][i] * weight;
        }
      }
    }
    
    private applyCrossfadeWindow(crossfadeProgress: number, interpolationProgress: number): number
  }
  ```
- **Features Required:**
  - Smooth IR transitions
  - Artifact-free crossfading
  - Time-domain interpolation
  - Perceptually optimized blending
- **Files:** `src/audio/ir-interpolator.ts` (new)

### 2. Enhance Multi-Source Audio Support
**Goal:** Support multiple simultaneous audio sources with individual spatial processing.

#### 2.1 Create Audio Source Manager
- **Description:** Manage multiple audio sources and their spatial properties
- **Implementation Details:**
  ```typescript
  class AudioSourceManager {
    private sources: Map<string, AudioSource> = new Map();
    private convolutionNodes: Map<string, ConvolverNode> = new Map();
    private mixerNode: GainNode;
    
    public addAudioSource(id: string, audioBuffer: AudioBuffer, position: vec3): AudioSource {
      const source = new AudioSource(id, audioBuffer, position);
      this.sources.set(id, source);
      
      // Create dedicated convolution chain for this source
      this.createConvolutionChain(source);
      
      return source;
    }
    
    public updateSourcePosition(id: string, newPosition: vec3): void {
      const source = this.sources.get(id);
      if (source) {
        source.setPosition(newPosition);
        // Trigger IR update for this source
        this.irUpdateManager.requestIRUpdate(source, this.listener, UpdatePriority.HIGH);
      }
    }
    
    public playSource(id: string, loop: boolean = false): void {
      const source = this.sources.get(id);
      if (source) {
        source.play(loop);
      }
    }
    
    private createConvolutionChain(source: AudioSource): void {
      const convolver = this.audioContext.createConvolver();
      const gainNode = this.audioContext.createGain();
      
      // Connect: Source → Convolver → Gain → Mixer
      source.connect(convolver);
      convolver.connect(gainNode);
      gainNode.connect(this.mixerNode);
      
      this.convolutionNodes.set(source.id, convolver);
    }
    
    public updateSourceIR(id: string, newIR: Float32Array[]): void
    public removeSource(id: string): void
    public setSourceVolume(id: string, volume: number): void
  }
  
  class AudioSource {
    public id: string;
    public position: vec3;
    public audioBuffer: AudioBuffer;
    private sourceNode: AudioBufferSourceNode | null = null;
    
    constructor(id: string, audioBuffer: AudioBuffer, position: vec3)
    public setPosition(position: vec3): void
    public play(loop: boolean = false): void
    public stop(): void
    public connect(destination: AudioNode): void
  }
  ```
- **Features Required:**
  - Multiple source management
  - Individual convolution chains
  - Dynamic source addition/removal
  - Position-based IR updates
- **Files:** `src/audio/audio-source-manager.ts` (new)

#### 2.2 Implement Source Prioritization
- **Description:** Prioritize processing for most important audio sources
- **Implementation Details:**
  ```typescript
  class SourcePrioritizer {
    public prioritizeSources(sources: AudioSource[], listener: Listener): PrioritizedSource[] {
      return sources.map(source => ({
        source,
        priority: this.calculatePriority(source, listener)
      })).sort((a, b) => b.priority - a.priority);
    }
    
    private calculatePriority(source: AudioSource, listener: Listener): number {
      let priority = 0;
      
      // Distance factor (closer = higher priority)
      const distance = vec3.distance(source.position, listener.position);
      priority += Math.max(0, 100 - distance * 10);
      
      // Volume factor
      priority += source.volume * 50;
      
      // Activity factor (playing sources get higher priority)
      if (source.isPlaying) priority += 25;
      
      // Recent movement factor
      if (source.hasRecentMovement()) priority += 15;
      
      return priority;
    }
  }
  
  interface PrioritizedSource {
    source: AudioSource;
    priority: number;
  }
  ```
- **Features Required:**
  - Distance-based prioritization
  - Volume-based weighting
  - Activity consideration
  - Movement detection
- **Files:** `src/audio/source-prioritizer.ts` (new)

### 3. Optimize Real-time Performance
**Goal:** Ensure smooth real-time operation without audio dropouts or frame rate issues.

#### 3.1 Implement Asynchronous IR Generation
- **Description:** Move IR generation to web workers to avoid blocking main thread
- **Implementation Details:**
  ```typescript
  class AsyncIRGenerator {
    private workers: Worker[] = [];
    private workerQueue: IRGenerationTask[] = [];
    private maxWorkers: number = navigator.hardwareConcurrency || 4;
    
    constructor() {
      this.initializeWorkers();
    }
    
    public async generateIRAsync(source: AudioSource, listener: Listener, 
                                room: Room): Promise<Float32Array[]> {
      return new Promise((resolve, reject) => {
        const task: IRGenerationTask = {
          id: this.generateTaskId(),
          source: source.serialize(),
          listener: listener.serialize(),
          room: room.serialize(),
          resolve,
          reject,
          timestamp: Date.now()
        };
        
        this.workerQueue.push(task);
        this.processQueue();
      });
    }
    
    private initializeWorkers(): void {
      for (let i = 0; i < this.maxWorkers; i++) {
        const worker = new Worker('/workers/ir-generation-worker.js');
        worker.onmessage = this.handleWorkerMessage.bind(this);
        this.workers.push(worker);
      }
    }
    
    private processQueue(): void {
      const availableWorker = this.workers.find(w => !w.busy);
      const nextTask = this.workerQueue.shift();
      
      if (availableWorker && nextTask) {
        availableWorker.busy = true;
        availableWorker.postMessage(nextTask);
      }
    }
    
    private handleWorkerMessage(event: MessageEvent): void
  }
  
  interface IRGenerationTask {
    id: string;
    source: SerializedAudioSource;
    listener: SerializedListener;
    room: SerializedRoom;
    resolve: (ir: Float32Array[]) => void;
    reject: (error: Error) => void;
    timestamp: number;
  }
  ```
- **Features Required:**
  - Web worker management
  - Task queuing and distribution
  - Serialization for worker communication
  - Error handling and timeouts
- **Files:** 
  - `src/audio/async-ir-generator.ts` (new)
  - `public/workers/ir-generation-worker.js` (new)

#### 3.2 Create Performance Monitor
- **Description:** Monitor system performance and adjust quality dynamically
- **Implementation Details:**
  ```typescript
  class PerformanceMonitor {
    private frameTimeHistory: number[] = [];
    private audioDropoutCount: number = 0;
    private memoryUsage: number = 0;
    
    public startMonitoring(): void {
      // Monitor frame times
      this.monitorFrameTimes();
      
      // Monitor audio performance
      this.monitorAudioPerformance();
      
      // Monitor memory usage
      this.monitorMemoryUsage();
    }
    
    public getPerformanceMetrics(): PerformanceMetrics {
      return {
        averageFrameTime: this.calculateAverageFrameTime(),
        audioDropouts: this.audioDropoutCount,
        memoryUsage: this.memoryUsage,
        cpuUsage: this.estimateCPUUsage(),
        recommendation: this.generateRecommendation()
      };
    }
    
    private monitorFrameTimes(): void {
      let lastTime = performance.now();
      
      const measureFrame = () => {
        const currentTime = performance.now();
        const frameTime = currentTime - lastTime;
        
        this.frameTimeHistory.push(frameTime);
        if (this.frameTimeHistory.length > 60) {
          this.frameTimeHistory.shift();
        }
        
        lastTime = currentTime;
        requestAnimationFrame(measureFrame);
      };
      
      requestAnimationFrame(measureFrame);
    }
    
    private generateRecommendation(): QualityRecommendation
  }
  
  interface PerformanceMetrics {
    averageFrameTime: number;
    audioDropouts: number;
    memoryUsage: number;
    cpuUsage: number;
    recommendation: QualityRecommendation;
  }
  
  interface QualityRecommendation {
    maxSources: number;
    irUpdateFrequency: number;
    rayCount: number;
    spatialResolution: number;
  }
  ```
- **Features Required:**
  - Real-time performance tracking
  - Quality recommendation system
  - Adaptive quality adjustment
  - Resource usage monitoring
- **Files:** `src/audio/performance-monitor.ts` (new)

### 4. Enhance User Interface and Controls
**Goal:** Provide intuitive controls for real-time auralization features.

#### 4.1 Create Audio Source Controls
- **Description:** UI for managing multiple audio sources
- **Implementation Details:**
  ```typescript
  class AudioSourceUI {
    private container: HTMLElement;
    private sourceManager: AudioSourceManager;
    
    public createSourceControls(): void {
      this.container = document.createElement('div');
      this.container.className = 'audio-source-controls';
      
      // Add source button
      const addButton = this.createAddSourceButton();
      this.container.appendChild(addButton);
      
      // Source list container
      const sourceList = document.createElement('div');
      sourceList.className = 'source-list';
      this.container.appendChild(sourceList);
      
      document.body.appendChild(this.container);
    }
    
    private createAddSourceButton(): HTMLElement {
      const button = document.createElement('button');
      button.textContent = 'Add Audio Source';
      button.onclick = () => this.showAddSourceDialog();
      return button;
    }
    
    private showAddSourceDialog(): void {
      // Create modal for adding new audio source
      const modal = this.createModal();
      
      // File input for audio
      const fileInput = this.createFileInput();
      
      // Position controls
      const positionControls = this.createPositionControls();
      
      // Add to modal and show
      modal.appendChild(fileInput);
      modal.appendChild(positionControls);
      this.showModal(modal);
    }
    
    public createSourceControlPanel(source: AudioSource): HTMLElement {
      const panel = document.createElement('div');
      panel.className = 'source-control-panel';
      
      // Position sliders
      const positionControls = this.createPositionSliders(source);
      
      // Volume control
      const volumeControl = this.createVolumeSlider(source);
      
      // Play/pause controls
      const playbackControls = this.createPlaybackControls(source);
      
      panel.appendChild(positionControls);
      panel.appendChild(volumeControl);
      panel.appendChild(playbackControls);
      
      return panel;
    }
  }
  ```
- **Features Required:**
  - Dynamic source addition
  - Position control sliders
  - Volume and playback controls
  - Visual feedback
- **Files:** `src/ui/audio-source-ui.ts` (new)

#### 4.2 Add Real-time Visualization
- **Description:** Visual feedback for real-time auralization
- **Implementation Details:**
  ```typescript
  class AuralizationVisualizer {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    
    public visualizeRealTimeAudio(sources: AudioSource[], listener: Listener): void {
      this.clearCanvas();
      
      // Draw room
      this.drawRoom();
      
      // Draw listener
      this.drawListener(listener);
      
      // Draw sources with activity indicators
      sources.forEach(source => {
        this.drawAudioSource(source);
        this.drawSoundWaves(source, listener);
      });
      
      // Draw performance metrics
      this.drawPerformanceMetrics();
    }
    
    private drawSoundWaves(source: AudioSource, listener: Listener): void {
      if (!source.isPlaying) return;
      
      const distance = vec3.distance(source.position, listener.position);
      const waveRadius = (Date.now() % 2000) / 2000 * distance;
      
      this.ctx.beginPath();
      this.ctx.arc(source.position[0], source.position[2], waveRadius, 0, Math.PI * 2);
      this.ctx.strokeStyle = `rgba(0, 255, 0, ${1 - waveRadius / distance})`;
      this.ctx.stroke();
    }
    
    private drawPerformanceMetrics(): void {
      const metrics = this.performanceMonitor.getPerformanceMetrics();
      
      this.ctx.fillStyle = 'white';
      this.ctx.font = '12px Arial';
      this.ctx.fillText(`Frame Time: ${metrics.averageFrameTime.toFixed(1)}ms`, 10, 20);
      this.ctx.fillText(`Audio Dropouts: ${metrics.audioDropouts}`, 10, 35);
      this.ctx.fillText(`Memory: ${(metrics.memoryUsage / 1024 / 1024).toFixed(1)}MB`, 10, 50);
    }
  }
  ```
- **Features Required:**
  - Real-time audio visualization
  - Performance metrics display
  - Sound wave animation
  - Source activity indicators
- **Files:** `src/ui/auralization-visualizer.ts` (new)

## Implementation Phases

### Phase 1: Dynamic IR System (Tasks 1.1-1.3)
**Duration:** 2-3 weeks
1. Create IR Update Manager
2. Implement Incremental IR Updates
3. Create IR Interpolation System

### Phase 2: Multi-Source Support (Tasks 2.1-2.2)
**Duration:** 2 weeks
4. Create Audio Source Manager
5. Implement Source Prioritization

### Phase 3: Performance Optimization (Tasks 3.1-3.2)
**Duration:** 2-3 weeks
6. Implement Asynchronous IR Generation
7. Create Performance Monitor

### Phase 4: User Interface (Tasks 4.1-4.2)
**Duration:** 1-2 weeks
8. Create Audio Source Controls
9. Add Real-time Visualization

## Expected Benefits

### Real-time Capabilities
- **Dynamic spatial audio**: Immediate response to position changes
- **Multiple source support**: Simultaneous processing of multiple audio sources
- **Smooth transitions**: Artifact-free updates during movement
- **Low latency**: Minimal delay between changes and audio output

### User Experience
- **Interactive control**: Real-time manipulation of audio sources
- **Visual feedback**: Clear indication of audio source positions and activity
- **Performance awareness**: Real-time performance monitoring
- **Quality adaptation**: Automatic adjustment based on system capabilities

### Technical Achievements
- **Efficient processing**: Optimized for real-time operation
- **Scalable architecture**: Support for varying numbers of sources
- **Robust performance**: Graceful handling of performance limitations
- **Professional quality**: Suitable for professional audio applications
