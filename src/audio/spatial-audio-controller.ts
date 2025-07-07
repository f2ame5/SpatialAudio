/**
 * Spatial Audio Controller - Manages the integration between 3D visualization and spatial audio
 */

import { vec3 } from 'gl-matrix';
import { audioManager } from './web-audio-manager';
import { createGainNode } from './audio-utils';
import { AcousticRaytracer, RaytracerConfig } from './acoustic-raytracer';
import { ImpulseResponseGenerator, IRGeneratorConfig } from './impulse-response-generator';
import { RoomAcoustics } from '../room/room-acoustics';
import { Room } from '../room/room';
import { Camera } from '../camera/camera';
import { Sphere } from '../objects/sphere';
import { performanceMonitor } from '../utils/performance-monitor';
import * as dat from 'dat.gui';

/**
 * Spatial audio configuration
 */
export interface SpatialAudioConfig {
    enabled: boolean;
    rayCount: number;
    maxBounces: number;
    impulseResponseLength: number;
    updateRate: number; // Hz
    visualizeRays: boolean;
    dryWetMix: number; // 0 = dry only, 1 = wet only
}

/**
 * Audio source configuration
 */
export interface AudioSourceConfig {
    file: string;
    gain: number;
    loop: boolean;
}

export class SpatialAudioController {
    private device: GPUDevice;
    private adapter: GPUAdapter;
    private room: Room;
    private roomAcoustics: RoomAcoustics;
    private camera: Camera;
    private sphere: Sphere;
    
    private raytracer: AcousticRaytracer | null = null;
    private irGenerator: ImpulseResponseGenerator;
    private config: SpatialAudioConfig;
    
    private initialized: boolean = false;
    private isProcessing: boolean = false;
    private lastUpdateTime: number = 0;
    private currentSourceId: string = 'test-source';
    private audioFiles = {
        'Snare': '/soundfile/snare.wav',
        'Loop': '/soundfile/loop.wav',
        'Top Loop': '/soundfile/top_loop.wav',
        'Test Tone (440Hz)': 'tone:440',
        'White Noise': 'noise:white'
    };
    private selectedAudioFile: string = 'Snare';
    
    // GUI elements
    private gui: dat.GUI;
    private audioFolder: dat.GUI | null = null;
    private statusElement: dat.GUIController | null = null;
    private acousticsInfo: any = null;
    private debugStats: any = null;
    
    constructor(
        device: GPUDevice,
        adapter: GPUAdapter,
        room: Room,
        camera: Camera,
        sphere: Sphere,
        gui: dat.GUI
    ) {
        this.device = device;
        this.adapter = adapter;
        this.room = room;
        this.camera = camera;
        this.sphere = sphere;
        this.gui = gui;
        
        // Initialize room acoustics
        this.roomAcoustics = new RoomAcoustics(room);
        
        // Default configuration
        this.config = {
            enabled: false,
            rayCount: 2048,
            maxBounces: 20,
            impulseResponseLength: 2.0,
            updateRate: 30,
            visualizeRays: false,
            dryWetMix: 0.7
        };
        
        // Initialize IR generator (will be properly initialized after audio context is created)
        const roomVolume = this.room.getDimensions()[0] * this.room.getDimensions()[1] * this.room.getDimensions()[2];
        this.irGenerator = new ImpulseResponseGenerator({
            sampleRate: 48000, // Default, will be updated
            maxLength: this.config.impulseResponseLength,
            roomVolume: roomVolume,
            usePoissonProcess: true,
            histogramResolution: 0.004 // 4ms bins
        });
        
        // Setup GUI
        this.setupGUI();
    }

    /**
     * Calculate appropriate listener radius based on room dimensions
     * The radius should be large enough to capture rays across the entire room
     */
    private calculateListenerRadius(): number {
        const dimensions = this.room.getDimensions();
        const [width, height, depth] = dimensions;

        // Calculate room diagonal - this ensures we can capture rays from any direction
        const roomDiagonal = Math.sqrt(width * width + height * height + depth * depth);

        // Use half the room diagonal as the listener radius
        // This ensures we capture rays from the entire room while not being excessive
        const listenerRadius = roomDiagonal * 0.5;

        console.log(`📐 Calculated listener radius: ${listenerRadius.toFixed(2)}m for room ${width}x${height}x${depth}m (diagonal: ${roomDiagonal.toFixed(2)}m)`);

        return listenerRadius;
    }

    /**
     * Initialize spatial audio system
     */
    async initialize(): Promise<void> {
        if (this.initialized) return;
        
        try {
            // Initialize audio manager
            await audioManager.initialize();
            
            // Update IR generator with correct sample rate
            const roomVolume = this.room.getDimensions()[0] * this.room.getDimensions()[1] * this.room.getDimensions()[2];
            this.irGenerator = new ImpulseResponseGenerator({
                sampleRate: audioManager.getSampleRate(),
                maxLength: this.config.impulseResponseLength,
                roomVolume: roomVolume,
                usePoissonProcess: true,
                histogramResolution: 0.004 // 4ms bins
            });
            
            // Create raytracer
            this.raytracer = new AcousticRaytracer(this.device, this.adapter, {
                maxRays: this.config.rayCount,
                maxBounces: this.config.maxBounces,
                impulseResponseLength: this.config.impulseResponseLength,
                sampleRate: audioManager.getSampleRate()
            });
            
            await this.raytracer.initialize();
            
            // Set room materials in raytracer
            this.raytracer.setMaterials(this.roomAcoustics.getMaterialsMap());

            // Set room bounds in raytracer
            this.raytracer.setRoomBounds(this.roomAcoustics.getRoomBounds());
            
            // Load test audio
            await this.loadTestAudio();
            
            this.initialized = true;
            this.updateStatus('Initialized');
            
        } catch (error) {
            console.error('Failed to initialize spatial audio:', error);
            this.updateStatus('Initialization failed');
            throw error;
        }
    }
    
    /**
     * Setup GUI controls
     */
    private setupGUI(): void {
        this.audioFolder = this.gui.addFolder('Spatial Audio');
        
        // Status display
        const status = { status: 'Not initialized' };
        this.statusElement = this.audioFolder.add(status, 'status').listen();
        
        // Enable/disable
        this.audioFolder.add(this.config, 'enabled')
            .name('Enable Spatial Audio')
            .onChange(async (value: boolean) => {
                if (value && !this.initialized) {
                    await this.initialize();
                }
                if (value) {
                    await audioManager.resume();
                } else {
                    await audioManager.suspend();
                }
            });
        
        // Ray tracing parameters
        const raytracingFolder = this.audioFolder.addFolder('Ray Tracing');
        raytracingFolder.add(this.config, 'rayCount', 512, 8192, 512)
            .name('Ray Count')
            .onChange((value: number) => {
                if (this.raytracer) {
                    this.raytracer.updateConfig({ maxRays: value });
                }
            });
        
        raytracingFolder.add(this.config, 'maxBounces', 5, 50, 1)
            .name('Max Bounces');
        
        raytracingFolder.add(this.config, 'impulseResponseLength', 0.5, 4.0, 0.1)
            .name('IR Length (s)')
            .onChange((value: number) => {
                if (this.initialized) {
                    // Recreate IR generator with new length
                    const roomVolume = this.room.getDimensions()[0] * this.room.getDimensions()[1] * this.room.getDimensions()[2];
                    this.irGenerator = new ImpulseResponseGenerator({
                        sampleRate: audioManager.getSampleRate(),
                        maxLength: value,
                        roomVolume: roomVolume,
                        usePoissonProcess: true,
                        histogramResolution: 0.004 // 4ms bins
                    });
                }
            });
        
        // Audio parameters
        const audioParamsFolder = this.audioFolder.addFolder('Audio Parameters');
        audioParamsFolder.add(this.config, 'updateRate', 10, 60, 10)
            .name('Update Rate (Hz)');
        
        audioParamsFolder.add(this.config, 'dryWetMix', 0, 1, 0.01)
            .name('Dry/Wet Mix')
            .onChange((value: number) => {
                this.updateDryWetMix(value);
            });
        
        // Master volume
        const masterVolume = { volume: audioManager.getMasterVolume() };
        audioParamsFolder.add(masterVolume, 'volume', 0, 1, 0.01)
            .name('Master Volume')
            .onChange((value: number) => {
                audioManager.setMasterVolume(value);
            });
        
        // Audio file selection
        audioParamsFolder.add(this, 'selectedAudioFile', Object.keys(this.audioFiles))
            .name('Audio File')
            .onChange(async () => {
                await this.loadSelectedAudio();
            });
        
        // Actions
        const actionsFolder = this.audioFolder.addFolder('Actions');
        
        // Generate impulse response button
        const actions = {
            generateIR: () => this.generateImpulseResponse(),
            playTestSound: () => this.playTestSound(),
            stopSound: () => this.stopSound(),
            exportIR: () => this.exportImpulseResponse()
        };
        
        actionsFolder.add(actions, 'generateIR').name('Generate IR');
        actionsFolder.add(actions, 'playTestSound').name('Play Test Sound');
        actionsFolder.add(actions, 'stopSound').name('Stop Sound');
        actionsFolder.add(actions, 'exportIR').name('Export IR as WAV');
        
        // Room acoustics info
        const acousticsFolder = this.audioFolder.addFolder('Room Acoustics');
        this.acousticsInfo = {
            rt60: '0.00 s',
            volume: '0 m³',
            surfaceArea: '0 m²'
        };
        
        acousticsFolder.add(this.acousticsInfo, 'rt60').name('RT60').listen();
        acousticsFolder.add(this.acousticsInfo, 'volume').name('Volume').listen();
        acousticsFolder.add(this.acousticsInfo, 'surfaceArea').name('Surface Area').listen();
        
        // Update acoustics info
        this.updateAcousticsInfo();

        // Debug section
        const debugFolder = this.audioFolder.addFolder('🔍 Debug & Monitoring');

        // Real-time statistics
        this.debugStats = {
            currentFrame: 0,
            raysGenerated: 0,
            raysActive: 0,
            raysCollected: 0,
            totalEnergy: 0,
            collectedEnergy: 0,
            listenerRadius: 0,
            averageArrivalTime: 0,
            performanceMs: 0,
            errors: 0,
            warnings: 0
        };

        debugFolder.add(this.debugStats, 'currentFrame').name('Current Frame').listen();
        debugFolder.add(this.debugStats, 'raysGenerated').name('Rays Generated').listen();
        debugFolder.add(this.debugStats, 'raysActive').name('Rays Active').listen();
        debugFolder.add(this.debugStats, 'raysCollected').name('Rays Collected').listen();
        debugFolder.add(this.debugStats, 'totalEnergy').name('Total Energy').listen();
        debugFolder.add(this.debugStats, 'collectedEnergy').name('Collected Energy').listen();
        debugFolder.add(this.debugStats, 'listenerRadius').name('Listener Radius (m)').listen();
        debugFolder.add(this.debugStats, 'averageArrivalTime').name('Avg Arrival Time (ms)').listen();
        debugFolder.add(this.debugStats, 'performanceMs').name('Performance (ms)').listen();
        debugFolder.add(this.debugStats, 'errors').name('Errors').listen();
        debugFolder.add(this.debugStats, 'warnings').name('Warnings').listen();

        // Debug controls
        const debugControls = {
            enableDetailedLogging: true,
            enableRayValidation: true,
            enableEnergyTracking: true,
            logInterval: 1,
            maxLoggedRays: 10,
            resetStats: () => this.resetDebugStats(),
            exportDebugData: () => this.exportDebugData()
        };

        debugFolder.add(debugControls, 'enableDetailedLogging').name('Detailed Logging')
            .onChange((value: boolean) => {
                if (this.raytracer) {
                    this.raytracer.setDebugConfig({ enableDetailedLogging: value });
                }
            });

        debugFolder.add(debugControls, 'enableRayValidation').name('Ray Validation')
            .onChange((value: boolean) => {
                if (this.raytracer) {
                    this.raytracer.setDebugConfig({ enableRayValidation: value });
                }
            });

        debugFolder.add(debugControls, 'enableEnergyTracking').name('Energy Tracking')
            .onChange((value: boolean) => {
                if (this.raytracer) {
                    this.raytracer.setDebugConfig({ enableEnergyTracking: value });
                }
            });

        debugFolder.add(debugControls, 'logInterval', 1, 30, 1).name('Log Interval (frames)')
            .onChange((value: number) => {
                if (this.raytracer) {
                    this.raytracer.setDebugConfig({ logInterval: value });
                }
            });

        debugFolder.add(debugControls, 'maxLoggedRays', 1, 50, 1).name('Max Logged Rays')
            .onChange((value: number) => {
                if (this.raytracer) {
                    this.raytracer.setDebugConfig({ maxLoggedRays: value });
                }
            });

        debugFolder.add(debugControls, 'resetStats').name('Reset Statistics');
        debugFolder.add(debugControls, 'exportDebugData').name('Export Debug Data');

        this.audioFolder.open();
        debugFolder.open(); // Open debug folder by default
    }
    
    /**
     * Update status display
     */
    private updateStatus(status: string): void {
        if (this.statusElement) {
            (this.statusElement as any).object.status = status;
        }
    }

    /**
     * Update debug statistics display
     */
    private updateDebugStats(): void {
        if (!this.debugStats || !this.raytracer) return;

        const stats = this.raytracer.getDebugStats();

        this.debugStats.currentFrame = stats.currentFrame;
        this.debugStats.raysGenerated = stats.raysGenerated;
        this.debugStats.raysActive = stats.raysActive;
        this.debugStats.raysCollected = stats.raysCollected;
        this.debugStats.totalEnergy = parseFloat(stats.totalEnergyGenerated.toFixed(4));
        this.debugStats.collectedEnergy = parseFloat(stats.totalEnergyCollected.toFixed(4));
        this.debugStats.listenerRadius = parseFloat(stats.collectionRadius.toFixed(2));
        this.debugStats.averageArrivalTime = parseFloat((stats.averageArrivalTime * 1000).toFixed(2)); // Convert to ms
        this.debugStats.performanceMs = parseFloat(stats.totalTime.toFixed(2));
        this.debugStats.errors = stats.errors.length;
        this.debugStats.warnings = stats.warnings.length;
    }

    /**
     * Reset debug statistics
     */
    private resetDebugStats(): void {
        if (this.raytracer) {
            // Reset the raytracer's debug stats
            this.raytracer.setDebugConfig({
                enableDetailedLogging: true,
                enableRayValidation: true,
                enableEnergyTracking: true
            });
        }
        console.log('🔄 Debug statistics reset');
    }

    /**
     * Export debug data to JSON
     */
    private exportDebugData(): void {
        if (!this.raytracer) {
            console.warn('No raytracer available for debug export');
            return;
        }

        const debugData = {
            timestamp: new Date().toISOString(),
            roomDimensions: this.room.getDimensions(),
            listenerRadius: this.calculateListenerRadius(),
            raytracerStats: this.raytracer.getDebugStats(),
            config: this.config
        };

        const dataStr = JSON.stringify(debugData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);

        const link = document.createElement('a');
        link.href = url;
        link.download = `raytracing-debug-${Date.now()}.json`;
        link.click();

        URL.revokeObjectURL(url);
        console.log('📊 Debug data exported');
    }

    /**
     * Load test audio files
     */
    private async loadTestAudio(): Promise<void> {
        await this.loadSelectedAudio();
    }
    
    /**
     * Load selected audio file
     */
    private async loadSelectedAudio(): Promise<void> {
        try {
            // Stop current playback if any
            if (audioManager.isSourcePlaying(this.currentSourceId)) {
                audioManager.stopSource(this.currentSourceId);
            }
            
            const audioPath = this.audioFiles[this.selectedAudioFile as keyof typeof this.audioFiles];
            
            // Check if it's a generated sound
            if (audioPath.startsWith('tone:') || audioPath.startsWith('noise:')) {
                await this.loadGeneratedSound(audioPath);
            } else {
                // Load selected sound file
                await audioManager.loadAudioSource(
                    this.currentSourceId,
                    audioPath,
                    {
                        loop: this.selectedAudioFile === 'Loop' || this.selectedAudioFile === 'Top Loop',
                        gain: 1.0,
                        useSpatialAudio: true,
                        position: { x: 0, y: 0, z: 0 }
                    }
                );
            }
            
            this.updateStatus(`Loaded: ${this.selectedAudioFile}`);
        } catch (error) {
            console.error('Failed to load audio:', error);
            this.updateStatus('Failed to load audio');
        }
    }
    
    /**
     * Load generated sound (tone or noise)
     */
    private async loadGeneratedSound(type: string): Promise<void> {
        const context = audioManager.getContext();
        const sampleRate = context.sampleRate;
        const duration = 2.0; // 2 seconds
        const buffer = context.createBuffer(1, sampleRate * duration, sampleRate);
        const data = buffer.getChannelData(0);
        
        if (type === 'tone:440') {
            // Generate 440Hz sine wave
            const frequency = 440;
            for (let i = 0; i < data.length; i++) {
                data[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 0.3;
            }
        } else if (type === 'noise:white') {
            // Generate white noise
            for (let i = 0; i < data.length; i++) {
                data[i] = (Math.random() * 2 - 1) * 0.2;
            }
        }
        
        // Remove existing source if it exists
        if ((audioManager as any).sources.has(this.currentSourceId)) {
            const existingSource = (audioManager as any).sources.get(this.currentSourceId);
            if (existingSource.node && existingSource.isPlaying) {
                existingSource.node.stop();
            }
            (audioManager as any).sources.delete(this.currentSourceId);
        }
        
        // Create a source from the buffer
        const gainNode = createGainNode(context, 1.0);
        
        // Connect directly to master gain (spatial audio comes from convolution)
        const masterGain = (audioManager as any).masterGain;
        if (masterGain) {
            gainNode.connect(masterGain);
        } else {
            gainNode.connect(context.destination);
        }
        
        const source = {
            id: this.currentSourceId,
            buffer: buffer,
            gainNode: gainNode,
            pannerNode: undefined,
            loop: false,
            isPlaying: false
        };
        
        // Store in audio manager's sources
        (audioManager as any).sources.set(this.currentSourceId, source);
    }
    
    /**
     * Generate impulse response
     */
    async generateImpulseResponse(): Promise<void> {
        if (!this.initialized || !this.raytracer || this.isProcessing) {
            return;
        }
        
        this.isProcessing = true;
        this.updateStatus('Generating IR...');
        
        try {
            performanceMonitor.startFrame();
            
            // Clear previous IR
            this.irGenerator.clear();
            
            // Get positions
            const sourcePos = this.sphere.getPosition();
            const listenerPos = this.camera.getPosition();
            const listenerForward = this.camera.getForward();
            const listenerUp = this.camera.getUp();
            
            // Source position is handled by ray tracing, not panner nodes
            
            // Update listener position
            audioManager.updateListenerTransform({
                position: { x: listenerPos[0], y: listenerPos[1], z: listenerPos[2] },
                forward: { x: listenerForward[0], y: listenerForward[1], z: listenerForward[2] },
                up: { x: listenerUp[0], y: listenerUp[1], z: listenerUp[2] }
            });
            
            // Calculate appropriate listener radius based on room size
            const listenerRadius = this.calculateListenerRadius();

            // Trace rays
            const irData = await this.raytracer.traceRays(
                sourcePos,
                {
                    position: listenerPos,
                    radius: listenerRadius,
                    forward: listenerForward,
                    up: listenerUp
                },
                0.016 // Frame time
            );

            // Update debug statistics
            this.updateDebugStats();

            // Use real raytracing to generate impulse response
            // (sourcePos and listenerPos are already defined above)

            const listenerConfig = {
                position: listenerPos,
                radius: listenerRadius,
                forward: listenerForward,
                up: listenerUp
            };

            // Run raytracing simulation
            console.log('Starting raytracing with source:', sourcePos, 'listener:', listenerConfig);

            let rayData: Float32Array;
            try {
                rayData = await this.raytracer.traceRays(
                    sourcePos,
                    listenerConfig,
                    0.016 // 16ms delta time
                );

                console.log(`Raytracing completed, got ${rayData.length} samples`);
                console.log('First 10 samples:', Array.from(rayData.slice(0, 10)));
                console.log('Max value in rayData:', Math.max(...rayData));
                console.log('Non-zero samples:', Array.from(rayData).filter(x => x > 0.001).length);
            } catch (error) {
                console.error('GPU Raytracing failed:', error);

                // Check if it's a device lost error
                if (error.message.includes('device lost')) {
                    console.warn('GPU device lost - attempting to reinitialize raytracer');
                    try {
                        // Try to reinitialize the raytracer
                        await this.raytracer.initialize();
                        console.log('Raytracer reinitialized successfully');
                    } catch (reinitError) {
                        console.error('Failed to reinitialize raytracer:', reinitError);
                    }
                }

                console.log('Falling back to test IR generation');
                // Fall back to test IR generation
                this.generateTestImpulseResponse();
                const irBuffer = this.irGenerator.generateImpulseResponse();

                // Apply IR to convolver
                await audioManager.setImpulseResponse(irBuffer);
                audioManager.createConvolverForSource(this.currentSourceId);

                const irStats = this.irGenerator.calculateStatistics();
                this.updateStatus(`IR Generated (CPU fallback) - RT60: ${irStats.rt60.toFixed(2)}s`);
                return;
            }

            // Clear previous IR data
            this.irGenerator.clear();

            // Convert ray data to impulse response
            const sampleRate = audioManager.getSampleRate();
            let processedSamples = 0;
            let totalEnergy = 0;

            for (let i = 0; i < rayData.length; i++) {
                if (rayData[i] > 0.001) { // Only process significant energy
                    const time = i / sampleRate; // Convert sample index to time
                    this.irGenerator.addRayContribution({
                        timeBin: time,
                        energy: rayData[i],
                        phase: 0, // Simplified - could add phase information
                        frequencyEnergy: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]).map(v => v * rayData[i]),
                        direction: vec3.fromValues(0, 0, -1) // Simplified direction
                    });
                    processedSamples++;
                    totalEnergy += rayData[i];
                }
            }

            console.log(`🎯 Processed ${processedSamples}/${rayData.length} ray samples, total energy: ${totalEnergy.toFixed(6)}`);

            // Add test energy data to verify histogram is working
            this.irGenerator.addTestEnergyData();

            // Debug: Show energy histogram statistics
            const histogramStats = this.irGenerator.getEnergyHistogramStatistics();
            const histogramData = this.irGenerator.getEnergyHistogramData();

            console.log('🔋 Energy Histogram Statistics:', histogramStats);
            console.log('📊 Energy Histogram Data (first 10 bins):', histogramData.slice(0, 10));

            // Create a simple ASCII visualization of the energy histogram
            if (histogramData.length > 0) {
                console.log('📈 Energy Histogram Visualization:');
                const maxEnergy = Math.max(...histogramData.map(d => d.energy));
                const scale = 50; // ASCII bar width

                histogramData.slice(0, 20).forEach((bin, i) => {
                    const barLength = Math.round((bin.energy / maxEnergy) * scale);
                    const bar = '█'.repeat(barLength) + '░'.repeat(scale - barLength);
                    console.log(`${bin.time.toFixed(3)}s: ${bar} ${bin.energy.toFixed(4)} (${bin.sampleCount} rays)`);
                });
            }

            // Generate simple impulse response from energy histogram
            const simpleIR = this.irGenerator.generateSimpleImpulseResponse();
            console.log('📈 Simple IR Stats:', {
                length: simpleIR.length,
                nonZeroSamples: Array.from(simpleIR).filter(x => Math.abs(x) > 0.001).length,
                maxValue: Math.max(...simpleIR.map(Math.abs)),
                totalEnergy: Array.from(simpleIR).reduce((sum, x) => sum + Math.abs(x), 0)
            });

            // Generate audio buffer from IR data
            const irBuffer = this.irGenerator.generateImpulseResponse();

            // Apply IR to convolver
            await audioManager.setImpulseResponse(irBuffer);
            
            // Create convolver for source if not exists
            audioManager.createConvolverForSource(this.currentSourceId);
            
            performanceMonitor.endFrame();
            const metrics = performanceMonitor.getMetrics();
            
            // Calculate IR statistics
            const irStats = this.irGenerator.calculateStatistics();
            
            this.updateStatus(
                `IR Generated - RT60: ${irStats.rt60.toFixed(2)}s, ` +
                `Time: ${metrics.frameTime.toFixed(1)}ms`
            );
            
        } catch (error) {
            console.error('Failed to generate impulse response:', error);
            this.updateStatus('IR generation failed');
        } finally {
            this.isProcessing = false;
        }
    }
    
    /**
     * Generate a simple test impulse response
     */
    private generateTestImpulseResponse(): void {
        // Clear previous data
        this.irGenerator.clear();
        
        // Generate some test impulse response data
        const sampleRate = audioManager.getSampleRate();
        const props = this.roomAcoustics.calculateAcousticProperties();
        
        // Direct sound (arrives immediately)
        this.irGenerator.addRayContribution({
            timeBin: 0.0, // Time in seconds
            energy: 1.0,
            phase: 0,
            frequencyEnergy: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]),
            direction: vec3.create()
        });
        
        // Generate some early reflections
        const numEarlyReflections = 20;
        for (let i = 0; i < numEarlyReflections; i++) {
            const time = (i + 1) * 0.005 + Math.random() * 0.01; // 5-15ms spacing
            const energy = Math.pow(0.8, i + 1) * (0.5 + Math.random() * 0.5);

            this.irGenerator.addRayContribution({
                timeBin: time, // Time in seconds, not pre-converted
                energy: energy,
                phase: Math.random() * Math.PI * 2,
                frequencyEnergy: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]).map(v => v * energy),
                direction: vec3.fromValues(
                    Math.random() * 2 - 1,
                    Math.random() * 2 - 1,
                    Math.random() * 2 - 1
                )
            });
        }
        
        // Generate late reverberation tail
        const rt60 = props.rt60;
        const numLateReflections = Math.floor(rt60 * sampleRate / 100); // Sparse late reflections

        for (let i = 0; i < numLateReflections; i++) {
            const time = 0.05 + i * (rt60 / numLateReflections) + Math.random() * 0.01;
            const decay = Math.exp(-3 * time / rt60); // Exponential decay
            const energy = decay * (0.1 + Math.random() * 0.2);

            this.irGenerator.addRayContribution({
                timeBin: time, // Time in seconds, not pre-converted
                energy: energy,
                phase: Math.random() * Math.PI * 2,
                frequencyEnergy: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]).map(v => v * energy),
                direction: vec3.fromValues(
                    Math.random() * 2 - 1,
                    Math.random() * 2 - 1,
                    Math.random() * 2 - 1
                )
            });
        }
    }
    
    /**
     * Play test sound
     */
    playTestSound(): void {
        if (!this.initialized) return;
        
        try {
            audioManager.playSource(this.currentSourceId);
            this.updateStatus('Playing sound');
        } catch (error) {
            console.error('Failed to play sound:', error);
            this.updateStatus('Playback failed');
        }
    }
    
    /**
     * Stop sound
     */
    stopSound(): void {
        audioManager.stopSource(this.currentSourceId);
        this.updateStatus('Stopped');
    }
    
    /**
     * Export impulse response as WAV
     */
    exportImpulseResponse(): void {
        const wavData = this.irGenerator.exportAsWAV();
        const blob = new Blob([wavData], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `impulse_response_${Date.now()}.wav`;
        a.click();
        
        URL.revokeObjectURL(url);
        this.updateStatus('IR exported');
    }
    
    /**
     * Update dry/wet mix
     */
    private updateDryWetMix(value: number): void {
        const source = (audioManager as any).sources.get(this.currentSourceId);
        if (source && source.dryGain && source.wetGain) {
            // value: 0 = fully dry, 1 = fully wet
            source.dryGain.gain.value = 1.0 - value;
            source.wetGain.gain.value = value;
        }
    }
    
    /**
     * Update acoustics info display
     */
    private updateAcousticsInfo(): void {
        if (!this.acousticsInfo) return;
        
        const props = this.roomAcoustics.calculateAcousticProperties();
        this.acousticsInfo.rt60 = `${props.rt60.toFixed(2)} s`;
        this.acousticsInfo.volume = `${this.room.getVolume().toFixed(1)} m³`;
        this.acousticsInfo.surfaceArea = `${this.room.getSurfaceArea().toFixed(1)} m²`;
    }
    
    /**
     * Update room (called when room dimensions change)
     */
    updateRoom(room: Room): void {
        this.room = room;
        this.roomAcoustics = new RoomAcoustics(room);

        if (this.raytracer) {
            this.raytracer.setMaterials(this.roomAcoustics.getMaterialsMap());
            this.raytracer.setRoomBounds(this.roomAcoustics.getRoomBounds());
        }

        // Update acoustics display
        this.updateAcousticsInfo();
    }
    
    /**
     * Update camera (for listener position)
     */
    updateCamera(camera: Camera): void {
        this.camera = camera;
    }
    
    /**
     * Update sphere (for source position)
     */
    updateSphere(sphere: Sphere): void {
        this.sphere = sphere;
    }
    
    /**
     * Frame update (for real-time processing)
     */
    update(deltaTime: number): void {
        if (!this.config.enabled || !this.initialized) return;
        
        // Check if we should update based on update rate
        const currentTime = performance.now() / 1000;
        const updateInterval = 1.0 / this.config.updateRate;
        
        if (currentTime - this.lastUpdateTime >= updateInterval) {
            this.lastUpdateTime = currentTime;
            
            // Update listener position in real-time
            const listenerPos = this.camera.getPosition();
            const listenerForward = this.camera.getForward();
            const listenerUp = this.camera.getUp();
            
            audioManager.updateListenerTransform({
                position: { x: listenerPos[0], y: listenerPos[1], z: listenerPos[2] },
                forward: { x: listenerForward[0], y: listenerForward[1], z: listenerForward[2] },
                up: { x: listenerUp[0], y: listenerUp[1], z: listenerUp[2] }
            });
            
            // Source position is handled by ray tracing impulse response
            // No need to update panner node positions
        }
    }
    
    /**
     * Dispose of resources
     */
    dispose(): void {
        this.raytracer?.dispose();
        audioManager.dispose();
    }
}
