/**
 * WebAudioManager - Central manager for Web Audio API context and nodes
 */

import { 
    createAudioContext, 
    updateListener, 
    createGainNode,
    createPannerNode,
    loadAudioFile,
    SAMPLE_RATES
} from './audio-utils';
import { AudioFileLoader, LoadOptions } from './audio-file-loader';

export interface AudioSource {
    id: string;
    buffer: AudioBuffer;
    node?: AudioBufferSourceNode;
    gainNode: GainNode;
    pannerNode?: PannerNode;
    convolverNode?: ConvolverNode;
    loop: boolean;
    isPlaying: boolean;
}

export interface ListenerConfig {
    position: { x: number; y: number; z: number };
    forward: { x: number; y: number; z: number };
    up: { x: number; y: number; z: number };
}

export class WebAudioManager {
    private context: AudioContext | null = null;
    private masterGain: GainNode | null = null;
    private sources: Map<string, AudioSource> = new Map();
    private impulseResponseBuffer: AudioBuffer | null = null;
    private initialized: boolean = false;
    private fileLoader: AudioFileLoader | null = null;

    /**
     * Initialize the Web Audio context
     */
    async initialize(): Promise<void> {
        if (this.initialized) {
            console.warn('WebAudioManager already initialized');
            return;
        }

        try {
            // Create audio context
            this.context = await createAudioContext();
            
            // Create master gain node
            this.masterGain = createGainNode(this.context, 1.0);
            this.masterGain.connect(this.context.destination);
            
            // Create file loader
            this.fileLoader = new AudioFileLoader(this.context);
            
            this.initialized = true;
            console.log('WebAudioManager initialized successfully');
            console.log(`Sample rate: ${this.context.sampleRate} Hz`);
        } catch (error) {
            console.error('Failed to initialize WebAudioManager:', error);
            throw error;
        }
    }

    /**
     * Get the audio context
     */
    getContext(): AudioContext {
        if (!this.context) {
            throw new Error('WebAudioManager not initialized');
        }
        return this.context;
    }

    /**
     * Load an audio file and create a source
     */
    async loadAudioSource(
        id: string, 
        url: string, 
        options: {
            loop?: boolean;
            gain?: number;
            position?: { x: number; y: number; z: number };
            useSpatialAudio?: boolean;
        } = {}
    ): Promise<AudioSource> {
        if (!this.context || !this.masterGain) {
            throw new Error('WebAudioManager not initialized');
        }

        // Remove existing source if it exists (for reloading)
        if (this.sources.has(id)) {
            const existingSource = this.sources.get(id)!;
            if (existingSource.node && existingSource.isPlaying) {
                existingSource.node.stop();
            }
            this.sources.delete(id);
        }

        try {
            // Load audio file
            const buffer = await loadAudioFile(this.context, url);
            
            // Create gain node
            const gainNode = createGainNode(this.context, options.gain || 1.0);
            
            // For spatial audio, we'll use convolution only (no panner)
            // The spatial information comes from the ray-traced impulse response
            gainNode.connect(this.masterGain);
            
            // Create source object
            const source: AudioSource = {
                id,
                buffer,
                gainNode,
                pannerNode: undefined, // We use convolution for spatial audio, not panning
                loop: options.loop || false,
                isPlaying: false
            };
            
            this.sources.set(id, source);
            console.log(`Loaded audio source '${id}' from ${url}`);
            
            return source;
        } catch (error) {
            console.error(`Failed to load audio source '${id}':`, error);
            throw error;
        }
    }

    /**
     * Play an audio source
     */
    playSource(id: string, when: number = 0): AudioBufferSourceNode {
        const source = this.sources.get(id);
        if (!source) {
            throw new Error(`Audio source '${id}' not found`);
        }

        if (!this.context) {
            throw new Error('WebAudioManager not initialized');
        }

        // Stop current playback if any
        if (source.node && source.isPlaying) {
            source.node.stop();
        }

        // Create new buffer source node
        const bufferSource = this.context.createBufferSource();
        bufferSource.buffer = source.buffer;
        bufferSource.loop = source.loop;
        
        // Connect to gain node
        bufferSource.connect(source.gainNode);
        
        // Start playback
        bufferSource.start(when);
        
        // Update source state
        source.node = bufferSource;
        source.isPlaying = true;
        
        // Handle playback end
        bufferSource.onended = () => {
            source.isPlaying = false;
            source.node = undefined;
        };
        
        return bufferSource;
    }

    /**
     * Stop an audio source
     */
    stopSource(id: string, when: number = 0): void {
        const source = this.sources.get(id);
        if (!source || !source.node || !source.isPlaying) {
            return;
        }

        source.node.stop(when);
        source.isPlaying = false;
        source.node = undefined;
    }

    /**
     * Update source position (for spatial audio)
     */
    updateSourcePosition(
        id: string, 
        position: { x: number; y: number; z: number }
    ): void {
        const source = this.sources.get(id);
        if (!source || !source.pannerNode) {
            return;
        }

        source.pannerNode.positionX.value = position.x;
        source.pannerNode.positionY.value = position.y;
        source.pannerNode.positionZ.value = position.z;
    }

    /**
     * Update source gain
     */
    updateSourceGain(id: string, gain: number, rampTime: number = 0): void {
        const source = this.sources.get(id);
        if (!source) {
            return;
        }

        if (!this.context) {
            return;
        }

        if (rampTime > 0) {
            source.gainNode.gain.linearRampToValueAtTime(
                gain, 
                this.context.currentTime + rampTime
            );
        } else {
            source.gainNode.gain.value = gain;
        }
    }

    /**
     * Update listener position and orientation
     */
    updateListenerTransform(config: ListenerConfig): void {
        if (!this.context) {
            return;
        }

        updateListener(
            this.context,
            config.position,
            config.forward,
            config.up
        );
    }

    /**
     * Set master volume
     */
    setMasterVolume(volume: number): void {
        if (!this.masterGain) {
            return;
        }
        this.masterGain.gain.value = Math.max(0, Math.min(1, volume));
    }

    /**
     * Get master volume
     */
    getMasterVolume(): number {
        return this.masterGain ? this.masterGain.gain.value : 0;
    }

    /**
     * Set impulse response for convolution reverb
     */
    async setImpulseResponse(buffer: AudioBuffer): Promise<void> {
        this.impulseResponseBuffer = buffer;
        
        // Update all sources with convolver nodes
        for (const source of this.sources.values()) {
            if (source.convolverNode) {
                source.convolverNode.buffer = buffer;
            }
        }
    }

    /**
     * Create a convolver node for a source
     */
    createConvolverForSource(id: string): ConvolverNode | null {
        const source = this.sources.get(id);
        if (!source || !this.context || !this.masterGain) {
            return null;
        }

        // If convolver already exists, just update the buffer
        if (source.convolverNode) {
            if (this.impulseResponseBuffer) {
                source.convolverNode.buffer = this.impulseResponseBuffer;
            }
            return source.convolverNode;
        }

        // Create convolver node
        const convolver = this.context.createConvolver();
        
        // Set impulse response if available
        if (this.impulseResponseBuffer) {
            convolver.buffer = this.impulseResponseBuffer;
        }
        
        // Create dry and wet gain nodes for mixing
        const dryGain = this.context.createGain();
        const wetGain = this.context.createGain();
        
        // Default mix: mostly wet for spatial audio
        dryGain.gain.value = 0.3;
        wetGain.gain.value = 0.7;
        
        // Reconnect audio graph with parallel dry/wet paths
        source.gainNode.disconnect();
        
        // Dry path: source -> dryGain -> masterGain
        source.gainNode.connect(dryGain);
        dryGain.connect(this.masterGain);
        
        // Wet path: source -> convolver -> wetGain -> masterGain
        source.gainNode.connect(convolver);
        convolver.connect(wetGain);
        wetGain.connect(this.masterGain);
        
        // Store convolver reference and gain nodes
        source.convolverNode = convolver;
        (source as any).dryGain = dryGain;
        (source as any).wetGain = wetGain;
        
        return convolver;
    }

    /**
     * Get current time from audio context
     */
    getCurrentTime(): number {
        return this.context ? this.context.currentTime : 0;
    }

    /**
     * Get sample rate
     */
    getSampleRate(): number {
        return this.context ? this.context.sampleRate : SAMPLE_RATES.DAT_QUALITY;
    }

    /**
     * Suspend audio context (pause all audio)
     */
    async suspend(): Promise<void> {
        if (this.context && this.context.state === 'running') {
            await this.context.suspend();
        }
    }

    /**
     * Resume audio context
     */
    async resume(): Promise<void> {
        if (this.context && this.context.state === 'suspended') {
            await this.context.resume();
        }
    }

    /**
     * Get audio context state
     */
    getState(): AudioContextState | 'uninitialized' {
        return this.context ? this.context.state : 'uninitialized';
    }

    /**
     * Cleanup and close audio context
     */
    async dispose(): Promise<void> {
        // Stop all sources
        for (const source of this.sources.values()) {
            if (source.node && source.isPlaying) {
                source.node.stop();
            }
        }
        
        // Clear sources
        this.sources.clear();
        
        // Close audio context
        if (this.context) {
            await this.context.close();
            this.context = null;
        }
        
        this.masterGain = null;
        this.impulseResponseBuffer = null;
        this.fileLoader = null;
        this.initialized = false;
        
        console.log('WebAudioManager disposed');
    }

    /**
     * Get list of loaded source IDs
     */
    getSourceIds(): string[] {
        return Array.from(this.sources.keys());
    }

    /**
     * Check if a source is playing
     */
    isSourcePlaying(id: string): boolean {
        const source = this.sources.get(id);
        return source ? source.isPlaying : false;
    }

    /**
     * Add audio file to loader (without loading)
     */
    addAudioFile(id: string, url: string): void {
        if (!this.fileLoader) {
            throw new Error('WebAudioManager not initialized');
        }
        this.fileLoader.addFile(id, url);
    }

    /**
     * Load audio file with progress tracking
     */
    async loadAudioFileWithProgress(
        id: string, 
        url: string,
        options: LoadOptions & {
            position?: { x: number; y: number; z: number };
            useSpatialAudio?: boolean;
            loop?: boolean;
            gain?: number;
        } = {}
    ): Promise<AudioSource> {
        if (!this.fileLoader || !this.context || !this.masterGain) {
            throw new Error('WebAudioManager not initialized');
        }

        // Add file to loader
        this.fileLoader.addFile(id, url);

        // Load with progress tracking
        const buffer = await this.fileLoader.loadFile(id, {
            priority: options.priority,
            onProgress: options.onProgress,
            onError: options.onError
        });

        // Create gain node
        const gainNode = createGainNode(this.context, options.gain || 1.0);
        
        // Create panner node if spatial audio is enabled
        let pannerNode: PannerNode | undefined;
        if (options.useSpatialAudio) {
            pannerNode = createPannerNode(this.context, options.position);
            gainNode.connect(pannerNode);
            pannerNode.connect(this.masterGain);
        } else {
            gainNode.connect(this.masterGain);
        }
        
        // Create source object
        const source: AudioSource = {
            id,
            buffer,
            gainNode,
            pannerNode,
            loop: options.loop || false,
            isPlaying: false
        };
        
        this.sources.set(id, source);
        return source;
    }

    /**
     * Preload multiple audio files
     */
    async preloadAudioFiles(
        files: Array<{ id: string; url: string }>,
        options: LoadOptions = {}
    ): Promise<void> {
        if (!this.fileLoader) {
            throw new Error('WebAudioManager not initialized');
        }

        // Add all files
        for (const file of files) {
            this.fileLoader.addFile(file.id, file.url);
        }

        // Load all files
        await this.fileLoader.loadFiles(
            files.map(f => f.id),
            options
        );
    }

    /**
     * Get file loading progress
     */
    getFileLoadingProgress(): number {
        return this.fileLoader ? this.fileLoader.getTotalProgress() : 0;
    }

    /**
     * Get file loader instance
     */
    getFileLoader(): AudioFileLoader | null {
        return this.fileLoader;
    }
}

// Export singleton instance
export const audioManager = new WebAudioManager();
