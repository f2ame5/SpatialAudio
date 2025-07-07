/**
 * Raytracing Integration Module
 * Integrates all raytracing components for complete spatial audio system
 */

import { vec3 } from 'gl-matrix';
import { AcousticRaytracer } from './acoustic-raytracer';
import { RaytracingDebugger } from '../debug/raytracing-debugger';
import { WebAudioManager } from './web-audio-manager';
import { ImpulseResponseGenerator } from './impulse-response-generator';
import { RayDistributionType } from './ray-types';
import { Room } from '../room/room';
import { RoomAcoustics } from '../room/room-acoustics';

export interface SpatialAudioConfig {
    raytracing: {
        maxRays: number;
        maxBounces: number;
        minEnergy: number;
        distributionType: RayDistributionType;
        updateRate: number; // Hz
    };
    audio: {
        sampleRate: number;
        impulseResponseLength: number; // seconds
        convolutionSize: number;
        enableHRTF: boolean;
    };
    performance: {
        targetFPS: number;
        adaptiveQuality: boolean;
        debugMode: boolean;
    };
}

export class SpatialAudioSystem {
    private device: GPUDevice;
    private canvas: HTMLCanvasElement;
    private room: Room;
    
    // Core components
    private raytracer: AcousticRaytracer;
    private audioManager: WebAudioManager;
    private irGenerator: ImpulseResponseGenerator;
    private debugger: RaytracingDebugger | null = null;
    
    // Configuration
    private config: SpatialAudioConfig;
    
    // State
    private initialized: boolean = false;
    private sourcePosition: vec3 = vec3.create();
    private listenerPosition: vec3 = vec3.create();
    private currentImpulseResponse: Float32Array | null = null;
    
    // Performance tracking
    private lastRaytracingUpdate: number = 0;
    private raytracingInterval: number = 100; // ms
    
    constructor(
        device: GPUDevice,
        canvas: HTMLCanvasElement,
        room: Room,
        config: Partial<SpatialAudioConfig> = {}
    ) {
        this.device = device;
        this.canvas = canvas;
        this.room = room;
        
        // Default configuration
        this.config = {
            raytracing: {
                maxRays: 2048,
                maxBounces: 20,
                minEnergy: 0.001,
                distributionType: RayDistributionType.UNIFORM_SPHERE,
                updateRate: 10
            },
            audio: {
                sampleRate: 48000,
                impulseResponseLength: 2.0,
                convolutionSize: 1024,
                enableHRTF: false
            },
            performance: {
                targetFPS: 60,
                adaptiveQuality: true,
                debugMode: false
            },
            ...config
        };
        
        this.raytracingInterval = 1000 / this.config.raytracing.updateRate;
    }

    /**
     * Calculate appropriate listener radius based on room dimensions
     */
    private calculateListenerRadius(): number {
        const dimensions = this.room.getDimensions();
        const [width, height, depth] = dimensions;

        // Calculate room diagonal to ensure we capture rays from entire room
        const roomDiagonal = Math.sqrt(width * width + height * height + depth * depth);

        // Use half the room diagonal as listener radius
        return roomDiagonal * 0.5;
    }

    /**
     * Initialize the spatial audio system
     */
    async initialize(): Promise<void> {
        if (this.initialized) {
            console.warn('SpatialAudioSystem already initialized');
            return;
        }
        
        try {
            // Initialize raytracer
            this.raytracer = new AcousticRaytracer(this.device, {
                maxRays: this.config.raytracing.maxRays,
                maxBounces: this.config.raytracing.maxBounces,
                minEnergy: this.config.raytracing.minEnergy,
                sampleRate: this.config.audio.sampleRate,
                impulseResponseLength: this.config.audio.impulseResponseLength
            });
            
            await this.raytracer.initialize();
            
            // Set room materials using RoomAcoustics
            const roomAcoustics = new RoomAcoustics(this.room);
            this.raytracer.setMaterials(roomAcoustics.getMaterialsMap());
            this.raytracer.setRoomBounds(roomAcoustics.getRoomBounds());
            
            // Initialize audio manager
            this.audioManager = new WebAudioManager();
            await this.audioManager.initialize();
            
            // Initialize impulse response generator
            this.irGenerator = new ImpulseResponseGenerator(
                this.config.audio.sampleRate,
                this.config.audio.impulseResponseLength
            );
            
            // Initialize debugger if enabled
            if (this.config.performance.debugMode) {
                this.debugger = new RaytracingDebugger(
                    this.device,
                    this.canvas,
                    this.raytracer
                );
                await this.debugger.initialize();
            }
            
            this.initialized = true;
            console.log('Spatial Audio System initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize Spatial Audio System:', error);
            throw error;
        }
    }
    
    /**
     * Update the spatial audio system
     */
    async update(deltaTime: number): Promise<void> {
        if (!this.initialized) return;
        
        const currentTime = performance.now();
        
        // Update raytracing at specified rate
        if (currentTime - this.lastRaytracingUpdate >= this.raytracingInterval) {
            await this.updateRaytracing();
            this.lastRaytracingUpdate = currentTime;
        }
        
        // Update debugger
        if (this.debugger) {
            await this.debugger.update(deltaTime);
        }
        
        // Apply adaptive quality if enabled
        if (this.config.performance.adaptiveQuality) {
            this.adjustQualityBasedOnPerformance(deltaTime);
        }
    }
    
    /**
     * Update raytracing and generate new impulse response
     */
    private async updateRaytracing(): Promise<void> {
        try {
            // Run raytracing simulation
            const listenerConfig = {
                position: this.listenerPosition,
                radius: this.calculateListenerRadius(), // Dynamic radius based on room size
                forward: vec3.fromValues(0, 0, -1),
                up: vec3.fromValues(0, 1, 0)
            };
            
            const rayData = await this.raytracer.traceRays(
                this.sourcePosition,
                listenerConfig,
                this.raytracingInterval / 1000
            );
            
            // Generate impulse response from ray data
            this.currentImpulseResponse = this.irGenerator.generateFromRayData(rayData);
            
            // Update audio convolution
            if (this.currentImpulseResponse && this.audioManager) {
                this.audioManager.updateImpulseResponse(this.currentImpulseResponse);
            }
            
        } catch (error) {
            console.error('Error updating raytracing:', error);
        }
    }
    
    /**
     * Adjust quality based on performance
     */
    private adjustQualityBasedOnPerformance(deltaTime: number): void {
        const targetFrameTime = 1000 / this.config.performance.targetFPS;
        const currentFrameTime = deltaTime * 1000;
        
        if (currentFrameTime > targetFrameTime * 1.2) {
            // Performance is poor - reduce quality
            if (this.config.raytracing.maxRays > 512) {
                this.config.raytracing.maxRays = Math.floor(this.config.raytracing.maxRays * 0.9);
                this.raytracer.updateConfig({ maxRays: this.config.raytracing.maxRays });
            }
            
            if (this.config.raytracing.updateRate > 5) {
                this.config.raytracing.updateRate = Math.max(5, this.config.raytracing.updateRate - 1);
                this.raytracingInterval = 1000 / this.config.raytracing.updateRate;
            }
            
        } else if (currentFrameTime < targetFrameTime * 0.8) {
            // Performance is good - increase quality
            if (this.config.raytracing.maxRays < 4096) {
                this.config.raytracing.maxRays = Math.floor(this.config.raytracing.maxRays * 1.1);
                this.raytracer.updateConfig({ maxRays: this.config.raytracing.maxRays });
            }
            
            if (this.config.raytracing.updateRate < 30) {
                this.config.raytracing.updateRate = Math.min(30, this.config.raytracing.updateRate + 1);
                this.raytracingInterval = 1000 / this.config.raytracing.updateRate;
            }
        }
    }
    
    /**
     * Set sound source position
     */
    setSourcePosition(position: vec3): void {
        vec3.copy(this.sourcePosition, position);
    }
    
    /**
     * Set listener position
     */
    setListenerPosition(position: vec3): void {
        vec3.copy(this.listenerPosition, position);
    }
    
    /**
     * Play audio with spatial processing
     */
    async playAudio(audioBuffer: AudioBuffer): Promise<void> {
        if (!this.audioManager) {
            throw new Error('Audio manager not initialized');
        }
        
        await this.audioManager.playWithConvolution(audioBuffer);
    }
    
    /**
     * Update room configuration
     */
    updateRoom(room: Room): void {
        this.room = room;

        if (this.raytracer) {
            const roomAcoustics = new RoomAcoustics(room);
            this.raytracer.setMaterials(roomAcoustics.getMaterialsMap());
            this.raytracer.setRoomBounds(roomAcoustics.getRoomBounds());
        }
    }
    
    /**
     * Get current acoustic metrics
     */
    getAcousticMetrics(): any {
        if (this.debugger) {
            // Return metrics from debugger
            return this.debugger.getAcousticMetrics?.();
        }
        
        // Calculate basic metrics from current impulse response
        if (this.currentImpulseResponse) {
            return this.irGenerator.calculateAcousticMetrics(this.currentImpulseResponse);
        }
        
        return null;
    }
    
    /**
     * Get current impulse response
     */
    getCurrentImpulseResponse(): Float32Array | null {
        return this.currentImpulseResponse;
    }
    
    /**
     * Enable/disable debug mode
     */
    async setDebugMode(enabled: boolean): Promise<void> {
        this.config.performance.debugMode = enabled;
        
        if (enabled && !this.debugger) {
            this.debugger = new RaytracingDebugger(
                this.device,
                this.canvas,
                this.raytracer
            );
            await this.debugger.initialize();
        } else if (!enabled && this.debugger) {
            this.debugger.dispose();
            this.debugger = null;
        }
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig: Partial<SpatialAudioConfig>): void {
        this.config = { ...this.config, ...newConfig };
        
        // Update raytracer configuration
        if (newConfig.raytracing) {
            this.raytracer?.updateConfig({
                maxRays: this.config.raytracing.maxRays,
                maxBounces: this.config.raytracing.maxBounces,
                minEnergy: this.config.raytracing.minEnergy
            });
            
            this.raytracingInterval = 1000 / this.config.raytracing.updateRate;
        }
        
        // Update audio configuration
        if (newConfig.audio && this.audioManager) {
            // Update audio manager settings
        }
    }
    
    /**
     * Dispose of all resources
     */
    dispose(): void {
        this.debugger?.dispose();
        this.raytracer?.dispose();
        this.audioManager?.dispose();
        
        this.initialized = false;
        console.log('Spatial Audio System disposed');
    }
}
