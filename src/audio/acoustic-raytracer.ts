/**
 * Acoustic Raytracer - GPU-accelerated acoustic ray tracing engine
 */

import { vec3 } from 'gl-matrix';
import {
    AcousticRay,
    RayGenerationParams,
    RayBouncingParams,
    ListenerConfig,
    RayBufferConfig,
    RAY_STRUCT_SIZE,
    calculateRayBufferSize,
    packRayForGPU,
    unpackRayFromGPU,
    generateRayDirections,
    RayDistribution,
    RayStatistics,
    calculateRayStatistics,
    RayDistributionType
} from './ray-types';
import { AcousticMaterial } from './acoustic-materials';
import { FREQUENCY_BANDS } from './audio-utils';

// Import shaders as text
import rayGenerationShader from '../shaders/ray-generation.wgsl?raw';
import rayBouncingShader from '../shaders/ray-bouncing.wgsl?raw';
import rayCollectionShader from '../shaders/ray-collection.wgsl?raw';

/**
 * Comprehensive debugging statistics for raytracing
 */
interface RaytracingDebugStats {
    // Frame info
    currentFrame: number;
    totalFrames: number;

    // Ray generation
    raysGenerated: number;
    raysActive: number;
    raysTerminated: number;

    // Ray bouncing
    totalBounces: number;
    averageBounces: number;
    maxBounces: number;

    // Energy tracking
    totalEnergyGenerated: number;
    totalEnergyCollected: number;
    energyLoss: number;

    // Collection stats
    raysCollected: number;
    collectionRadius: number;
    averageArrivalTime: number;

    // Performance
    generationTime: number;
    bouncingTime: number;
    collectionTime: number;
    totalTime: number;

    // GPU state
    bufferSizes: {
        rayBuffer: number;
        impulseBuffer: number;
        stagingBuffer: number;
    };

    // Validation
    errors: string[];
    warnings: string[];
}

/**
 * Debug configuration for raytracing
 */
interface RaytracingDebugConfig {
    enableDetailedLogging: boolean;
    enableRayValidation: boolean;
    enableEnergyTracking: boolean;
    enablePerformanceMonitoring: boolean;
    enableGPUStateValidation: boolean;
    logInterval: number; // frames between detailed logs
    maxLoggedRays: number; // max rays to log in detail
}

/**
 * GPU buffer bindings
 */
enum BufferBinding {
    RAYS_STORAGE = 0,
    GENERATION_PARAMS = 1,
    BOUNCING_PARAMS = 2,
    MATERIALS = 3,
    IMPULSE_RESPONSE = 4,
    STATISTICS = 5
}

/**
 * Acoustic raytracer configuration
 */
export interface RaytracerConfig {
    maxRays: number;
    maxBounces: number;
    minEnergy: number;
    speedOfSound: number;
    sampleRate: number;
    impulseResponseLength: number; // seconds
    workgroupSize: number;
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: RaytracerConfig = {
    maxRays: 2048,
    maxBounces: 20,
    minEnergy: 0.001,
    speedOfSound: 343.0, // m/s at 20°C
    sampleRate: 48000,
    impulseResponseLength: 2.0,
    workgroupSize: 64
};

export class AcousticRaytracer {
    private device: GPUDevice;
    private config: RaytracerConfig;
    
    // Buffers
    private rayBuffer: GPUBuffer | null = null;
    private rayBufferStaging: GPUBuffer | null = null;
    private generationParamsBuffer: GPUBuffer | null = null;
    private bouncingParamsBuffer: GPUBuffer | null = null;
    private collectionParamsBuffer: GPUBuffer | null = null;
    private materialsBuffer: GPUBuffer | null = null;
    private impulseResponseBuffer: GPUBuffer | null = null;
    private statisticsBuffer: GPUBuffer | null = null;
    private statisticsBufferStaging: GPUBuffer | null = null;
    
    // Pipelines
    private generationPipeline: GPUComputePipeline | null = null;
    private bouncingPipeline: GPUComputePipeline | null = null;
    private collectionPipeline: GPUComputePipeline | null = null;
    private normalizationPipeline: GPUComputePipeline | null = null;
    
    // Bind groups
    private generationBindGroup: GPUBindGroup | null = null;
    private bouncingBindGroup: GPUBindGroup | null = null;
    private collectionBindGroup: GPUBindGroup | null = null;
    private normalizationBindGroup: GPUBindGroup | null = null;
    
    // State
    private initialized: boolean = false;
    private rays: AcousticRay[] = [];
    private materials: Map<number, AcousticMaterial> = new Map();
    private currentFrame: number = 0;
    private roomBounds: { min: vec3; max: vec3 } | null = null;

    // Debug system
    private debugConfig: RaytracingDebugConfig = {
        enableDetailedLogging: true,
        enableRayValidation: true,
        enableEnergyTracking: true,
        enablePerformanceMonitoring: true,
        enableGPUStateValidation: true,
        logInterval: 1, // Log every frame for now
        maxLoggedRays: 10
    };
    private debugStats: RaytracingDebugStats = this.createEmptyDebugStats();
    private frameStartTime: number = 0;
    
    constructor(device: GPUDevice, config: Partial<RaytracerConfig> = {}) {
        this.device = device;
        this.config = { ...DEFAULT_CONFIG, ...config };
    }
    
    /**
     * Initialize the raytracer
     */
    async initialize(): Promise<void> {
        if (this.initialized) {
            console.warn('AcousticRaytracer already initialized');
            return;
        }
        
        // Create buffers
        this.createBuffers();
        
        // Load and compile shaders
        await this.createPipelines();
        
        // Create bind groups
        this.createBindGroups();
        
        // Initialize rays
        this.initializeRays();

        // Test basic GPU functionality
        const gpuTestPassed = await this.testGPUBasics();
        if (!gpuTestPassed) {
            console.warn('GPU basic test failed - raytracing may not work properly');
        }

        this.initialized = true;
        console.log('AcousticRaytracer initialized successfully');
        console.log('- Generation pipeline:', !!this.generationPipeline);
        console.log('- Bouncing pipeline:', !!this.bouncingPipeline);
        console.log('- Collection pipeline:', !!this.collectionPipeline);
        console.log('- All bind groups created:', !!this.generationBindGroup && !!this.bouncingBindGroup && !!this.collectionBindGroup);
        console.log('- GPU basic test:', gpuTestPassed ? 'PASSED' : 'FAILED');
    }
    
    /**
     * Create GPU buffers
     */
    private createBuffers(): void {
        const { maxRays, impulseResponseLength, sampleRate } = this.config;
        
        // Ray buffer (double the size for ping-pong)
        const rayBufferSize = calculateRayBufferSize(maxRays) * 2;
        this.rayBuffer = this.device.createBuffer({
            size: rayBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        
        // Staging buffer for reading rays back
        this.rayBufferStaging = this.device.createBuffer({
            size: calculateRayBufferSize(maxRays),
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        
        // Generation parameters (aligned for vec4 frequency weights)
        this.generationParamsBuffer = this.device.createBuffer({
            size: 112, // Increased size for vec4 frequency weights (32 bytes) + other params
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        
        // Bouncing parameters (aligned for vec4 air absorption)
        this.bouncingParamsBuffer = this.device.createBuffer({
            size: 144, // Increased size for vec4 air absorption and surface materials
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Collection parameters
        this.collectionParamsBuffer = this.device.createBuffer({
            size: 64, // listener position, radius, time bin size, etc.
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Materials buffer (space for 64 materials)
        const materialSize = 4 * (8 + 8 + 2); // absorption + scattering + impedance/roughness
        this.materialsBuffer = this.device.createBuffer({
            size: materialSize * 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        
        // Impulse response buffer
        const irSamples = Math.floor(impulseResponseLength * sampleRate);
        // ImpulseBin structure: energy(1) + phase_real(1) + phase_imag(1) + sample_count(1) + freq_low(4) + freq_high(4) + padding(3) = 15 floats
        const irSize = irSamples * 15 * 4; // 15 floats per bin * 4 bytes per float
        this.impulseResponseBuffer = this.device.createBuffer({
            size: irSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        
        // Statistics buffer
        this.statisticsBuffer = this.device.createBuffer({
            size: 64, // Various statistics
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Statistics staging buffer for readback
        this.statisticsBufferStaging = this.device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
    }
    
    /**
     * Create compute pipelines
     */
    private async createPipelines(): Promise<void> {
        try {
            // Load ray generation shader
            const generationShaderCode = this.getShaderCode('ray-generation.wgsl');
            const generationModule = this.device.createShaderModule({
                code: generationShaderCode
            });

            // Create bind group layout for generation
            const generationBindGroupLayout = this.device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0, // rays storage
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'storage' }
                    },
                    {
                        binding: 1, // generation params
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'uniform' }
                    }
                ]
            });

            // Create generation pipeline
            this.generationPipeline = this.device.createComputePipeline({
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [generationBindGroupLayout]
                }),
                compute: {
                    module: generationModule,
                    entryPoint: 'main'
                }
            });

            console.log('Ray generation pipeline created successfully');

            // Load ray bouncing shader
            const bouncingShaderCode = this.getShaderCode('ray-bouncing.wgsl');
            const bouncingModule = this.device.createShaderModule({
                code: bouncingShaderCode
            });

            // Create bind group layout for bouncing
            const bouncingBindGroupLayout = this.device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0, // rays storage
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'storage' }
                    },
                    {
                        binding: 1, // bouncing params
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'uniform' }
                    },
                    {
                        binding: 2, // materials storage
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'read-only-storage' }
                    }
                ]
            });

            // Create bouncing pipeline
            this.bouncingPipeline = this.device.createComputePipeline({
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [bouncingBindGroupLayout]
                }),
                compute: {
                    module: bouncingModule,
                    entryPoint: 'main'
                }
            });

            console.log('Ray bouncing pipeline created successfully');

            // Load ray collection shader with error handling
            const collectionShaderCode = this.getShaderCode('ray-collection.wgsl');

            // Push error scope to catch shader compilation errors
            this.device.pushErrorScope('validation');
            const collectionModule = this.device.createShaderModule({
                code: collectionShaderCode
            });

            // Check for compilation errors
            const shaderError = await this.device.popErrorScope();
            if (shaderError) {
                console.error('❌ Collection Shader Compilation Error:', shaderError);
                throw new Error(`Collection shader compilation failed: ${shaderError.message}`);
            } else {
                console.log('✅ Collection shader compiled successfully');
            }

            // Create bind group layout for collection
            const collectionBindGroupLayout = this.device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0, // rays storage (read-only for collection)
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'read-only-storage' }
                    },
                    {
                        binding: 1, // collection params
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'uniform' }
                    },
                    {
                        binding: 2, // impulse response storage
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'storage' }
                    },
                    {
                        binding: 3, // statistics storage
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'storage' }
                    }
                ]
            });

            // Create collection pipeline
            this.collectionPipeline = this.device.createComputePipeline({
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [collectionBindGroupLayout]
                }),
                compute: {
                    module: collectionModule,
                    entryPoint: 'main'
                }
            });

            console.log('Ray collection pipeline created successfully');

            // Create normalization pipeline (uses same shader, different entry point)
            this.normalizationPipeline = this.device.createComputePipeline({
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [collectionBindGroupLayout]
                }),
                compute: {
                    module: collectionModule,
                    entryPoint: 'normalize_statistics'
                }
            });

            console.log('Ray normalization pipeline created successfully');

        } catch (error) {
            console.error('Failed to create compute pipelines:', error);
            console.error('Error details:', error);

            // Log shader compilation details
            if (error.message.includes('shader')) {
                console.error('Shader compilation failed. Check shader syntax and bindings.');
            }

            // Don't throw - continue with partial initialization
            console.warn('Continuing with partial pipeline initialization');
        }
    }

    /**
     * Get shader code by filename
     */
    private getShaderCode(filename: string): string {
        switch (filename) {
            case 'ray-generation.wgsl':
                return rayGenerationShader;
            case 'ray-bouncing.wgsl':
                return rayBouncingShader;
            case 'ray-collection.wgsl':
                return rayCollectionShader;
            default:
                throw new Error(`Unknown shader: ${filename}`);
        }
    }
    
    /**
     * Create bind groups
     */
    private createBindGroups(): void {
        if (!this.generationPipeline || !this.rayBuffer || !this.generationParamsBuffer) {
            console.warn('Cannot create bind groups: missing pipelines or buffers');
            return;
        }

        // Create generation bind group
        this.generationBindGroup = this.device.createBindGroup({
            layout: this.generationPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.rayBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.generationParamsBuffer
                    }
                }
            ]
        });

        console.log('Ray generation bind group created successfully');

        // Create bouncing bind group
        if (this.bouncingPipeline && this.bouncingParamsBuffer && this.materialsBuffer) {
            this.bouncingBindGroup = this.device.createBindGroup({
                layout: this.bouncingPipeline.getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.rayBuffer
                        }
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.bouncingParamsBuffer
                        }
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: this.materialsBuffer
                        }
                    }
                ]
            });
            console.log('Ray bouncing bind group created successfully');
        }

        // Create collection bind group
        if (this.collectionPipeline && this.collectionParamsBuffer && this.impulseResponseBuffer && this.statisticsBuffer) {
            this.collectionBindGroup = this.device.createBindGroup({
                layout: this.collectionPipeline.getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.rayBuffer
                        }
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.collectionParamsBuffer
                        }
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: this.impulseResponseBuffer
                        }
                    },
                    {
                        binding: 3,
                        resource: {
                            buffer: this.statisticsBuffer
                        }
                    }
                ]
            });
            console.log('Ray collection bind group created successfully');
        }

        // Create normalization bind group (same as collection)
        if (this.normalizationPipeline && this.collectionParamsBuffer && this.impulseResponseBuffer && this.statisticsBuffer) {
            this.normalizationBindGroup = this.device.createBindGroup({
                layout: this.normalizationPipeline.getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.rayBuffer
                        }
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.collectionParamsBuffer
                        }
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: this.impulseResponseBuffer
                        }
                    },
                    {
                        binding: 3,
                        resource: {
                            buffer: this.statisticsBuffer
                        }
                    }
                ]
            });
            console.log('Ray normalization bind group created successfully');
        }
    }
    
    /**
     * Initialize ray array
     */
    private initializeRays(): void {
        this.rays = [];
        for (let i = 0; i < this.config.maxRays; i++) {
            this.rays.push({
                origin: vec3.create(),
                direction: vec3.create(),
                energy: 0,
                phase: 0,
                frequencyEnergy: new Float32Array(8),
                pathLength: 0,
                arrivalTime: 0,
                bounceCount: 0,
                active: 0,
                lastMaterialId: -1,
                padding: new Float32Array(2)
            });
        }
    }
    
    /**
     * Set room materials
     */
    setMaterials(materials: Map<number, AcousticMaterial>): void {
        this.materials = new Map(materials);
        this.uploadMaterials();
    }

    /**
     * Set room bounds for raytracing
     */
    setRoomBounds(bounds: { min: vec3; max: vec3 }): void {
        this.roomBounds = {
            min: vec3.clone(bounds.min),
            max: vec3.clone(bounds.max)
        };
    }
    
    /**
     * Upload materials to GPU
     */
    private uploadMaterials(): void {
        if (!this.materialsBuffer) return;
        
        const materialData = new Float32Array(64 * (8 + 8 + 2));
        let offset = 0;
        
        for (const [id, material] of this.materials) {
            if (id >= 64) continue; // Max 64 materials
            
            const baseOffset = id * (8 + 8 + 2);
            
            // Absorption coefficients
            for (let i = 0; i < 8; i++) {
                materialData[baseOffset + i] = material.absorption[i];
            }
            
            // Scattering coefficients
            for (let i = 0; i < 8; i++) {
                materialData[baseOffset + 8 + i] = material.scattering[i];
            }
            
            // Impedance and roughness
            materialData[baseOffset + 16] = material.impedance;
            materialData[baseOffset + 17] = material.roughness;
        }
        
        this.device.queue.writeBuffer(this.materialsBuffer, 0, materialData);
    }
    
    /**
     * Trace rays for one frame
     */
    async traceRays(
        sourcePosition: vec3,
        listenerConfig: ListenerConfig,
        deltaTime: number
    ): Promise<Float32Array> {
        if (!this.isFullyInitialized()) {
            throw new Error('AcousticRaytracer not fully initialized - missing pipelines or buffers');
        }

        // Start debug timing
        this.startDebugTiming();
        this.debugStats.currentFrame = this.currentFrame;
        this.debugStats.collectionRadius = listenerConfig.radius;

        console.log('🚀 traceRays called with:', {
            frame: this.currentFrame,
            sourcePosition: Array.from(sourcePosition),
            listenerPosition: Array.from(listenerConfig.position),
            listenerRadius: listenerConfig.radius,
            deltaTime,
            roomBounds: this.roomBounds ? {
                min: Array.from(this.roomBounds.min),
                max: Array.from(this.roomBounds.max)
            } : null
        });

        try {
            // Validate GPU state before operations
            if (this.debugConfig.enableGPUStateValidation) {
                this.validateGPUState();
            }

            const encoder = this.device.createCommandEncoder();
        
        // Update generation parameters
        this.updateGenerationParams(sourcePosition);
        
        // Update bouncing parameters
        this.updateBouncingParams();

        // Clear impulse response buffer
        this.clearImpulseResponseBuffer(encoder);

        // Generation pass - run when needed
        const shouldGenerate = this.currentFrame === 0 || this.shouldRegenerateRays();
        if (shouldGenerate) {
            console.log(`🔄 Running ray generation pass (frame ${this.currentFrame})`);
            const genStart = performance.now();
            this.runGenerationPass(encoder);
            this.debugStats.generationTime = performance.now() - genStart;
            this.debugStats.raysGenerated = this.config.maxRays;
        }

        // Submit generation commands
        this.device.queue.submit([encoder.finish()]);

        // Debug: Check rays immediately after generation
        if (shouldGenerate) {
            console.log('🔍 Checking rays after generation...');
            const firstRays = await this.getFirstRays(5);
            console.log('First 5 rays after generation:', firstRays.map((ray, i) => ({
                index: i,
                energy: ray.energy.toFixed(4),
                active: ray.active,
                pathLength: ray.pathLength.toFixed(3),
                arrivalTime: ray.arrivalTime.toFixed(4),
                bounceCount: ray.bounceCount,
                origin: Array.from(ray.origin).map(x => x.toFixed(2)),
                direction: Array.from(ray.direction).map(x => x.toFixed(2)),
                frequencyEnergyLow: Array.from(ray.frequencyEnergy.slice(0, 4)).map(x => x.toFixed(3)),
                frequencyEnergyHigh: Array.from(ray.frequencyEnergy.slice(4, 8)).map(x => x.toFixed(3))
            })));
        }

        // Validate rays after generation if enabled
        if (shouldGenerate && this.debugConfig.enableRayValidation) {
            await this.validateRayData();
        }

        // Bouncing pass - run every frame for active rays
        console.log(`🏀 Running ray bouncing pass (frame ${this.currentFrame})`);
        const bouncingEncoder = this.device.createCommandEncoder();
        const bounceStart = performance.now();
        this.runBouncingPass(bouncingEncoder, deltaTime);
        this.debugStats.bouncingTime = performance.now() - bounceStart;
        this.device.queue.submit([bouncingEncoder.finish()]);

        // Collection pass - run every frame
        console.log(`🎯 Running ray collection pass (frame ${this.currentFrame})`);
        const collectionEncoder = this.device.createCommandEncoder();

        // Clear impulse response buffer before collection
        this.clearImpulseResponseBuffer(collectionEncoder);

        // Clear statistics buffer before collection
        collectionEncoder.clearBuffer(this.statisticsBuffer!, 0, 16); // Clear first 16 bytes (4 u32 values)

        const collectStart = performance.now();
        this.runCollectionPass(collectionEncoder, listenerConfig);
        this.debugStats.collectionTime = performance.now() - collectStart;
        this.device.queue.submit([collectionEncoder.finish()]);

        // Read back statistics for debugging
        const stats = await this.readStatistics();
        console.log('📊 Collection shader debug stats (raw):', Array.from(stats));
        console.log('📊 Collection shader debug stats:', {
            shaderRunning: stats[0],    // Should be 999 if shader ran
            totalRaysProcessed: stats[1], // Total rays processed
            raysCollected: stats[2],    // Number of rays collected
            totalEnergyX1000: stats[3], // Total energy * 1000 (as integer)
            totalEnergyFloat: stats[3] / 1000.0 // Convert back to float
        });

        // Read back impulse response from GPU
        const impulseResponse = await this.readImpulseResponse();

        // Update debug statistics
        this.debugStats.totalTime = performance.now() - this.frameStartTime;
        this.debugStats.totalFrames++;

        // Validate final results if enabled
        if (this.debugConfig.enableRayValidation) {
            await this.validateRayData();
        }

        // Log debug information
        this.logDebugInfo();

        // Detailed ray analysis for first few frames
        if (this.currentFrame < 3 && this.debugConfig.enableDetailedLogging) {
            const detailedRays = await this.getFirstRays(10);
            console.log('🔬 Detailed ray analysis:', {
                frame: this.currentFrame,
                listenerPos: Array.from(listenerConfig.position),
                listenerRadius: listenerConfig.radius,
                rays: detailedRays.map((ray, i) => {
                    const distance = vec3.distance(ray.origin, listenerConfig.position);
                    return {
                        index: i,
                        active: ray.active,
                        energy: ray.energy.toFixed(4),
                        bounces: ray.bounceCount,
                        pathLength: ray.pathLength.toFixed(3),
                        arrivalTime: ray.arrivalTime.toFixed(4),
                        distanceToListener: distance.toFixed(3),
                        withinRadius: distance <= listenerConfig.radius,
                        origin: Array.from(ray.origin).map(x => x.toFixed(2)),
                        direction: Array.from(ray.direction).map(x => x.toFixed(2))
                    };
                })
            });
        }

        this.currentFrame++;
        return impulseResponse;

        } catch (error) {
            this.debugStats.errors.push(`GPU raytracing failed: ${error}`);
            console.error('❌ GPU raytracing failed:', error);
            throw new Error(`Raytracing execution failed: ${error.message}`);
        }
    }
    
    /**
     * Update generation parameters
     */
    private updateGenerationParams(sourcePosition: vec3, distributionType: RayDistributionType = RayDistributionType.UNIFORM_SPHERE): void {
        if (!this.generationParamsBuffer) return;

        // Buffer layout must match WGSL struct layout rules.
        // Specifically, vec3 is aligned to 16 bytes.
        const buffer = new ArrayBuffer(112);
        const floatView = new Float32Array(buffer);
        const uintView = new Uint32Array(buffer);

        // Offset 0: source_position: vec3<f32> (requires 16-byte alignment)
        floatView[0] = sourcePosition[0];
        floatView[1] = sourcePosition[1];
        floatView[2] = sourcePosition[2];
        // floatView[3] is padding

        // Offset 16: source_radius: f32
        floatView[4] = 0.1;

        // Offset 20: ray_count: u32
        uintView[5] = this.config.maxRays;

        // Offset 24: initial_energy: f32
        floatView[6] = 1.0;

        // Offset 28: time: f32
        floatView[7] = this.currentFrame * 0.016;

        // Offset 32: seed: u32
        uintView[8] = Math.floor(Math.random() * 0xFFFFFF);
        // From offset 36 to 48 is padding for the next vec4

        // Offset 48: frequency_weights_low: vec4<f32>
        floatView[12] = 1.0 / 8.0;
        floatView[13] = 1.0 / 8.0;
        floatView[14] = 1.0 / 8.0;
        floatView[15] = 1.0 / 8.0;

        // Offset 64: frequency_weights_high: vec4<f32>
        floatView[16] = 1.0 / 8.0;
        floatView[17] = 1.0 / 8.0;
        floatView[18] = 1.0 / 8.0;
        floatView[19] = 1.0 / 8.0;

        // Offset 80: distribution_type: u32
        uintView[20] = distributionType;

        // Offset 84: cone_angle: f32
        floatView[21] = Math.PI / 4;

        this.device.queue.writeBuffer(this.generationParamsBuffer, 0, buffer);

        // Debug generation parameters on first frame
        if (this.currentFrame === 0) {
            console.log('Generation params (Corrected Layout):', {
                sourcePosition: [floatView[0], floatView[1], floatView[2]],
                sourceRadius: floatView[4],
                rayCount: uintView[5],
                initialEnergy: floatView[6],
                time: floatView[7],
                seed: uintView[8],
                distributionType: uintView[20],
                frequencyWeightsLow: [floatView[12], floatView[13], floatView[14], floatView[15]],
                frequencyWeightsHigh: [floatView[16], floatView[17], floatView[18], floatView[19]]
            });
        }
    }
    
    /**
     * Update bouncing parameters
     */
    private updateBouncingParams(): void {
        if (!this.bouncingParamsBuffer) return;

        const params = new Float32Array(36); // 144 bytes / 4 bytes per float

        // Room bounds - use actual room bounds if available, otherwise default
        if (this.roomBounds) {
            params[0] = this.roomBounds.min[0]; params[1] = this.roomBounds.min[1]; params[2] = this.roomBounds.min[2]; // room_min
            params[3] = 0; // padding for alignment
            params[4] = this.roomBounds.max[0]; params[5] = this.roomBounds.max[1]; params[6] = this.roomBounds.max[2]; // room_max
            params[7] = 0; // padding for alignment
        } else {
            // Default room bounds if not set
            params[0] = -5; params[1] = 0; params[2] = -5; // room_min
            params[3] = 0; // padding for alignment
            params[4] = 5; params[5] = 3; params[6] = 5; // room_max
            params[7] = 0; // padding for alignment
        }

        params[8] = this.config.maxBounces;
        params[9] = this.config.minEnergy;
        params[10] = this.config.speedOfSound;
        params[11] = 0.016; // time step

        // Air absorption coefficients (split into two vec4s)
        // Low frequency air absorption (125, 250, 500, 1k Hz)
        params[12] = 0.0001; // 125 Hz
        params[13] = 0.0002; // 250 Hz
        params[14] = 0.0004; // 500 Hz
        params[15] = 0.0008; // 1 kHz

        // High frequency air absorption (2k, 4k, 8k, 16k Hz)
        params[16] = 0.0016; // 2 kHz
        params[17] = 0.0032; // 4 kHz
        params[18] = 0.0064; // 8 kHz
        params[19] = 0.0128; // 16 kHz

        // Surface materials (material IDs for each face)
        params[20] = 0; // +X face (right wall)
        params[21] = 0; // -X face (left wall)
        params[22] = 1; // +Y face (ceiling)
        params[23] = 2; // -Y face (floor)

        // Additional surface materials
        params[24] = 0; // +Z face (front wall)
        params[25] = 0; // -Z face (back wall)

        // Padding to reach 144 bytes
        for (let i = 26; i < 36; i++) {
            params[i] = 0.0;
        }

        this.device.queue.writeBuffer(this.bouncingParamsBuffer, 0, params);

        // Debug bouncing parameters on first frame
        if (this.currentFrame === 0) {
            console.log('Bouncing params:', {
                roomMin: [params[0], params[1], params[2]],
                roomMax: [params[4], params[5], params[6]],
                maxBounces: params[8],
                minEnergy: params[9],
                speedOfSound: params[10],
                roomBoundsSet: !!this.roomBounds
            });
        }
    }

    /**
     * Update collection parameters
     */
    private updateCollectionParams(listenerConfig: ListenerConfig): void {
        if (!this.collectionParamsBuffer) return;

        const params = new Float32Array(16); // 64 bytes / 4 bytes per float

        // Listener position and radius
        params[0] = listenerConfig.position[0];
        params[1] = listenerConfig.position[1];
        params[2] = listenerConfig.position[2];
        params[3] = listenerConfig.radius;

        // Collection parameters (matching WGSL struct)
        params[4] = this.config.sampleRate; // sample_rate
        params[5] = this.config.impulseResponseLength; // ir_length in seconds

        const timeBinSize = 1.0 / this.config.sampleRate; // One sample duration
        params[6] = timeBinSize; // time_bin_size

        const maxBins = Math.floor(this.config.impulseResponseLength * this.config.sampleRate);
        params[7] = maxBins; // max_bins (as float, will be cast to u32 in shader)

        params[8] = this.config.minEnergy; // energy_threshold
        params[9] = 0; // padding

        // Additional padding to reach 64 bytes
        for (let i = 10; i < 16; i++) {
            params[i] = 0;
        }

        console.log('Collection params:', {
            listenerPos: [params[0], params[1], params[2]],
            radius: params[3],
            sampleRate: params[4],
            irLength: params[5],
            timeBinSize: params[6],
            maxBins: params[7],
            energyThreshold: params[8],
            timeBinSizeMs: params[6] * 1000,
            maxTimeMs: params[5] * 1000,
            bufferSizeBytes: this.impulseResponseBuffer?.size,
            expectedBins: Math.floor(this.config.impulseResponseLength * this.config.sampleRate)
        });

        this.device.queue.writeBuffer(this.collectionParamsBuffer, 0, params);
    }
    
    /**
     * Check if rays should be regenerated
     */
    private shouldRegenerateRays(): boolean {
        // For debugging: regenerate every frame to ensure we always have rays
        return true;

        // Original logic: Regenerate every 10 frames for variety
        // return this.currentFrame % 10 === 0;
    }
    
    /**
     * Run ray generation compute pass
     */
    private runGenerationPass(encoder: GPUCommandEncoder): void {
        if (!this.generationPipeline || !this.generationBindGroup) return;
        
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.generationPipeline);
        pass.setBindGroup(0, this.generationBindGroup);
        
        const workgroups = Math.ceil(this.config.maxRays / this.config.workgroupSize);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
    }
    
    /**
     * Run ray bouncing compute pass
     */
    private runBouncingPass(encoder: GPUCommandEncoder, deltaTime: number): void {
        if (!this.bouncingPipeline || !this.bouncingBindGroup) return;
        
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.bouncingPipeline);
        pass.setBindGroup(0, this.bouncingBindGroup);
        
        const workgroups = Math.ceil(this.config.maxRays / this.config.workgroupSize);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
    }
    
    /**
     * Clear impulse response buffer
     */
    private clearImpulseResponseBuffer(encoder: GPUCommandEncoder): void {
        if (!this.impulseResponseBuffer) return;

        encoder.clearBuffer(this.impulseResponseBuffer);
    }

    /**
     * Run ray collection compute pass
     */
    private runCollectionPass(encoder: GPUCommandEncoder, listenerConfig: ListenerConfig): void {
        if (!this.collectionPipeline || !this.collectionBindGroup) {
            console.error('❌ Collection pass failed: missing pipeline or bind group', {
                pipeline: !!this.collectionPipeline,
                bindGroup: !!this.collectionBindGroup
            });
            return;
        }

        // Update listener parameters
        this.updateCollectionParams(listenerConfig);

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.collectionPipeline);
        pass.setBindGroup(0, this.collectionBindGroup);

        // Dispatch workgroups to cover all rays
        const workgroups = Math.ceil(this.config.maxRays / this.config.workgroupSize);
        console.log(`🎯 Dispatching collection pass: ${workgroups} workgroups for ${this.config.maxRays} rays`);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
    }
    
    /**
     * Run normalization compute pass
     */
    private runNormalizationPass(encoder: GPUCommandEncoder): void {
        if (!this.normalizationPipeline || !this.normalizationBindGroup) return;

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.normalizationPipeline);
        pass.setBindGroup(0, this.normalizationBindGroup);

        // Single workgroup for normalization
        pass.dispatchWorkgroups(1);
        pass.end();
    }

    /**
     * Read impulse response from GPU
     */
    private async readImpulseResponse(): Promise<Float32Array> {
        if (!this.impulseResponseBuffer) {
            throw new Error('Impulse response buffer not available');
        }

        const irLength = Math.floor(this.config.impulseResponseLength * this.config.sampleRate);

        // Create staging buffer for readback
        const stagingBuffer = this.device.createBuffer({
            size: this.impulseResponseBuffer.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // Copy impulse response data to staging buffer
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(
            this.impulseResponseBuffer, 0,
            stagingBuffer, 0,
            this.impulseResponseBuffer.size
        );
        this.device.queue.submit([encoder.finish()]);

        // Read back the data
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const rawData = new Float32Array(stagingBuffer.getMappedRange());

        // Convert from ImpulseBin format to simple Float32Array
        const impulseResponse = new Float32Array(irLength);
        const floatsPerBin = 15; // ImpulseBin has 15 floats total

        for (let i = 0; i < irLength && i < rawData.length / floatsPerBin; i++) {
            const binIndex = i * floatsPerBin;
            const energy = rawData[binIndex]; // energy field
            const phaseReal = rawData[binIndex + 1]; // phase_real field
            const phaseImag = rawData[binIndex + 2]; // phase_imag field

            // Calculate magnitude from complex phase representation
            const magnitude = Math.sqrt(phaseReal * phaseReal + phaseImag * phaseImag);
            impulseResponse[i] = energy * magnitude;
        }

        // Proper cleanup
        stagingBuffer.unmap();
        stagingBuffer.destroy();

        // Detailed logging for debugging
        const nonZeroSamples = Array.from(impulseResponse).filter(x => Math.abs(x) > 0.001).length;
        const maxValue = Math.max(...impulseResponse.map(Math.abs));
        const totalEnergy = Array.from(impulseResponse).reduce((sum, x) => sum + Math.abs(x), 0);

        console.log(`IR readback detailed:`, {
            frame: this.currentFrame,
            nonZeroSamples: `${nonZeroSamples}/${irLength}`,
            maxValue: maxValue.toFixed(6),
            totalEnergy: totalEnergy.toFixed(6),
            timeBinSize: (1.0 / this.config.sampleRate).toFixed(8),
            maxBins: Math.floor(this.config.impulseResponseLength * this.config.sampleRate)
        });

        // Show first few non-zero samples for debugging
        const firstNonZero = [];
        for (let i = 0; i < Math.min(1000, impulseResponse.length); i++) {
            if (Math.abs(impulseResponse[i]) > 0.001) {
                firstNonZero.push({ index: i, value: impulseResponse[i].toFixed(6), timeMs: (i / this.config.sampleRate * 1000).toFixed(2) });
                if (firstNonZero.length >= 10) break;
            }
        }
        if (firstNonZero.length > 0) {
            console.log('First non-zero IR samples:', firstNonZero);
        }

        // If no energy was collected, add a simple test impulse for debugging
        if (maxValue === 0 && this.currentFrame < 3) {
            console.log('No energy collected, adding test impulse for debugging');
            impulseResponse[0] = 1.0; // Direct sound
            impulseResponse[Math.floor(0.01 * this.config.sampleRate)] = 0.5; // Early reflection at 10ms
            impulseResponse[Math.floor(0.05 * this.config.sampleRate)] = 0.2; // Later reflection at 50ms
        }

        return impulseResponse;
    }
    
    /**
     * Get ray statistics
     */
    async getRayStatistics(): Promise<RayStatistics> {
        if (!this.rayBuffer || !this.rayBufferStaging) {
            return calculateRayStatistics([]);
        }

        // Copy ray buffer to staging
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(
            this.rayBuffer, 0,
            this.rayBufferStaging, 0,
            calculateRayBufferSize(this.config.maxRays)
        );
        this.device.queue.submit([encoder.finish()]);

        // Read back rays
        await this.rayBufferStaging.mapAsync(GPUMapMode.READ);
        const rayData = new Float32Array(this.rayBufferStaging.getMappedRange());

        // Unpack rays
        const rays: AcousticRay[] = [];
        const floatsPerRay = RAY_STRUCT_SIZE / 4;
        for (let i = 0; i < this.config.maxRays; i++) {
            rays.push(unpackRayFromGPU(rayData, i * floatsPerRay));
        }

        this.rayBufferStaging.unmap();

        return calculateRayStatistics(rays);
    }

    /**
     * Get first N rays for debugging
     */
    async getFirstRays(count: number): Promise<AcousticRay[]> {
        if (!this.rayBuffer || !this.rayBufferStaging) {
            return [];
        }

        // Copy ray buffer to staging
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(
            this.rayBuffer, 0,
            this.rayBufferStaging, 0,
            calculateRayBufferSize(this.config.maxRays)
        );
        this.device.queue.submit([encoder.finish()]);

        // Read back rays
        await this.rayBufferStaging.mapAsync(GPUMapMode.READ);
        const rayData = new Float32Array(this.rayBufferStaging.getMappedRange());

        // Unpack first N rays
        const rays: AcousticRay[] = [];
        const floatsPerRay = RAY_STRUCT_SIZE / 4;
        const numRays = Math.min(count, this.config.maxRays);
        for (let i = 0; i < numRays; i++) {
            rays.push(unpackRayFromGPU(rayData, i * floatsPerRay));
        }

        this.rayBufferStaging.unmap();

        return rays;
    }
    
    /**
     * Set ray distribution strategy
     */
    setRayDistribution(distribution: RayDistribution): void {
        // Generate new ray directions
        const directions = generateRayDirections(this.config.maxRays, distribution);
        
        // Update ray directions
        for (let i = 0; i < this.rays.length && i < directions.length; i++) {
            vec3.copy(this.rays[i].direction, directions[i]);
        }
    }
    
    /**
     * Update configuration
     */
    updateConfig(config: Partial<RaytracerConfig>): void {
        const oldMaxRays = this.config.maxRays;
        this.config = { ...this.config, ...config };
        
        // Recreate buffers if ray count changed
        if (this.config.maxRays !== oldMaxRays && this.initialized) {
            this.dispose();
            this.initialize();
        }
    }
    
    /**
     * Test ray generation (for debugging)
     */
    async testRayGeneration(sourcePosition: vec3, distributionType: RayDistributionType = RayDistributionType.UNIFORM_SPHERE): Promise<AcousticRay[]> {
        if (!this.initialized) {
            throw new Error('AcousticRaytracer not initialized');
        }

        if (!this.generationPipeline || !this.generationBindGroup) {
            throw new Error('Ray generation pipeline not ready');
        }

        // Update generation parameters
        this.updateGenerationParams(sourcePosition, distributionType);

        // Create command encoder
        const encoder = this.device.createCommandEncoder();

        // Run generation pass
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.generationPipeline);
        pass.setBindGroup(0, this.generationBindGroup);

        const workgroups = Math.ceil(this.config.maxRays / this.config.workgroupSize);
        pass.dispatchWorkgroups(workgroups);
        pass.end();

        // Copy rays to staging buffer for readback
        if (this.rayBufferStaging) {
            encoder.copyBufferToBuffer(
                this.rayBuffer!, 0,
                this.rayBufferStaging, 0,
                calculateRayBufferSize(this.config.maxRays)
            );
        }

        // Submit commands
        this.device.queue.submit([encoder.finish()]);

        // Read back rays
        if (!this.rayBufferStaging) {
            throw new Error('Staging buffer not available');
        }

        await this.rayBufferStaging.mapAsync(GPUMapMode.READ);
        const rayData = new Float32Array(this.rayBufferStaging.getMappedRange());

        // Unpack rays
        const rays: AcousticRay[] = [];
        const floatsPerRay = RAY_STRUCT_SIZE / 4;
        for (let i = 0; i < this.config.maxRays; i++) {
            rays.push(unpackRayFromGPU(rayData, i * floatsPerRay));
        }

        this.rayBufferStaging.unmap();

        return rays;
    }

    /**
     * Test basic GPU functionality
     */
    async testGPUBasics(): Promise<boolean> {
        try {
            // Create a simple test buffer
            const testBuffer = this.device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });

            // Write test data
            const testData = new Float32Array([1, 2, 3, 4]);
            this.device.queue.writeBuffer(testBuffer, 0, testData);

            // Create staging buffer for readback
            const stagingBuffer = this.device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
            });

            // Copy and read back
            const encoder = this.device.createCommandEncoder();
            encoder.copyBufferToBuffer(testBuffer, 0, stagingBuffer, 0, 16);
            this.device.queue.submit([encoder.finish()]);

            await stagingBuffer.mapAsync(GPUMapMode.READ);
            const result = new Float32Array(stagingBuffer.getMappedRange());

            const success = result[0] === 1 && result[1] === 2 && result[2] === 3 && result[3] === 4;

            stagingBuffer.unmap();
            testBuffer.destroy();
            stagingBuffer.destroy();

            console.log('GPU basic test:', success ? 'PASSED' : 'FAILED');
            return success;

        } catch (error) {
            console.error('GPU basic test failed:', error);
            return false;
        }
    }

    /**
     * Check if raytracer is properly initialized
     */
    isFullyInitialized(): boolean {
        return this.initialized &&
               !!this.rayBuffer &&
               !!this.generationPipeline &&
               !!this.bouncingPipeline &&
               !!this.collectionPipeline &&
               !!this.generationBindGroup &&
               !!this.bouncingBindGroup &&
               !!this.collectionBindGroup;
    }

    /**
     * Validate GPU state before operations
     */
    private validateGPUState(): void {
        // Check for any pending GPU errors (but don't fail on device lost)
        this.device.pushErrorScope('validation');
        this.device.pushErrorScope('out-of-memory');
        this.device.pushErrorScope('internal');
    }

    /**
     * Check for GPU errors after operations
     */
    private async checkGPUErrors(): Promise<void> {
        const internalError = await this.device.popErrorScope();
        const memoryError = await this.device.popErrorScope();
        const validationError = await this.device.popErrorScope();

        if (internalError) {
            console.error('GPU internal error:', internalError);
        }
        if (memoryError) {
            console.error('GPU memory error:', memoryError);
        }
        if (validationError) {
            console.error('GPU validation error:', validationError);
        }
    }

    /**
     * Read statistics buffer for debugging
     */
    private async readStatistics(): Promise<Uint32Array> {
        if (!this.statisticsBuffer || !this.statisticsBufferStaging) {
            return new Uint32Array(4);
        }

        // Copy statistics buffer to staging
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(
            this.statisticsBuffer, 0,
            this.statisticsBufferStaging, 0,
            16 // 4 atomic u32 values = 16 bytes
        );
        this.device.queue.submit([encoder.finish()]);

        // Read back statistics as u32 array (atomic values)
        await this.statisticsBufferStaging.mapAsync(GPUMapMode.READ);
        const statsData = new Uint32Array(this.statisticsBufferStaging.getMappedRange().slice(0, 16));
        const result = new Uint32Array(statsData); // Copy the data
        this.statisticsBufferStaging.unmap();

        return result;
    }

    /**
     * Create a fake impulse response from ray data (CPU-side processing)
     */
    private createFakeImpulseResponseFromRays(rays: AcousticRay[]): Float32Array {
        const irLength = Math.floor(this.config.impulseResponseLength * this.config.sampleRate);
        const impulseResponse = new Float32Array(irLength);

        console.log('Creating fake IR from rays:', {
            rayCount: rays.length,
            irLength,
            sampleRate: this.config.sampleRate
        });

        // Process each ray
        for (let i = 0; i < rays.length; i++) {
            const ray = rays[i];

            if (ray.active > 0 && ray.energy > 0) {
                // Put ray energy at different time positions
                const timeBin = Math.floor((i * 0.01) * this.config.sampleRate); // 10ms apart

                if (timeBin < irLength) {
                    impulseResponse[timeBin] += ray.energy * 0.1; // Scale down energy
                    console.log(`Ray ${i}: energy=${ray.energy.toFixed(3)}, timeBin=${timeBin}, time=${(timeBin/this.config.sampleRate*1000).toFixed(1)}ms`);
                }
            }
        }

        // Add a test impulse at the beginning
        impulseResponse[0] = 1.0;

        const nonZero = Array.from(impulseResponse).filter(x => Math.abs(x) > 0.001).length;
        console.log(`Fake IR created: ${nonZero}/${irLength} non-zero samples`);

        return impulseResponse;
    }

    /**
     * Test writing a ray directly from CPU to verify buffer works
     */
    private async testDirectRayWrite(): Promise<void> {
        if (!this.rayBuffer) return;

        console.log('Writing test ray directly to GPU buffer...');

        // Create a test ray with known values
        const testRayData = new Float32Array(28); // 28 floats = 112 bytes per ray
        let offset = 0;

        // Origin (vec4)
        testRayData[offset++] = 1.0; // x
        testRayData[offset++] = 2.0; // y
        testRayData[offset++] = 3.0; // z
        testRayData[offset++] = 0.0; // w padding

        // Direction (vec4)
        testRayData[offset++] = 0.0; // x
        testRayData[offset++] = 1.0; // y
        testRayData[offset++] = 0.0; // z
        testRayData[offset++] = 0.0; // w padding

        // Energy and phase (vec4)
        testRayData[offset++] = 999.0; // energy - unique value
        testRayData[offset++] = 0.0;   // phase
        testRayData[offset++] = 0.0;   // z padding
        testRayData[offset++] = 0.0;   // w padding

        // Frequency energy low (vec4)
        testRayData[offset++] = 1.0; // 125 Hz
        testRayData[offset++] = 1.0; // 250 Hz
        testRayData[offset++] = 1.0; // 500 Hz
        testRayData[offset++] = 1.0; // 1 kHz

        // Frequency energy high (vec4)
        testRayData[offset++] = 1.0; // 2 kHz
        testRayData[offset++] = 1.0; // 4 kHz
        testRayData[offset++] = 1.0; // 8 kHz
        testRayData[offset++] = 1.0; // 16 kHz

        // Path data (vec4)
        testRayData[offset++] = 0.0; // path_length
        testRayData[offset++] = 0.0; // arrival_time
        testRayData[offset++] = 0.0; // bounce_count
        testRayData[offset++] = 1.0; // active

        // Material data (vec4)
        testRayData[offset++] = -1.0; // last_material_id
        testRayData[offset++] = 0.0;  // y padding
        testRayData[offset++] = 0.0;  // z padding
        testRayData[offset++] = 0.0;  // w padding

        // Write to the last ray slot to avoid conflicts
        const lastRayOffset = (this.config.maxRays - 1) * 28 * 4; // 28 floats * 4 bytes per float
        this.device.queue.writeBuffer(this.rayBuffer, lastRayOffset, testRayData);

        // Read it back to verify
        const testStats = await this.getRayStatistics();
        const testRays = await this.getFirstRays(this.config.maxRays);
        const lastRay = testRays[this.config.maxRays - 1];

        console.log('Test ray verification:', {
            lastRayEnergy: lastRay.energy,
            lastRayActive: lastRay.active,
            lastRayOrigin: Array.from(lastRay.origin),
            expectedEnergy: 999.0,
            bufferWorking: lastRay.energy === 999.0
        });
    }

    /**
     * Create empty debug stats structure
     */
    private createEmptyDebugStats(): RaytracingDebugStats {
        return {
            currentFrame: 0,
            totalFrames: 0,
            raysGenerated: 0,
            raysActive: 0,
            raysTerminated: 0,
            totalBounces: 0,
            averageBounces: 0,
            maxBounces: 0,
            totalEnergyGenerated: 0,
            totalEnergyCollected: 0,
            energyLoss: 0,
            raysCollected: 0,
            collectionRadius: 0,
            averageArrivalTime: 0,
            generationTime: 0,
            bouncingTime: 0,
            collectionTime: 0,
            totalTime: 0,
            bufferSizes: {
                rayBuffer: 0,
                impulseBuffer: 0,
                stagingBuffer: 0
            },
            errors: [],
            warnings: []
        };
    }

    /**
     * Start debug timing for current frame
     */
    private startDebugTiming(): void {
        if (this.debugConfig.enablePerformanceMonitoring) {
            this.frameStartTime = performance.now();
        }
    }

    /**
     * Log detailed debug information
     */
    private logDebugInfo(): void {
        if (!this.debugConfig.enableDetailedLogging) return;
        if (this.currentFrame % this.debugConfig.logInterval !== 0) return;

        console.group(`🔍 Raytracing Debug - Frame ${this.currentFrame}`);

        // Basic stats
        console.log(`📊 Ray Stats:`, {
            generated: this.debugStats.raysGenerated,
            active: this.debugStats.raysActive,
            collected: this.debugStats.raysCollected,
            terminated: this.debugStats.raysTerminated
        });

        // Energy tracking
        if (this.debugConfig.enableEnergyTracking) {
            console.log(`⚡ Energy Stats:`, {
                generated: this.debugStats.totalEnergyGenerated.toFixed(4),
                collected: this.debugStats.totalEnergyCollected.toFixed(4),
                loss: this.debugStats.energyLoss.toFixed(4),
                efficiency: this.debugStats.totalEnergyGenerated > 0 ?
                    ((this.debugStats.totalEnergyCollected / this.debugStats.totalEnergyGenerated) * 100).toFixed(2) + '%' : '0%'
            });
        }

        // Performance
        if (this.debugConfig.enablePerformanceMonitoring) {
            console.log(`⏱️ Performance:`, {
                total: this.debugStats.totalTime.toFixed(2) + 'ms',
                generation: this.debugStats.generationTime.toFixed(2) + 'ms',
                bouncing: this.debugStats.bouncingTime.toFixed(2) + 'ms',
                collection: this.debugStats.collectionTime.toFixed(2) + 'ms'
            });
        }

        // Errors and warnings
        if (this.debugStats.errors.length > 0) {
            console.error(`❌ Errors:`, this.debugStats.errors);
        }
        if (this.debugStats.warnings.length > 0) {
            console.warn(`⚠️ Warnings:`, this.debugStats.warnings);
        }

        console.groupEnd();
    }

    /**
     * Validate ray data for debugging
     */
    private async validateRayData(): Promise<void> {
        if (!this.debugConfig.enableRayValidation) return;

        try {
            const rays = await this.getFirstRays(Math.min(this.debugConfig.maxLoggedRays, this.config.maxRays));

            let activeCount = 0;
            let totalEnergy = 0;
            let maxBounces = 0;

            for (const ray of rays) {
                if (ray.active) {
                    activeCount++;
                    totalEnergy += ray.energy;
                    maxBounces = Math.max(maxBounces, ray.bounceCount);
                }
            }

            this.debugStats.raysActive = activeCount;
            this.debugStats.totalEnergyGenerated = totalEnergy;
            this.debugStats.maxBounces = maxBounces;

            // Log first few rays for detailed inspection
            if (this.currentFrame % (this.debugConfig.logInterval * 5) === 0) {
                console.log(`🔬 Ray Details (first ${Math.min(3, rays.length)}):`,
                    rays.slice(0, 3).map(ray => ({
                        active: ray.active,
                        energy: ray.energy.toFixed(4),
                        bounces: ray.bounceCount,
                        origin: [ray.origin[0].toFixed(2), ray.origin[1].toFixed(2), ray.origin[2].toFixed(2)],
                        direction: [ray.direction[0].toFixed(2), ray.direction[1].toFixed(2), ray.direction[2].toFixed(2)]
                    }))
                );
            }

        } catch (error) {
            this.debugStats.errors.push(`Ray validation failed: ${error}`);
        }
    }

    /**
     * Get current debug statistics
     */
    getDebugStats(): RaytracingDebugStats {
        return { ...this.debugStats };
    }

    /**
     * Update debug configuration
     */
    setDebugConfig(config: Partial<RaytracingDebugConfig>): void {
        this.debugConfig = { ...this.debugConfig, ...config };
    }

    /**
     * Dispose of GPU resources
     */
    dispose(): void {
        this.rayBuffer?.destroy();
        this.rayBufferStaging?.destroy();
        this.generationParamsBuffer?.destroy();
        this.bouncingParamsBuffer?.destroy();
        this.materialsBuffer?.destroy();
        this.impulseResponseBuffer?.destroy();
        this.statisticsBuffer?.destroy();

        this.rayBuffer = null;
        this.rayBufferStaging = null;
        this.generationParamsBuffer = null;
        this.bouncingParamsBuffer = null;
        this.materialsBuffer = null;
        this.impulseResponseBuffer = null;
        this.statisticsBuffer = null;

        this.generationPipeline = null;
        this.bouncingPipeline = null;
        this.collectionPipeline = null;
        this.normalizationPipeline = null;

        this.generationBindGroup = null;
        this.bouncingBindGroup = null;
        this.collectionBindGroup = null;
        this.normalizationBindGroup = null;

        this.initialized = false;
    }
}
