/**
 * Acoustic Raytracer - WebGPU-based acoustic ray simulation
 * Generates and simulates acoustic rays for spatial audio
 */

import { vec3 } from 'gl-matrix';
import { RoomAcoustics } from '../room/room-acoustics';

export interface RaytracerConfig {
    maxRays: number;
    maxBounces: number;
    minEnergy: number;
    speedOfSound: number;
    initialEnergy: number;
    sourceRadius: number;
}

export interface RayData {
    origin: vec3;
    direction: vec3;
    energy: number;
    pathLength: number;
    bounceCount: number;
    active: boolean;
}

export class AcousticRaytracer {
    private device: GPUDevice;
    private config: RaytracerConfig;
    private roomAcoustics: RoomAcoustics;
    
    // WebGPU resources
    private rayBuffer: GPUBuffer | null = null;
    private generationPipeline: GPUComputePipeline | null = null;
    private bouncingPipeline: GPUComputePipeline | null = null;
    private generationBindGroup: GPUBindGroup | null = null;
    private bouncingBindGroup: GPUBindGroup | null = null;
    private generationUniformBuffer: GPUBuffer | null = null;
    private bouncingUniformBuffer: GPUBuffer | null = null;
    private materialBuffer: GPUBuffer | null = null;
    private materialNameToIdMap: Map<string, number> = new Map();
    
    // Current ray data for visualization
    private currentRays: RayData[] = [];
    private isInitialized = false;

    constructor(device: GPUDevice, roomAcoustics: RoomAcoustics, config?: Partial<RaytracerConfig>) {
        this.device = device;
        this.roomAcoustics = roomAcoustics;
        
        // Default configuration
        this.config = {
            maxRays: 1, // Start with just 1 ray
            maxBounces: 50, // More bounces to see the ray travel
            minEnergy: 0.001,
            speedOfSound: 343.0,
            initialEnergy: 1.0,
            sourceRadius: 0.01,
            ...config
        };
    }

    /**
     * Initialize the raytracer
     */
    async initialize(): Promise<void> {
        if (this.isInitialized) return;

        try {
            await this.createBuffers();
            await this.createPipelines();
            await this.createBindGroups();
            
            this.isInitialized = true;
            console.log('Acoustic raytracer initialized successfully');
        } catch (error) {
            console.error('Failed to initialize acoustic raytracer:', error);
            throw error;
        }
    }

    /**
     * Create GPU buffers
     */
    private async createBuffers(): Promise<void> {
        // Ray buffer - stores ray data
        // Ray struct: 7 vec4<f32> = 7 * 16 bytes = 112 bytes per ray
        const rayBufferSize = this.config.maxRays * 112;
        this.rayBuffer = this.device.createBuffer({
            size: rayBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: 'Ray Buffer'
        });

        // Uniform buffer for ray generation parameters (128 bytes)
        this.generationUniformBuffer = this.device.createBuffer({
            size: 128, // RayGenerationParams structure size
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Ray Generation Uniform Buffer'
        });

        // Uniform buffer for ray bouncing parameters (128 bytes)
        this.bouncingUniformBuffer = this.device.createBuffer({
            size: 128, // RayBouncingParams structure size
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Ray Bouncing Uniform Buffer'
        });

        // Material buffer
        const materialsMap = this.roomAcoustics.getMaterialsMap();
        const materials = Array.from(materialsMap.values());
        const materialData = new Float32Array(materials.length * 20); // 20 floats per material

        materials.forEach((material, index: number) => {
            this.materialNameToIdMap.set(material.name, index);
            const offset = index * 20;
            // Absorption coefficients (8 frequencies)
            materialData.set(material.absorption, offset);
            // Scattering coefficients (8 frequencies)
            materialData.set(material.scattering, offset + 8);
            // Properties: impedance, roughness
            materialData[offset + 16] = material.impedance;
            materialData[offset + 17] = material.roughness;
        });

        this.materialBuffer = this.device.createBuffer({
            size: materialData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Material Buffer'
        });
        
        this.device.queue.writeBuffer(this.materialBuffer, 0, materialData);
    }

    /**
     * Create compute pipelines
     */
    private async createPipelines(): Promise<void> {
        // Load shaders - use dynamic imports for Vite
        const generationShaderImport = await import('../shaders/ray-generation.wgsl?raw');
        const generationShaderCode = generationShaderImport.default;

        const bouncingShaderImport = await import('../shaders/ray-bouncing-simple.wgsl?raw');
        const bouncingShaderCode = bouncingShaderImport.default;

        // Create shader modules
        const generationShaderModule = this.device.createShaderModule({
            code: generationShaderCode,
            label: 'Ray Generation Shader'
        });

        const bouncingShaderModule = this.device.createShaderModule({
            code: bouncingShaderCode,
            label: 'Ray Bouncing Shader'
        });

        // Create pipelines
        this.generationPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: generationShaderModule,
                entryPoint: 'main'
            },
            label: 'Ray Generation Pipeline'
        });

        this.bouncingPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: bouncingShaderModule,
                entryPoint: 'main'
            },
            label: 'Ray Bouncing Pipeline'
        });
    }

    /**
     * Create bind groups
     */
    private async createBindGroups(): Promise<void> {
        if (!this.generationPipeline || !this.bouncingPipeline || !this.rayBuffer || !this.generationUniformBuffer || !this.bouncingUniformBuffer || !this.materialBuffer) {
            throw new Error('Pipelines or buffers not created');
        }

        // Generation bind group
        this.generationBindGroup = this.device.createBindGroup({
            layout: this.generationPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.rayBuffer } },
                { binding: 1, resource: { buffer: this.generationUniformBuffer } }
            ],
            label: 'Ray Generation Bind Group'
        });

        // Bouncing bind group
        this.bouncingBindGroup = this.device.createBindGroup({
            layout: this.bouncingPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.rayBuffer } },
                { binding: 1, resource: { buffer: this.bouncingUniformBuffer } },
                { binding: 2, resource: { buffer: this.materialBuffer } }
            ],
            label: 'Ray Bouncing Bind Group'
        });
    }

    /**
     * Generate and simulate rays from a source position
     */
    async simulateRays(sourcePosition: vec3, listenerPosition: vec3): Promise<RayData[]> {
        if (!this.isInitialized) {
            console.log('Initializing raytracer...');
            await this.initialize();
        }

        try {
            // Update uniform buffer with current parameters for ray generation
            await this.updateGenerationUniforms(sourcePosition, listenerPosition);

            // Generate initial rays
            await this.generateRays();

            // Update uniform buffer with current parameters for ray bouncing
            await this.updateBouncingUniforms();

            // Simulate ray bouncing
            await this.simulateBouncing();

            // Read back ray data for visualization
            await this.readRayData();

            console.log(`Ray simulation complete. Generated ${this.currentRays.length} active rays.`);
            return this.currentRays;
        } catch (error) {
            console.error('Ray simulation failed:', error);
            return [];
        }
    }

    /**
     * Update uniform buffer with current parameters for ray generation
     */
    private async updateGenerationUniforms(sourcePosition: vec3, _listenerPosition: vec3): Promise<void> {
        if (!this.generationUniformBuffer) return;

        // Create uniform data matching RayGenerationParams structure
        const uniformData = new Float32Array(32); // 128 bytes / 4 bytes per float
        let offset = 0;

        // source_position: vec3<f32> + source_radius: f32
        uniformData[offset++] = sourcePosition[0];
        uniformData[offset++] = sourcePosition[1];
        uniformData[offset++] = sourcePosition[2];
        uniformData[offset++] = this.config.sourceRadius;

        // ray_count: u32
        uniformData[offset++] = this.config.maxRays;

        // initial_energy: f32
        uniformData[offset++] = this.config.initialEnergy;

        // time: f32
        uniformData[offset++] = 0.0; // Current time

        // seed: u32
        uniformData[offset++] = Math.floor(Math.random() * 1000000);

        // frequency_weights_low: vec4<f32> (125, 250, 500, 1k Hz)
        uniformData[offset++] = 1.0; // 125 Hz
        uniformData[offset++] = 1.0; // 250 Hz
        uniformData[offset++] = 1.0; // 500 Hz
        uniformData[offset++] = 1.0; // 1 kHz

        // frequency_weights_high: vec4<f32> (2k, 4k, 8k, 16k Hz)
        uniformData[offset++] = 1.0; // 2 kHz
        uniformData[offset++] = 1.0; // 4 kHz
        uniformData[offset++] = 1.0; // 8 kHz
        uniformData[offset++] = 1.0; // 16 kHz

        // distribution_type: u32 (0 = uniform sphere)
        uniformData[offset++] = 0;

        // cone_angle: f32
        uniformData[offset++] = 0.0;

        // padding: vec2<f32>
        uniformData[offset++] = 0.0;
        uniformData[offset++] = 0.0;

        this.device.queue.writeBuffer(this.generationUniformBuffer, 0, uniformData);
    }

    /**
     * Update uniform buffer with current parameters for ray bouncing
     */
    private async updateBouncingUniforms(): Promise<void> {
        if (!this.bouncingUniformBuffer) return;

        // Create uniform data matching RayBouncingParams structure
        // room_min: vec3<f32>
        // room_max: vec3<f32>
        // max_bounces: u32
        // min_energy: f32
        // speed_of_sound: f32
        // time_step: f32
        // air_absorption_low: vec4<f32>
        // air_absorption_high: vec4<f32>
        // surface_materials: vec4<u32>
        // surface_materials_zw: vec2<u32>
        // padding: vec2<f32>

        const uniformData = new Float32Array(32); // Max size for uniform buffer, 128 bytes
        let offset = 0;

        const roomDimensions = this.roomAcoustics.getRoomDimensions();
        uniformData[offset++] = roomDimensions.min[0];
        uniformData[offset++] = roomDimensions.min[1];
        uniformData[offset++] = roomDimensions.min[2];
        offset++; // Padding for vec3 alignment

        uniformData[offset++] = roomDimensions.max[0];
        uniformData[offset++] = roomDimensions.max[1];
        uniformData[offset++] = roomDimensions.max[2];
        offset++; // Padding for vec3 alignment

        uniformData[offset++] = this.config.maxBounces;
        uniformData[offset++] = this.config.minEnergy;
        uniformData[offset++] = this.config.speedOfSound;
        uniformData[offset++] = 0.01; // time_step - placeholder, adjust as needed

        // Air absorption coefficients (example values, adjust based on real data)
        // These are typically frequency-dependent.
        // For simplicity, using placeholder values.
        uniformData[offset++] = 0.0001; // 125 Hz
        uniformData[offset++] = 0.0002; // 250 Hz
        uniformData[offset++] = 0.0004; // 500 Hz
        uniformData[offset++] = 0.0008; // 1k Hz

        uniformData[offset++] = 0.0016; // 2k Hz
        uniformData[offset++] = 0.0032; // 4k Hz
        uniformData[offset++] = 0.0064; // 8k Hz
        uniformData[offset++] = 0.0128; // 16k Hz

        // Surface material IDs
        const surfaceMaterialNames = this.roomAcoustics.getRoom().getSurfaceMaterialNames();
        const surfaceMaterialIds: number[] = [];
        for (const name of surfaceMaterialNames) {
            const id = this.materialNameToIdMap.get(name);
            if (id !== undefined) {
                surfaceMaterialIds.push(id);
            } else {
                console.warn(`Material ID not found for name: ${name}. Using 0.`);
                surfaceMaterialIds.push(0); // Default to material 0 if not found
            }
        }

        uniformData[offset++] = surfaceMaterialIds[0]; // +X
        uniformData[offset++] = surfaceMaterialIds[1]; // -X
        uniformData[offset++] = surfaceMaterialIds[2]; // +Y
        uniformData[offset++] = surfaceMaterialIds[3]; // -Y

        uniformData[offset++] = surfaceMaterialIds[4]; // +Z
        uniformData[offset++] = surfaceMaterialIds[5]; // -Z
        uniformData[offset++] = 0; // Padding
        uniformData[offset++] = 0; // Padding

        this.device.queue.writeBuffer(this.bouncingUniformBuffer, 0, uniformData);
    }

    /**
     * Update uniform buffer with current parameters for ray bouncing
     */
    private async updateBouncingUniforms(): Promise<void> {
        if (!this.uniformBuffer) return;

        // Create uniform data matching RayBouncingParams structure
        // room_min: vec3<f32>
        // room_max: vec3<f32>
        // max_bounces: u32
        // min_energy: f32
        // speed_of_sound: f32
        // time_step: f32
        // air_absorption_low: vec4<f32>
        // air_absorption_high: vec4<f32>
        // surface_materials: vec4<u32>
        // surface_materials_zw: vec2<u32>
        // padding: vec2<f32>

        const uniformData = new Float32Array(32); // Max size for uniform buffer, 128 bytes
        let offset = 0;

        const roomDimensions = this.roomAcoustics.getRoomDimensions();
        uniformData[offset++] = roomDimensions.min[0];
        uniformData[offset++] = roomDimensions.min[1];
        uniformData[offset++] = roomDimensions.min[2];
        offset++; // Padding for vec3 alignment

        uniformData[offset++] = roomDimensions.max[0];
        uniformData[offset++] = roomDimensions.max[1];
        uniformData[offset++] = roomDimensions.max[2];
        offset++; // Padding for vec3 alignment

        uniformData[offset++] = this.config.maxBounces;
        uniformData[offset++] = this.config.minEnergy;
        uniformData[offset++] = this.config.speedOfSound;
        uniformData[offset++] = 0.01; // time_step - placeholder, adjust as needed

        // Air absorption coefficients (example values, adjust based on real data)
        // These are typically frequency-dependent.
        // For simplicity, using placeholder values.
        uniformData[offset++] = 0.0001; // 125 Hz
        uniformData[offset++] = 0.0002; // 250 Hz
        uniformData[offset++] = 0.0004; // 500 Hz
        uniformData[offset++] = 0.0008; // 1k Hz

        uniformData[offset++] = 0.0016; // 2k Hz
        uniformData[offset++] = 0.0032; // 4k Hz
        uniformData[offset++] = 0.0064; // 8k Hz
        uniformData[offset++] = 0.0128; // 16k Hz

        // Surface material IDs
        const surfaceMaterialNames = this.roomAcoustics.getRoom().getSurfaceMaterialNames();
        const surfaceMaterialIds: number[] = [];
        for (const name of surfaceMaterialNames) {
            const id = this.materialNameToIdMap.get(name);
            if (id !== undefined) {
                surfaceMaterialIds.push(id);
            } else {
                console.warn(`Material ID not found for name: ${name}. Using 0.`);
                surfaceMaterialIds.push(0); // Default to material 0 if not found
            }
        }

        uniformData[offset++] = surfaceMaterialIds[0]; // +X
        uniformData[offset++] = surfaceMaterialIds[1]; // -X
        uniformData[offset++] = surfaceMaterialIds[2]; // +Y
        uniformData[offset++] = surfaceMaterialIds[3]; // -Y

        uniformData[offset++] = surfaceMaterialIds[4]; // +Z
        uniformData[offset++] = surfaceMaterialIds[5]; // -Z
        uniformData[offset++] = 0; // Padding
        uniformData[offset++] = 0; // Padding

        this.device.queue.writeBuffer(this.generationUniformBuffer, 0, uniformData);
    }

    /**
     * Update bouncing uniform buffer
     */
    private async updateBouncingUniforms(): Promise<void> {
        if (!this.bouncingUniformBuffer) return;

        const roomBounds = this.roomAcoustics.getRoomBounds();

        // Create uniform data matching RayBouncingParams structure
        const uniformData = new Float32Array(32); // 128 bytes / 4 bytes per float
        let offset = 0;

        // room_min: vec3<f32>
        uniformData[offset++] = roomBounds.min[0];
        uniformData[offset++] = roomBounds.min[1];
        uniformData[offset++] = roomBounds.min[2];
        uniformData[offset++] = 0; // padding

        // room_max: vec3<f32>
        uniformData[offset++] = roomBounds.max[0];
        uniformData[offset++] = roomBounds.max[1];
        uniformData[offset++] = roomBounds.max[2];
        uniformData[offset++] = 0; // padding

        // max_bounces: f32
        uniformData[offset++] = this.config.maxBounces;

        // min_energy: f32
        uniformData[offset++] = this.config.minEnergy;

        // speed_of_sound: f32
        uniformData[offset++] = this.config.speedOfSound;

        // time_step: f32
        uniformData[offset++] = 0.1; // Fixed time step

        // air_absorption_low: vec4<f32> (125, 250, 500, 1k Hz)
        const airAbsorption = this.roomAcoustics.getAirAbsorption();
        uniformData[offset++] = airAbsorption[0];
        uniformData[offset++] = airAbsorption[1];
        uniformData[offset++] = airAbsorption[2];
        uniformData[offset++] = airAbsorption[3];

        // air_absorption_high: vec4<f32> (2k, 4k, 8k, 16k Hz)
        uniformData[offset++] = airAbsorption[4];
        uniformData[offset++] = airAbsorption[5];
        uniformData[offset++] = airAbsorption[6];
        uniformData[offset++] = airAbsorption[7];

        this.device.queue.writeBuffer(this.bouncingUniformBuffer, 0, uniformData);
    }

    /**
     * Generate initial rays
     */
    private async generateRays(): Promise<void> {
        if (!this.generationPipeline || !this.generationBindGroup) {
            console.error('Generation pipeline or bind group not available');
            return;
        }

        const commandEncoder = this.device.createCommandEncoder({ label: 'Ray Generation' });
        const computePass = commandEncoder.beginComputePass({ label: 'Generate Rays' });

        computePass.setPipeline(this.generationPipeline);
        computePass.setBindGroup(0, this.generationBindGroup);

        // Dispatch enough workgroups to cover all rays
        const workgroupCount = Math.ceil(this.config.maxRays / 64);
        computePass.dispatchWorkgroups(workgroupCount);
        computePass.end();

        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    /**
     * Simulate ray bouncing
     */
    private async simulateBouncing(): Promise<void> {
        if (!this.bouncingPipeline || !this.bouncingBindGroup) {
            console.error('Bouncing pipeline or bind group not available');
            return;
        }

        // Simulate multiple bounces
        for (let bounce = 0; bounce < this.config.maxBounces; bounce++) {
            const commandEncoder = this.device.createCommandEncoder({ label: `Ray Bouncing ${bounce}` });
            const computePass = commandEncoder.beginComputePass({ label: `Bounce ${bounce}` });

            computePass.setPipeline(this.bouncingPipeline);
            computePass.setBindGroup(0, this.bouncingBindGroup);

            // Dispatch enough workgroups to cover all rays
            const workgroupCount = Math.ceil(this.config.maxRays / 64);
            computePass.dispatchWorkgroups(workgroupCount);
            computePass.end();

            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
        }
    }

    /**
     * Read ray data back from GPU for visualization
     */
    private async readRayData(): Promise<void> {
        if (!this.rayBuffer) return;

        // Create staging buffer
        const stagingBuffer = this.device.createBuffer({
            size: this.rayBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            label: 'Ray Staging Buffer'
        });

        // Copy data
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.rayBuffer, 0, stagingBuffer, 0, this.rayBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);

        // Read data
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = stagingBuffer.getMappedRange();
        const data = new Float32Array(arrayBuffer);

        // Parse ray data
        this.currentRays = [];
        for (let i = 0; i < this.config.maxRays; i++) {
            // Ray struct: 7 vec4<f32> = 28 floats per ray
            const offset = i * 28;

            const ray: RayData = {
                // origin: vec4<f32> at offset 0-3
                origin: vec3.fromValues(data[offset], data[offset + 1], data[offset + 2]),
                // direction: vec4<f32> at offset 4-7
                direction: vec3.fromValues(data[offset + 4], data[offset + 5], data[offset + 6]),
                // energy_phase: vec4<f32> at offset 8-11 (energy at index 8)
                energy: data[offset + 8],
                // path_data: vec4<f32> at offset 20-23 (path_length at 20, bounce_count at 22, active at 23)
                pathLength: data[offset + 20],
                bounceCount: data[offset + 22],
                active: data[offset + 23] > 0.5
            };

            // Debug: Log first ray data
            if (i === 0) {
                console.log('Ray data:', {
                    origin: `[${ray.origin[0].toFixed(2)}, ${ray.origin[1].toFixed(2)}, ${ray.origin[2].toFixed(2)}]`,
                    direction: `[${ray.direction[0].toFixed(2)}, ${ray.direction[1].toFixed(2)}, ${ray.direction[2].toFixed(2)}]`,
                    energy: ray.energy.toFixed(3),
                    pathLength: ray.pathLength.toFixed(2),
                    bounceCount: ray.bounceCount,
                    active: ray.active
                });
            }

            if (ray.active && ray.energy > this.config.minEnergy) {
                this.currentRays.push(ray);
            }
        }

        stagingBuffer.unmap();
        stagingBuffer.destroy();
    }

    /**
     * Get current ray data for visualization
     */
    getCurrentRays(): RayData[] {
        return this.currentRays;
    }

    /**
     * Cleanup resources
     */
    destroy(): void {
        this.rayBuffer?.destroy();
        this.generationUniformBuffer?.destroy();
        this.bouncingUniformBuffer?.destroy();
        this.materialBuffer?.destroy();
        this.isInitialized = false;
    }
}
