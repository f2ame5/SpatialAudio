import { vec3 } from 'gl-matrix';
import { Camera } from '../camera/camera';

interface RoomAcoustics {
    rt60Low: number;
    rt60Mid: number;
    rt60High: number;
    airAbsorptionLow: number;
    airAbsorptionMid: number;
    airAbsorptionHigh: number;
    earlyReflectionTime: number;
    roomVolume: number;
    totalSurfaceArea: number;
}

interface RayHit {
    position: vec3;
    time: number;
    energyLow: number;
    energyMid: number;
    energyHigh: number;
    bounces: number;
    phase: number;        // Phase of the wave at hit point
    frequency: number;    // Frequency of the ray
    dopplerShift: number; // Doppler shift at this point
}

interface WaveProperties {
    phase: number;
    frequency: number;
    dopplerShift: number;
}

export class SpatialAudioProcessor {
    private device: GPUDevice;
    private computePipeline!: GPUComputePipeline;
    private bindGroup!: GPUBindGroup;
    private listenerBuffer: GPUBuffer;
    private rayHitsBuffer!: GPUBuffer;
    private spatialIRBuffer!: GPUBuffer;
    private paramsBuffer: GPUBuffer;
    private acousticsBuffer: GPUBuffer;
    private wavePropertiesBuffer: GPUBuffer;
    private outputBuffer: GPUBuffer;
    private sampleRate: number;
    private readonly WORKGROUP_SIZE = 256;

    constructor(device: GPUDevice, sampleRate: number = 44100) {
        this.device = device;
        this.sampleRate = sampleRate;

        // Create uniform buffers with proper alignment and labels
        this.listenerBuffer = device.createBuffer({
            size: 64,  // 4 vec3f (16 bytes each)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Listener Data Buffer'
        });

        this.paramsBuffer = device.createBuffer({
            size: 48,  // 12 floats * 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Audio Params Buffer'
        });

        this.acousticsBuffer = device.createBuffer({
            size: 128, // 32 floats * 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Room Acoustics Buffer'
        });

        this.wavePropertiesBuffer = device.createBuffer({
            size: 16, // 4 floats * 4 bytes per wave property
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Wave Properties Buffer'
        });

        this.outputBuffer = device.createBuffer({
            size: 16, // 4 floats * 4 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Output Buffer'
        });

        // Initialize the pipeline and other buffers
        this.initializeAsync();
    }

    private async initializeAsync(): Promise<void> {
        await this.createPipeline();
    }

    private async createPipeline(): Promise<void> {
        const shaderModule = this.device.createShaderModule({
            code: await fetch('/src/raytracer/shaders/spatial_audio.wgsl').then(r => r.text()),
            label: 'Spatial Audio Shader'
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform', minBindingSize: 64 }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage', minBindingSize: 96 } // Updated size for RayHit with wave properties
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform', minBindingSize: 48 }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform', minBindingSize: 108 }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage', minBindingSize: 128 }
                },
                {
                    binding: 6,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage', minBindingSize: 16 }
                }
            ],
            label: 'Spatial Audio Bind Group Layout'
        });

        // Create pipeline layout and compute pipeline
        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
            label: 'Spatial Audio Pipeline Layout'
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            },
            label: 'Spatial Audio Pipeline'
        });
    }

    private createOrResizeBuffers(hitCount: number): void {
        // Calculate buffer sizes with proper alignment
        const rayHitSize = 80; // Size of RayHit struct in shader (20 floats * 4 bytes)
        const rayHitsSize = Math.max(hitCount * rayHitSize, rayHitSize);
        const wavePropsSize = Math.max(hitCount * 16, 16); // 4 floats * 4 bytes per wave property
        const spatialIRSize = Math.max(hitCount * 16, 16); // vec4f per hit

        // Create or resize ray hits buffer
        if (!this.rayHitsBuffer || this.rayHitsBuffer.size < rayHitsSize) {
            if (this.rayHitsBuffer) this.rayHitsBuffer.destroy();
            this.rayHitsBuffer = this.device.createBuffer({
                size: rayHitsSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'Ray Hits Buffer'
            });
        }

        // Create or resize wave properties buffer
        if (!this.wavePropertiesBuffer || this.wavePropertiesBuffer.size < wavePropsSize) {
            if (this.wavePropertiesBuffer) this.wavePropertiesBuffer.destroy();
            this.wavePropertiesBuffer = this.device.createBuffer({
                size: wavePropsSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'Wave Properties Buffer'
            });
        }

        // Create or resize spatial IR buffer
        if (!this.spatialIRBuffer || this.spatialIRBuffer.size < spatialIRSize) {
            if (this.spatialIRBuffer) this.spatialIRBuffer.destroy();
            this.spatialIRBuffer = this.device.createBuffer({
                size: spatialIRSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                label: 'Spatial IR Buffer'
            });

            // Initialize spatialIR buffer with zeros
            const zeros = new Float32Array(hitCount * 4); // vec4f per hit
            this.device.queue.writeBuffer(this.spatialIRBuffer, 0, zeros);
        }

        this.updateBindGroup();
    }

    private updateBindGroup(): void {
        this.bindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.listenerBuffer } },
                { binding: 1, resource: { buffer: this.rayHitsBuffer } },
                { binding: 2, resource: { buffer: this.spatialIRBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } },
                { binding: 4, resource: { buffer: this.acousticsBuffer } },
                { binding: 5, resource: { buffer: this.wavePropertiesBuffer } },
                { binding: 6, resource: { buffer: this.outputBuffer } }
            ],
            label: 'Spatial Audio Bind Group'
        });
    }

    public async processSpatialAudio(
        camera: Camera,
        rayHits: Array<{
            position: vec3;
            time: number;
            energyLow: number;
            energyMid: number;
            energyHigh: number;
            phase: number;
            frequency: number;
            dopplerShift: number
        }>,
        params: any,
        room: Room
    ): Promise<[Float32Array, Float32Array]> {
        if (rayHits.length === 0) {
            console.warn('No ray hits to process');
            return [new Float32Array(0), new Float32Array(0)];
        }

        // Verify ray hit data
        console.log(`Processing ${rayHits.length} ray hits`);
        console.log('Sample ray hit:', rayHits[0]);

        // Ensure buffers are properly sized
        this.createOrResizeBuffers(rayHits.length);

        // Prepare ray hits data
        const rayHitsData = new Float32Array(rayHits.length * 20);
        const wavePropsData = new Float32Array(rayHits.length * 4);

        // Fill ray hits and wave properties data with validation
        rayHits.forEach((hit, i) => {
            const baseIndex = i * 20;
            
            // Validate position
            const position = hit.position || vec3.create();
            rayHitsData[baseIndex] = position[0];
            rayHitsData[baseIndex + 1] = position[1];
            rayHitsData[baseIndex + 2] = position[2];
            
            // Validate time
            rayHitsData[baseIndex + 3] = Math.max(hit.time || 0, 0);
            
            // Calculate and validate energy values
            const maxEnergy = Math.max(
                Math.max(hit.energyLow || 0, hit.energyMid || 0),
                hit.energyHigh || 0
            );
            
            if (maxEnergy === 0) {
                console.warn(`Ray hit ${i} has zero energy`);
            }

            // Store energy values
            rayHitsData[baseIndex + 7] = maxEnergy;
            
            // Store frequency bands with validation
            rayHitsData[baseIndex + 8] = Math.max(hit.energyLow || 0, 0);
            rayHitsData[baseIndex + 9] = Math.max(hit.energyLow * 0.8 + (hit.energyMid || 0) * 0.2, 0);
            rayHitsData[baseIndex + 10] = Math.max(hit.energyLow * 0.7 + (hit.energyMid || 0) * 0.3, 0);
            rayHitsData[baseIndex + 11] = Math.max(hit.energyMid || 0, 0);
            rayHitsData[baseIndex + 12] = Math.max(hit.energyMid || 0, 0);
            rayHitsData[baseIndex + 13] = Math.max((hit.energyMid || 0) * 0.3 + (hit.energyHigh || 0) * 0.7, 0);
            rayHitsData[baseIndex + 14] = Math.max((hit.energyHigh || 0) * 0.8 + (hit.energyMid || 0) * 0.2, 0);
            rayHitsData[baseIndex + 15] = Math.max(hit.energyHigh || 0, 0);

            // Store wave properties with validation
            rayHitsData[baseIndex + 16] = hit.phase || 0;
            rayHitsData[baseIndex + 17] = Math.max(hit.frequency || 440, 20); // Minimum 20Hz
            rayHitsData[baseIndex + 18] = Math.max(hit.dopplerShift || 1, 0.1); // Minimum 0.1
            rayHitsData[baseIndex + 19] = 1.0;

            // Store wave properties in separate buffer
            const waveBaseIndex = i * 4;
            wavePropsData[waveBaseIndex] = hit.phase || 0;
            wavePropsData[waveBaseIndex + 1] = Math.max(hit.frequency || 440, 20);
            wavePropsData[waveBaseIndex + 2] = Math.max(hit.dopplerShift || 1, 0.1);
            wavePropsData[waveBaseIndex + 3] = 1.0;
        });

        // Write validated data to GPU buffers
        this.device.queue.writeBuffer(this.rayHitsBuffer, 0, rayHitsData);
        this.device.queue.writeBuffer(this.wavePropertiesBuffer, 0, wavePropsData);

        // Update listener data
        const front = camera.getFront();
        const forward = new Float32Array([front[0], front[1], front[2]]);
        const position = camera.getPosition();
        const up = camera.getUp();
        
        // Calculate right vector as cross product of front and up
        const right = vec3.create();
        vec3.cross(right, front, up);
        vec3.normalize(right, right);
        
        const listenerData = new Float32Array([
            position[0], position[1], position[2],
            0, // padding
            forward[0], forward[1], forward[2],
            0, // padding
            up[0], up[1], up[2],
            0, // padding
            right[0], right[1], right[2],
            0  // padding
        ]);
        this.device.queue.writeBuffer(this.listenerBuffer, 0, listenerData);

        // Create command encoder and pass
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();

        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.bindGroup);

        // Dispatch workgroups
        const workgroupCount = Math.ceil(rayHits.length / this.WORKGROUP_SIZE);
        computePass.dispatchWorkgroups(workgroupCount);
        computePass.end();

        // Create buffer for reading results
        const readbackBuffer = this.device.createBuffer({
            size: rayHits.length * 16, // vec4f per hit
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        // Copy results to readback buffer
        commandEncoder.copyBufferToBuffer(
            this.spatialIRBuffer,
            0,
            readbackBuffer,
            0,
            rayHits.length * 16
        );

        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);

        // Read results
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const results = new Float32Array(readbackBuffer.getMappedRange());

        // Separate left and right channels
        const leftChannel = new Float32Array(rayHits.length);
        const rightChannel = new Float32Array(rayHits.length);
        for (let i = 0; i < rayHits.length; i++) {
            leftChannel[i] = results[i * 4];
            rightChannel[i] = results[i * 4 + 1];
        }

        // Cleanup
        readbackBuffer.unmap();
        readbackBuffer.destroy();

        return [leftChannel, rightChannel];
    }
}