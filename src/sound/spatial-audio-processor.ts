import { vec3 } from 'gl-matrix';
import { Camera } from '../camera/camera';

// Define the 8 frequency bands explicitly for clarity and consistency
export const FREQUENCY_BANDS_8 = [63, 125, 250, 500, 1000, 2000, 4000, 8000] as const;
export type FrequencyBandType = typeof FREQUENCY_BANDS_8[number];

// Updated RayHit interface to use 8 bands
interface RayHit {
    position: vec3;
    time: number;
    energy: Record<FrequencyBandType, number>; // Use Record for structured access
    bounces: number;
    phase: number;        // Phase of the wave at hit point
    frequency: number;    // Frequency of the primary ray (Hz)
    dopplerShift: number; // Doppler shift factor
}

// Updated RoomAcoustics interface for 8 bands
interface RoomAcoustics {
    rt60_63: number;
    rt60_125: number;
    rt60_250: number;
    rt60_500: number;
    rt60_1k: number;
    rt60_2k: number;
    rt60_4k: number;
    rt60_8k: number;
    airAbsorption_63: number;
    airAbsorption_125: number;
    airAbsorption_250: number;
    airAbsorption_500: number;
    airAbsorption_1k: number;
    airAbsorption_2k: number;
    airAbsorption_4k: number;
    airAbsorption_8k: number;
    earlyReflectionTime: number;
    roomVolume: number;
    totalSurfaceArea: number;
}

// Assumes the Room type is defined elsewhere in the project
interface Room {
    acoustics: RoomAcoustics;
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
    private sampleRate: number;
    private readonly WORKGROUP_SIZE = 256;

    constructor(device: GPUDevice, sampleRate: number = 44100) {
        this.device = device;
        this.sampleRate = sampleRate;

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
            // 8 RT60s + 8 Air Absorptions + 3 others = 19 floats. Padded to 80 bytes.
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Room Acoustics Buffer (8-Band)'
        });

        this.initializeAsync();
    }

    private async initializeAsync(): Promise<void> {
        await this.createPipeline();
    }

    private async createPipeline(): Promise<void> {
        // NOTE: The accompanying WGSL shader must be updated to match these changes.
        const shaderModule = this.device.createShaderModule({
            code: await fetch('/src/raytracer/shaders/spatial_audio.wgsl').then(r => r.text()),
            label: 'Spatial Audio Shader'
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { // Listener data (camera pos, orientation)
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                },
                { // Input ray hits from the ray tracer
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                { // Output impulse response data
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                },
                { // General audio parameters
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                },
                { // Room acoustic properties (8-band)
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                }
            ],
            label: 'Spatial Audio Bind Group Layout'
        });

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
        // Size of RayHit struct in shader: 16 floats * 4 bytes/float = 64 bytes
        const rayHitStructSize = 64;
        const rayHitsSize = Math.max(hitCount * rayHitStructSize, rayHitStructSize);
        const spatialIRSize = Math.max(hitCount * 16, 16); // vec4f per hit

        if (!this.rayHitsBuffer || this.rayHitsBuffer.size < rayHitsSize) {
            if (this.rayHitsBuffer) this.rayHitsBuffer.destroy();
            this.rayHitsBuffer = this.device.createBuffer({
                size: rayHitsSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'Ray Hits Buffer'
            });
        }

        if (!this.spatialIRBuffer || this.spatialIRBuffer.size < spatialIRSize) {
            if (this.spatialIRBuffer) this.spatialIRBuffer.destroy();
            this.spatialIRBuffer = this.device.createBuffer({
                size: spatialIRSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                label: 'Spatial IR Buffer'
            });

            const zeros = new Float32Array(hitCount * 4);
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
            ],
            label: 'Spatial Audio Bind Group'
        });
    }

    public async processSpatialAudio(
        camera: Camera,
        rayHits: Array<RayHit>,
        params: any,
        room: Room
    ): Promise<[Float32Array, Float32Array]> {
        if (!rayHits || rayHits.length === 0) {
            console.warn('No ray hits to process');
            return [new Float32Array(0), new Float32Array(0)];
        }

        this.createOrResizeBuffers(rayHits.length);

        // Prepare ray hits data for the GPU buffer (16 floats per hit)
        const rayHitsData = new Float32Array(rayHits.length * 16);

        rayHits.forEach((hit, i) => {
            const baseIndex = i * 16;
            
            const position = hit.position || vec3.create();
            rayHitsData[baseIndex] = position[0];
            rayHitsData[baseIndex + 1] = position[1];
            rayHitsData[baseIndex + 2] = position[2];
            
            rayHitsData[baseIndex + 3] = Math.max(hit.time || 0, 0);

            // Pack 8 energy bands, ensuring order matches FREQUENCY_BANDS_8
            FREQUENCY_BANDS_8.forEach((band, bandIndex) => {
                rayHitsData[baseIndex + 4 + bandIndex] = Math.max(hit.energy?.[band] || 0, 0);
            });

            rayHitsData[baseIndex + 12] = hit.bounces || 0;
            rayHitsData[baseIndex + 13] = hit.phase || 0;
            rayHitsData[baseIndex + 14] = Math.max(hit.frequency || 440, 20);
            rayHitsData[baseIndex + 15] = Math.max(hit.dopplerShift || 1, 0.1);
        });

        this.device.queue.writeBuffer(this.rayHitsBuffer, 0, rayHitsData);

        const front = camera.getFront();
        const position = camera.getPosition();
        const up = camera.getUp();
        
        const right = vec3.create();
        vec3.cross(right, front, up);
        vec3.normalize(right, right);
        
        const listenerData = new Float32Array([
            position[0], position[1], position[2], 0,
            front[0],    front[1],    front[2],    0,
            up[0],       up[1],       up[2],       0,
            right[0],    right[1],    right[2],    0
        ]);
        this.device.queue.writeBuffer(this.listenerBuffer, 0, listenerData);

        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.bindGroup);

        const workgroupCount = Math.ceil(rayHits.length / this.WORKGROUP_SIZE);
        computePass.dispatchWorkgroups(workgroupCount);
        computePass.end();

        const readbackBuffer = this.device.createBuffer({
            size: this.spatialIRBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        commandEncoder.copyBufferToBuffer(
            this.spatialIRBuffer, 0,
            readbackBuffer, 0,
            this.spatialIRBuffer.size
        );

        this.device.queue.submit([commandEncoder.finish()]);

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const resultsData = new Float32Array(readbackBuffer.getMappedRange());

        const leftChannel = new Float32Array(rayHits.length);
        const rightChannel = new Float32Array(rayHits.length);
        for (let i = 0; i < rayHits.length; i++) {
            leftChannel[i] = resultsData[i * 4];
            rightChannel[i] = resultsData[i * 4 + 1];
        }

        readbackBuffer.unmap();
        readbackBuffer.destroy();

        return [leftChannel, rightChannel];
    }
}