# WebGPU Spatial Audio Implementation Guide
## Real-time Room Impulse Response Generation using Raytracing Shaders

This comprehensive guide explains how to implement GPU-accelerated spatial audio using WebGPU compute shaders for realistic room acoustics simulation.

## Table of Contents
1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Geometry Data Structure](#geometry-data-structure)
4. [WebGPU Compute Shader Implementation](#webgpu-compute-shader-implementation)
5. [Ray-Triangle Intersection Algorithm](#ray-triangle-intersection-algorithm)
6. [Acoustic Ray Tracing](#acoustic-ray-tracing)
7. [Impulse Response Generation](#impulse-response-generation)
8. [Ambisonic Encoding](#ambisonic-encoding)
9. [Filter Tree Processing](#filter-tree-processing)
10. [Implementation Steps](#implementation-steps)

## Overview

This system generates **Directional Room Impulse Responses (DRIR)** by:
- Using **reverse ray tracing** (listener → source) for spatial accuracy
- Processing **8 frequency bands** simultaneously for realistic material absorption
- Encoding results in **first-order Ambisonics** for 3D spatial audio
- Applying **Linkwitz-Riley crossover filters** for frequency band combination
- Converting to **binaural audio** via HRTF convolution

### Key Innovation: Reverse Ray Tracing
Unlike traditional acoustic ray tracing (source → listener), this implementation traces rays from the **listener to the source**. This provides crucial directional information needed for spatial audio encoding.

## Core Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   3D Geometry   │───▶│  WebGPU Compute  │───▶│  Impulse Data   │
│   (Triangles)   │    │     Shader       │    │ (Power/Length)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Binaural Audio  │◀───│  Filter Tree +   │◀───│   Ambisonic     │
│   (Stereo)      │    │  HRTF Processing │    │   Encoding      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Geometry Data Structure

### Triangle Data Layout
Each triangle requires **36 float values** (9 elements × 4 components):

```
Triangle Data Structure (36 floats):
┌─────────────────────────────────────────────────────────────────┐
│ Vertex 1 (12 floats)  │ Vertex 2 (12 floats)  │ Vertex 3 (12 floats) │
├─────────────────────────────────────────────────────────────────┤
│ pos(3) norm(3) uv(2) mat(4) │ pos(3) norm(3) uv(2) mat(4) │ pos(3) norm(3) uv(2) mat(4) │
└─────────────────────────────────────────────────────────────────┘
```

### Texture Encoding for GPU
Geometry is packed into a 2D texture for shader access:

```typescript
// Texture dimensions: (ELEMENTS_PER_TRIANGLE/4) × num_triangles
const ELEMENTS_PER_TRIANGLE = 36; // 9 vec4s per triangle
const textureWidth = 9;  // 36/4 = 9 vec4s per row
const textureHeight = numTriangles;

// Texture layout per triangle row:
// [0]: vertex1.position + vertex1.normal.x
// [1]: vertex1.normal.yz + vertex1.uv + vertex1.material.x
// [2]: vertex1.material.yzw + vertex2.position.x
// [3]: vertex2.position.yz + vertex2.normal.xy
// [4]: vertex2.normal.z + vertex2.uv + vertex2.material.xy
// [5]: vertex2.material.zw + vertex3.position.xy
// [6]: vertex3.position.z + vertex3.normal
// [7]: vertex3.uv + vertex3.material.xy
// [8]: vertex3.material.zw + padding
```

## WebGPU Compute Shader Implementation

### Shader Structure
```wgsl
// WebGPU Compute Shader for Acoustic Ray Tracing
@group(0) @binding(0) var<storage, read> geometryData: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputData: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

struct Uniforms {
    observerPos: vec3<f32>,
    sourcePos: vec3<f32>,
    numTriangles: u32,
    time: f32,
    resolution: vec2<f32>,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    length: f32,
    power: f32,
    lastPoint: vec3<f32>,
}

// Frequency-dependent absorption coefficients
const ABSORPTION_BANDS = array<f32, 8>(0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05);
const MAX_BOUNCES = 20;
const SAMPLES_PER_PIXEL = 10;
const FREQ_BANDS = 8;
```

### Main Compute Function
```wgsl
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let pixelCoord = vec2<i32>(globalId.xy);
    let resolution = vec2<i32>(uniforms.resolution);
    
    if (pixelCoord.x >= resolution.x || pixelCoord.y >= resolution.y) {
        return;
    }
    
    // Convert pixel to spherical coordinates
    let uv = vec2<f32>(pixelCoord) / vec2<f32>(resolution);
    let spherical = uvToSpherical(uv);
    let rayDirection = sphericalToCartesian(spherical);
    
    // Process multiple frequency bands
    var results = array<vec4<f32>, 4>();
    
    for (var band = 0u; band < FREQ_BANDS; band += 2u) {
        let bandIndex = band / 2u;
        results[bandIndex] = processFrequencyBand(rayDirection, band);
    }
    
    // Write results to output buffer
    let outputIndex = (pixelCoord.y * resolution.x + pixelCoord.x) * 16;
    for (var i = 0u; i < 4u; i++) {
        outputData[outputIndex + i * 4 + 0] = results[i].x;
        outputData[outputIndex + i * 4 + 1] = results[i].y;
        outputData[outputIndex + i * 4 + 2] = results[i].z;
        outputData[outputIndex + i * 4 + 3] = results[i].w;
    }
}
```

## Ray-Triangle Intersection Algorithm

### Möller-Trumbore Algorithm Implementation
```wgsl
fn rayTriangleIntersect(
    rayOrigin: vec3<f32>,
    rayDirection: vec3<f32>,
    v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>
) -> IntersectionResult {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(rayDirection, edge2);
    let a = dot(edge1, h);
    
    if (abs(a) < 0.00001) {
        return IntersectionResult(false, 0.0, vec2<f32>(0.0), vec3<f32>(0.0));
    }
    
    let f = 1.0 / a;
    let s = rayOrigin - v0;
    let u = f * dot(s, h);
    
    if (u < 0.0 || u > 1.0) {
        return IntersectionResult(false, 0.0, vec2<f32>(0.0), vec3<f32>(0.0));
    }
    
    let q = cross(s, edge1);
    let v = f * dot(rayDirection, q);
    
    if (v < 0.0 || u + v > 1.0) {
        return IntersectionResult(false, 0.0, vec2<f32>(0.0), vec3<f32>(0.0));
    }
    
    let t = f * dot(edge2, q);
    
    if (t > 0.00001) {
        let normal = normalize(cross(edge1, edge2));
        return IntersectionResult(true, t, vec2<f32>(u, v), normal);
    }
    
    return IntersectionResult(false, 0.0, vec2<f32>(0.0), vec3<f32>(0.0));
}
```

### Scene Intersection
```wgsl
fn intersectScene(ray: Ray) -> SceneIntersection {
    var closestHit = SceneIntersection();
    closestHit.distance = 999999.0;
    closestHit.hit = false;
    
    for (var triangleId = 0u; triangleId < uniforms.numTriangles; triangleId++) {
        let triangle = loadTriangle(triangleId);
        let intersection = rayTriangleIntersect(
            ray.origin, ray.direction,
            triangle.v0.position, triangle.v1.position, triangle.v2.position
        );
        
        if (intersection.hit && intersection.distance < closestHit.distance) {
            closestHit.hit = true;
            closestHit.distance = intersection.distance;
            closestHit.point = ray.origin + ray.direction * intersection.distance;
            closestHit.normal = intersection.normal;
            closestHit.material = triangle.material;
        }
    }
    
    return closestHit;
}
```

## Acoustic Ray Tracing

### Ray Bouncing with Material Properties
```wgsl
fn traceAcousticRay(initialRay: Ray, frequencyBand: u32) -> vec2<f32> {
    var ray = initialRay;
    var totalPower = 1.0;
    var totalLength = 0.0;
    var hitSource = false;
    
    for (var bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        // Check source intersection first
        let sourceIntersection = intersectSphere(ray.origin, ray.direction, 
                                               uniforms.sourcePos, 1.5);
        
        // Check scene intersection
        let sceneHit = intersectScene(ray);
        
        if (sceneHit.hit) {
            // If source is closer than scene, we hit the source
            if (sourceIntersection.hit && sourceIntersection.distance < sceneHit.distance) {
                totalLength += sourceIntersection.distance;
                hitSource = true;
                break;
            }
            
            // Update ray properties
            totalLength += sceneHit.distance;
            totalPower *= (1.0 - ABSORPTION_BANDS[frequencyBand]);
            
            // Generate new ray direction (specular + diffuse reflection)
            let reflectedDir = reflect(ray.direction, sceneHit.normal);
            let randomDir = generateRandomDirection(sceneHit.point, uniforms.time);
            let newDirection = normalize(mix(reflectedDir, randomDir, 0.2));
            
            // Update ray for next bounce
            ray.origin = sceneHit.point + sceneHit.normal * 0.001;
            ray.direction = newDirection;
        } else {
            break; // Ray escaped scene
        }
    }
    
    if (!hitSource) {
        totalPower = 0.0;
    }
    
    return vec2<f32>(totalPower / totalLength, totalLength);
}
```

### Random Number Generation for Diffuse Reflections
```wgsl
fn generateRandomDirection(seed: vec3<f32>, time: f32) -> vec3<f32> {
    let hash = fract(sin(dot(seed.xy + time, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let theta = hash * 2.0 * 3.14159265;
    let phi = acos(sqrt(fract(hash * 1.618033988749)));

    return vec3<f32>(
        sin(phi) * cos(theta),
        cos(phi),
        sin(phi) * sin(theta)
    );
}
```

## Impulse Response Generation

### Spherical Coordinate Conversion
```wgsl
fn uvToSpherical(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        uv.x * 2.0 * 3.14159265,  // Azimuth: 0 to 2π
        uv.y * 3.14159265         // Elevation: 0 to π
    );
}

fn sphericalToCartesian(spherical: vec2<f32>) -> vec3<f32> {
    let theta = spherical.x;  // Azimuth
    let phi = spherical.y;    // Elevation

    return vec3<f32>(
        sin(phi) * sin(theta),
        -cos(phi),
        sin(phi) * cos(theta)
    );
}
```

### Frequency Band Processing
```wgsl
fn processFrequencyBand(rayDirection: vec3<f32>, bandIndex: u32) -> vec4<f32> {
    var accumulator = vec4<f32>(0.0);

    for (var sample = 0u; sample < SAMPLES_PER_PIXEL; sample++) {
        let ray = Ray(
            uniforms.observerPos,
            rayDirection,
            0.0,
            1.0,
            uniforms.observerPos
        );

        // Process two frequency bands per iteration
        let result1 = traceAcousticRay(ray, bandIndex);
        let result2 = traceAcousticRay(ray, bandIndex + 1u);

        accumulator.x += result1.x; // Power/distance for band 1
        accumulator.y += result1.y; // Travel time for band 1
        accumulator.z += result2.x; // Power/distance for band 2
        accumulator.w += result2.y; // Travel time for band 2
    }

    return accumulator / f32(SAMPLES_PER_PIXEL);
}
```

## Ambisonic Encoding

### First-Order Ambisonics (B-Format)
The directional impulse data is encoded into 4-channel Ambisonic format:

```typescript
// Ambisonic encoding coefficients (SN3D normalization, ACN ordering)
function encodeAmbisonics(
    power: number,
    azimuth: number,
    elevation: number
): [number, number, number, number] {
    const sqrt3 = Math.sqrt(3);

    return [
        power,                                           // W (omnidirectional)
        sqrt3 * Math.cos(elevation) * Math.sin(azimuth) * power,  // X (front-back)
        sqrt3 * Math.sin(elevation) * power,                      // Y (up-down)
        sqrt3 * Math.cos(elevation) * Math.cos(azimuth) * power   // Z (left-right)
    ];
}
```

### Creating Ambisonic Buffers
```typescript
function createAmbisonicBuffer(
    sphereData: SpherePoint[],
    sampleRate: number
): AudioBuffer[] {
    const maxTime = Math.max(...sphereData.map(p =>
        Math.max(...p.characteristics.map(c => c.length))
    ));
    const duration = maxTime / 343; // Speed of sound

    const buffers: AudioBuffer[] = [];
    const numBands = sphereData[0].characteristics.length;

    for (let band = 0; band < numBands; band++) {
        const buffer = audioContext.createBuffer(4, sampleRate * duration, sampleRate);

        // Get channel data for W, X, Y, Z
        const channels = [
            buffer.getChannelData(0), // W
            buffer.getChannelData(1), // X
            buffer.getChannelData(2), // Y
            buffer.getChannelData(3)  // Z
        ];

        for (const point of sphereData) {
            const sampleIndex = Math.floor(
                (point.characteristics[band].length / 343) * sampleRate
            );

            const power = point.characteristics[band].power;
            const theta = point.sphere_point.azimuth;
            const phi = point.sphere_point.elevation;

            const [w, x, y, z] = encodeAmbisonics(power, theta, phi);

            channels[0][sampleIndex] += w;
            channels[1][sampleIndex] += x;
            channels[2][sampleIndex] += y;
            channels[3][sampleIndex] += z;
        }

        buffers.push(buffer);
    }

    return buffers;
}
```

## Filter Tree Processing

### Linkwitz-Riley Crossover Filters
```typescript
class LinkwitzRileyTree {
    private offlineContext: OfflineAudioContext;
    private sources: AudioBufferSourceNode[] = [];
    private filters: BiquadFilterNode[] = [];

    constructor(
        sampleRate: number,
        duration: number,
        frequencyBands: number[],
        channels: number
    ) {
        this.offlineContext = new OfflineAudioContext(
            channels,
            sampleRate * duration,
            sampleRate
        );

        this.createFilterTree(frequencyBands);
    }

    private createFilterTree(frequencies: number[]): void {
        // Stage 1: Create butterworth filters for each frequency pair
        for (let i = 0; i < frequencies.length; i += 2) {
            const crossoverFreq = Math.sqrt(frequencies[i] * frequencies[i + 1]);

            // Create source nodes
            const lowSource = this.offlineContext.createBufferSource();
            const highSource = this.offlineContext.createBufferSource();

            // Create 4th-order Butterworth filters (2 biquads each)
            const lowpass1 = this.createButterworthStage(crossoverFreq, 'lowpass');
            const lowpass2 = this.createButterworthStage(crossoverFreq, 'lowpass');
            const highpass1 = this.createButterworthStage(crossoverFreq, 'highpass');
            const highpass2 = this.createButterworthStage(crossoverFreq, 'highpass');

            // Connect filter chains
            lowSource.connect(lowpass1).connect(lowpass2);
            highSource.connect(highpass1).connect(highpass2);

            this.sources.push(lowSource, highSource);
        }

        // Stage 2: Combine and apply final processing
        this.createOutputStage();
    }

    private createButterworthStage(
        frequency: number,
        type: BiquadFilterType
    ): BiquadFilterNode {
        const filter = this.offlineContext.createBiquadFilter();
        filter.frequency.setValueAtTime(frequency, this.offlineContext.currentTime);
        filter.type = type;
        filter.Q.setValueAtTime(0.7071, this.offlineContext.currentTime); // Butterworth Q
        return filter;
    }

    async generateIR(ambisonicBuffers: AudioBuffer[]): Promise<AudioBuffer> {
        // Assign buffers to sources
        for (let i = 0; i < ambisonicBuffers.length; i++) {
            this.sources[i].buffer = ambisonicBuffers[i];
            this.sources[i].start();
        }

        return await this.offlineContext.startRendering();
    }
}
```

## Implementation Steps

### 1. WebGPU Setup
```typescript
async function initializeWebGPU(): Promise<{device: GPUDevice, context: GPUCanvasContext}> {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();

    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    const context = canvas.getContext('webgpu')!;

    context.configure({
        device,
        format: 'bgra8unorm',
        alphaMode: 'premultiplied'
    });

    return { device, context };
}
```

### 2. Geometry Buffer Creation
```typescript
function createGeometryBuffer(device: GPUDevice, triangles: Triangle[]): GPUBuffer {
    const geometryData = new Float32Array(triangles.length * 36);

    for (let i = 0; i < triangles.length; i++) {
        const triangle = triangles[i];
        const offset = i * 36;

        // Pack triangle data: positions, normals, UVs, materials
        geometryData.set(triangle.v0.position, offset + 0);
        geometryData.set(triangle.v0.normal, offset + 3);
        geometryData.set(triangle.v0.uv, offset + 6);
        geometryData.set(triangle.v0.material, offset + 8);

        geometryData.set(triangle.v1.position, offset + 12);
        geometryData.set(triangle.v1.normal, offset + 15);
        geometryData.set(triangle.v1.uv, offset + 18);
        geometryData.set(triangle.v1.material, offset + 20);

        geometryData.set(triangle.v2.position, offset + 24);
        geometryData.set(triangle.v2.normal, offset + 27);
        geometryData.set(triangle.v2.uv, offset + 30);
        geometryData.set(triangle.v2.material, offset + 32);
    }

    const buffer = device.createBuffer({
        size: geometryData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(buffer, 0, geometryData);
    return buffer;
}
```

### 3. Compute Pipeline Setup
```typescript
function createComputePipeline(device: GPUDevice, shaderCode: string): GPUComputePipeline {
    const shaderModule = device.createShaderModule({ code: shaderCode });

    return device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });
}
```

### 4. Execution and Data Retrieval
```typescript
async function generateImpulseResponse(
    device: GPUDevice,
    pipeline: GPUComputePipeline,
    geometryBuffer: GPUBuffer,
    observerPos: [number, number, number],
    sourcePos: [number, number, number],
    resolution: [number, number]
): Promise<Float32Array[]> {

    const outputBuffer = device.createBuffer({
        size: resolution[0] * resolution[1] * 16 * 4, // 4 vec4s per pixel
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const uniformBuffer = device.createBuffer({
        size: 64, // Uniforms size
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Write uniform data
    const uniformData = new Float32Array([
        ...observerPos, 0,
        ...sourcePos, 0,
        triangles.length, Date.now() * 0.001, 0, 0,
        ...resolution, 0, 0
    ]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: geometryBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
            { binding: 2, resource: { buffer: uniformBuffer } }
        ]
    });

    // Dispatch compute shader
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();

    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(
        Math.ceil(resolution[0] / 8),
        Math.ceil(resolution[1] / 8),
        1
    );
    computePass.end();

    // Copy results back
    const stagingBuffer = device.createBuffer({
        size: outputBuffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputBuffer.size);
    device.queue.submit([commandEncoder.finish()]);

    // Read results
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = stagingBuffer.getMappedRange();
    const results = new Float32Array(arrayBuffer);

    // Split into frequency band textures
    const bandTextures: Float32Array[] = [];
    const pixelCount = resolution[0] * resolution[1];

    for (let band = 0; band < 4; band++) {
        const bandData = new Float32Array(pixelCount * 4);
        for (let i = 0; i < pixelCount; i++) {
            const srcOffset = i * 16 + band * 4;
            const dstOffset = i * 4;
            bandData.set(results.subarray(srcOffset, srcOffset + 4), dstOffset);
        }
        bandTextures.push(bandData);
    }

    stagingBuffer.unmap();
    return bandTextures;
}
```

### 5. Integration with Web Audio API
```typescript
async function processAudioResults(
    bandTextures: Float32Array[],
    resolution: [number, number],
    audioContext: AudioContext
): Promise<AudioBuffer> {

    // Convert GPU results to sphere points
    const spherePoints = texturesToSpherePoints(bandTextures, resolution);

    // Create ambisonic buffers
    const ambisonicBuffers = createAmbisonicBuffer(spherePoints, audioContext.sampleRate);

    // Process through filter tree
    const filterTree = new LinkwitzRileyTree(
        audioContext.sampleRate,
        4, // 4 second duration
        [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
        2  // Stereo output
    );

    // Generate final impulse response
    const finalIR = await filterTree.generateIR(ambisonicBuffers);

    return finalIR;
}
```

## Performance Considerations

### Optimization Strategies
1. **Workgroup Size**: Use 8×8 workgroups for optimal GPU utilization
2. **Memory Coalescing**: Structure data access patterns for cache efficiency
3. **Early Ray Termination**: Stop rays when power drops below threshold
4. **Adaptive Sampling**: Use fewer samples for distant/weak reflections
5. **Level-of-Detail**: Reduce triangle count for distant geometry

### Memory Management
- Use storage buffers for large geometry data
- Implement double-buffering for real-time updates
- Consider texture compression for material properties
- Pool AudioBuffer objects to reduce GC pressure

## Conclusion

This implementation provides a complete framework for real-time spatial audio using WebGPU compute shaders. The system achieves realistic room acoustics by:

- Leveraging GPU parallelism for thousands of simultaneous ray calculations
- Processing multiple frequency bands for accurate material simulation
- Using reverse ray tracing for optimal spatial audio encoding
- Integrating seamlessly with Web Audio API for real-time playback

The modular design allows for easy extension with additional acoustic phenomena like diffraction, atmospheric absorption, and dynamic material properties.
```
