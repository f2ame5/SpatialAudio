# WebGPU Compute Shader Architecture for Acoustic Raytracing

## Overview

This document defines the complete compute shader architecture for GPU-accelerated acoustic raytracing. The system uses a multi-stage pipeline to generate, bounce, and collect acoustic rays for realistic spatial audio simulation.

## Pipeline Stages

### Stage 1: Ray Generation
**Shader**: `ray-generation.wgsl`
**Purpose**: Generate acoustic rays from spherical sound source
**Workgroup Size**: 64 threads
**Dispatch**: `ceil(rayCount / 64)` workgroups

### Stage 2: Ray Intersection & Bouncing  
**Shader**: `ray-bouncing.wgsl`
**Purpose**: Calculate ray-surface intersections and reflections
**Workgroup Size**: 64 threads
**Dispatch**: `ceil(activeRayCount / 64)` workgroups

### Stage 3: Ray Collection
**Shader**: `ray-collection.wgsl`
**Purpose**: Collect rays at listener position for impulse response
**Workgroup Size**: 64 threads
**Dispatch**: `ceil(rayCount / 64)` workgroups

## Buffer Layout Architecture

### Buffer Binding Layout
```
Group 0 (Generation):
├── Binding 0: Ray Storage Buffer (read_write)
├── Binding 1: Generation Params (uniform)
└── Binding 2: Material Database (read_only_storage)

Group 1 (Bouncing):
├── Binding 0: Ray Storage Buffer (read_write)
├── Binding 1: Bouncing Params (uniform)
├── Binding 2: Material Database (read_only_storage)
└── Binding 3: Room Geometry (read_only_storage)

Group 2 (Collection):
├── Binding 0: Ray Storage Buffer (read_only)
├── Binding 1: Collection Params (uniform)
├── Binding 2: Impulse Response Buffer (read_write)
└── Binding 3: Statistics Buffer (read_write)
```

## Memory Layout Specifications

### Ray Structure (96 bytes, vec4-aligned)
```wgsl
struct Ray {
    origin: vec4<f32>,              // xyz=position, w=padding (16 bytes)
    direction: vec4<f32>,           // xyz=direction, w=padding (16 bytes)
    energy_phase: vec4<f32>,        // x=energy, y=phase, zw=padding (16 bytes)
    frequency_energy_low: vec4<f32>, // 125,250,500,1k Hz (16 bytes)
    frequency_energy_high: vec4<f32>, // 2k,4k,8k,16k Hz (16 bytes)
    path_data: vec4<f32>,           // x=path_length, y=arrival_time, z=bounce_count, w=active (16 bytes)
}
```

### Material Structure (72 bytes, vec4-aligned)
```wgsl
struct Material {
    absorption_low: vec4<f32>,      // 125,250,500,1k Hz absorption (16 bytes)
    absorption_high: vec4<f32>,     // 2k,4k,8k,16k Hz absorption (16 bytes)
    scattering_low: vec4<f32>,      // 125,250,500,1k Hz scattering (16 bytes)
    scattering_high: vec4<f32>,     // 2k,4k,8k,16k Hz scattering (16 bytes)
    properties: vec4<f32>,          // x=impedance, y=roughness, zw=padding (8 bytes)
}
```

### Room Geometry Structure
```wgsl
struct RoomGeometry {
    bounds_min: vec4<f32>,          // xyz=min bounds, w=padding
    bounds_max: vec4<f32>,          // xyz=max bounds, w=padding
    surface_count: u32,             // Number of surfaces
    padding: vec3<u32>,             // Alignment padding
}

struct Surface {
    normal: vec4<f32>,              // xyz=normal, w=distance from origin
    material_id: u32,               // Material index
    surface_type: u32,              // 0=wall, 1=floor, 2=ceiling
    padding: vec2<u32>,             // Alignment
}
```

## Workgroup Size Strategy

### Optimal Workgroup Sizes by GPU Architecture
- **Desktop GPUs**: 64 threads (2 warps/wavefronts)
- **Mobile GPUs**: 32 threads (1 wavefront)
- **Integrated GPUs**: 32-64 threads (adaptive)

### Memory Access Patterns
- **Coalesced Access**: Rays processed in sequential order
- **Bank Conflicts**: Avoided by vec4 alignment
- **Cache Efficiency**: Materials accessed via texture cache

## Dispatch Parameters

### Ray Generation Dispatch
```typescript
const workgroups = Math.ceil(rayCount / WORKGROUP_SIZE);
pass.dispatchWorkgroups(workgroups, 1, 1);
```

### Ray Bouncing Dispatch (Multi-pass)
```typescript
for (let bounce = 0; bounce < maxBounces; bounce++) {
    const activeRays = countActiveRays(); // GPU query
    const workgroups = Math.ceil(activeRays / WORKGROUP_SIZE);
    pass.dispatchWorkgroups(workgroups, 1, 1);
}
```

### Ray Collection Dispatch
```typescript
const irSamples = Math.floor(irLength * sampleRate);
const workgroups = Math.ceil(irSamples / WORKGROUP_SIZE);
pass.dispatchWorkgroups(workgroups, 1, 1);
```

## Synchronization Strategy

### Pipeline Barriers
```typescript
// Between generation and bouncing
encoder.insertDebugMarker("Generation -> Bouncing Barrier");

// Between bouncing iterations
encoder.insertDebugMarker("Bounce Iteration Barrier");

// Between bouncing and collection
encoder.insertDebugMarker("Bouncing -> Collection Barrier");
```

### Double Buffering
- **Ping-Pong Buffers**: Alternate between two ray buffers
- **Frame Overlap**: Process frame N+1 while reading frame N
- **Memory Efficiency**: Reduce GPU-CPU synchronization

## Performance Optimization

### GPU Memory Hierarchy
1. **Registers**: Ray data during processing
2. **Shared Memory**: Material cache per workgroup
3. **Global Memory**: Ray buffers, impulse response
4. **Texture Cache**: Material database lookup

### Occupancy Optimization
- **Thread Divergence**: Minimize branching in shaders
- **Memory Bandwidth**: Optimize buffer access patterns
- **ALU Utilization**: Balance compute vs memory operations

### Adaptive Quality
```typescript
interface QualitySettings {
    rayCount: number;           // 512-8192 rays
    maxBounces: number;         // 5-50 bounces
    workgroupSize: number;      // 32-64 threads
    precisionMode: 'fast' | 'precise';
}
```

## Error Handling & Debugging

### Validation Checks
- Ray count within GPU limits
- Buffer size alignment
- Workgroup size compatibility
- Shader compilation errors

### Debug Features
- Ray visualization buffers
- Performance counters
- Energy conservation validation
- Convergence monitoring

## Implementation Phases

### Phase 1: Basic Pipeline (Week 1)
- [ ] Ray generation shader
- [ ] Simple room intersection
- [ ] Basic energy tracking

### Phase 2: Advanced Physics (Week 2)
- [ ] Frequency-dependent absorption
- [ ] Scattering calculations
- [ ] Phase tracking

### Phase 3: Optimization (Week 3)
- [ ] Double buffering
- [ ] Adaptive quality
- [ ] Performance profiling

## Detailed Shader Module Structure

### Ray Generation Module
```wgsl
// Common structures (shared across all shaders)
struct Ray {
    origin: vec4<f32>,
    direction: vec4<f32>,
    energy_phase: vec4<f32>,
    frequency_energy_low: vec4<f32>,
    frequency_energy_high: vec4<f32>,
    path_data: vec4<f32>,
}

struct RayGenerationParams {
    source_position: vec3<f32>,
    source_radius: f32,
    ray_count: u32,
    initial_energy: f32,
    time: f32,
    seed: u32,
    frequency_weights: array<f32, 8>,
}

@group(0) @binding(0) var<storage, read_write> rays: array<Ray>;
@group(0) @binding(1) var<uniform> params: RayGenerationParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Implementation details in actual shader
}
```

### Ray Bouncing Module
```wgsl
struct RayBouncingParams {
    room_min: vec3<f32>,
    room_max: vec3<f32>,
    max_bounces: u32,
    min_energy: f32,
    speed_of_sound: f32,
    time_step: f32,
    air_absorption: array<f32, 8>,
}

struct Material {
    absorption_low: vec4<f32>,
    absorption_high: vec4<f32>,
    scattering_low: vec4<f32>,
    scattering_high: vec4<f32>,
    properties: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> rays: array<Ray>;
@group(0) @binding(1) var<uniform> params: RayBouncingParams;
@group(0) @binding(2) var<storage, read> materials: array<Material>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Implementation details in actual shader
}
```

### Ray Collection Module
```wgsl
struct CollectionParams {
    listener_position: vec3<f32>,
    listener_radius: f32,
    sample_rate: f32,
    ir_length: f32,
    time_bin_size: f32,
    padding: vec3<f32>,
}

struct ImpulseSample {
    energy: f32,
    phase: f32,
    frequency_energy: array<f32, 8>,
}

@group(0) @binding(0) var<storage, read> rays: array<Ray>;
@group(0) @binding(1) var<uniform> params: CollectionParams;
@group(0) @binding(2) var<storage, read_write> impulse_response: array<ImpulseSample>;
@group(0) @binding(3) var<storage, read_write> statistics: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Implementation details in actual shader
}
```

## TypeScript Pipeline Integration

### Bind Group Layout Definitions
```typescript
const generationBindGroupLayout = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' }
        },
        {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform' }
        }
    ]
});

const bouncingBindGroupLayout = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' }
        },
        {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform' }
        },
        {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' }
        }
    ]
});

const collectionBindGroupLayout = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' }
        },
        {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform' }
        },
        {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' }
        },
        {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' }
        }
    ]
});
```

### Pipeline Creation Strategy
```typescript
class ShaderPipelineManager {
    private device: GPUDevice;
    private pipelines: Map<string, GPUComputePipeline> = new Map();
    private bindGroupLayouts: Map<string, GPUBindGroupLayout> = new Map();

    async createPipelines(): Promise<void> {
        // Load shader modules
        const generationModule = await this.loadShaderModule('ray-generation.wgsl');
        const bouncingModule = await this.loadShaderModule('ray-bouncing.wgsl');
        const collectionModule = await this.loadShaderModule('ray-collection.wgsl');

        // Create pipelines
        this.pipelines.set('generation', this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.bindGroupLayouts.get('generation')!]
            }),
            compute: {
                module: generationModule,
                entryPoint: 'main'
            }
        }));

        // Similar for bouncing and collection...
    }
}
```

## Next Implementation Steps

1. **Complete ray-generation.wgsl** with spherical distribution algorithm
2. **Implement room intersection** in ray-bouncing.wgsl with material lookup
3. **Create material lookup system** with efficient GPU access patterns
4. **Add impulse response accumulation** in ray-collection.wgsl with temporal binning
5. **Integrate with TypeScript pipeline** management and buffer orchestration
6. **Add performance monitoring** and adaptive quality controls
