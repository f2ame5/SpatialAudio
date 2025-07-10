# RayTracer Class Documentation

## Overview
The `RayTracer` class implements acoustic ray tracing for spatial audio simulation using WebGPU. It calculates ray paths from a sound source, handling reflections and energy decay across multiple frequency bands to simulate realistic sound propagation in a room.

## Key Components

### Configuration Interface
```typescript
interface RayTracerConfig {
    numRays: number;    // Number of rays to emit (default: 1000)
    maxBounces: number; // Maximum number of bounces per ray (default: 50)
    minEnergy: number;  // Minimum energy threshold for ray termination (default: 0.001)
}
```

### Ray Hit Structure
```typescript
interface RayHit {
    position: [number, number, number];  // Hit position in 3D space
    energy_low: number;                  // Energy in low frequency band
    energy_mid: number;                  // Energy in mid frequency band
    energy_high: number;                 // Energy in high frequency band
    time: number;                        // Time of hit from source
}
```

## Methods

### Constructor
```typescript
constructor(device: GPUDevice, soundSource: SoundSource, room: Room, config: RayTracerConfig)
```
Initializes a new ray tracer with:
- WebGPU device for GPU-accelerated computations
- Sound source for ray origin
- Room for boundary information
- Configuration parameters

### calculateRayPaths
```typescript
async calculateRayPaths(numRays: number = 1000): Promise<void>
```
Performs GPU-accelerated ray tracing:
- Initializes rays from sound source with uniform spherical distribution
- Traces rays through multiple bounces using compute shader
- Collects ray hits with energy levels for each frequency band
- Passes hit data to audio processor for histogram generation

### setAudioProcessor
```typescript
setAudioProcessor(processor: AudioProcessor): void
```
Sets the audio processor that will handle the ray hit data and generate the audio output.

## Implementation Details

### GPU Buffers
1. **Rays Buffer**: Stores ray data
   - Origin (vec3)
   - Direction (vec3)
   - Energy levels (vec3 for low/mid/high frequencies)
   - Path length
   - Bounce count
   - Active flag

2. **Surfaces Buffer**: Stores room surface data
   - Normal (vec3)
   - Position (vec3)
   - Absorption coefficients (vec3 for low/mid/high frequencies)
   - Roughness
   - Scattering coefficient

3. **Ray Hits Buffer**: Stores intersection data
   - Position (vec3)
   - Energy levels (vec3)
   - Time

4. **Hit Counter**: Atomic counter for tracking number of hits

### Ray Tracing Process
1. **Initialization**
   - Rays are emitted from sound source with uniform spherical distribution
   - Each ray starts with full energy (1.0) in all frequency bands

2. **GPU Compute Pass**
   - Rays are processed in parallel on the GPU
   - Each ray is traced through multiple bounces until:
     - Maximum bounce count reached
     - Energy falls below threshold
     - Ray exits room boundaries

3. **Energy Calculations**
   - Frequency-dependent absorption per surface
   - Distance-based attenuation
   - Surface roughness affects reflection direction
   - Scattering coefficient determines energy distribution

4. **Data Collection**
   - Ray hits are stored with position, energy, and time
   - Hit data is passed to audio processor for:
     - Energy histogram generation
     - Audio synthesis
     - Real-time visualization

### Material Properties
Surface materials are defined with frequency-dependent properties:
```typescript
interface MaterialProperties {
    absorptionLow: number;   // Low frequency absorption (125Hz)
    absorptionMid: number;   // Mid frequency absorption (1kHz)
    absorptionHigh: number;  // High frequency absorption (4kHz)
    roughness: number;       // Surface roughness (0-1)
    scattering: number;      // Scattering coefficient (0-1)
}
```

## Usage Example
```typescript
// Initialize ray tracer
const rayTracer = new RayTracer(device, soundSource, room, {
    numRays: 1000,
    maxBounces: 50,
    minEnergy: 0.001
});

// Connect to audio processor
rayTracer.setAudioProcessor(audioProcessor);

// Calculate ray paths and generate audio
await rayTracer.calculateRayPaths(1000);
```

## Recent Updates
- Implemented GPU-accelerated ray tracing using compute shaders
- Added frequency-dependent absorption and energy tracking
- Integrated with audio processor for real-time audio synthesis
- Added support for surface roughness and scattering
- Improved energy histogram visualization
