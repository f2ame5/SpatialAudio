# Spatial Audio System Overview

This document provides an in-depth overview of our spatial audio system, focusing on how we create realistic impulse responses from raytracing, the raytracing process itself, and the division of audio processing tasks between GPU and CPU.

## Table of Contents

1. [Introduction](#introduction)
2. [Raytracing for Spatial Audio](#raytracing-for-spatial-audio)
3. [Impulse Response Generation](#impulse-response-generation)
4. [GPU-CPU Task Division](#gpu-cpu-task-division)
5. [Audio Processing Pipeline](#audio-processing-pipeline)
6. [Technical Implementation Details](#technical-implementation-details)
7. [Code Examples](#code-examples)
8. [Mathematical Models and Formulas](#mathematical-models-and-formulas)
9. [Visualization Techniques](#visualization-techniques)
10. [Performance Benchmarks and Optimization](#performance-benchmarks-and-optimization)
11. [Real-time Parameter Adjustment](#real-time-parameter-adjustment)
12. [Validation and Testing](#validation-and-testing)
13. [Future Enhancements](#future-enhancements)

## Introduction

Our spatial audio system aims to create realistic 3D audio by simulating how sound waves propagate through a virtual environment. We use a physics-based approach that combines:

- Geometric acoustic raytracing to model sound propagation
- Frequency-dependent material properties for realistic reflections
- Binaural rendering for immersive 3D audio perception
- GPU acceleration for computationally intensive tasks

The end result is a system that can generate realistic impulse responses (IRs) for any position in a virtual room, which can then be used to spatialize audio sources through convolution.

## Raytracing for Spatial Audio

### Core Principles

Raytracing for spatial audio follows similar principles to visual raytracing but with important acoustic considerations:

1. **Sound Source Emission**: Rays are emitted from a sound source in all directions
2. **Ray Propagation**: Rays travel through the environment at the speed of sound (343 m/s)
3. **Surface Interactions**: Rays interact with surfaces based on acoustic properties:
   - Reflection (specular and diffuse)
   - Absorption (frequency-dependent)
   - Scattering (frequency-dependent)
4. **Energy Tracking**: Each ray carries energy across multiple frequency bands (125Hz-16kHz)
5. **Listener Detection**: Rays that reach the listener contribute to the impulse response

### Raytracing Implementation

Our raytracing implementation uses WebGPU compute shaders for parallel processing:

1. **Ray Generation**:
   - Rays are generated from the sound source position
   - Initial ray directions are distributed uniformly using spherical coordinates
   - Each ray carries energy across 8 frequency bands (125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz, 8kHz, 16kHz)

2. **Ray Propagation**:
   - The `raytracer.wgsl` compute shader processes rays in parallel
   - Rays are traced through the environment until they:
     - Reach maximum bounce count (typically 50)
     - Energy falls below a threshold (typically 0.05)
     - Exit the environment

3. **Surface Interaction**:
   - When a ray hits a surface, it calculates:
     - Reflection direction (based on surface normal)
     - Energy attenuation (based on material properties)
     - Frequency-dependent absorption
     - Scattering effects

4. **Ray Hit Collection**:
   - All ray hits are collected and stored with:
     - Position
     - Time (arrival time relative to direct sound)
     - Direction
     - Energy (across frequency bands)
     - Surface normal
     - Bounce count

## Impulse Response Generation

### From Ray Hits to Impulse Response

The process of converting ray hits to an impulse response involves:

1. **Temporal Mapping**:
   - Each ray hit is mapped to a specific time in the impulse response
   - Time is calculated based on the ray's travel distance and the speed of sound

2. **Spatial Processing**:
   - Ray hits are processed with spatial audio cues:
     - Interaural Time Difference (ITD)
     - Interaural Level Difference (ILD)
     - Direction-dependent filtering

3. **Energy Distribution**:
   - Energy from ray hits is distributed across the impulse response
   - Early reflections (< 100ms) are processed individually
   - Late reflections are processed statistically

### GPU-Accelerated IR Generation

The `spatial_audio.wgsl` compute shader performs key calculations for IR generation:

1. **Wave Contribution Calculation**:
   - Calculates the contribution of each ray hit to the impulse response
   - Applies phase, frequency, and amplitude modulation
   - Accounts for Doppler shift and distance attenuation

2. **HRTF Approximation**:
   - Calculates simplified HRTF (Head-Related Transfer Function) effects
   - Generates left and right channel gains based on ray direction
   - Applies directional attenuation based on listener orientation

3. **Frequency-Dependent Processing**:
   - Processes each frequency band with appropriate characteristics
   - Applies frequency-dependent air absorption
   - Implements frequency-dependent energy decay

## GPU-CPU Task Division

Our system divides tasks between GPU and CPU to optimize performance:

### GPU Tasks (WebGPU Compute Shaders)

1. **Ray Propagation and Physics** (`raytracer.wgsl`):
   - Ray-surface intersection calculations
   - Reflection vector computation
   - Energy propagation across frequency bands
   - Air absorption and attenuation calculations

2. **Initial IR Processing** (`spatial_audio.wgsl`):
   - Wave contribution calculations
   - Basic HRTF approximation
   - Frequency-dependent energy decay
   - Parallel processing of ray hits

### CPU Tasks (JavaScript/TypeScript)

1. **Audio Buffer Management** (`audio-processor_modified.ts`):
   - Creating and managing AudioContext and AudioBuffers
   - Setting up the Web Audio API convolution pipeline
   - Handling audio file loading and playback

2. **Advanced Spatial Processing**:
   - Gain and ITD (Interaural Time Difference) calculations
   - Temporal spreading for natural sound
   - Crossfading between early reflections and late reverberation

3. **Late Reverberation** (`diffuse-field-model_modified.ts`):
   - Statistical modeling of late reverberation
   - Feedback delay networks for dense reverberation
   - Stereo decorrelation for spatial impression

4. **Audio Rendering and Playback**:
   - Convolution of audio sources with the impulse response
   - Real-time audio playback
   - Visualization of impulse responses

## Audio Processing Pipeline

The complete audio processing pipeline follows these steps:

1. **Ray Tracing Simulation**:
   - `calculateIR()` in `main.ts` initiates the process
   - `rayTracer.calculateRayPaths()` simulates sound propagation
   - Ray hit data is collected with position, time, energy, and direction information

2. **Audio Processing**:
   - Ray hit data is passed to `audioProcessor.processRayHits()`
   - `processRayHitsInternal()` processes hits into a stereo impulse response
   - Early reflections (< 100ms) use the Gain + ITD model with temporal spreading
   - Late reverberation (≥ 100ms) is generated using `DiffuseFieldModelModified`

3. **Impulse Response Assembly**:
   - Early reflections and late reverberation are combined with a crossfade (80-120ms)
   - The final impulse response is normalized and stored in an `AudioBuffer`
   - The impulse response is visualized using the waveform renderer

4. **Audio Spatialization**:
   - Audio sources are convolved with the impulse response
   - The result is played back through the Web Audio API
   - Real-time updates occur as the listener or source moves

## Technical Implementation Details

### Frequency-Dependent Material Properties

Materials in our system have frequency-dependent properties:

- **Absorption Coefficients**: How much energy is absorbed at each frequency band
- **Scattering Coefficients**: How much energy is scattered vs. specularly reflected
- **Transmission Coefficients**: How much energy passes through the material (future work)

### Air Absorption Modeling

Sound attenuation in air is modeled with:

- Temperature-dependent absorption
- Humidity-dependent absorption
- Frequency-dependent absorption (higher frequencies attenuate more quickly)

### Wave Phenomena

Our system models several wave phenomena:

- **Phase Effects**: Constructive and destructive interference
- **Doppler Shift**: Frequency changes due to relative motion
- **Diffraction**: Simplified modeling around edges (future enhancement)

### Performance Optimizations

To maintain real-time performance:

1. **Compute Shader Optimizations**:
   - Workgroup size of 256 for efficient GPU utilization
   - Minimized divergent execution paths
   - Optimized memory access patterns

2. **Hybrid Approach**:
   - Image source method for early reflections (up to 2nd order)
   - Stochastic ray tracing for late reflections
   - Statistical modeling for very late reverberation

## Code Examples

This section provides code examples and pseudocode for key functions in our spatial audio system to illustrate the implementation details of critical calculations.

### 1. Wave Contribution Calculation (GPU)

The following WGSL code from `spatial_audio.wgsl` shows how we calculate the wave contribution for each ray hit:

```wgsl
fn calculateWaveContribution(
    time: f32,
    phase: f32,
    frequency: f32,
    dopplerShift: f32,
    amplitude: f32,
    distance: f32
) -> f32 {
    // Validate inputs to prevent NaN/Infinity
    let validFreq = max(frequency, 20.0);
    let validAmplitude = max(amplitude, 0.0);
    let validDistance = max(distance, 0.001);

    // Apply Doppler shift to frequency
    let shiftedFreq = validFreq * max(dopplerShift, 0.1);

    // Calculate wavelength and phase based on distance
    let wavelength = SPEED_OF_SOUND / shiftedFreq;
    let distancePhase = 2.0 * 3.14159 * validDistance / wavelength;
    let totalPhase = phase + distancePhase;

    // Apply window function for temporal shaping
    let windowPos = clamp(time / (validDistance / SPEED_OF_SOUND), 0.0, 1.0);
    let window = 0.8 * (1.0 - cos(2.0 * 3.14159 * windowPos));

    // Apply distance-based attenuation
    let earlyBoost = 3.0;
    let distanceAttenuation = 1.0 / max(validDistance * validDistance, 0.01);

    // Apply frequency-dependent boost based on room acoustics
    var freqBoost = 1.0;
    if (validFreq < 250.0) {
        freqBoost = mix(1.2, 1.0, (validFreq - 20.0) / 230.0);  // Bass boost
    } else if (validFreq < 4000.0) {
        freqBoost = mix(1.0, 1.1, (validFreq - 250.0) / 3750.0);  // Mid boost
    } else {
        freqBoost = mix(1.1, 0.9, (validFreq - 4000.0) / 12000.0);  // High attenuation
    }

    // Combine all factors and apply sinusoidal oscillation
    return validAmplitude * earlyBoost * window * sin(totalPhase) * distanceAttenuation * freqBoost;
}
```

This function models how sound waves propagate through space, accounting for:
- Phase changes based on distance
- Doppler shift effects
- Distance-based attenuation (inverse square law)
- Frequency-dependent boost/attenuation
- Temporal windowing for natural sound

### 2. Interaural Time Difference Calculation (CPU)

The following TypeScript code from `audio-processor_modified.ts` shows how we calculate the Interaural Time Difference (ITD):

```typescript
// Calculate ITD in samples based on azimuth angle
private calculateITDsamples(azimuthRad: number, sampleRate: number): number {
    // Constants
    const HEAD_RADIUS = 0.0875; // Average human head radius in meters
    const SPEED_OF_SOUND = 343.0; // Speed of sound in m/s

    // Woodworth's formula (simplified)
    // ITD = (r/c) * (θ + sin(θ)) where r is head radius, c is speed of sound, θ is azimuth

    // Ensure azimuth is within -π to π range
    const normalizedAzimuth = Math.atan2(Math.sin(azimuthRad), Math.cos(azimuthRad));

    // Calculate ITD in seconds
    // Use absolute value for calculation, then restore sign
    const absAzimuth = Math.abs(normalizedAzimuth);
    const itdSeconds = (HEAD_RADIUS / SPEED_OF_SOUND) * (absAzimuth + Math.sin(absAzimuth));

    // Convert to samples and apply sign based on original azimuth
    const itdSamples = itdSeconds * sampleRate * Math.sign(normalizedAzimuth);

    // Clamp to reasonable range (typically max ~1ms or ~44 samples at 44.1kHz)
    const MAX_ITD_SAMPLES = 44;
    return Math.max(-MAX_ITD_SAMPLES, Math.min(MAX_ITD_SAMPLES, itdSamples));
}
```

This function implements Woodworth's formula for ITD calculation, which models the time difference between when sound reaches each ear based on the azimuth angle. The ITD is a critical cue for horizontal sound localization.

### 3. Frequency-Dependent Energy Decay (GPU)

The following WGSL code shows how we calculate frequency-dependent energy decay:

```wgsl
fn calculateEnergyDecay(time: f32, distance: f32, direction: vec3f, normal: vec3f) -> FrequencyBands {
    // Base decay factor (time-dependent)
    let timeDecay = exp(-3.0 * time);

    // Distance-dependent decay
    let distanceDecay = 1.0 / max(distance * distance, 0.01);

    // Direction-dependent decay (grazing angle effect)
    let directionFactor = max(dot(direction, normal), 0.01);

    // Frequency-dependent decay factors
    // Lower frequencies decay more slowly than higher frequencies
    let band125decay = timeDecay * 0.95;
    let band250decay = timeDecay * 0.92;
    let band500decay = timeDecay * 0.90;
    let band1kdecay = timeDecay * 0.87;
    let band2kdecay = timeDecay * 0.84;
    let band4kdecay = timeDecay * 0.80;
    let band8kdecay = timeDecay * 0.75;
    let band16kdecay = timeDecay * 0.70;

    // Apply distance and direction factors
    let commonFactor = distanceDecay * directionFactor;

    return FrequencyBands(
        band125decay * commonFactor,
        band250decay * commonFactor,
        band500decay * commonFactor,
        band1kdecay * commonFactor,
        band2kdecay * commonFactor,
        band4kdecay * commonFactor,
        band8kdecay * commonFactor,
        band16kdecay * commonFactor
    );
}
```

This function models how sound energy decays over time, with different rates for different frequency bands. Higher frequencies decay faster than lower frequencies, which is consistent with real-world acoustics.

### 4. Ray-Surface Interaction with Material Properties (GPU)

The following pseudocode illustrates how ray-surface interactions are computed with material properties:

```wgsl
fn calculateRayReflection(
    ray: Ray,
    hitPoint: vec3f,
    normal: vec3f,
    material: Material
) -> ReflectionResult {
    // Calculate reflection direction (specular component)
    let incidentDir = normalize(ray.direction);
    let specularDir = reflect(incidentDir, normal);

    // Calculate diffuse (scattered) direction
    let randomVec = generateRandomVector(ray.seed);
    let diffuseDir = normalize(normal + randomVec);

    // Initialize energy bands for the reflected ray
    var reflectedEnergy: FrequencyBands;

    // Apply frequency-dependent absorption for each band
    reflectedEnergy.band125 = ray.energy.band125 * (1.0 - material.absorption_125);
    reflectedEnergy.band250 = ray.energy.band250 * (1.0 - material.absorption_250);
    reflectedEnergy.band500 = ray.energy.band500 * (1.0 - material.absorption_500);
    reflectedEnergy.band1k = ray.energy.band1k * (1.0 - material.absorption_1k);
    reflectedEnergy.band2k = ray.energy.band2k * (1.0 - material.absorption_2k);
    reflectedEnergy.band4k = ray.energy.band4k * (1.0 - material.absorption_4k);
    reflectedEnergy.band8k = ray.energy.band8k * (1.0 - material.absorption_8k);
    reflectedEnergy.band16k = ray.energy.band16k * (1.0 - material.absorption_16k);

    // Apply scattering - mix between specular and diffuse reflection
    let finalDirection: vec3f;
    for (var i = 0u; i < 8u; i++) {
        // Get scattering coefficient for this frequency band
        let scatteringCoeff = getScatteringCoeff(material, i);

        // Mix between specular and diffuse based on scattering coefficient
        if (random(ray.seed + i) < scatteringCoeff) {
            // Use diffuse direction for this frequency component
            // (In practice, we use one direction for all bands but with energy adjustment)
        }
    }

    // Determine final direction (simplified - in practice we'd use energy splitting)
    let avgScattering = calculateAverageScattering(material);
    finalDirection = mix(specularDir, diffuseDir, avgScattering);

    // Create reflection result
    return ReflectionResult(
        hitPoint,           // New ray origin
        finalDirection,     // New ray direction
        reflectedEnergy,    // Attenuated energy
        ray.bounces + 1u    // Increment bounce count
    );
}
```

This pseudocode demonstrates how we handle ray-surface interactions, including:
- Specular reflection calculation
- Diffuse (scattered) reflection
- Frequency-dependent absorption
- Material-dependent scattering
- Energy attenuation across frequency bands

These calculations are critical for realistic acoustic simulation as they determine how sound energy propagates through the environment after hitting surfaces with different acoustic properties.
