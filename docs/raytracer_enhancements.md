# Ray Tracer Enhancements for Impulse Response Generation

This document outlines the plan to enhance the ray tracer to simulate sound waves more realistically and generate an impulse response.

## Goal

Simulate sound wave behavior, including phase and interference, to produce a realistic impulse response.

## Plan

1.  **Modify `RayHit` Struct:**

    *   Add a `phase: f32` field to the `RayHit` struct in `spatialAudio/src/raytracer/shaders/raytracer.wgsl`. This will store the phase of the sound wave at the point of impact.

2.  **Initialize Phase:**

    *   When a ray is initialized (likely in `spatialAudio/src/raytracer/raytracer.ts`), set its initial phase to 0.  (Consider adding an option for a random initial phase later, to simulate less coherent sources).

3.  **Update Phase in Shader:**

    *   In `spatialAudio/src/raytracer/shaders/raytracer.wgsl`, when a ray hits a surface, update the `phase` of the `RayHit`.
    *   Introduce a uniform variable (e.g., `@group(0) @binding(1) var<uniform> frequency: f32;`) to pass the sound wave's frequency to the shader.
    *   Calculate the phase change using the formula: `new_phase = old_phase + 2.0 * PI * frequency * time;` where `time` is the existing `RayHit.time` field.

4.  **Process Hits (Outside Shader):**

    *   After the shader runs (likely in `spatialAudio/src/raytracer/raytracer.ts` or a related file), process the `hits` data.
    *   For each hit, generate a sample of a sine wave: `amplitude * sin(2.0 * PI * frequency * time + phase)`.
        *   `amplitude` will be derived from the `RayHit.energy` field (e.g., `amplitude = sqrt(energy)` or `amplitude = energy`).
        *   `frequency`, `time`, and `phase` are obtained from the `RayHit` struct and the uniform `frequency`.

5.  **Accumulate Impulse Response:**

    *   Create an array (or a suitable data structure) to represent the impulse response. The size of this array will depend on the desired length of the impulse response and the chosen sampling rate.
    *   For each sine wave sample generated in step 4, add it to the impulse response array at the appropriate time index.  This index will be determined by the `time` value from the `RayHit` and the sampling rate.  For example, `index = floor(time * sampling_rate)`.
    *   Ensure that samples from multiple hits that fall within the same time index are summed together.

6.  **Sampling Rate:**

    *   Determine an appropriate sampling rate for the impulse response (e.g., 44100 Hz, 48000 Hz). This will affect the accuracy and length of the impulse response.

7. **Multiple Frequencies (Future Enhancement):**
    * Consider how to handle multiple frequencies. Options include:
        * Running the simulation multiple times for different frequencies.
        * Modifying the shader and host code to handle an array of frequencies (more complex).

## Implementation Steps

1. Modify the `RayHit` struct (WGSL).
2. Add the frequency uniform (WGSL).
3. Update the phase calculation in the shader (WGSL).
4. Modify the ray initialization to set the initial phase (TypeScript).
5. Implement the hit processing and impulse response accumulation (TypeScript).

## User notes

Implementation Steps

1
RayHit Struct Modification

struct RayHit {
    // Existing fields
    time: f32,
    energy: f32,

    // New field for phase tracking
    phase: f32,
};

2
Phase Initialization (TypeScript)

interface RayInitParams {
    initialPhase?: number;  // Optional random initial phase
    frequency: number;
}

function initializeRay(params: RayInitParams): RayHit {
    return {
        time: 0.0,
        energy: 1.0,
        phase: params.initialPhase ?? 0.0
    };
}

3
Shader Phase Calculation (WGSL)

fn updatePhase(hit: ptr<function, RayHit>) {
    let time = hit.time;
    let old_phase = hit.phase;
    let new_phase = old_phase + 2.0 * PI * frequency * time;
    hit.phase = new_phase;
}

4
Impulse Response Accumulation (TypeScript)

interface ImpulseResponseConfig {
    samplingRate: number;
    lengthSeconds: number;
}

class ImpulseResponseAccumulator {
    private readonly samples: Float32Array;

    constructor(config: ImpulseResponseConfig) {
        this.samples = new Float32Array(
            config.samplingRate * config.lengthSeconds
        );
    }

    accumulateSample(hit: RayHit, frequency: number): void {
        const index = Math.floor(hit.time * this.config.samplingRate);
        const amplitude = Math.sqrt(hit.energy);
        const sampleValue = amplitude * Math.sin(
            2.0 * Math.PI * frequency * hit.time + hit.phase
        );

        if (index >= 0 && index < this.samples.length) {
            this.samples[index] += sampleValue;
        }
    }
}