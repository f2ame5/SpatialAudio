# Ray Tracing Modification Plan (Revised Again)

## Goal

Modify the ray tracer to make the rays behave like sound waves.

## Clarification and Considerations

*   **Attenuation:** Attenuation should depend on the room's geometry and materials. The impulse response will characterize the room and source location. We need to calculate attenuation based on ray path length.
*   **Phase Difference:**  Adding phase differences between rays is crucial for simulating wave interference, which significantly impacts the perceived sound. This should be included.
* **Speed of Sound:** The speed of sound is relevant to both attenuation (time delay) and phase calculations. We need to incorporate this constant.

## Information Gathering

1.  **Examine existing code:**
    *   Review `raytracer.ts` to understand how energy loss and the `energyLow/Mid/High` values are currently handled. Check for any existing distance calculations.
    *   Check `room-materials.ts` to see how materials are defined.
    *   Look for any existing code related to diffraction or distance-based attenuation.
2.  **Ask clarifying questions (in the plan, for now):**
    *   How are the `energyLow`, `energyMid`, and `energyHigh` values intended to be used? Do the materials have corresponding absorption coefficients for these frequency bands?
    * Should we consider air absorption, which also attenuates sound, especially at higher frequencies?

## Aspects of Sound Wave Behavior

We need to simulate the following:

*   **Reflection:** Sound waves reflect off surfaces, losing energy with each reflection. The current ray tracer already does this.
*   **Diffraction:** Sound waves bend around obstacles. The current ray tracer *does not* do this. Implementing diffraction would be a significant change.
*   **Attenuation:** Sound intensity decreases with distance. The current ray tracer has energy loss on reflection, but perhaps not a continuous distance-based attenuation.
*   **Frequency-dependent absorption:** Different materials absorb different sound frequencies differently. The current ray tracer has `energyLow`, `energyMid`, and `energyHigh`, suggesting some frequency-dependent behavior, but it's not clear how it's used.
* **Speed of sound**: The current ray tracer does not take into account the speed of sound.
* **Phase:** The phase of the sound waves.

## File to Modify

`spatialAudio/src/raytracer/raytracer.ts` (and potentially others, depending on the clarification)

## Changes (Placeholder - will be refined after clarification)

Likely changes include:

*   Adding a `speedOfSound` constant.
*   Tracking the distance traveled by each ray.
*   Calculating attenuation based on distance and possibly air absorption.
*   Tracking the phase of each ray, likely based on distance traveled and wavelength (which relates to frequency).
*   Modifying how reflections affect phase (e.g., a 180-degree phase shift on reflection from a rigid surface).
* Implementing diffraction.

## Implementation Approach

1.  **Distance and Speed:**
    *   Introduce a `speedOfSound` constant.
    *   Modify the ray structure to include a `distanceTraveled` property.
    *   Update the ray tracing logic to increment `distanceTraveled` as the ray moves.

2.  **Basic Attenuation:**
    *   Implement a simple distance-based attenuation factor (e.g., inverse square law).

3.  **Phase:**
    *   Add a `phase` property to the ray.
    *   Calculate phase based on `distanceTraveled` and wavelength.
    *   Update phase on reflection.

4.  **Frequency-Dependent Absorption:**
    *   Clarify how `energyLow/Mid/High` are used.
    *   Implement material-specific absorption coefficients.
    *   Apply absorption during reflection based on frequency and material.

5.  **Diffraction:**
    *   Research diffraction algorithms (this is a complex topic).
    *   Implement a suitable diffraction model. This will likely be the most challenging part.

## Implementation Steps

1.  Create this revised plan document (`raytracing_modification_plan.md`).
2.  Request user approval of the plan (including the information gathering steps).
3.  Gather information (examine code and, once possible, ask clarifying questions).
4.  Refine the "Changes" section of this plan based on the information gathered.
5.  Request a mode switch to implement the changes.