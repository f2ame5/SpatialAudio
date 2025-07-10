# Ray Tracing Explanation

This document explains the ray tracing process implemented in `spatialAudio/src/raytracer/raytracer.ts` and `spatialAudio/src/raytracer/ray.ts`, focusing on how the `rayHits` data is generated and used.

## Overview

The goal of ray tracing in this context is to simulate how sound propagates in a 3D environment (the `Room`). This simulation is used to create a realistic spatial audio experience. The process involves tracing the paths of sound rays from a source to a listener, accounting for reflections off surfaces and energy loss due to material absorption and air absorption.

## The `RayTracer` Class

The `RayTracer` class in `raytracer.ts` is the core of the simulation. It manages the generation, propagation, and intersection of rays. Key methods and properties include:

*   **`config`:**  A `RayTracerConfig` object that defines parameters like the number of rays (`numRays`), maximum bounces (`maxBounces`), and minimum energy (`minEnergy`).
*   **`rays`:** An array of `Ray` objects.
*   **`hits`:** An array of `RayHit` objects. This is the array that is ultimately passed to the shader, and is the focus of this explanation.
*   **`generateRays()`:** Creates the initial rays originating from the sound source with random directions.
*   **`calculateRayPaths()`:** Orchestrates the ray tracing process, calling `calculateEarlyReflections()` and `calculateLateReflections()`.
*   **`calculateEarlyReflections()`:**  Calculates early reflections using the image source method (currently incomplete).
*   **`calculateLateReflections()`:** Handles the stochastic ray tracing for late reflections. This is where the majority of the `rayHits` data is generated.
*   **`getRayHits()`:** Returns the `hits` array.
*   **`render()`:** Visualizes the ray paths using the `RayRenderer`.

## The `Ray` Class

The `Ray` class in `ray.ts` represents a single sound ray.  It stores information about the ray's:

*   `origin`:  The starting point of the ray.
*   `direction`: The direction the ray is traveling.
*   `energyLow`, `energyMid`, `energyHigh`: The energy of the ray in three frequency bands (low, mid, high).
*   `pathLength`: The total distance the ray has traveled.
*   `bounces`: The number of times the ray has bounced off a surface.
*   `isActive`: A flag indicating whether the ray is still active.

The `updateRay()` method is crucial for updating the ray's state after each bounce:

```typescript
// From ray.ts
public updateRay(
    newOrigin: vec3,
    newDirection: vec3,
    energyLoss: {low: number, mid: number, high: number},
    distance: number,
    temperature: number = 20,
    humidity: number = 50
): void {
    vec3.copy(this.origin, newOrigin);
    vec3.normalize(this.direction, newDirection);

    // Apply frequency-dependent air absorption
    const airAbsorption = this.calculateAirAbsorption(distance, temperature, humidity);

    // Update energy levels with both material absorption and air absorption
    this.energyLow *= (1 - energyLoss.low) * airAbsorption.low;
    this.energyMid *= (1 - energyLoss.mid) * airAbsorption.mid;
    this.energyHigh *= (1 - energyLoss.high) * airAbsorption.high;

    this.pathLength += distance;
    this.bounces++;
}
```

This method updates the ray's properties based on the intersection with a surface. It accounts for both material absorption (`energyLoss`) and air absorption (`calculateAirAbsorption()`).

## Generating `rayHits`

The `rayHits` array is populated within the `calculateLateReflections()` method of the `RayTracer` class:

```typescript
// From raytracer.ts
private calculateLateReflections(): void {
    // ... (setup code) ...

    for (const ray of this.rays) {
        // ... (ray loop) ...

        while (ray.isRayActive() && ray.getBounces() < this.config.maxBounces && ray.getEnergy() > this.config.minEnergy) {
            // ... (intersection logic) ...

            if (intersection) {
                // ... (hit point and reflection calculations) ...

                // **KEY LINE: Populating the hits array**
                this.hits.push({
                    position: hitPoint,
                    energy: ray.getEnergy(),
                    time: ray.getPathLength() / 343, // Speed of sound = 343 m/s
                    energyLow: ray.getEnergyLow(),
                    energyMid: ray.getEnergyMid(),
                    energyHigh: ray.getEnergyHigh(),
                });

                // ... (update ray properties) ...
            } else {
                // ... (deactivate ray) ...
            }
        }
    }
}
```

Each time a ray intersects a surface, a `RayHit` object is created and added to the `hits` array.  The `RayHit` interface is defined as:

```typescript
//From raytracer.ts
export interface RayHit {
    position: vec3;
    energy: number;
    time: number;
    energyLow: number;
    energyMid: number;
    energyHigh: number;
}
```

The `RayHit` object contains:

*   `position`: The 3D coordinates of the intersection point.
*   `energy`: The average energy of the ray across all frequency bands.
*   `time`: The time it took for the ray to reach the intersection point (calculated as `pathLength / speed of sound`).
*   `energyLow`, `energyMid`, `energyHigh`: The energy of the ray in each frequency band.

## Early vs. Late Reflections

*   **Early Reflections:** These are the first few reflections of a sound ray. They are typically calculated using the image source method, which provides a deterministic and accurate representation of these reflections. The `calculateEarlyReflections()` method is intended for this, but its implementation is incomplete.
*   **Late Reflections:**  These are the numerous reflections that occur after the initial reflections. They are often modeled stochastically (using random sampling) due to the complexity of tracking every possible reflection. The `calculateLateReflections()` method handles this using the stochastic ray tracing algorithm described above.

## Visualization

The `render()` method in `RayTracer` uses a separate `RayRenderer` class to visualize the ray paths. This is primarily for debugging and understanding the ray tracing process, and is separate from the core spatial audio calculations.

## Summary

The `rayHits` array, which is passed to the shader, is populated by the `calculateLateReflections()` method in the `RayTracer` class. Each `RayHit` object represents a point where a ray intersects a surface in the room. The data within each `RayHit` (position, energy, time, and frequency-dependent energy) is calculated based on the ray's properties (origin, direction, energy, path length) and the material properties of the intersected surface (absorption coefficients). The ray's properties are updated after each bounce, accounting for both material absorption and air absorption. The `calculateEarlyReflections` function *would* also contribute to this array, but its implementation is incomplete.