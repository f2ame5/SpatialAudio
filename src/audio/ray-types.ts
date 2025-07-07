/**
 * Ray Types - Data structures for acoustic ray tracing
 */

import { vec3 } from 'gl-matrix';

/**
 * Acoustic ray structure for GPU storage
 * Must match WGSL struct layout exactly
 */
export interface AcousticRay {
    // Ray geometry
    origin: vec3;           // Ray origin position
    direction: vec3;        // Ray direction (normalized)
    
    // Energy properties
    energy: number;         // Total ray energy (0-1)
    phase: number;          // Phase offset in radians
    
    // Frequency-dependent energy (8 bands)
    frequencyEnergy: Float32Array; // [125, 250, 500, 1k, 2k, 4k, 8k, 16k] Hz
    
    // Path tracking
    pathLength: number;     // Total distance traveled
    arrivalTime: number;    // Time of arrival at listener (seconds)
    bounceCount: number;    // Number of reflections
    
    // State
    active: number;         // 1 if ray is still active, 0 if terminated
    
    // Material history (for accurate reflection modeling)
    lastMaterialId: number; // ID of last hit material
    
    // Padding for GPU alignment (vec4 alignment)
    padding: Float32Array;  // 2 floats for alignment
}

/**
 * Ray distribution types
 */
export enum RayDistributionType {
    UNIFORM_SPHERE = 0,    // Uniform distribution over full sphere
    HEMISPHERE = 1,        // Upper hemisphere only (z >= 0)
    CONE = 2,             // Directional cone distribution
    FIBONACCI = 3         // Fibonacci spiral (deterministic, very uniform)
}

/**
 * Ray generation parameters
 */
export interface RayGenerationParams {
    sourcePosition: vec3;   // Sound source position
    sourceRadius: number;   // Source sphere radius
    rayCount: number;       // Total number of rays to generate
    initialEnergy: number;  // Initial energy per ray
    time: number;          // Current simulation time
    seed: number;          // Random seed for deterministic generation
    frequencyWeights: Float32Array; // Energy distribution across frequencies (8 bands)
    distributionType: RayDistributionType; // Ray distribution strategy
    coneAngle: number;     // Cone angle in radians (for directional sources)
}

/**
 * Ray bouncing parameters
 */
export interface RayBouncingParams {
    roomMin: vec3;         // Room bounding box minimum
    roomMax: vec3;         // Room bounding box maximum
    maxBounces: number;    // Maximum allowed bounces
    minEnergy: number;     // Energy threshold for termination
    speedOfSound: number;  // Speed of sound (m/s)
    airAbsorption: Float32Array; // Frequency-dependent air absorption
    timeStep: number;      // Simulation time step
}

/**
 * Listener configuration for ray collection
 */
export interface ListenerConfig {
    position: vec3;        // Listener position
    radius: number;        // Collection sphere radius
    forward: vec3;         // Forward direction (for HRTF)
    up: vec3;             // Up direction
}

/**
 * Impulse response sample
 */
export interface ImpulseResponseSample {
    timeBin: number;       // Time bin index
    energy: number;        // Total energy
    phase: number;         // Phase information
    frequencyEnergy: Float32Array; // Energy per frequency band
    direction: vec3;       // Arrival direction (for spatial processing)
}

/**
 * Ray buffer configuration
 */
export interface RayBufferConfig {
    maxRays: number;       // Maximum number of rays
    doubleBuffering: boolean; // Use double buffering for updates
    gpuOnly: boolean;      // Keep data on GPU only
}

/**
 * Create an empty ray
 */
export function createRay(): AcousticRay {
    return {
        origin: vec3.create(),
        direction: vec3.create(),
        energy: 0,
        phase: 0,
        frequencyEnergy: new Float32Array(8),
        pathLength: 0,
        arrivalTime: 0,
        bounceCount: 0,
        active: 0,
        lastMaterialId: -1,
        padding: new Float32Array(2)
    };
}

/**
 * Ray struct size in bytes (for GPU buffer allocation)
 * New WGSL structure with vec4 alignment:
 * - origin: vec4<f32> = 16 bytes
 * - direction: vec4<f32> = 16 bytes
 * - energy_phase: vec4<f32> = 16 bytes
 * - frequency_energy_low: vec4<f32> = 16 bytes
 * - frequency_energy_high: vec4<f32> = 16 bytes
 * - path_data: vec4<f32> = 16 bytes
 * - material_data: vec4<f32> = 16 bytes
 * Total: 112 bytes per ray (7 * 16 bytes)
 */
export const RAY_STRUCT_SIZE = 112; // bytes

/**
 * Calculate required buffer size for rays
 */
export function calculateRayBufferSize(rayCount: number): number {
    return rayCount * RAY_STRUCT_SIZE;
}

/**
 * Pack ray data for GPU upload (matches WGSL Ray struct layout)
 */
export function packRayForGPU(ray: AcousticRay): Float32Array {
    const data = new Float32Array(RAY_STRUCT_SIZE / 4); // 28 floats = 112 bytes
    let offset = 0;

    // Origin (vec4, xyz used, w padding)
    data[offset++] = ray.origin[0];
    data[offset++] = ray.origin[1];
    data[offset++] = ray.origin[2];
    data[offset++] = 0; // w padding

    // Direction (vec4, xyz used, w padding)
    data[offset++] = ray.direction[0];
    data[offset++] = ray.direction[1];
    data[offset++] = ray.direction[2];
    data[offset++] = 0; // w padding

    // Energy and phase (vec4, xy used, zw padding)
    data[offset++] = ray.energy;
    data[offset++] = ray.phase;
    data[offset++] = 0; // z padding
    data[offset++] = 0; // w padding

    // Frequency energy low (vec4, 125, 250, 500, 1k Hz)
    data[offset++] = ray.frequencyEnergy[0]; // 125 Hz
    data[offset++] = ray.frequencyEnergy[1]; // 250 Hz
    data[offset++] = ray.frequencyEnergy[2]; // 500 Hz
    data[offset++] = ray.frequencyEnergy[3]; // 1 kHz

    // Frequency energy high (vec4, 2k, 4k, 8k, 16k Hz)
    data[offset++] = ray.frequencyEnergy[4]; // 2 kHz
    data[offset++] = ray.frequencyEnergy[5]; // 4 kHz
    data[offset++] = ray.frequencyEnergy[6]; // 8 kHz
    data[offset++] = ray.frequencyEnergy[7]; // 16 kHz

    // Path data (vec4, x=path_length, y=arrival_time, z=bounce_count, w=active)
    data[offset++] = ray.pathLength;
    data[offset++] = ray.arrivalTime;
    data[offset++] = ray.bounceCount;
    data[offset++] = ray.active;

    // Material data (vec4, x=last_material_id, yzw=padding)
    data[offset++] = ray.lastMaterialId;
    data[offset++] = 0; // y padding
    data[offset++] = 0; // z padding
    data[offset++] = 0; // w padding
    data[offset++] = 0; // final padding
    
    return data;
}

/**
 * Unpack ray data from GPU (matches WGSL Ray struct layout)
 */
export function unpackRayFromGPU(data: Float32Array, offset: number = 0): AcousticRay {
    const ray = createRay();
    let idx = offset;

    // Ray struct in WGSL:
    // origin: vec4<f32>          (16 bytes)
    // direction: vec4<f32>       (16 bytes)
    // energy_phase: vec4<f32>    (16 bytes)
    // frequency_energy_low: vec4<f32>  (16 bytes)
    // frequency_energy_high: vec4<f32> (16 bytes)
    // path_data: vec4<f32>       (16 bytes)
    // material_data: vec4<f32>   (16 bytes)
    // Total: 112 bytes = 28 floats

    // Origin (vec4, xyz used, w padding)
    ray.origin[0] = data[idx++];
    ray.origin[1] = data[idx++];
    ray.origin[2] = data[idx++];
    idx++; // skip w padding

    // Direction (vec4, xyz used, w padding)
    ray.direction[0] = data[idx++];
    ray.direction[1] = data[idx++];
    ray.direction[2] = data[idx++];
    idx++; // skip w padding

    // Energy and phase (vec4, xy used, zw padding)
    ray.energy = data[idx++];
    ray.phase = data[idx++];
    idx += 2; // skip zw padding

    // Frequency energy low (vec4, 125, 250, 500, 1k Hz)
    ray.frequencyEnergy[0] = data[idx++]; // 125 Hz
    ray.frequencyEnergy[1] = data[idx++]; // 250 Hz
    ray.frequencyEnergy[2] = data[idx++]; // 500 Hz
    ray.frequencyEnergy[3] = data[idx++]; // 1 kHz

    // Frequency energy high (vec4, 2k, 4k, 8k, 16k Hz)
    ray.frequencyEnergy[4] = data[idx++]; // 2 kHz
    ray.frequencyEnergy[5] = data[idx++]; // 4 kHz
    ray.frequencyEnergy[6] = data[idx++]; // 8 kHz
    ray.frequencyEnergy[7] = data[idx++]; // 16 kHz

    // Path data (vec4, x=path_length, y=arrival_time, z=bounce_count, w=active)
    ray.pathLength = data[idx++];
    ray.arrivalTime = data[idx++];
    ray.bounceCount = data[idx++];
    ray.active = data[idx++];

    // Material data (vec4, x=last_material_id, yzw=padding)
    ray.lastMaterialId = data[idx++];
    idx += 3; // skip yzw padding

    return ray;
}

/**
 * Ray statistics for debugging
 */
export interface RayStatistics {
    totalRays: number;
    activeRays: number;
    averageBounces: number;
    averagePathLength: number;
    energyConserved: number;
    terminationReasons: {
        maxBounces: number;
        minEnergy: number;
        escaped: number;
        absorbed: number;
    };
}

/**
 * Calculate ray statistics from ray buffer
 */
export function calculateRayStatistics(rays: AcousticRay[]): RayStatistics {
    let activeCount = 0;
    let totalBounces = 0;
    let totalPathLength = 0;
    let totalEnergy = 0;
    const terminations = {
        maxBounces: 0,
        minEnergy: 0,
        escaped: 0,
        absorbed: 0
    };
    
    for (const ray of rays) {
        if (ray.active > 0) {
            activeCount++;
        }
        totalBounces += ray.bounceCount;
        totalPathLength += ray.pathLength;
        totalEnergy += ray.energy;
    }
    
    return {
        totalRays: rays.length,
        activeRays: activeCount,
        averageBounces: rays.length > 0 ? totalBounces / rays.length : 0,
        averagePathLength: rays.length > 0 ? totalPathLength / rays.length : 0,
        energyConserved: totalEnergy,
        terminationReasons: terminations
    };
}

/**
 * Ray direction generation strategies
 */
export enum RayDistribution {
    UNIFORM_SPHERE = 'uniform_sphere',
    RANDOM_SPHERE = 'random_sphere',
    FIBONACCI_SPHERE = 'fibonacci_sphere',
    ICOSAHEDRON = 'icosahedron',
    DIRECTIONAL = 'directional'
}

/**
 * Generate ray directions based on distribution strategy
 */
export function generateRayDirections(
    count: number,
    distribution: RayDistribution = RayDistribution.FIBONACCI_SPHERE
): vec3[] {
    const directions: vec3[] = [];
    
    switch (distribution) {
        case RayDistribution.FIBONACCI_SPHERE:
            // Fibonacci sphere - even distribution
            const phi = Math.PI * (3 - Math.sqrt(5)); // Golden angle
            for (let i = 0; i < count; i++) {
                const y = 1 - (i / (count - 1)) * 2; // -1 to 1
                const radius = Math.sqrt(1 - y * y);
                const theta = phi * i;
                
                const dir = vec3.fromValues(
                    Math.cos(theta) * radius,
                    y,
                    Math.sin(theta) * radius
                );
                vec3.normalize(dir, dir);
                directions.push(dir);
            }
            break;
            
        case RayDistribution.UNIFORM_SPHERE:
            // Uniform grid on sphere
            const samples = Math.ceil(Math.sqrt(count));
            for (let i = 0; i < samples; i++) {
                for (let j = 0; j < samples; j++) {
                    if (directions.length >= count) break;
                    
                    const theta = (i / samples) * Math.PI;
                    const phi = (j / samples) * 2 * Math.PI;
                    
                    const dir = vec3.fromValues(
                        Math.sin(theta) * Math.cos(phi),
                        Math.cos(theta),
                        Math.sin(theta) * Math.sin(phi)
                    );
                    directions.push(dir);
                }
            }
            break;
            
        case RayDistribution.RANDOM_SPHERE:
            // Random distribution
            for (let i = 0; i < count; i++) {
                const theta = Math.random() * Math.PI;
                const phi = Math.random() * 2 * Math.PI;
                
                const dir = vec3.fromValues(
                    Math.sin(theta) * Math.cos(phi),
                    Math.cos(theta),
                    Math.sin(theta) * Math.sin(phi)
                );
                directions.push(dir);
            }
            break;
    }
    
    return directions;
}
