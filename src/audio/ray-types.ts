/**
 * Ray Types - Data structures for acoustic raytracing
 */

import { vec3 } from 'gl-matrix';

/**
 * Acoustic ray data structure
 */
export interface AcousticRay {
    // Position and direction
    origin: vec3;
    direction: vec3;
    
    // Energy and phase information
    energy: number;
    phase: number;
    
    // Frequency-dependent energy (8 bands: 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz)
    frequencyEnergy: number[];
    
    // Path information
    pathLength: number;
    arrivalTime: number;
    bounceCount: number;
    active: boolean;
    
    // Material interaction history
    lastMaterialId: number;
    lastSurfaceNormal?: vec3;
}

/**
 * Ray distribution types for generation
 */
export enum RayDistributionType {
    UNIFORM_SPHERICAL = 'uniform_spherical',
    FIBONACCI_SPHERICAL = 'fibonacci_spherical',
    RANDOM_SPHERICAL = 'random_spherical',
    HEMISPHERICAL = 'hemispherical'
}

/**
 * Ray visualization data for rendering
 */
export interface RayVisualizationData {
    start: vec3;
    end: vec3;
    energy: number;
    bounceCount: number;
    frequency: number; // Dominant frequency for coloring
    active: boolean;
}

/**
 * Ray intersection result
 */
export interface RayIntersection {
    hit: boolean;
    distance: number;
    point: vec3;
    normal: vec3;
    surfaceId: number;
    materialId: number;
}

/**
 * Ray generation parameters
 */
export interface RayGenerationParams {
    sourcePosition: vec3;
    sourceRadius: number;
    rayCount: number;
    initialEnergy: number;
    distributionType: RayDistributionType;
    frequencyWeights: number[]; // 8 frequency bands
    randomSeed: number;
}

/**
 * Ray bouncing parameters
 */
export interface RayBouncingParams {
    roomMin: vec3;
    roomMax: vec3;
    maxBounces: number;
    minEnergy: number;
    speedOfSound: number;
    timeStep: number;
    airAbsorption: number[]; // 8 frequency bands
    surfaceMaterials: number[]; // Material IDs for each surface
}

/**
 * Ray collection parameters for impulse response generation
 */
export interface RayCollectionParams {
    listenerPosition: vec3;
    listenerRadius: number;
    sampleRate: number;
    irLength: number; // seconds
    timeBinSize: number;
}

/**
 * Impulse response bin data
 */
export interface ImpulseBin {
    energy: number;
    phase: number;
    frequencyEnergy: number[]; // 8 frequency bands
    sampleCount: number;
}

/**
 * Ray statistics for debugging
 */
export interface RayStatistics {
    totalRaysGenerated: number;
    activeRays: number;
    averageEnergy: number;
    averageBounces: number;
    averagePathLength: number;
    raysCollected: number;
    totalEnergyCollected: number;
}

/**
 * Material interaction result
 */
export interface MaterialInteraction {
    absorbed: boolean;
    reflected: boolean;
    scattered: boolean;
    newDirection: vec3;
    energyLoss: number;
    frequencyResponse: number[]; // 8 frequency bands
}

/**
 * Ray path segment for visualization
 */
export interface RayPathSegment {
    start: vec3;
    end: vec3;
    energy: number;
    segmentIndex: number;
    materialId: number;
    surfaceId: number;
}

/**
 * Complete ray path for full visualization
 */
export interface RayPath {
    rayId: number;
    segments: RayPathSegment[];
    totalLength: number;
    totalTime: number;
    finalEnergy: number;
    bounceCount: number;
}

/**
 * Ray batch for GPU processing
 */
export interface RayBatch {
    rays: AcousticRay[];
    batchId: number;
    timestamp: number;
    processed: boolean;
}

/**
 * Acoustic material properties
 */
export interface AcousticMaterial {
    id: number;
    name: string;
    absorption: number[]; // 8 frequency bands (0-1)
    scattering: number[]; // 8 frequency bands (0-1)
    impedance: number;
    roughness: number; // 0-1, affects reflection type
}

/**
 * Room surface definition
 */
export interface RoomSurface {
    id: number;
    name: string;
    materialId: number;
    normal: vec3;
    area: number;
}

/**
 * Environmental conditions affecting ray propagation
 */
export interface EnvironmentalConditions {
    temperature: number; // Celsius
    humidity: number; // Percentage (0-100)
    pressure: number; // Pa
    windVelocity?: vec3; // m/s (optional)
}

/**
 * Ray simulation configuration
 */
export interface RaySimulationConfig {
    generation: RayGenerationParams;
    bouncing: RayBouncingParams;
    collection: RayCollectionParams;
    environment: EnvironmentalConditions;
    visualization: {
        enabled: boolean;
        maxRaysToShow: number;
        colorByEnergy: boolean;
        colorByFrequency: boolean;
        colorByBounceCount: boolean;
    };
}
