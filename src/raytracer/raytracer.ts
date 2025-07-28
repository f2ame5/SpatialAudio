import { vec3 } from 'gl-matrix';
import { Ray } from './ray';
import { Room } from '../room/room';
import { Sphere } from '../objects/sphere';
import { RayRenderer } from './ray-renderer';
import { WallMaterial } from '../room/room-materials';
import { loadHRTFData } from '../sound/hrtf-loader';
import { Camera } from '../camera/camera';

// Helper interface for ISM generation
interface ImageSourceInternal extends ImageSource {
    lastReflectionIndex: number;
}

// Type definitions for image source method
interface ImageSource {
    position: vec3;
    imageOrder: number;
    reflectionSequence: number[];
}

interface RayPath {
    origin: vec3;
    direction: vec3;
    energy: number;
    bounces: number[];
    time: number;
}

export interface RayTracerConfig {
    numRays: number;
    maxBounces: number;
    minEnergy: number;
}

// Updated RayHit interface for 8 bands
export interface RayHit {
    position: vec3;
    energy: number; // Average energy
    // Energy for each of the 8 frequency bands
    energy63: number;
    energy125: number;
    energy250: number;
    energy500: number;
    energy1k: number;
    energy2k: number;
    energy4k: number;
    energy8k: number;
    time: number;      // Arrival time
    // Phase for each of the 8 frequency bands
    phase63: number;
    phase125: number;
    phase250: number;
    phase500: number;
    phase1k: number;
    phase2k: number;
    phase4k: number;
    phase8k: number;
    // Frequency is now implicit in the band
    dopplerShift: number; // Doppler shift at this point
    hrtfIndex?: number; // HRTF lookup index for spatial audio
    incomingDirection: vec3; // Direction of the ray when it hit
}

// Updated RayPathPoint interface for 8 bands
export interface RayPathPoint extends RayHit {
    bounceNumber: number;
    rayIndex: number;
    direction: vec3;      // Add direction to resolve TypeScript error
}

export interface ImpulseResponse {
    time: Float32Array;     // Time points
    amplitude: Float32Array; // Amplitude values
    sampleRate: number;     // Sample rate of the impulse response
    frequencies: Float32Array; // Frequency content at each time point
}

export class RayTracer {
    private device: GPUDevice;
    private soundSource: Sphere;
    private room: Room;
    private config: RayTracerConfig;
    private rays: Ray[] = [];
    private hits: RayHit[] = [];
    private rayPaths: { origin: vec3, direction: vec3, energy: number }[] = [];
    private rayPathPoints: RayPathPoint[] = []; // Store all points along ray paths
    private readonly VISIBILITY_THRESHOLD = 0.05; // Rays below 5% energy become invisible
    private readonly SPEED_OF_SOUND = 343.0; // Speed of sound in m/s at 20°C
    private readonly AIR_TEMPERATURE = 20.0;  // Air temperature in Celsius
    private rayRenderer: RayRenderer;
    private listenerBuffer!: GPUBuffer;
    private rayHitsBuffer!: GPUBuffer;
    private spatialIRBuffer!: GPUBuffer;
    private paramsBuffer!: GPUBuffer;
    private acousticsBuffer!: GPUBuffer;
    private wavePropertiesBuffer!: GPUBuffer;
    private roomMaterialsBuffer!: GPUBuffer;
    private computePipeline!: GPUComputePipeline;
    private hrtfBuffer!: GPUBuffer; // Ensure only one declaration
    
    // Define the 8 frequency bands
    private static readonly FREQUENCY_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000];

    constructor(
        device: GPUDevice,
        soundSource: Sphere,
        room: Room,
        config: RayTracerConfig = {
            numRays: 10000,
            maxBounces: 50,
            minEnergy: 0.01
        }
    ) {
        this.device = device;
        this.soundSource = soundSource;
        this.room = room;
        this.config = config;
        this.rayRenderer = new RayRenderer(device);
        // Removed 'await' from constructor
    }

    private generateRays(): void {
        this.rays = [];
        const sourcePos = this.soundSource.getPosition();
        const sphereRadius = this.soundSource.getRadius();
        
        for (let i = 0; i < this.config.numRays; i++) {
            // Generate random direction using spherical coordinates
            const theta = 2 * Math.PI * Math.random();
            const phi = Math.acos(2 * Math.random() - 1);
            const direction = vec3.fromValues(
                Math.sin(phi) * Math.cos(theta),
                Math.sin(phi) * Math.sin(theta),
                Math.cos(phi)
            );
            
            // Calculate the ray origin on the sphere's surface
            const rayOrigin = vec3.create();
            vec3.scale(rayOrigin, direction, sphereRadius);
            vec3.add(rayOrigin, rayOrigin, sourcePos);
            
            // Distribute rays across frequency bands
            // In a more advanced implementation, you might want to create separate rays for each band
            // For now, we'll use one ray and handle all bands in the Ray class
            this.rays.push(new Ray(rayOrigin, direction, 1.0));
        }
    }

    private isPointInRoom(point: vec3): boolean {
        const { width, height, depth } = this.room.config.dimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;
        return point[0] >= -halfWidth && point[0] <= halfWidth &&
               point[1] >= 0 && point[1] <= height &&
               point[2] >= -halfDepth && point[2] <= halfDepth;
    }

    public async calculateRayPaths(camera?: Camera): Promise<void> {
        // Clear previous data
        this.hits = [];
        this.rays = [];
        this.rayPaths = [];
        this.rayPathPoints = [];
        
        // Generate initial rays
        this.generateRays();
        
        // Store initial ray paths
        for (const ray of this.rays) {
            this.rayPaths.push({
                origin: ray.getOrigin(),
                direction: ray.getDirection(),
                energy: ray.getEnergy()
            });
        }
        
        // Calculate early reflections using image source method
        this.calculateEarlyReflections(camera);
        
        // Calculate late reflections using stochastic ray tracing
        this.calculateLateReflections(camera);
        
        console.log(`Completed ray tracing with ${this.hits.length} total hits.`);
    }

    private calculateEarlyReflections(camera?: Camera): void {
        if (!camera) return;
        const sourcePos = this.soundSource.getPosition();
        const listenerPos = camera.getPosition();
        const maxOrder = 4; // Maximum reflection order for image sources
        const { width, height, depth } = this.room.config.dimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;
        const planes = [
            { normal: vec3.fromValues(1, 0, 0), point: vec3.fromValues(-halfWidth, 0, 0), material: this.room.config.materials.walls },   // 0: Left
            { normal: vec3.fromValues(-1, 0, 0), point: vec3.fromValues(halfWidth, 0, 0), material: this.room.config.materials.walls },  // 1: Right
            { normal: vec3.fromValues(0, 1, 0), point: vec3.fromValues(0, 0, 0), material: this.room.config.materials.floor },   // 2: Floor
            { normal: vec3.fromValues(0, -1, 0), point: vec3.fromValues(0, height, 0), material: this.room.config.materials.ceiling }, // 3: Ceiling
            { normal: vec3.fromValues(0, 0, 1), point: vec3.fromValues(0, 0, -halfDepth), material: this.room.config.materials.walls },   // 4: Back
            { normal: vec3.fromValues(0, 0, -1), point: vec3.fromValues(0, 0, halfDepth), material: this.room.config.materials.walls },  // 5: Front
        ];
        
        // Generate image sources up to maxOrder
        const imageSources = this.generateImageSources(sourcePos, maxOrder);
        
        // Process each image source
        for (const imageSource of imageSources) {
            // Path from listener to virtual source
            const dirToSource = vec3.subtract(vec3.create(), imageSource.position, listenerPos);
            const totalDist = vec3.length(dirToSource);
            vec3.normalize(dirToSource, dirToSource);
            
            // Basic visibility check: is there a wall between listener and image source?
            // This is a simple but effective occlusion check for convex rooms.
            let isOccluded = false;
            for (const plane of planes) {
                const denom = vec3.dot(dirToSource, plane.normal);
                if (Math.abs(denom) > 1e-6) {
                    const t = (vec3.dot(plane.point, plane.normal) - vec3.dot(listenerPos, plane.normal)) / denom;
                    // If intersection is between listener and virtual source
                    if (t > 1e-6 && t < totalDist - 1e-6) {
                        isOccluded = true;
                        break;
                    }
                }
            }
            if (isOccluded) continue;
            
            // If visible, calculate properties
            const travelTime = totalDist / this.SPEED_OF_SOUND;
            
            // Phase 3: Apply frequency-dependent absorption from materials
            // Initialize energy for all bands
            let energy63 = 1.0, energy125 = 1.0, energy250 = 1.0, energy500 = 1.0;
            let energy1k = 1.0, energy2k = 1.0, energy4k = 1.0, energy8k = 1.0;
            
            for (const planeIndex of imageSource.reflectionSequence) {
                const material = planes[planeIndex].material;
                energy63  *= (1.0 - (material.absorption63 ?? 0));
                energy125 *= (1.0 - (material.absorption125 ?? 0));
                energy250 *= (1.0 - (material.absorption250 ?? 0));
                energy500 *= (1.0 - (material.absorption500 ?? 0));
                energy1k  *= (1.0 - (material.absorption1k ?? 0));
                energy2k  *= (1.0 - (material.absorption2k ?? 0));
                energy4k  *= (1.0 - (material.absorption4k ?? 0));
                energy8k  *= (1.0 - (material.absorption8k ?? 0));
            }
            
            const distAtten = 1.0 / (1.0 + totalDist * totalDist);
            energy63  *= distAtten;
            energy125 *= distAtten;
            energy250 *= distAtten;
            energy500 *= distAtten;
            energy1k  *= distAtten;
            energy2k  *= distAtten;
            energy4k  *= distAtten;
            energy8k  *= distAtten;
            
            // Calculate average energy
            const averageEnergy = (energy63 + energy125 + energy250 + energy500 +
                                 energy1k + energy2k + energy4k + energy8k) / 8;
            
            this.hits.push({
                position: listenerPos, // For ISM, the "hit" is at the listener
                energy: averageEnergy,
                energy63: energy63,
                energy125: energy125,
                energy250: energy250,
                energy500: energy500,
                energy1k: energy1k,
                energy2k: energy2k,
                energy4k: energy4k,
                energy8k: energy8k,
                time: travelTime,
                // Initialize phase for all bands (random for ISM)
                phase63: Math.random() * 2 * Math.PI,
                phase125: Math.random() * 2 * Math.PI,
                phase250: Math.random() * 2 * Math.PI,
                phase500: Math.random() * 2 * Math.PI,
                phase1k: Math.random() * 2 * Math.PI,
                phase2k: Math.random() * 2 * Math.PI,
                phase4k: Math.random() * 2 * Math.PI,
                phase8k: Math.random() * 2 * Math.PI,
                dopplerShift: 1.0,
                incomingDirection: dirToSource,
                // hrtfIndex can be calculated here based on dirToSource
            });
        }
    }

    private generateImageSources(sourcePos: vec3, maxOrder: number): ImageSource[] {
        const { width, height, depth } = this.room.config.dimensions;
        const reflectionPlanes = [
            { point: vec3.fromValues(-width / 2, 0, 0), normal: vec3.fromValues(1, 0, 0) }, // 0: Left
            { point: vec3.fromValues(width / 2, 0, 0), normal: vec3.fromValues(-1, 0, 0) }, // 1: Right
            { point: vec3.fromValues(0, 0, 0), normal: vec3.fromValues(0, 1, 0) }, // 2: Floor
            { point: vec3.fromValues(0, height, 0), normal: vec3.fromValues(0, -1, 0) }, // 3: Ceiling
            { point: vec3.fromValues(0, 0, -depth / 2), normal: vec3.fromValues(0, 0, 1) }, // 4: Back
            { point: vec3.fromValues(0, 0, depth / 2), normal: vec3.fromValues(0, 0, -1) }, // 5: Front
        ];
        
        let currentSources: ImageSourceInternal[] = [{
            position: sourcePos,
            imageOrder: 0,
            reflectionSequence: [],
            lastReflectionIndex: -1
        }];
        
        const allImageSources: ImageSource[] = [];
        let sourcesToProcess = currentSources;
        
        for (let order = 1; order <= maxOrder; order++) {
            const nextSources: ImageSourceInternal[] = [];
            for (const source of sourcesToProcess) {
                for (let i = 0; i < reflectionPlanes.length; i++) {
                    if (i === source.lastReflectionIndex) continue;
                    const plane = reflectionPlanes[i];
                    const reflectedPos = vec3.create();
                    const v = vec3.subtract(vec3.create(), source.position, plane.point);
                    const dot_v_n = vec3.dot(v, plane.normal);
                    vec3.scaleAndAdd(reflectedPos, source.position, plane.normal, -2 * dot_v_n);
                    const newSource = {
                        position: reflectedPos,
                        imageOrder: order,
                        reflectionSequence: [...source.reflectionSequence, i],
                        lastReflectionIndex: i,
                    };
                    nextSources.push(newSource);
                }
            }
            allImageSources.push(...nextSources);
            sourcesToProcess = nextSources;
        }
        
        return allImageSources;
    }

    private calculateLateReflections(camera?: Camera): void {
        const { width, height, depth } = this.room.config.dimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;
        
        // Define room planes (normal points inward)
        const planes = [
            // Use walls material for all wall surfaces
            { normal: vec3.fromValues(1, 0, 0), d: halfWidth,  material: this.room.config.materials.walls },   // Left wall
            { normal: vec3.fromValues(-1, 0, 0), d: halfWidth, material: this.room.config.materials.walls },  // Right wall
            { normal: vec3.fromValues(0, 1, 0), d: 0,         material: this.room.config.materials.floor },   // Floor
            { normal: vec3.fromValues(0, -1, 0), d: height,   material: this.room.config.materials.ceiling }, // Ceiling
            { normal: vec3.fromValues(0, 0, 1), d: halfDepth, material: this.room.config.materials.walls },   // Back wall
            { normal: vec3.fromValues(0, 0, -1), d: halfDepth, material: this.room.config.materials.walls },  // Front wall
        ];
        
        // Process each ray
        for (let rayIndex = 0; rayIndex < this.rays.length; rayIndex++) {
            const ray = this.rays[rayIndex];
            let bounces = 0;
            let currentTime = 0; // Track cumulative time for this ray
            
            // Calculate HRTF index for initial point if camera is provided
            let initialHrtfIndex: number | undefined;
            if (camera) {
                const listenerPos = camera.getPosition();
                const toListener = vec3.subtract(vec3.create(), listenerPos, ray.getOrigin());
                vec3.normalize(toListener, toListener);
                // Convert to spherical coordinates
                const azimuth = Math.atan2(toListener[2], toListener[0]); // -π to π
                const elevation = Math.asin(toListener[1]); // -π/2 to π/2
                // Map to HRTF indices (assuming 360° azimuth, 180° elevation)
                const azimuthIndex = Math.floor(((azimuth + Math.PI) / (2 * Math.PI)) * 360) % 360;
                const elevationIndex = Math.floor(((elevation + Math.PI/2) / Math.PI) * 180) % 180;
                initialHrtfIndex = elevationIndex * 360 + azimuthIndex;
            }
            
            // Store initial point
            this.rayPathPoints.push({
                // Initial ray position without hit data (no surface interaction yet)
                position: ray.getOrigin(),
                direction: ray.getDirection(), // Add missing direction property
                energy: ray.getEnergy(),
                energy63: ray.getEnergy63(),
                energy125: ray.getEnergy125(),
                energy250: ray.getEnergy250(),
                energy500: ray.getEnergy500(),
                energy1k: ray.getEnergy1k(),
                energy2k: ray.getEnergy2k(),
                energy4k: ray.getEnergy4k(),
                energy8k: ray.getEnergy8k(),
                time: 0,
                // Store phase for all bands
                phase63: ray.getPhase63(),
                phase125: ray.getPhase125(),
                phase250: ray.getPhase250(),
                phase500: ray.getPhase500(),
                phase1k: ray.getPhase1k(),
                phase2k: ray.getPhase2k(),
                phase4k: ray.getPhase4k(),
                phase8k: ray.getPhase8k(),
                dopplerShift: 1.0, // Default doppler shift (no shift)
                bounceNumber: 0,
                rayIndex: rayIndex,
                incomingDirection: ray.getDirection(),
                hrtfIndex: initialHrtfIndex
            });
            
            while (ray.isRayActive() && bounces < this.config.maxBounces && ray.getEnergy() > this.config.minEnergy) {
                let closestT = Infinity;
                let closestPlane = null;
                const origin = ray.getOrigin();
                const direction = ray.getDirection();
                
                for (const plane of planes) {
                    const denom = vec3.dot(direction, plane.normal);
                    if (Math.abs(denom) > 0.0001) { // Avoid parallel rays
                        const t = -(vec3.dot(origin, plane.normal) + plane.d) / denom;
                        if (t > 0.0001 && t < closestT) {
                            closestT = t;
                            closestPlane = plane;
                        }
                    }
                }
                
                if (closestPlane && closestPlane.material) {
                    const hitPoint = vec3.scaleAndAdd(vec3.create(), origin, direction, closestT - 0.0001);
                    
                    // Calculate time taken for the ray to reach this point
                    const distanceTraveled = vec3.distance(origin, hitPoint);
                    const travelTime = this.calculateTravelTime(distanceTraveled);
                    currentTime += travelTime;
                    
                    // Calculate phase change based on distance and frequency for each band
                    const phaseChange63 = (2 * Math.PI * RayTracer.FREQUENCY_BANDS[0] * distanceTraveled) / this.calculateSpeedOfSound();
                    const phaseChange125 = (2 * Math.PI * RayTracer.FREQUENCY_BANDS[1] * distanceTraveled) / this.calculateSpeedOfSound();
                    const phaseChange250 = (2 * Math.PI * RayTracer.FREQUENCY_BANDS[2] * distanceTraveled) / this.calculateSpeedOfSound();
                    const phaseChange500 = (2 * Math.PI * RayTracer.FREQUENCY_BANDS[3] * distanceTraveled) / this.calculateSpeedOfSound();
                    const phaseChange1k = (2 * Math.PI * RayTracer.FREQUENCY_BANDS[4] * distanceTraveled) / this.calculateSpeedOfSound();
                    const phaseChange2k = (2 * Math.PI * RayTracer.FREQUENCY_BANDS[5] * distanceTraveled) / this.calculateSpeedOfSound();
                    const phaseChange4k = (2 * Math.PI * RayTracer.FREQUENCY_BANDS[6] * distanceTraveled) / this.calculateSpeedOfSound();
                    const phaseChange8k = (2 * Math.PI * RayTracer.FREQUENCY_BANDS[7] * distanceTraveled) / this.calculateSpeedOfSound();
                    
                    const newPhase63 = (ray.getPhase63() + phaseChange63) % (2 * Math.PI);
                    const newPhase125 = (ray.getPhase125() + phaseChange125) % (2 * Math.PI);
                    const newPhase250 = (ray.getPhase250() + phaseChange250) % (2 * Math.PI);
                    const newPhase500 = (ray.getPhase500() + phaseChange500) % (2 * Math.PI);
                    const newPhase1k = (ray.getPhase1k() + phaseChange1k) % (2 * Math.PI);
                    const newPhase2k = (ray.getPhase2k() + phaseChange2k) % (2 * Math.PI);
                    const newPhase4k = (ray.getPhase4k() + phaseChange4k) % (2 * Math.PI);
                    const newPhase8k = (ray.getPhase8k() + phaseChange8k) % (2 * Math.PI);
                    
                    // Calculate HRTF index for hit point if camera is provided
                    let hitHrtfIndex: number | undefined;
                    if (camera) {
                        const listenerPos = camera.getPosition();
                        const toListener = vec3.subtract(vec3.create(), listenerPos, hitPoint);
                        vec3.normalize(toListener, toListener);
                        // Convert to spherical coordinates
                        const azimuth = Math.atan2(toListener[2], toListener[0]); // -π to π
                        const elevation = Math.asin(toListener[1]); // -π/2 to π/2
                        // Map to HRTF indices (assuming 360° azimuth, 180° elevation)
                        const azimuthIndex = Math.floor(((azimuth + Math.PI) / (2 * Math.PI)) * 360) % 360;
                        const elevationIndex = Math.floor(((elevation + Math.PI/2) / Math.PI) * 180) % 180;
                        hitHrtfIndex = elevationIndex * 360 + azimuthIndex;
                    }
                    
                    // Explicitly construct hit data instead of spreading undefined 'hit'
                    this.rayPathPoints.push({
                        position: vec3.clone(hitPoint),
                        direction: direction, // Add direction property
                        energy: ray.getEnergy(),
                        energy63: ray.getEnergy63(),
                        energy125: ray.getEnergy125(),
                        energy250: ray.getEnergy250(),
                        energy500: ray.getEnergy500(),
                        energy1k: ray.getEnergy1k(),
                        energy2k: ray.getEnergy2k(),
                        energy4k: ray.getEnergy4k(),
                        energy8k: ray.getEnergy8k(),
                        time: currentTime,
                        // Store phase for all bands
                        phase63: newPhase63,
                        phase125: newPhase125,
                        phase250: newPhase250,
                        phase500: newPhase500,
                        phase1k: newPhase1k,
                        phase2k: newPhase2k,
                        phase4k: newPhase4k,
                        phase8k: newPhase8k,
                        dopplerShift: 1.0, // Default doppler shift (no shift)
                        bounceNumber: bounces + 1, // Increment bounce count
                        rayIndex: rayIndex,
                        incomingDirection: direction, // Define incoming direction
                        hrtfIndex: hitHrtfIndex
                    });
                    
                    // Update ray properties with new time and phase
                    ray.updateTime(currentTime);
                    ray.updatePhase(
                        newPhase63, newPhase125, newPhase250, newPhase500,
                        newPhase1k, newPhase2k, newPhase4k, newPhase8k
                    );
                    
                    // Calculate reflection
                    const reflected = vec3.create();
                    const dot = vec3.dot(direction, closestPlane.normal);
                    vec3.scale(reflected, closestPlane.normal, -2 * dot);
                    vec3.add(reflected, direction, reflected);
                    
                    // Apply scattering based on frequency-dependent coefficients
                    const material = closestPlane.material;                    
                    const scatteringCoeffs = [
                        material.scattering63 ?? 0, material.scattering125 ?? 0, material.scattering250 ?? 0,
                        material.scattering500 ?? 0, material.scattering1k ?? 0, material.scattering2k ?? 0,
                        material.scattering4k ?? 0, material.scattering8k ?? 0
                    ];

                    // To better model how different frequencies scatter, we calculate a reflected
                    // direction for each band and then average them to find the new ray path.
                    // This is a simplification to avoid splitting the ray into multiple paths.
                    const averageReflectedDirection = vec3.create();
                    let totalScattering = 0;

                    for (let i = 0; i < scatteringCoeffs.length; i++) {
                        const scattering = scatteringCoeffs[i];
                        totalScattering += scattering;

                        // Start with the pure specular reflection for this band's calculation
                        const bandReflected = vec3.clone(reflected);

                        if (scattering > 0) {
                            // Generate a unique random diffuse direction for each band
                            const randomDir = this.generateRandomDirection(closestPlane.normal, material.roughness);
                            // Blend specular and diffuse based on the band's scattering coefficient
                            vec3.lerp(bandReflected, bandReflected, randomDir, scattering);
                        }
                        // Add the (potentially scattered) direction for this band to the accumulator
                        vec3.add(averageReflectedDirection, averageReflectedDirection, bandReflected);
                    }

                    if (totalScattering > 0) {
                        vec3.normalize(averageReflectedDirection, averageReflectedDirection);
                        vec3.copy(reflected, averageReflectedDirection);
                    }
                    
                    // Move new origin slightly away from the surface
                    const newOrigin = vec3.scaleAndAdd(vec3.create(), hitPoint, closestPlane.normal, 0.0001);
                    
                    // Get material properties for frequency-dependent absorption
                    // The energyLoss object is used in updateRay to apply material absorption
                    // along with air absorption to the ray's energy values
                    const energyLoss = {
                        band63: material.absorption63 ?? 0,
                        band125: material.absorption125 ?? 0,
                        band250: material.absorption250 ?? 0,
                        band500: material.absorption500 ?? 0,
                        band1k: material.absorption1k ?? 0,
                        band2k: material.absorption2k ?? 0,
                        band4k: material.absorption4k ?? 0,
                        band8k: material.absorption8k ?? 0
                    };
                    
                    // Calculate HRTF index if camera is provided
                    let hrtfIndex: number | undefined;
                    if (camera) {
                        const listenerPos = camera.getPosition();
                        const toListener = vec3.subtract(vec3.create(), listenerPos, hitPoint);
                        vec3.normalize(toListener, toListener);
                        // Convert to spherical coordinates
                        const azimuth = Math.atan2(toListener[2], toListener[0]); // -π to π
                        const elevation = Math.asin(toListener[1]); // -π/2 to π/2
                        // Map to HRTF indices (assuming 360° azimuth, 180° elevation)
                        const azimuthIndex = Math.floor(((azimuth + Math.PI) / (2 * Math.PI)) * 360) % 360;
                        const elevationIndex = Math.floor(((elevation + Math.PI/2) / Math.PI) * 180) % 180;
                        hrtfIndex = elevationIndex * 360 + azimuthIndex;
                    }
                    
                    // Record hit with frequency-dependent energies
                    this.hits.push({
                        position: hitPoint,
                        energy: ray.getEnergy(),
                        time: currentTime,
                        energy63: ray.getEnergy63(),
                        energy125: ray.getEnergy125(),
                        energy250: ray.getEnergy250(),
                        energy500: ray.getEnergy500(),
                        energy1k: ray.getEnergy1k(),
                        energy2k: ray.getEnergy2k(),
                        energy4k: ray.getEnergy4k(),
                        energy8k: ray.getEnergy8k(),
                        // Store phase for all bands
                        phase63: newPhase63,
                        phase125: newPhase125,
                        phase250: newPhase250,
                        phase500: newPhase500,
                        phase1k: newPhase1k,
                        phase2k: newPhase2k,
                        phase4k: newPhase4k,
                        phase8k: newPhase8k,
                        dopplerShift: 1.0, // Default doppler shift (no shift)
                        hrtfIndex: hrtfIndex,
                        incomingDirection: vec3.clone(direction)
                    });
                    
                    // Update ray with environmental parameters
                    ray.updateRay(
                        newOrigin,
                        reflected,
                        energyLoss,
                        closestT,
                        this.room.getTemperature(),
                        this.room.getHumidity()
                    );
                    
                    // Store ray segment if energy is above visibility threshold
                    if (ray.getEnergy() > this.VISIBILITY_THRESHOLD) {
                        this.rayPaths.push({
                            origin: newOrigin,
                            direction: vec3.clone(reflected),
                            energy: ray.getEnergy()
                        });
                    }
                    
                    bounces++;
                } else {
                    // No intersection or no material, deactivate ray
                    ray.deactivate();
                }
            }
        }
        
        console.log(`Completed late reflections`);
    }

    private calculateSpeedOfSound(temperature: number = this.AIR_TEMPERATURE): number {
        // Speed of sound formula: c = 331.3 + 0.606 * T
        // where T is temperature in Celsius
        return 331.3 + 0.606 * temperature;
    }

    private calculateTravelTime(distance: number): number {
        // Time = Distance / Speed
        return distance / this.calculateSpeedOfSound();
    }

    private generateRandomDirection(normal: vec3, roughness: number): vec3 {
        // Create orthonormal basis with normal
        const basis = this.createOrthonormalBasis(normal);
        // Generate random direction with roughness-based perturbation
        const phi = 2 * Math.PI * Math.random();
        const r = roughness * Math.sqrt(-Math.log(Math.random()));
        const x = r * Math.cos(phi);
        const y = r * Math.sin(phi);
        const z = Math.sqrt(Math.max(0.0, 1 - x*x - y*y));
        // Transform direction from local space to world space
        const result = vec3.create();
        vec3.scale(result, basis[0], x);
        vec3.scaleAndAdd(result, result, basis[1], y);
        vec3.scaleAndAdd(result, result, basis[2], z);
        return vec3.normalize(vec3.create(), result);
    }

    private createOrthonormalBasis(normal: vec3): vec3[] {
        const tangent = vec3.create();
        const bitangent = vec3.create();
        // Find least dominant axis of normal
        const absX = Math.abs(normal[0]);
        const absY = Math.abs(normal[1]);
        const absZ = Math.abs(normal[2]);
        if (absX <= absY && absX <= absZ) {
            vec3.set(tangent, 0, -normal[2], normal[1]);
        } else if (absY <= absX && absY <= absZ) {
            vec3.set(tangent, -normal[2], 0, normal[0]);
        } else {
            vec3.set(tangent, -normal[1], normal[0], 0);
        }
        vec3.normalize(tangent, tangent);
        vec3.cross(bitangent, normal, tangent);
        vec3.normalize(bitangent, bitangent);
        return [tangent, bitangent, normal];
    }

    public getRayHits(): RayHit[] {
        return this.hits;
    }

    async initializeHRTF(): Promise<void> {
        const hrtfData = await loadHRTFData();
        this.hrtfBuffer = this.device.createBuffer({
            size: hrtfData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'HRTF Coefficients'
        });
        this.device.queue.writeBuffer(this.hrtfBuffer, 0, hrtfData as BufferSource);
    }

    private applyWindowFunction(t: number, windowSize: number): number {
        // Hann window function: 0.5 * (1 - cos(2π * t/N))
        return 0.5 * (1 - Math.cos(2 * Math.PI * t / windowSize));
    }

    private calculateDopplerFrequency(point: RayPathPoint, nextPoint: RayPathPoint | null): number {
        if (!nextPoint) return 1000; // Return a default frequency if no next point
        const speedOfSound = this.calculateSpeedOfSound();
        // Calculate ray velocity between points
        const distance = vec3.distance(point.position, nextPoint.position);
        const timeDiff = nextPoint.time - point.time;
        const rayVelocity = distance / timeDiff;
        // Calculate direction vector between points
        const direction = vec3.create();
        vec3.subtract(direction, nextPoint.position, point.position);
        vec3.normalize(direction, direction);
        // Project ray velocity onto direction to receiver
        // For simplicity, assume receiver is at (0,0,0)
        const toReceiver = vec3.create();
        vec3.negate(toReceiver, point.position);
        vec3.normalize(toReceiver, toReceiver);
        const relativeVelocity = rayVelocity * vec3.dot(direction, toReceiver);
        // Doppler equation: f' = f * (c / (c ± v))
        // where c is speed of sound, v is relative velocity
        // Use average frequency for Doppler calculation
        const avgFrequency = (point.energy63 * 63 + point.energy125 * 125 + point.energy250 * 250 +
                            point.energy500 * 500 + point.energy1k * 1000 + point.energy2k * 2000 +
                            point.energy4k * 4000 + point.energy8k * 8000) /
                           (point.energy63 + point.energy125 + point.energy250 + point.energy500 +
                            point.energy1k + point.energy2k + point.energy4k + point.energy8k || 1);
        return avgFrequency * (speedOfSound / (speedOfSound - relativeVelocity));
    }

    public generateImpulseResponse(sampleRate: number = 44100): ImpulseResponse {
        if (this.rayPathPoints.length === 0) {
            throw new Error("No ray path points available. Run calculateRayPaths first.");
        }
        
        // Sort ray path points by time
        const sortedPoints = [...this.rayPathPoints].sort((a, b) => a.time - b.time);
        
        // Find the maximum time to determine the length of the impulse response
        const maxTime = sortedPoints[sortedPoints.length - 1].time;
        const numSamples = Math.ceil(maxTime * sampleRate);
        
        // Create arrays for the impulse response
        const timeArray = new Float32Array(numSamples);
        const amplitudeArray = new Float32Array(numSamples);
        const frequencyArray = new Float32Array(numSamples);
        
        // Fill time array
        for (let i = 0; i < numSamples; i++) {
            timeArray[i] = i / sampleRate;
        }
        
        // Process each path point and add its contribution to the impulse response
        for (let i = 0; i < sortedPoints.length; i++) {
            const point = sortedPoints[i];
            const nextPoint = i < sortedPoints.length - 1 ? sortedPoints[i + 1] : null;
            const startSample = Math.floor(point.time * sampleRate);
            // Convert energy to amplitude (using average energy)
            const averageEnergy = (point.energy63 + point.energy125 + point.energy250 + point.energy500 +
                                 point.energy1k + point.energy2k + point.energy4k + point.energy8k) / 8;
            const amplitude = Math.sqrt(Math.max(averageEnergy, 0)); // Convert energy to amplitude
            
            // Calculate Doppler-shifted frequency
            const dopplerFrequency = this.calculateDopplerFrequency(point, nextPoint);
            
            // Add the contribution of this point to nearby samples
            const windowSize = Math.ceil(sampleRate / Math.max(dopplerFrequency, 1));
            for (let j = 0; j < windowSize && (startSample + j) < numSamples; j++) {
                const t = timeArray[startSample + j];
                // Apply window function to avoid discontinuities
                const windowValue = this.applyWindowFunction(j, windowSize);
                // Calculate phase with Doppler-shifted frequency
                // Use average phase for simplicity
                const avgPhase = (point.phase63 + point.phase125 + point.phase250 + point.phase500 +
                                point.phase1k + point.phase2k + point.phase4k + point.phase8k) / 8;
                const phase = avgPhase + 2 * Math.PI * dopplerFrequency * (t - point.time);
                // Apply frequency-dependent absorption based on distance and material
                const distanceAttenuation = Math.exp(-0.1 * dopplerFrequency * point.time); // Simple frequency-dependent air absorption
                // Combine all factors for final amplitude
                const contribution = amplitude * windowValue * distanceAttenuation * Math.sin(phase);
                // Add to impulse response
                amplitudeArray[startSample + j] += contribution;
                // Store frequency information (weighted average based on energy)
                const currentFreq = frequencyArray[startSample + j];
                const currentEnergy = Math.pow(amplitudeArray[startSample + j], 2);
                const newEnergy = Math.pow(contribution, 2);
                frequencyArray[startSample + j] = 
                    (currentFreq * currentEnergy + dopplerFrequency * newEnergy) / 
                    (currentEnergy + newEnergy || 1);
            }
        }
        
        // Normalize the impulse response
        const maxAmplitude = Math.max(...Array.from(amplitudeArray).map(Math.abs));
        if (maxAmplitude > 0) {
            for (let i = 0; i < numSamples; i++) {
                amplitudeArray[i] /= maxAmplitude;
            }
        }
        
        return {
            time: timeArray,
            amplitude: amplitudeArray,
            sampleRate: sampleRate,
            frequencies: frequencyArray
        };
    }

    public render(pass: GPURenderPassEncoder, viewProjection: Float32Array): void {
        this.rayRenderer.render(
            pass,
            viewProjection,
            this.rayPaths,
            this.room.config.dimensions
        );
    }

    // Add bind group entry setup for HRTF buffer
    private setupComputeBindGroup(): GPUBindGroup {
        return this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
               { binding: 0, resource: { buffer: this.listenerBuffer } },
               { binding: 1, resource: { buffer: this.rayHitsBuffer } },
               { binding: 2, resource: { buffer: this.spatialIRBuffer } },
               { binding: 3, resource: { buffer: this.paramsBuffer } },
               { binding: 4, resource: { buffer: this.acousticsBuffer } },
               { binding: 5, resource: { buffer: this.wavePropertiesBuffer } },
               { binding: 6, resource: { buffer: this.roomMaterialsBuffer } },
               { binding: 7, resource: { buffer: this.hrtfBuffer } },
            ],
        });
    }
}