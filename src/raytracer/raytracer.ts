import { vec3 } from 'gl-matrix';
import { Ray } from './ray';
import { Room } from '../room/room';
import { Sphere } from '../objects/sphere';
import { RayRenderer } from './ray-renderer';
import { WallMaterial } from '../room/room-materials';

export interface RayTracerConfig {
    numRays: number;
    maxBounces: number;
    minEnergy: number;
}

export interface RayHit {
    position: vec3;
    energy: number;
    energyLow: number;
    energyMid: number;
    energyHigh: number;
    time: number;      // Arrival time
    phase: number;     // Phase at hit
    frequency: number; // Frequency of the ray
}

export interface RayPathPoint extends RayHit {
    bounceNumber: number;  // Which bounce this point represents
    rayIndex: number;     // Which ray this point belongs to
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

    constructor(
        device: GPUDevice,
        soundSource: Sphere,
        room: Room,
        config: RayTracerConfig = {
            numRays: 1000,
            maxBounces: 50,
            minEnergy: 0.01
        }
    ) {
        this.device = device;
        this.soundSource = soundSource;
        this.room = room;
        this.config = config;
        this.rayRenderer = new RayRenderer(device);
    }

    private generateRays(): void {
        this.rays = [];
        const sourcePos = this.soundSource.getPosition();
        const sphereRadius = this.soundSource.getRadius();

        // Define frequency bands for analysis
        const frequencies = [250, 500, 1000, 2000, 4000]; // Hz

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
            const frequency = frequencies[i % frequencies.length];
            
            // Create ray starting from the sphere's surface
            this.rays.push(new Ray(rayOrigin, direction, 1.0, frequency));
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

    public async calculateRayPaths(): Promise<void> {
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
        await this.calculateEarlyReflections();

        // Calculate late reflections using stochastic ray tracing
        await this.calculateLateReflections();

        console.log(`Completed ray tracing with early and late reflections`);
    }

    private async calculateEarlyReflections(): Promise<void> {
        const sourcePos = this.soundSource.getPosition();
        const maxOrder = 3; // Maximum reflection order for image sources

        // Generate image sources up to maxOrder
        const imageSources = this.generateImageSources(sourcePos, maxOrder);

        // Process each image source
        for (const imageSource of imageSources) {
            const path = this.validateImageSourcePath(imageSource);
            if (path) {
                this.processImageSourcePath(path);
            }
        }
    }

    private generateImageSources(sourcePos: vec3, maxOrder: number): ImageSource[] {
        const imageSources: ImageSource[] = [];
        // Implementation of image source generation
        // This is a recursive process that mirrors the source across room surfaces
        return imageSources;
    }

    private validateImageSourcePath(imageSource: ImageSource): RayPath | null {
        // Validate visibility and calculate reflection path
        // Returns null if path is invalid (blocked)
        return null; // Placeholder
    }

    private processImageSourcePath(path: RayPath): void {
        // Calculate energy and add to hits
        // Early reflections are more precise than stochastic rays
    }

    private async calculateLateReflections(): Promise<void> {
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
            
            // Store initial point
            this.rayPathPoints.push({
                position: ray.getOrigin(),
                energy: ray.getEnergy(),
                energyLow: ray.getEnergyLow(),
                energyMid: ray.getEnergyMid(),
                energyHigh: ray.getEnergyHigh(),
                time: currentTime,
                phase: ray.getPhase(),
                frequency: ray.getFrequency(),
                bounceNumber: bounces,
                rayIndex: rayIndex
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

                    // Calculate phase change based on distance and frequency
                    const wavelength = this.calculateSpeedOfSound() / ray.getFrequency();
                    const phaseChange = (2 * Math.PI * distanceTraveled) / wavelength;
                    const newPhase = (ray.getPhase() + phaseChange) % (2 * Math.PI);

                    // Store the hit point before reflection
                    this.rayPathPoints.push({
                        position: vec3.clone(hitPoint),
                        energy: ray.getEnergy(),
                        energyLow: ray.getEnergyLow(),
                        energyMid: ray.getEnergyMid(),
                        energyHigh: ray.getEnergyHigh(),
                        time: currentTime,
                        phase: newPhase,
                        frequency: ray.getFrequency(),
                        bounceNumber: bounces,
                        rayIndex: rayIndex
                    });

                    // Update ray properties with new time and phase
                    ray.updateTime(currentTime);
                    ray.updatePhase(newPhase);

                    // Calculate reflection
                    const reflected = vec3.create();
                    const dot = vec3.dot(direction, closestPlane.normal);
                    vec3.scale(reflected, closestPlane.normal, -2 * dot);
                    vec3.add(reflected, direction, reflected);

                    // Apply scattering based on frequency-dependent coefficients
                    const material = closestPlane.material;
                    const avgScattering = (material.scatteringLow + material.scatteringMid + material.scatteringHigh) / 3;

                    if (avgScattering > 0) {
                        // Generate random vector for scattering
                        const randomDir = this.generateRandomDirection(closestPlane.normal, material.roughness);

                        // Blend reflected direction with random direction based on scattering coefficient
                        vec3.lerp(reflected, reflected, randomDir, avgScattering);
                        vec3.normalize(reflected, reflected);
                    }

                    // Move new origin slightly away from the surface
                    const newOrigin = vec3.scaleAndAdd(vec3.create(), hitPoint, closestPlane.normal, 0.0001);

                    // Get material properties for frequency-dependent absorption
                    const energyLoss = {
                        low: material.absorptionLow,
                        mid: material.absorptionMid,
                        high: material.absorptionHigh
                    };

                    // Record hit with frequency-dependent energies
                    this.hits.push({
                        position: hitPoint,
                        energy: ray.getEnergy(),
                        time: currentTime,
                        energyLow: ray.getEnergyLow(),
                        energyMid: ray.getEnergyMid(),
                        energyHigh: ray.getEnergyHigh(),
                        phase: newPhase,
                        frequency: ray.getFrequency()
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

    private applyWindowFunction(t: number, windowSize: number): number {
        // Hann window function: 0.5 * (1 - cos(2π * t/N))
        return 0.5 * (1 - Math.cos(2 * Math.PI * t / windowSize));
    }

    private calculateDopplerFrequency(point: RayPathPoint, nextPoint: RayPathPoint | null): number {
        if (!nextPoint) return point.frequency;

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
        return point.frequency * (speedOfSound / (speedOfSound - relativeVelocity));
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
            const amplitude = Math.sqrt(point.energy); // Convert energy to amplitude

            // Calculate Doppler-shifted frequency
            const dopplerFrequency = this.calculateDopplerFrequency(point, nextPoint);

            // Add the contribution of this point to nearby samples
            const windowSize = Math.ceil(sampleRate / dopplerFrequency);
            
            for (let j = 0; j < windowSize && (startSample + j) < numSamples; j++) {
                const t = timeArray[startSample + j];
                
                // Apply window function to avoid discontinuities
                const windowValue = this.applyWindowFunction(j, windowSize);
                
                // Calculate phase with Doppler-shifted frequency
                const phase = point.phase + 2 * Math.PI * dopplerFrequency * (t - point.time);
                
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
}