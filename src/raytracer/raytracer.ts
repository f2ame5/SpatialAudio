import { vec3, mat4 } from 'gl-matrix';
import { Ray, FrequencyBands } from './ray';
import { Room } from '../room/room';
import { Sphere } from '../objects/sphere';
import { RayRenderer } from './ray-renderer';
import { Camera } from '../camera/camera';
import { WallMaterial } from '../room/room-materials';

interface Edge {
    start: vec3;
    end: vec3;
    adjacentSurfaces: number[];
}

interface ImageSource {
    position: vec3;
    order: number;
    reflectionPath: vec3[];
    surfaces: number[];
}

export interface RayTracerConfig {
    numRays: number;
    maxBounces: number;
    minEnergy: number;
    enableDiffraction: boolean;
    diffractionAttenuationFactor: number;
}

export interface RayHit {
    position: vec3;
    energies: FrequencyBands;
    time: number;
    phase: number;
    frequency: number;
    dopplerShift: number;
    bounces: number;
    distance: number;
    direction: vec3;
    type: 'reflection' | 'diffraction' | 'direct';
}

export interface RayPathSegment {
    origin: vec3;
    direction: vec3;
    energies: FrequencyBands;
    type: 'reflection' | 'diffraction' | 'initial';
}


export interface RayPathPoint {
    position: vec3;
    energies: FrequencyBands;
    time: number;
    phase: number;
    frequency: number;
    dopplerShift: number;
    bounces: number;
    distance: number;
    direction: vec3;
    rayIndex: number;
}

export interface ImpulseResponse {
    time: Float32Array;
    amplitude: Float32Array;
    sampleRate: number;
    frequencies: Float32Array;
}

const DIFFRACTION_PROXIMITY_THRESHOLD_SQ = 0.05 * 0.05;

export class RayTracer {
    private soundSource: Sphere;
    private room: Room;
    private camera: Camera;
    private config: RayTracerConfig;
    private rays: Ray[] = [];
    private leftEarHits: RayHit[] = [];
    private rightEarHits: RayHit[] = [];
    private earLeftPos!: vec3;
    private earRightPos!: vec3;
    private listenerEarRadiusSq!: number;
    private rayPaths: RayPathSegment[] = [];
    private rayPathPoints: RayPathPoint[] = [];
    private rayRenderer: RayRenderer;
    private readonly SPEED_OF_SOUND = 343.0;
    private readonly HEAD_RADIUS = 0.0875;
    private readonly AIR_TEMPERATURE = 20.0;
    private edges: Edge[] = [];
    private imageSources: ImageSource[] = [];

    constructor(
        device: GPUDevice,
        soundSource: Sphere,
        room: Room,
        camera: Camera,
        config: RayTracerConfig = {
            numRays: 1000,
            maxBounces: 50,
            minEnergy: 0.05,
            enableDiffraction: true,
            diffractionAttenuationFactor: 0.5
        }
    ) {
        this.soundSource = soundSource;
        this.room = room;
        this.camera = camera;
        this.config = config;
        this.rayRenderer = new RayRenderer(device);
    }
    
    // NEW: Add getConfig method
    public getConfig(): RayTracerConfig {
        return this.config;
    }

    // NEW: Add setConfig method
    public setConfig(config: RayTracerConfig): void {
        this.config = config;
        console.log('[RayTracer] Config updated:', this.config);
    }

    private generateRays(): void {
        this.rays = [];
        const sourcePos = this.soundSource.getPosition();
        const frequencies = [125, 250, 500, 1000, 2000, 4000, 8000, 16000];

        for (let i = 0; i < this.config.numRays; i++) {
            const theta = 2 * Math.PI * Math.random();
            const phi = Math.acos(2 * Math.random() - 1);
            const direction = vec3.fromValues(
                Math.sin(phi) * Math.cos(theta),
                Math.sin(phi) * Math.sin(theta),
                Math.cos(phi)
            );
            const rayOrigin = vec3.scaleAndAdd(vec3.create(), sourcePos, direction, this.soundSource.getRadius());
            const frequency = frequencies[i % frequencies.length];
            this.rays.push(new Ray(rayOrigin, direction, 1.0, frequency));
        }
    }

    public async calculateRayPaths(): Promise<void> {
        this.rays = [];
        this.leftEarHits = [];
        this.rightEarHits = [];
        this.rayPaths = [];
        this.rayPathPoints = [];

        const listenerRight = this.camera.getRight();
        const headRadius = 0.0875; // 8.75 cm

        this.earLeftPos = vec3.scaleAndAdd(vec3.create(), this.camera.getPosition(), listenerRight, -headRadius);
        this.earRightPos = vec3.scaleAndAdd(vec3.create(), this.camera.getPosition(), listenerRight, headRadius);

        const listenerPos = this.camera.getPosition();
        const sourcePos = this.soundSource.getPosition();

        // Direct path calculation (important for initial sound)
        if (!this.checkForObstruction(sourcePos, listenerPos, this.getReflectionPlanes())) {
            const directDist = vec3.distance(sourcePos, listenerPos);
            const directTimeToListener = directDist / this.SPEED_OF_SOUND;
            const directEnergies: FrequencyBands = {
                energy125Hz: 1.0, energy250Hz: 1.0, energy500Hz: 1.0, energy1kHz: 1.0,
                energy2kHz: 1.0, energy4kHz: 1.0, energy8kHz: 1.0, energy16kHz: 1.0
            };
            const directAttenuation = 1.0 / Math.max(0.01, directDist * directDist);
            for (const key in directEnergies) {
                (directEnergies as any)[key] *= directAttenuation;
            }

            this.leftEarHits.push(this.createListenerRelativeHit(
                sourcePos, directEnergies, 0,
                0, 1000, 1.0, 0, 'direct', this.earLeftPos
            ));

            this.rightEarHits.push(this.createListenerRelativeHit(
                sourcePos, directEnergies, 0,
                0, 1000, 1.0, 0, 'direct', this.earRightPos
            ));
        }

        this.generateRays();
        await this.calculateLateReflections();
    }

    private getReflectionPlanes(): any[] {
        const { width, height, depth } = this.room.config.dimensions;
        const hW = width / 2, hD = depth / 2;
        const materials = this.room.config.materials;
        return [
            { normal: vec3.fromValues(1,0,0), d: -hW, material: materials.walls }, { normal: vec3.fromValues(-1,0,0), d: -hW, material: materials.walls },
            { normal: vec3.fromValues(0,1,0), d: 0, material: materials.floor }, { normal: vec3.fromValues(0,-1,0), d: -height, material: materials.ceiling },
            { normal: vec3.fromValues(0,0,1), d: -hD, material: materials.walls }, { normal: vec3.fromValues(0,0,-1), d: -hD, material: materials.walls }
        ];
    }
    
    // NEW: Helper method to check for obstructions
    private checkForObstruction(start: vec3, end: vec3, planes: any[]): boolean {
        const direction = vec3.subtract(vec3.create(), end, start);
        const distanceToEnd = vec3.length(direction);
        vec3.normalize(direction, direction);
    
        for (const plane of planes) {
            const denom = vec3.dot(direction, plane.normal);
            // Check for intersection with the plane
            if (Math.abs(denom) > 0.0001) {
                const t = -(vec3.dot(start, plane.normal) + plane.d) / denom;
                // If there's an intersection between the start and end point
                if (t > 0.0001 && t < distanceToEnd - 0.0001) {
                    return true; // The path is obstructed
                }
            }
        }
        return false; // Path is clear
    }

    // REWRITTEN: More efficient late reflection calculation
    private async calculateLateReflections(): Promise<void> {
        const reflectionPlanes = this.getReflectionPlanes();
        const { width, height, depth } = this.room.config.dimensions;
        const hW = width / 2, hD = depth / 2;

        for (const ray of this.rays) {
            let currentRay = ray;
            let bounces = 0;

            while (currentRay.isRayActive() && bounces < this.config.maxBounces && this.calculateAverageEnergy(currentRay.getEnergies()) > this.config.minEnergy) {
                let closestT = Infinity;
                let hitDetails: { plane: any, hitPoint: vec3, distance: number } | null = null;
                const P0 = currentRay.getOrigin();
                const D = currentRay.getDirection();

                // Find the closest wall intersection
                for (const plane of reflectionPlanes) {
                    const denom = vec3.dot(D, plane.normal);
                    if (Math.abs(denom) > 0.0001) {
                        const t = -(vec3.dot(P0, plane.normal) + plane.d) / denom;
                        if (t > 0.0001 && t < closestT) {
                            const hitPoint = vec3.scaleAndAdd(vec3.create(), P0, D, t);
                            if (Math.abs(hitPoint[0]) <= hW + 0.01 && hitPoint[1] >= -0.01 && hitPoint[1] <= height + 0.01 && Math.abs(hitPoint[2]) <= hD + 0.01) {
                                closestT = t;
                                hitDetails = { plane, hitPoint, distance: t };
                            }
                        }
                    }
                }

                if (hitDetails) {
                    const { plane, hitPoint, distance } = hitDetails;
                    const timeAtBounce = currentRay.getTime() + distance / this.SPEED_OF_SOUND;

                    // ADD CONTRIBUTION: Check for line of sight and add hit
                    if (!this.checkForObstruction(hitPoint, this.earLeftPos, reflectionPlanes)) {
                        this.leftEarHits.push(this.createListenerRelativeHit(hitPoint, currentRay.getEnergies(), timeAtBounce, currentRay.getPhase(), currentRay.getFrequency(), 1.0, bounces + 1, 'reflection', this.earLeftPos));
                    }
                    if (!this.checkForObstruction(hitPoint, this.earRightPos, reflectionPlanes)) {
                        this.rightEarHits.push(this.createListenerRelativeHit(hitPoint, currentRay.getEnergies(), timeAtBounce, currentRay.getPhase(), currentRay.getFrequency(), 1.0, bounces + 1, 'reflection', this.earRightPos));
                    }

                    // Update the ray for its next bounce
                    const normal = plane.normal;
                    const specularDir = vec3.create();
                    vec3.scaleAndAdd(specularDir, D, normal, -2 * vec3.dot(D, normal));
                    vec3.normalize(specularDir, specularDir);

                    const randomDir = vec3.fromValues(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1);
                    if (vec3.dot(randomDir, normal) < 0) vec3.negate(randomDir, randomDir);
                    vec3.normalize(randomDir, randomDir);

                    const avgScattering = (plane.material.scattering1kHz + plane.material.scattering4kHz) / 2.0;
                    const newDirection = vec3.lerp(vec3.create(), specularDir, randomDir, avgScattering);
                    vec3.normalize(newDirection, newDirection);
                    
                    const offsetOrigin = vec3.scaleAndAdd(vec3.create(), hitPoint, newDirection, 0.0001);
                    currentRay.updateRay(offsetOrigin, newDirection, plane.material, distance, this.AIR_TEMPERATURE, 50);
                    bounces++;
                } else {
                    currentRay.deactivate(); // No intersection found, ray escapes
                }
            }
        }
    }
    
    private createListenerRelativeHit(
        interactionPointWorld: vec3, energiesAtInteraction: FrequencyBands, timeAtInteraction: number,
        phaseAtInteraction: number, frequencyAtInteraction: number, dopplerShiftAtInteraction: number,
        bounces: number, type: 'reflection' | 'diffraction' | 'direct', earPos: vec3
    ): RayHit {
        const energiesAtListener = { ...energiesAtInteraction }; // Initialize here

        const vecToListener = vec3.subtract(vec3.create(), earPos, interactionPointWorld);
        const distanceToListener = vec3.length(vecToListener);
        const directionFromInteractionToListener = vec3.normalize(vec3.create(), vecToListener);
        const travelTimeToListener = distanceToListener / this.SPEED_OF_SOUND;
        const totalTimeAtListener = timeAtInteraction + travelTimeToListener;

        const airAbsRay = new Ray(vec3.create(), vec3.create(), 1.0, frequencyAtInteraction);
        const airAbsorptionAmplitudeFactors = airAbsRay.calculateAirAbsorption(distanceToListener, this.AIR_TEMPERATURE, 50);

        // Apply general distance attenuation (moved here to apply before head shadow)
        const distanceAttenuationFactor = 1.0 / Math.max(0.01, distanceToListener * distanceToListener);
        for (const key of Object.keys(energiesAtListener) as Array<keyof FrequencyBands>) {
            energiesAtListener[key] *= distanceAttenuationFactor;
            
            const bandKeyLookup = `absorption${key.replace('energy', '')}` as keyof typeof airAbsorptionAmplitudeFactors;
            if (airAbsorptionAmplitudeFactors.hasOwnProperty(bandKeyLookup)) {
                energiesAtListener[key] *= Math.pow((airAbsorptionAmplitudeFactors as any)[bandKeyLookup], 2);
            }
        }

        // Calculate head shadow based on ear position for Interaural Level Difference (ILD)
        const directionToEar = vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), earPos, interactionPointWorld));
        const listenerRight = this.camera.getRight();
        const listenerFront = this.camera.getFront();

        const azimuthRad = Math.atan2(vec3.dot(directionToEar, listenerRight), vec3.dot(directionToEar, listenerFront));
        const frequencies = [125, 250, 500, 1000, 2000, 4000, 8000, 16000];
        const energyKeys = Object.keys(energiesAtListener) as Array<keyof FrequencyBands>;

        const pathDifference = this.HEAD_RADIUS * (Math.abs(azimuthRad) + Math.sin(Math.abs(azimuthRad)));
        
        for (let i = 0; i < frequencies.length; i++) {
            const freq = frequencies[i];
            const key = energyKeys[i];
            const wavelength = this.SPEED_OF_SOUND / freq;
            
            const shadowEffect = 1.0 - 0.7 * Math.min(1.0, Math.max(0, pathDifference / wavelength));
            
            const isLeftEarHit = vec3.equals(earPos, this.earLeftPos);
            const isRightEarHit = vec3.equals(earPos, this.earRightPos);

            if (azimuthRad > 0 && isLeftEarHit) { // Sound from right, hitting left ear
                energiesAtListener[key] *= shadowEffect;
            } else if (azimuthRad < 0 && isRightEarHit) { // Sound from left, hitting right ear
                energiesAtListener[key] *= shadowEffect;
            }
        }
        const phaseAtListener = (phaseAtInteraction + (2 * Math.PI * frequencyAtInteraction * travelTimeToListener)) % (2 * Math.PI);

        return {
            position: vec3.clone(interactionPointWorld), energies: energiesAtListener, time: totalTimeAtListener,
            phase: phaseAtListener, frequency: frequencyAtInteraction, dopplerShift: dopplerShiftAtInteraction,
            bounces: bounces, distance: distanceToListener, direction: directionFromInteractionToListener, type: type
        };
    }

    private calculateAverageEnergy(energies: FrequencyBands): number {
        const values = Object.values(energies);
        return values.reduce((sum, energy) => sum + energy, 0) / values.length;
    }

    public getRayHits(): [RayHit[], RayHit[]] {
        console.log(`[RayTracer getRayHits] Total hits Left: ${this.leftEarHits.length}, Right: ${this.rightEarHits.length}`);
        return [this.leftEarHits, this.rightEarHits];
    }

    public render(pass: GPURenderPassEncoder, viewProjection: Float32Array): void {
        const renderablePaths = this.rayPaths.map(p => ({
            origin: p.origin, direction: p.direction, energies: p.energies
        }));
        this.rayRenderer.render(pass, viewProjection, renderablePaths, this.room.config.dimensions);
    }

    public recalculateRays(): void {
        this.rayRenderer.resetRender();
        this.calculateRayPaths();
    }

    private detectEdges(): void {
        this.edges = [];
        const { width, height, depth } = this.room.config.dimensions;
        const hW = width / 2, hD = depth / 2;
        const c = [
            vec3.fromValues(-hW, 0, -hD), vec3.fromValues(hW, 0, -hD), vec3.fromValues(hW, 0, hD), vec3.fromValues(-hW, 0, hD),
            vec3.fromValues(-hW, height, -hD), vec3.fromValues(hW, height, -hD), vec3.fromValues(hW, height, hD), vec3.fromValues(-hW, height, hD)
        ];
        this.edges.push({ start: c[0], end: c[1], adjacentSurfaces: [] } as Edge);
        this.edges.push({ start: c[1], end: c[2], adjacentSurfaces: [] } as Edge);
        this.edges.push({ start: c[2], end: c[3], adjacentSurfaces: [] } as Edge);
        this.edges.push({ start: c[3], end: c[0], adjacentSurfaces: [] } as Edge);
        this.edges.push({ start: c[4], end: c[5], adjacentSurfaces: [] } as Edge);
        this.edges.push({ start: c[5], end: c[6], adjacentSurfaces: [] } as Edge);
        this.edges.push({ start: c[6], end: c[7], adjacentSurfaces: [] } as Edge);
        this.edges.push({ start: c[7], end: c[4], adjacentSurfaces: [] } as Edge);
        this.edges.push({ start: c[0], end: c[4], adjacentSurfaces: [] } as Edge);
        this.edges.push({ start: c[1], end: c[5], adjacentSurfaces: [] } as Edge);
        this.edges.push({ start: c[2], end: c[6], adjacentSurfaces: [] } as Edge);
        this.edges.push({ start: c[3], end: c[7], adjacentSurfaces: [] } as Edge);
    }
}
