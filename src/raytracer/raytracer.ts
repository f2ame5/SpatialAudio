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
    private hits: RayHit[] = [];
    private rayPaths: RayPathSegment[] = [];
    private rayPathPoints: RayPathPoint[] = [];
    private rayRenderer: RayRenderer;
    private readonly SPEED_OF_SOUND = 343.0;
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
        this.hits = [];
        this.rays = [];
        this.rayPaths = [];
        this.rayPathPoints = [];

        const listenerPos = this.camera.getPosition();
        const sourcePos = this.soundSource.getPosition();

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

        this.hits.push({
            position: vec3.clone(sourcePos),
            energies: directEnergies,
            time: directTimeToListener,
            phase: (2 * Math.PI * 1000 * directTimeToListener) % (2 * Math.PI),
            frequency: 1000,
            dopplerShift: 1.0,
            bounces: 0,
            distance: directDist,
            direction: vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), listenerPos, sourcePos)),
            type: 'direct'
        });
        this.rayPaths.push({
            origin: vec3.clone(sourcePos),
            direction: vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), listenerPos, sourcePos)),
            energies: directEnergies,
            type: 'initial'
        });

        this.generateImageSources(2);
        await this.calculateEarlyReflections();
        this.detectEdges();
        this.generateRays();

        for (const ray of this.rays) {
            this.rayPaths.push({
                origin: ray.getOrigin(),
                direction: ray.getDirection(),
                energies: ray.getEnergies(),
                type: 'initial'
            });
        }
        await this.calculateLateReflections();
    }

    private generateImageSources(maxOrder: number = 2): void {
        this.imageSources = [];
        const sourcePos = this.soundSource.getPosition();
        this.imageSources.push({
            position: vec3.clone(sourcePos), order: 0, reflectionPath: [vec3.clone(sourcePos)], surfaces: []
        });

        const { width, height, depth } = this.room.config.dimensions;
        const hW = width / 2, hD = depth / 2;
        const planes = [
            { normal: vec3.fromValues(1, 0, 0), d: -hW, index: 0 }, { normal: vec3.fromValues(-1, 0, 0), d: -hW, index: 1 },
            { normal: vec3.fromValues(0, 1, 0), d: 0, index: 2 }, { normal: vec3.fromValues(0, -1, 0), d: -height, index: 3 },
            { normal: vec3.fromValues(0, 0, 1), d: -hD, index: 4 }, { normal: vec3.fromValues(0, 0, -1), d: -hD, index: 5 }
        ];

        let currentSources = [...this.imageSources];
        for (let order = 1; order <= maxOrder; order++) {
            const newSources: ImageSource[] = [];
            for (const source of currentSources) {
                if (source.order === order - 1) {
                    for (let i = 0; i < planes.length; i++) {
                        const plane = planes[i];
                        if (source.surfaces.length > 0 && source.surfaces[source.surfaces.length - 1] === plane.index) continue;
                        const reflectedPos = vec3.create();
                        const distToPlane = vec3.dot(source.position, plane.normal) + plane.d;
                        vec3.scaleAndAdd(reflectedPos, source.position, plane.normal, -2 * distToPlane);
                        const midPoint = vec3.scaleAndAdd(vec3.create(), source.position, plane.normal, -distToPlane);
                        newSources.push({
                            position: reflectedPos, order: order,
                            reflectionPath: [...source.reflectionPath, midPoint], surfaces: [...source.surfaces, plane.index]
                        });
                    }
                }
            }
            this.imageSources.push(...newSources);
            currentSources = [...this.imageSources];
        }
    }

    private async calculateEarlyReflections(): Promise<void> {
        const listenerPos = this.camera.getPosition();
        const materials = [
            this.room.config.materials.walls, this.room.config.materials.walls,
            this.room.config.materials.floor, this.room.config.materials.ceiling,
            this.room.config.materials.walls, this.room.config.materials.walls
        ];

        for (const source of this.imageSources) {
            if (source.order === 0) continue;
            const pathFromImageSourceToListener = vec3.subtract(vec3.create(), listenerPos, source.position);
            const distance = vec3.length(pathFromImageSourceToListener);
            const directionToListener = vec3.normalize(vec3.create(), pathFromImageSourceToListener);
            const timeOfArrival = distance / this.SPEED_OF_SOUND;

            let energies: FrequencyBands = {
                energy125Hz: 1.0, energy250Hz: 1.0, energy500Hz: 1.0, energy1kHz: 1.0,
                energy2kHz: 1.0, energy4kHz: 1.0, energy8kHz: 1.0, energy16kHz: 1.0
            };
            const initialAttenuation = 1.0 / Math.max(0.01, distance * distance);
            for (const key in energies) (energies as any)[key] *= initialAttenuation;

            for (let i = 0; i < source.surfaces.length; i++) {
                const material = materials[source.surfaces[i]];
                if (material) {
                    energies.energy125Hz *= (1.0 - material.absorption125Hz);
                    energies.energy250Hz *= (1.0 - material.absorption250Hz);
                    energies.energy500Hz *= (1.0 - material.absorption500Hz);
                    energies.energy1kHz *= (1.0 - material.absorption1kHz);
                    energies.energy2kHz *= (1.0 - material.absorption2kHz);
                    energies.energy4kHz *= (1.0 - material.absorption4kHz);
                    energies.energy8kHz *= (1.0 - material.absorption8kHz);
                    energies.energy16kHz *= (1.0 - material.absorption16kHz);
                }
            }
            const reflectionBoost = 2.0 / (source.order + 1);
            for (const key in energies) (energies as any)[key] *= reflectionBoost;

            let visualPathStart = this.soundSource.getPosition();
            for(let i=0; i < source.reflectionPath.length; ++i) {
                if (i === 0 && source.order > 0) continue;
                const visualPathEnd = source.reflectionPath[i];
                 this.rayPaths.push({
                    origin: vec3.clone(visualPathStart),
                    direction: vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), visualPathEnd, visualPathStart)),
                    energies: { ...energies }, type: 'reflection'
                });
                visualPathStart = visualPathEnd;
            }
            this.rayPaths.push({
                origin: vec3.clone(visualPathStart),
                direction: vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), listenerPos, visualPathStart)),
                energies: { ...energies }, type: 'reflection'
            });

            this.hits.push({
                position: vec3.clone(listenerPos), energies: { ...energies }, time: timeOfArrival,
                phase: (2 * Math.PI * 1000 * timeOfArrival) % (2 * Math.PI), frequency: 1000,
                dopplerShift: 1.0, bounces: source.order, distance: distance,
                direction: directionToListener, type: 'reflection'
            });
        }
    }

    private closestPointOnSegment(p: vec3, a: vec3, b: vec3): vec3 {
        const ap = vec3.subtract(vec3.create(), p, a);
        const ab = vec3.subtract(vec3.create(), b, a);
        const abLenSq = vec3.squaredLength(ab);
        if (abLenSq < 0.000001) return vec3.clone(a);
        let t = vec3.dot(ap, ab) / abLenSq;
        t = Math.max(0, Math.min(1, t));
        return vec3.scaleAndAdd(vec3.create(), a, ab, t);
    }

    private findClosestDiffractionEvent(ray: Ray, reflectionPlanes: any[]): { t: number, point: vec3, edge: Edge, diffractedDir: vec3 } | null {
        if (!this.config.enableDiffraction || !this.edges || this.edges.length === 0) return null;

        let closestT = Infinity;
        let bestEvent = null;
        const P0 = ray.getOrigin();
        const D = ray.getDirection();

        for (const edge of this.edges) {
            const pointOnEdgeSegment = this.closestPointOnSegment(P0, edge.start, edge.end);
            const vecToEdgePoint = vec3.subtract(vec3.create(), pointOnEdgeSegment, P0);
            const tCandidate = vec3.dot(vecToEdgePoint, D);

            if (tCandidate > 0.0001 && tCandidate < closestT) {
                const pointOnRay = vec3.scaleAndAdd(vec3.create(), P0, D, tCandidate);
                if (vec3.squaredDistance(pointOnRay, pointOnEdgeSegment) < DIFFRACTION_PROXIMITY_THRESHOLD_SQ) {
                    let clearPathToEdge = true;
                    for (const plane of reflectionPlanes) {
                        const denom = vec3.dot(D, plane.normal);
                        if (Math.abs(denom) > 0.0001) {
                            const t_plane = -(vec3.dot(P0, plane.normal) + plane.d) / denom;
                            if (t_plane > 0.0001 && t_plane < tCandidate) {
                                clearPathToEdge = false; break;
                            }
                        }
                    }
                    if (!clearPathToEdge) continue;

                    closestT = tCandidate;
                    const diffractionPoint = pointOnEdgeSegment;
                    const fromSourceToEdge = vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), diffractionPoint, this.soundSource.getPosition()));
                    const randomPerturbation = vec3.fromValues(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5);
                    vec3.scale(randomPerturbation, randomPerturbation, 0.8);
                    const diffractedDir = vec3.normalize(vec3.create(), vec3.add(vec3.create(), fromSourceToEdge, randomPerturbation));
                    bestEvent = { t: closestT, point: diffractionPoint, edge, diffractedDir };
                }
            }
        }
        return bestEvent;
    }

    private async calculateLateReflections(): Promise<void> {
        const { width, height, depth } = this.room.config.dimensions;
        const hW = width / 2, hD = depth / 2;
        const materials = this.room.config.materials;
        const reflectionPlanes = [
            { normal: vec3.fromValues(-1,0,0), d: hW, material: materials.walls }, { normal: vec3.fromValues(1,0,0),  d: hW, material: materials.walls },
            { normal: vec3.fromValues(0,-1,0), d: height, material: materials.ceiling },{ normal: vec3.fromValues(0,1,0),  d: 0, material: materials.floor },
            { normal: vec3.fromValues(0,0,-1), d: hD, material: materials.walls }, { normal: vec3.fromValues(0,0,1),  d: hD, material: materials.walls }
        ];

        for (let rayIndex = 0; rayIndex < this.rays.length; rayIndex++) {
            const ray = this.rays[rayIndex];
            let bounces = 0;
            let currentTime = 0;

            while (ray.isRayActive() && bounces < this.config.maxBounces && this.calculateAverageEnergy(ray.getEnergies()) > this.config.minEnergy) {
                let closestReflectionT = Infinity;
                let reflectionPlaneDetails: { plane: any, hitPoint: vec3, distance: number } | null = null;
                const P0_reflect = ray.getOrigin();
                const D_reflect = ray.getDirection();

                for (const plane of reflectionPlanes) {
                    const denom = vec3.dot(D_reflect, plane.normal);
                    if (Math.abs(denom) > 0.0001) {
                        const t = -(vec3.dot(P0_reflect, plane.normal) + plane.d) / denom;
                        if (t > 0.0001 && t < closestReflectionT) {
                            const hitPoint = vec3.scaleAndAdd(vec3.create(), P0_reflect, D_reflect, t);
                            if (Math.abs(hitPoint[0]) <= hW + 0.01 && hitPoint[1] >= -0.01 && hitPoint[1] <= height + 0.01 && Math.abs(hitPoint[2]) <= hD + 0.01) {
                                closestReflectionT = t;
                                reflectionPlaneDetails = { plane, hitPoint, distance: t };
                            }
                        }
                    }
                }

                const diffractionEvent = this.findClosestDiffractionEvent(ray, reflectionPlanes);
                const choseDiffraction = diffractionEvent && diffractionEvent.t < closestReflectionT;

                if (choseDiffraction && diffractionEvent) {
                    const distanceTraveled = diffractionEvent.t;
                    currentTime += distanceTraveled / this.SPEED_OF_SOUND;
                    this.rayPaths.push({
                        origin: vec3.clone(ray.getOrigin()),
                        direction: vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), diffractionEvent.point, ray.getOrigin())),
                        energies: ray.getEnergies(), type: 'diffraction'
                    });

                    const baseDiffCoeff = 1.0 - this.config.diffractionAttenuationFactor;
                    const diffractionEnergyLoss: WallMaterial = { // Using WallMaterial structure for convenience
                        absorption125Hz: 1.0-(baseDiffCoeff*0.9), absorption250Hz: 1.0-(baseDiffCoeff*0.8),
                        absorption500Hz: 1.0-(baseDiffCoeff*0.7), absorption1kHz:  1.0-(baseDiffCoeff*0.6),
                        absorption2kHz:  1.0-(baseDiffCoeff*0.5), absorption4kHz:  1.0-(baseDiffCoeff*0.4),
                        absorption8kHz:  1.0-(baseDiffCoeff*0.3), absorption16kHz: 1.0-(baseDiffCoeff*0.2),
                        scattering125Hz: 0, scattering250Hz: 0, scattering500Hz: 0, scattering1kHz: 0,
                        scattering2kHz: 0, scattering4kHz: 0, scattering8kHz: 0, scattering16kHz: 0,
                        roughness: 0, phaseShift: 0, phaseRandomization: 0
                    };
                    ray.updateRay(diffractionEvent.point, diffractionEvent.diffractedDir, diffractionEnergyLoss,
                                  distanceTraveled, this.AIR_TEMPERATURE, 50);

                    this.rayPaths.push({
                        origin: vec3.clone(ray.getOrigin()), direction: vec3.clone(ray.getDirection()),
                        energies: ray.getEnergies(), type: 'diffraction'
                    });
                    this.hits.push(this.createListenerRelativeHit(ray.getOrigin(), ray.getEnergies(), currentTime,
                                   ray.getPhase(), ray.getFrequency(), 1.0, bounces + 1, 'diffraction'));
                    bounces++;
                } else if (reflectionPlaneDetails) {
                    const { plane: closestPlane, hitPoint, distance: distanceTraveled } = reflectionPlaneDetails;
                    currentTime += distanceTraveled / this.SPEED_OF_SOUND;
                    this.rayPaths.push({
                        origin: vec3.clone(ray.getOrigin()),
                        direction: vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), hitPoint, ray.getOrigin())),
                        energies: ray.getEnergies(), type: 'reflection'
                    });

                    const D_orig = ray.getDirection();
                    const reflectedDir = vec3.create();

                    // Manual reflection: r = v - 2 * dot(v, n) * n
                    const dot_v_n = vec3.dot(D_orig, closestPlane.normal);
                    const term2_scalar_mult_n = vec3.create();
                    vec3.scale(term2_scalar_mult_n, closestPlane.normal, 2 * dot_v_n);
                    vec3.subtract(reflectedDir, D_orig, term2_scalar_mult_n);
                    vec3.normalize(reflectedDir, reflectedDir);
                    
                    const offsetOrigin = vec3.scaleAndAdd(vec3.create(), hitPoint, reflectedDir, 0.0001);
                    ray.updateRay(offsetOrigin, reflectedDir, closestPlane.material, distanceTraveled, this.AIR_TEMPERATURE, 50);

                    this.rayPaths.push({
                        origin: vec3.clone(ray.getOrigin()), direction: vec3.clone(ray.getDirection()),
                        energies: ray.getEnergies(), type: 'reflection'
                    });
                    this.hits.push(this.createListenerRelativeHit(ray.getOrigin(), ray.getEnergies(), currentTime,
                                   ray.getPhase(), ray.getFrequency(), 1.0, bounces + 1, 'reflection'));
                    bounces++;
                } else {
                    ray.deactivate();
                }
                if (this.calculateAverageEnergy(ray.getEnergies()) <= this.config.minEnergy) ray.deactivate();
            }
        }
    }

    private createListenerRelativeHit(
        interactionPointWorld: vec3, energiesAtInteraction: FrequencyBands, timeAtInteraction: number,
        phaseAtInteraction: number, frequencyAtInteraction: number, dopplerShiftAtInteraction: number,
        bounces: number, type: 'reflection' | 'diffraction' | 'direct'
    ): RayHit {
        const listenerPos = this.camera.getPosition();
        const vecToListener = vec3.subtract(vec3.create(), listenerPos, interactionPointWorld);
        const distanceToListener = vec3.length(vecToListener);
        const directionFromInteractionToListener = vec3.normalize(vec3.create(), vecToListener);
        const travelTimeToListener = distanceToListener / this.SPEED_OF_SOUND;
        const totalTimeAtListener = timeAtInteraction + travelTimeToListener;

        const airAbsRay = new Ray(vec3.create(),vec3.create(), 1.0, frequencyAtInteraction);
        const airAbs = airAbsRay.calculateAirAbsorption(distanceToListener, this.AIR_TEMPERATURE, 50);

        const energiesAtListener = { ...energiesAtInteraction };
        const distanceAttenuation = 1.0 / Math.max(0.01, distanceToListener * distanceToListener);
        for (const key of Object.keys(energiesAtListener) as Array<keyof FrequencyBands>) {
            energiesAtListener[key] *= distanceAttenuation;
            energiesAtListener[key] *= (airAbs as any)[`absorption${key.replace('energy', '')}`];
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

    public getRayHits(): RayHit[] {
        return this.hits.filter(hit => hit && hit.position && hit.energies);
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