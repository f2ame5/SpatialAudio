/**
 * Room Acoustics - Integrates acoustic materials with room geometry
 */

import { vec3 } from 'gl-matrix';
import { Room, RoomConfig } from './room';
import { 
    AcousticMaterial, 
    ACOUSTIC_MATERIALS, 
    getMaterial,
    calculateNRC,
    getAverageAbsorption 
} from '../audio/acoustic-materials';

/**
 * Surface types in a room
 */
export enum SurfaceType {
    FLOOR = 0,
    CEILING = 1,
    WALL_NORTH = 2,
    WALL_SOUTH = 3,
    WALL_EAST = 4,
    WALL_WEST = 5
}

/**
 * Acoustic room configuration extending base room config
 */
export interface AcousticRoomConfig extends RoomConfig {
    acousticMaterials?: {
        floor?: string;
        ceiling?: string;
        wallNorth?: string;
        wallSouth?: string;
        wallEast?: string;
        wallWest?: string;
    };
    temperature?: number;  // Celsius
    humidity?: number;     // Percentage
    airPressure?: number;  // Pa
}

/**
 * Room acoustics properties
 */
export interface RoomAcousticProperties {
    rt60: number;           // Reverberation time
    averageAbsorption: number;
    totalAbsorption: number;
    criticalDistance: number;
    roomConstant: number;
    meanFreePath: number;
}

export class RoomAcoustics {
    private room: Room;
    private materials: Map<SurfaceType, AcousticMaterial>;
    private temperature: number;
    private humidity: number;
    private airPressure: number;
    
    constructor(room: Room, config?: AcousticRoomConfig) {
        this.room = room;
        this.materials = new Map();
        
        // Set environmental conditions
        this.temperature = config?.temperature ?? 20;
        this.humidity = config?.humidity ?? 50;
        this.airPressure = config?.airPressure ?? 101325;
        
        // Initialize materials
        this.initializeMaterials(config?.acousticMaterials);
    }
    
    /**
     * Initialize acoustic materials for room surfaces
     */
    private initializeMaterials(materials?: AcousticRoomConfig['acousticMaterials']): void {
        // Default materials
        const defaults = {
            floor: 'wood_floor',
            ceiling: 'plaster',
            walls: 'plaster'
        };
        
        // Set floor material
        const floorMaterial = getMaterial(materials?.floor || defaults.floor);
        if (floorMaterial) {
            this.materials.set(SurfaceType.FLOOR, floorMaterial);
        }
        
        // Set ceiling material
        const ceilingMaterial = getMaterial(materials?.ceiling || defaults.ceiling);
        if (ceilingMaterial) {
            this.materials.set(SurfaceType.CEILING, ceilingMaterial);
        }
        
        // Set wall materials
        const wallMaterials = {
            [SurfaceType.WALL_NORTH]: materials?.wallNorth || defaults.walls,
            [SurfaceType.WALL_SOUTH]: materials?.wallSouth || defaults.walls,
            [SurfaceType.WALL_EAST]: materials?.wallEast || defaults.walls,
            [SurfaceType.WALL_WEST]: materials?.wallWest || defaults.walls
        };
        
        for (const [surface, materialId] of Object.entries(wallMaterials)) {
            const material = getMaterial(materialId);
            if (material) {
                this.materials.set(Number(surface) as SurfaceType, material);
            }
        }
    }
    
    /**
     * Set material for a specific surface
     */
    setMaterial(surface: SurfaceType, materialId: string): boolean {
        const material = getMaterial(materialId);
        if (material) {
            this.materials.set(surface, material);
            return true;
        }
        return false;
    }
    
    /**
     * Get material for a specific surface
     */
    getMaterial(surface: SurfaceType): AcousticMaterial | undefined {
        return this.materials.get(surface);
    }
    
    /**
     * Get all materials as a map for the raytracer
     */
    getMaterialsMap(): Map<number, AcousticMaterial> {
        const map = new Map<number, AcousticMaterial>();
        for (const [surface, material] of this.materials) {
            map.set(surface, material);
        }
        return map;
    }
    
    /**
     * Calculate surface areas
     */
    getSurfaceAreas(): Map<SurfaceType, number> {
        const { width, height, depth } = this.room.config.dimensions;
        const areas = new Map<SurfaceType, number>();
        
        areas.set(SurfaceType.FLOOR, width * depth);
        areas.set(SurfaceType.CEILING, width * depth);
        areas.set(SurfaceType.WALL_NORTH, width * height);
        areas.set(SurfaceType.WALL_SOUTH, width * height);
        areas.set(SurfaceType.WALL_EAST, depth * height);
        areas.set(SurfaceType.WALL_WEST, depth * height);
        
        return areas;
    }
    
    /**
     * Calculate room acoustic properties
     */
    calculateAcousticProperties(): RoomAcousticProperties {
        const volume = this.room.getVolume();
        const areas = this.getSurfaceAreas();
        
        // Calculate total absorption for each frequency band
        const totalAbsorptionByFreq = new Array(8).fill(0);
        let totalSurfaceArea = 0;
        
        for (const [surface, area] of areas) {
            const material = this.materials.get(surface);
            if (material) {
                for (let i = 0; i < 8; i++) {
                    totalAbsorptionByFreq[i] += area * material.absorption[i];
                }
            }
            totalSurfaceArea += area;
        }
        
        // Average absorption across all frequencies
        const averageAbsorption = totalAbsorptionByFreq.reduce((a, b) => a + b, 0) / 
                                (8 * totalSurfaceArea);
        
        // Total absorption (Sabins) at 1kHz (index 3)
        const totalAbsorption = totalAbsorptionByFreq[3];
        
        // Sabine's formula for RT60
        const rt60 = 0.161 * volume / totalAbsorption;
        
        // Room constant
        const roomConstant = totalAbsorption / (1 - averageAbsorption);
        
        // Critical distance
        const criticalDistance = 0.057 * Math.sqrt(roomConstant);
        
        // Mean free path
        const meanFreePath = 4 * volume / totalSurfaceArea;
        
        return {
            rt60,
            averageAbsorption,
            totalAbsorption,
            criticalDistance,
            roomConstant,
            meanFreePath
        };
    }
    
    /**
     * Calculate speed of sound based on temperature and humidity
     */
    getSpeedOfSound(): number {
        // Simplified formula for speed of sound in air
        const T = this.temperature + 273.15; // Convert to Kelvin
        const baseSpeed = 331.3 * Math.sqrt(T / 273.15);
        
        // Humidity correction (simplified)
        const humidityCorrection = 0.05 * this.humidity / 100;
        
        return baseSpeed * (1 + humidityCorrection);
    }
    
    /**
     * Calculate air absorption coefficients
     */
    getAirAbsorption(): number[] {
        // Simplified air absorption model (per meter)
        // Based on ISO 9613-1
        const frequencies = [125, 250, 500, 1000, 2000, 4000, 8000, 16000];
        const absorption = new Array(8);
        
        for (let i = 0; i < frequencies.length; i++) {
            const f = frequencies[i];
            // Simplified formula - actual calculation is more complex
            absorption[i] = 0.0000008 * f * f / (this.humidity + 10);
        }
        
        return absorption;
    }
    
    /**
     * Get room bounds for raytracing
     */
    getRoomBounds(): { min: vec3; max: vec3 } {
        const { width, height, depth } = this.room.config.dimensions;
        return {
            min: vec3.fromValues(-width / 2, 0, -depth / 2),
            max: vec3.fromValues(width / 2, height, depth / 2)
        };
    }
    
    /**
     * Check which surface a point is closest to
     */
    getClosestSurface(point: vec3): SurfaceType | null {
        const { width, height, depth } = this.room.config.dimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;
        
        const distances = [
            { surface: SurfaceType.FLOOR, distance: point[1] },
            { surface: SurfaceType.CEILING, distance: height - point[1] },
            { surface: SurfaceType.WALL_NORTH, distance: halfDepth - point[2] },
            { surface: SurfaceType.WALL_SOUTH, distance: point[2] + halfDepth },
            { surface: SurfaceType.WALL_EAST, distance: halfWidth - point[0] },
            { surface: SurfaceType.WALL_WEST, distance: point[0] + halfWidth }
        ];
        
        // Find minimum distance
        let minDistance = Infinity;
        let closestSurface: SurfaceType | null = null;
        
        for (const { surface, distance } of distances) {
            if (distance >= 0 && distance < minDistance) {
                minDistance = distance;
                closestSurface = surface;
            }
        }
        
        return closestSurface;
    }
    
    /**
     * Get surface normal for a given surface
     */
    getSurfaceNormal(surface: SurfaceType): vec3 {
        switch (surface) {
            case SurfaceType.FLOOR:
                return vec3.fromValues(0, 1, 0);
            case SurfaceType.CEILING:
                return vec3.fromValues(0, -1, 0);
            case SurfaceType.WALL_NORTH:
                return vec3.fromValues(0, 0, -1);
            case SurfaceType.WALL_SOUTH:
                return vec3.fromValues(0, 0, 1);
            case SurfaceType.WALL_EAST:
                return vec3.fromValues(-1, 0, 0);
            case SurfaceType.WALL_WEST:
                return vec3.fromValues(1, 0, 0);
            default:
                return vec3.fromValues(0, 1, 0);
        }
    }
    
    /**
     * Calculate optimal listener position (acoustic sweet spot)
     */
    getOptimalListenerPosition(): vec3 {
        const { width, height, depth } = this.room.config.dimensions;
        
        // Simple rule: 38% from front wall, centered, ear height
        return vec3.fromValues(
            0,                    // Centered
            1.2,                  // Typical ear height when seated
            -depth * 0.12         // 38% from back wall = 12% from center
        );
    }
    
    /**
     * Export room acoustic data
     */
    exportAcousticData(): object {
        const properties = this.calculateAcousticProperties();
        const areas = this.getSurfaceAreas();
        const materialsData: any = {};
        
        for (const [surface, material] of this.materials) {
            materialsData[SurfaceType[surface]] = {
                id: material.id,
                name: material.name,
                nrc: calculateNRC(material),
                averageAbsorption: getAverageAbsorption(material)
            };
        }
        
        return {
            dimensions: this.room.config.dimensions,
            volume: this.room.getVolume(),
            surfaceAreas: Object.fromEntries(areas),
            materials: materialsData,
            environmentalConditions: {
                temperature: this.temperature,
                humidity: this.humidity,
                airPressure: this.airPressure,
                speedOfSound: this.getSpeedOfSound()
            },
            acousticProperties: properties
        };
    }
}
