/**
 * Acoustic Materials - Material properties for spatial audio
 */

import { FREQUENCY_BANDS } from './audio-utils';

/**
 * Acoustic material properties interface
 */
export interface AcousticMaterial {
    id: string;
    name: string;
    category: MaterialCategory;
    
    // Frequency-dependent absorption coefficients (0-1)
    // One value per frequency band (125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz, 8kHz, 16kHz)
    absorption: number[];
    
    // Frequency-dependent scattering coefficients (0-1)
    scattering: number[];
    
    // Material impedance (kg/m²s)
    impedance: number;
    
    // Surface roughness (0-1, affects scattering behavior)
    roughness: number;
    
    // Transmission loss (dB) - for materials that allow sound transmission
    transmissionLoss?: number;
    
    // Visual properties for UI
    color?: string;
    texture?: string;
}

/**
 * Material categories for organization
 */
export enum MaterialCategory {
    HARD_SURFACES = 'hard_surfaces',
    SOFT_SURFACES = 'soft_surfaces',
    WOOD = 'wood',
    FABRIC = 'fabric',
    ACOUSTIC_TREATMENT = 'acoustic_treatment',
    GLASS = 'glass',
    METAL = 'metal',
    SPECIAL = 'special'
}

/**
 * Predefined acoustic materials database
 * Absorption and scattering coefficients based on research data
 */
export const ACOUSTIC_MATERIALS: Record<string, AcousticMaterial> = {
    // Hard surfaces
    concrete: {
        id: 'concrete',
        name: 'Concrete (Painted)',
        category: MaterialCategory.HARD_SURFACES,
        absorption: [0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03],
        scattering: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        impedance: 1.8e6,
        roughness: 0.1,
        color: '#808080'
    },
    
    brick: {
        id: 'brick',
        name: 'Brick Wall',
        category: MaterialCategory.HARD_SURFACES,
        absorption: [0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.07, 0.08],
        scattering: [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.50],
        impedance: 1.5e6,
        roughness: 0.3,
        color: '#8B4513'
    },
    
    plaster: {
        id: 'plaster',
        name: 'Plaster on Concrete',
        category: MaterialCategory.HARD_SURFACES,
        absorption: [0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05],
        scattering: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        impedance: 1.4e6,
        roughness: 0.05,
        color: '#F5F5DC'
    },
    
    // Wood surfaces
    wood_floor: {
        id: 'wood_floor',
        name: 'Wood Floor (Hardwood)',
        category: MaterialCategory.WOOD,
        absorption: [0.04, 0.04, 0.07, 0.06, 0.06, 0.07, 0.07, 0.08],
        scattering: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        impedance: 5.0e5,
        roughness: 0.1,
        color: '#8B4513'
    },
    
    wood_panel: {
        id: 'wood_panel',
        name: 'Wood Paneling (Thin)',
        category: MaterialCategory.WOOD,
        absorption: [0.28, 0.22, 0.17, 0.09, 0.10, 0.11, 0.13, 0.15],
        scattering: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        impedance: 3.0e5,
        roughness: 0.15,
        color: '#A0522D'
    },
    
    // Fabric and soft materials
    carpet_thick: {
        id: 'carpet_thick',
        name: 'Thick Carpet on Concrete',
        category: MaterialCategory.FABRIC,
        absorption: [0.08, 0.24, 0.57, 0.69, 0.71, 0.73, 0.75, 0.78],
        scattering: [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85],
        impedance: 2.0e4,
        roughness: 0.8,
        color: '#8B0000'
    },
    
    curtain_heavy: {
        id: 'curtain_heavy',
        name: 'Heavy Curtains',
        category: MaterialCategory.FABRIC,
        absorption: [0.14, 0.35, 0.55, 0.72, 0.70, 0.65, 0.65, 0.65],
        scattering: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        impedance: 1.5e4,
        roughness: 0.7,
        color: '#4B0082'
    },
    
    // Acoustic treatment
    acoustic_foam: {
        id: 'acoustic_foam',
        name: 'Acoustic Foam (5cm)',
        category: MaterialCategory.ACOUSTIC_TREATMENT,
        absorption: [0.11, 0.28, 0.68, 0.90, 0.95, 0.96, 0.97, 0.98],
        scattering: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        impedance: 5.0e3,
        roughness: 0.9,
        color: '#2F4F4F'
    },
    
    bass_trap: {
        id: 'bass_trap',
        name: 'Bass Trap (Corner)',
        category: MaterialCategory.ACOUSTIC_TREATMENT,
        absorption: [0.80, 0.90, 0.95, 0.95, 0.90, 0.85, 0.80, 0.75],
        scattering: [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90],
        impedance: 3.0e3,
        roughness: 0.95,
        color: '#000000'
    },
    
    diffuser: {
        id: 'diffuser',
        name: 'QRD Diffuser',
        category: MaterialCategory.ACOUSTIC_TREATMENT,
        absorption: [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22],
        scattering: [0.40, 0.60, 0.80, 0.90, 0.95, 0.95, 0.90, 0.85],
        impedance: 4.0e5,
        roughness: 1.0,
        color: '#D2691E'
    },
    
    // Glass
    glass_window: {
        id: 'glass_window',
        name: 'Glass Window (6mm)',
        category: MaterialCategory.GLASS,
        absorption: [0.18, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02],
        scattering: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        impedance: 1.2e7,
        roughness: 0.01,
        transmissionLoss: 25,
        color: '#87CEEB'
    },
    
    // Metal
    steel: {
        id: 'steel',
        name: 'Steel Plate',
        category: MaterialCategory.METAL,
        absorption: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        scattering: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        impedance: 4.0e7,
        roughness: 0.02,
        color: '#C0C0C0'
    },
    
    // Special materials
    water: {
        id: 'water',
        name: 'Water Surface',
        category: MaterialCategory.SPECIAL,
        absorption: [0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04],
        scattering: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        impedance: 1.5e6,
        roughness: 0.1,
        color: '#00CED1'
    },
    
    audience: {
        id: 'audience',
        name: 'Audience Area',
        category: MaterialCategory.SPECIAL,
        absorption: [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.85, 0.85],
        scattering: [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85],
        impedance: 5.0e4,
        roughness: 0.8,
        color: '#FFB6C1'
    }
};

/**
 * Get material by ID
 */
export function getMaterial(id: string): AcousticMaterial | undefined {
    return ACOUSTIC_MATERIALS[id];
}

/**
 * Get materials by category
 */
export function getMaterialsByCategory(category: MaterialCategory): AcousticMaterial[] {
    return Object.values(ACOUSTIC_MATERIALS).filter(m => m.category === category);
}

/**
 * Calculate average absorption coefficient
 */
export function getAverageAbsorption(material: AcousticMaterial): number {
    const sum = material.absorption.reduce((a, b) => a + b, 0);
    return sum / material.absorption.length;
}

/**
 * Calculate NRC (Noise Reduction Coefficient)
 * Average of absorption coefficients at 250, 500, 1000, and 2000 Hz
 */
export function calculateNRC(material: AcousticMaterial): number {
    // Indices for 250Hz, 500Hz, 1kHz, 2kHz in our frequency bands
    const indices = [1, 2, 3, 4];
    const sum = indices.reduce((acc, i) => acc + material.absorption[i], 0);
    return Math.round(sum / 4 * 100) / 100; // Round to 2 decimal places
}

/**
 * Interpolate material properties between two materials
 */
export function interpolateMaterials(
    material1: AcousticMaterial,
    material2: AcousticMaterial,
    factor: number // 0 = material1, 1 = material2
): AcousticMaterial {
    const t = Math.max(0, Math.min(1, factor));
    
    return {
        id: `${material1.id}_${material2.id}_${t}`,
        name: `Mix: ${material1.name} / ${material2.name}`,
        category: material1.category,
        absorption: material1.absorption.map((v, i) => 
            v * (1 - t) + material2.absorption[i] * t
        ),
        scattering: material1.scattering.map((v, i) => 
            v * (1 - t) + material2.scattering[i] * t
        ),
        impedance: material1.impedance * (1 - t) + material2.impedance * t,
        roughness: material1.roughness * (1 - t) + material2.roughness * t
    };
}

/**
 * Create custom material
 */
export function createCustomMaterial(
    id: string,
    name: string,
    properties: Partial<AcousticMaterial>
): AcousticMaterial {
    return {
        id,
        name,
        category: properties.category || MaterialCategory.SPECIAL,
        absorption: properties.absorption || new Array(8).fill(0.1),
        scattering: properties.scattering || new Array(8).fill(0.1),
        impedance: properties.impedance || 1e6,
        roughness: properties.roughness || 0.5,
        ...properties
    };
}

/**
 * Material presets for common room types
 */
export const ROOM_PRESETS = {
    recording_studio: {
        walls: 'acoustic_foam',
        floor: 'carpet_thick',
        ceiling: 'acoustic_foam',
        bass_traps: 'bass_trap'
    },
    concert_hall: {
        walls: 'wood_panel',
        floor: 'wood_floor',
        ceiling: 'plaster',
        audience: 'audience'
    },
    living_room: {
        walls: 'plaster',
        floor: 'carpet_thick',
        ceiling: 'plaster',
        windows: 'glass_window'
    },
    bathroom: {
        walls: 'plaster',
        floor: 'concrete',
        ceiling: 'plaster'
    }
};

/**
 * Export material data as JSON
 */
export function exportMaterialsJSON(): string {
    return JSON.stringify(ACOUSTIC_MATERIALS, null, 2);
}

/**
 * Import materials from JSON
 */
export function importMaterialsJSON(json: string): Record<string, AcousticMaterial> {
    try {
        return JSON.parse(json);
    } catch (error) {
        console.error('Failed to parse materials JSON:', error);
        return {};
    }
}
