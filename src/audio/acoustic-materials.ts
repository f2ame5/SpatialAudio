/**
 * Acoustic Materials Database
 * Frequency-dependent absorption and scattering coefficients for common building materials
 */

export interface AcousticMaterial {
    id: string;
    name: string;
    description: string;
    absorption: number[]; // 8 frequency bands: 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz
    scattering: number[]; // 8 frequency bands: 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz
    impedance: number;    // Acoustic impedance (Pa·s/m)
    roughness: number;    // Surface roughness factor (0-1)
    density: number;      // Material density (kg/m³)
}

/**
 * Acoustic materials database
 */
export const ACOUSTIC_MATERIALS: { [key: string]: AcousticMaterial } = {
    // Wall materials
    'concrete': {
        id: 'concrete',
        name: 'Concrete',
        description: 'Smooth concrete wall',
        absorption: [0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        scattering: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        impedance: 1.8e6,
        roughness: 0.1,
        density: 2400
    },
    
    'brick': {
        id: 'brick',
        name: 'Brick Wall',
        description: 'Painted brick wall',
        absorption: [0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.09, 0.10],
        scattering: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        impedance: 1.2e6,
        roughness: 0.2,
        density: 1800
    },
    
    'plaster': {
        id: 'plaster',
        name: 'Plaster Wall',
        description: 'Smooth plaster on lath',
        absorption: [0.02, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.02],
        scattering: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        impedance: 8.5e5,
        roughness: 0.05,
        density: 1200
    },
    
    'drywall': {
        id: 'drywall',
        name: 'Drywall',
        description: 'Painted gypsum board',
        absorption: [0.05, 0.06, 0.07, 0.09, 0.08, 0.08, 0.08, 0.08],
        scattering: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        impedance: 4.2e5,
        roughness: 0.05,
        density: 800
    },
    
    // Floor materials
    'hardwood': {
        id: 'hardwood',
        name: 'Hardwood Floor',
        description: 'Polished hardwood flooring',
        absorption: [0.04, 0.04, 0.07, 0.06, 0.06, 0.07, 0.07, 0.07],
        scattering: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        impedance: 3.8e5,
        roughness: 0.1,
        density: 700
    },
    
    'carpet': {
        id: 'carpet',
        name: 'Carpet',
        description: 'Heavy carpet on concrete',
        absorption: [0.02, 0.06, 0.14, 0.37, 0.60, 0.65, 0.70, 0.75],
        scattering: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        impedance: 2.1e4,
        roughness: 0.8,
        density: 400
    },
    
    'tile': {
        id: 'tile',
        name: 'Ceramic Tile',
        description: 'Glazed ceramic tile on concrete',
        absorption: [0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02],
        scattering: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        impedance: 2.2e6,
        roughness: 0.05,
        density: 2300
    },
    
    // Ceiling materials
    'acoustic_tile': {
        id: 'acoustic_tile',
        name: 'Acoustic Ceiling Tile',
        description: 'Perforated acoustic ceiling tile',
        absorption: [0.17, 0.86, 0.99, 0.93, 0.85, 0.85, 0.85, 0.85],
        scattering: [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        impedance: 1.8e4,
        roughness: 0.6,
        density: 300
    },
    
    'gypsum_board': {
        id: 'gypsum_board',
        name: 'Gypsum Board Ceiling',
        description: 'Painted gypsum board ceiling',
        absorption: [0.05, 0.06, 0.07, 0.09, 0.08, 0.08, 0.08, 0.08],
        scattering: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        impedance: 4.2e5,
        roughness: 0.05,
        density: 800
    },
    
    // Special materials
    'glass': {
        id: 'glass',
        name: 'Glass Window',
        description: 'Large glass window',
        absorption: [0.18, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02],
        scattering: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        impedance: 1.8e7,
        roughness: 0.01,
        density: 2500
    },
    
    'curtain': {
        id: 'curtain',
        name: 'Heavy Curtain',
        description: 'Heavy fabric curtain',
        absorption: [0.07, 0.31, 0.49, 0.75, 0.70, 0.60, 0.50, 0.40],
        scattering: [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        impedance: 3.2e4,
        roughness: 0.9,
        density: 200
    }
};

/**
 * Get material by ID
 */
export function getMaterial(id: string): AcousticMaterial | undefined {
    return ACOUSTIC_MATERIALS[id];
}

/**
 * Get all available material IDs
 */
export function getMaterialIds(): string[] {
    return Object.keys(ACOUSTIC_MATERIALS);
}

/**
 * Get materials by category
 */
export function getMaterialsByCategory(category: 'wall' | 'floor' | 'ceiling'): AcousticMaterial[] {
    const categoryMaterials: { [key: string]: string[] } = {
        wall: ['concrete', 'brick', 'plaster', 'drywall', 'glass'],
        floor: ['hardwood', 'carpet', 'tile'],
        ceiling: ['acoustic_tile', 'gypsum_board', 'plaster']
    };
    
    return categoryMaterials[category]?.map(id => ACOUSTIC_MATERIALS[id]).filter(Boolean) || [];
}

/**
 * Calculate Noise Reduction Coefficient (NRC)
 * Average of absorption coefficients at 250, 500, 1000, and 2000 Hz
 */
export function calculateNRC(material: AcousticMaterial): number {
    const frequencies = [1, 2, 3, 4]; // Indices for 250, 500, 1k, 2k Hz
    const sum = frequencies.reduce((acc, idx) => acc + material.absorption[idx], 0);
    return Math.round((sum / 4) * 20) / 20; // Round to nearest 0.05
}

/**
 * Calculate average absorption across all frequencies
 */
export function getAverageAbsorption(material: AcousticMaterial): number {
    const sum = material.absorption.reduce((acc, val) => acc + val, 0);
    return sum / material.absorption.length;
}

/**
 * Calculate average scattering across all frequencies
 */
export function getAverageScattering(material: AcousticMaterial): number {
    const sum = material.scattering.reduce((acc, val) => acc + val, 0);
    return sum / material.scattering.length;
}

/**
 * Get frequency labels for the 8 bands
 */
export function getFrequencyLabels(): string[] {
    return ['125 Hz', '250 Hz', '500 Hz', '1 kHz', '2 kHz', '4 kHz', '8 kHz', '16 kHz'];
}

/**
 * Get frequency values for the 8 bands
 */
export function getFrequencyValues(): number[] {
    return [125, 250, 500, 1000, 2000, 4000, 8000, 16000];
}

/**
 * Default material assignments for room surfaces
 */
export const DEFAULT_ROOM_MATERIALS = {
    floor: 'hardwood',
    ceiling: 'gypsum_board',
    walls: 'plaster'
};
