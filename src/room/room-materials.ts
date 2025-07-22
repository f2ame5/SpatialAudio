export interface RoomMaterials {
    left:   WallMaterial;
    right:  WallMaterial;
    top:    WallMaterial;
    bottom: WallMaterial;
    front:  WallMaterial;
    back:   WallMaterial;
}

export interface WallMaterial {
    absorptionLow: number;   // 125-500Hz
    absorptionMid: number;   // 500-2000Hz
    absorptionHigh: number;  // 2000-8000Hz

    scatteringLow: number;   // Low frequency scattering
    scatteringMid: number;   // Mid frequency scattering
    scatteringHigh: number;  // High frequency scattering

    roughness: number;       // 0-1, affects reflection pattern
    phaseShift: number;      // Fixed phase shift on reflection (radians)
    phaseRandomization: number; // Max random phase variation (radians)
}

// Add standard material presets
export const MATERIAL_PRESETS = {
    CONCRETE: {
        absorptionLow: 0.08,
        absorptionMid: 0.08,
        absorptionHigh: 0.04,
        scatteringLow: 0.1,
        scatteringMid: 0.15,
        scatteringHigh: 0.2,
        roughness: 0.25
    },
    WOOD: {
        absorptionLow: 0.15,
        absorptionMid: 0.10,
        absorptionHigh: 0.07,
        scatteringLow: 0.2,
        scatteringMid: 0.3,
        scatteringHigh: 0.4,
        roughness: 0.3
    },
    
};