export interface RoomMaterials {
    walls: WallMaterial;
    ceiling: WallMaterial;
    floor: WallMaterial;
}

export interface WallMaterial {
    // Keep the simple properties for manual UI control
    absorptionLow?: number;
    absorptionMid?: number;
    absorptionHigh?: number;

    // Add the detailed 8-band properties from your presets
    absorption63?: number;
    absorption125?: number;
    absorption250?: number;
    absorption500?: number;
    absorption1k?: number;
    absorption2k?: number;
    absorption4k?: number;
    absorption8k?: number;

    // --- We will do the same for scattering ---
    scatteringLow?: number;
    scatteringMid?: number;
    scatteringHigh?: number;
    scattering63?: number;
    scattering125?: number;
    scattering250?: number;
    scattering500?: number;
    scattering1k?: number;
    scattering2k?: number;
    scattering4k?: number;
    scattering8k?: number;

    roughness: number;
    phaseShift: number;
    phaseRandomization: number;
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