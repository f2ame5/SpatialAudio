export interface RoomMaterials {
    left:   WallMaterial;
    right:  WallMaterial;
    top:    WallMaterial;
    bottom: WallMaterial;
    front:  WallMaterial;
    back:   WallMaterial;
}

export interface WallMaterial {
    color: number[]; // RGB color values [r, g, b]
}

// Simple material presets for visual rendering
export const MATERIAL_PRESETS = {
    CONCRETE: {
        color: [0.7, 0.7, 0.7] // Light gray
    },
    WOOD: {
        color: [0.6, 0.4, 0.2] // Brown
    },
    CARPET: {
        color: [0.5, 0.3, 0.3] // Dark red
    },
    GLASS: {
        color: [0.8, 0.9, 1.0] // Light blue
    },
    BRICK: {
        color: [0.8, 0.4, 0.3] // Red-brown
    }
};