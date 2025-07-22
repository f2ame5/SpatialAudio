
interface RoomConfig {
    dimensions: {
        width: number;
        height: number;
        depth: number;
    };
    materials: {
        walls: { absorptionLow: number, absorptionMid: number, absorptionHigh: number };
        ceiling: { absorptionLow: number, absorptionMid: number, absorptionHigh: number };
        floor: { absorptionLow: number, absorptionMid: number, absorptionHigh: number };
    };
}

interface RoomDimensions {
    width: number;
    height: number;
    depth: number;
}

interface RoomMaterials {
    walls: { absorptionLow: number, absorptionMid: number, absorptionHigh: number };
    ceiling: { absorptionLow: number, absorptionMid: number, absorptionHigh: number };
    floor: { absorptionLow: number, absorptionMid: number, absorptionHigh: number };
}

export enum Surface {
    FLOOR = 0,
    CEILING = 1,
    WALL_FRONT_BACK = 2,
    WALL_LEFT_RIGHT = 3
}