
interface RoomConfig {
    dimensions: {
        width: number;
        height: number;
        depth: number;
    };
    materials: {
        walls: { color: number[] };
        ceiling: { color: number[] };
        floor: { color: number[] };
    };
}

interface RoomDimensions {
    width: number;
    height: number;
    depth: number;
}

interface RoomMaterials {
    walls: { color: number[] };
    ceiling: { color: number[] };
    floor: { color: number[] };
}

export enum Surface {
    FLOOR = 0,
    CEILING = 1,
    WALL_FRONT_BACK = 2,
    WALL_LEFT_RIGHT = 3
}