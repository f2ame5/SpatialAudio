# Room Boundaries Documentation

## Overview
The `RoomBoundaries` class defines the geometric boundaries of the room using planes. Each plane is defined by a normal vector and distance from origin, following the plane equation ax + by + cz + d = 0.

## Key Components

### RoomBoundaries Class
Manages room geometry and surface properties.

#### Constructor
```typescript
constructor(width: number, height: number, depth: number)
```
Creates room boundaries with specified dimensions.

### Plane Definition
Each plane is defined with:
- Normal vector (pointing inward)
- Distance from origin
- Surface type

### Surface Types
```typescript
enum SurfaceType {
    FLOOR,
    CEILING,
    WALL_FRONT,
    WALL_BACK,
    WALL_LEFT,
    WALL_RIGHT
}
```

### Room Planes
Room is defined by six planes with normals pointing inward:
1. Floor (Y-): normal = (0, 1, 0)
2. Ceiling (Y+): normal = (0, -1, 0)
3. Front Wall (Z+): normal = (0, 0, -1)
4. Back Wall (Z-): normal = (0, 0, 1)
5. Left Wall (X-): normal = (1, 0, 0)
6. Right Wall (X+): normal = (-1, 0, 0)

### Key Methods

#### updateDimensions
```typescript
updateDimensions(width: number, height: number, depth: number): void
```
Updates room dimensions and recalculates plane equations.

#### getDimensions
```typescript
getDimensions(): { width: number, height: number, depth: number }
```
Returns current room dimensions.

#### getPlanes
```typescript
getPlanes(): Plane[]
```
Returns array of room boundary planes.

### Implementation Details

#### Plane Updates
When dimensions change:
1. Recalculate plane distances
2. Normalize all plane normals
3. Update vertex buffers for rendering

#### Coordinate System
- Origin at room center
- Y-axis up
- Z-axis forward
- X-axis right

#### Numerical Stability
1. All plane normals are normalized
2. Dimensions are halved for centered coordinate system
3. Plane distances are negative of half-dimensions

## Usage Example
```typescript
const boundaries = new RoomBoundaries(10, 5, 8);

// Update room size
boundaries.updateDimensions(12, 6, 10);

// Get planes for intersection testing
const planes = boundaries.getPlanes();
```

## Ray Intersection
For ray intersection testing:
1. Get normalized plane normal and distance
2. Calculate intersection using plane equation
3. Validate intersection point is within bounds
4. Use surface type for material properties

## Visualization
Room boundaries are rendered using:
1. Vertex buffer for geometry
2. Different colors per surface type
3. Wireframe mode for debugging
