# Room System Documentation

## Overview
The room system provides a 3D visualization of the acoustic space with proper surface rendering and material properties.

## Components

### Room Class (`src/room/room.ts`)
Main class responsible for room rendering and management.

#### Properties
- `device`: GPUDevice - WebGPU device instance
- `pipeline`: GPURenderPipeline - WebGPU render pipeline
- `vertexBuffer`: GPUBuffer - Buffer containing vertex data
- `indexBuffer`: GPUBuffer - Buffer containing index data
- `uniformBuffer`: GPUBuffer - Buffer for view-projection matrix
- `config`: RoomConfig - Room configuration object

### Surface Types
```typescript
enum Surface {
    FLOOR = 0,
    CEILING = 1,
    WALL_FRONT_BACK = 2,
    WALL_LEFT_RIGHT = 3
}
```

## Rendering Features

### Double-Sided Rendering
- Walls are rendered on both sides
- Proper face orientation for interior/exterior viewing
- Disabled face culling for complete visibility

### Surface Materials
- Floor: Dark grey (0.35) for solid foundation
- Ceiling: Light grey (0.75) for brightness
- Front/Back walls: Medium grey (0.55)
- Side walls: Slightly darker grey (0.45)

### Lighting System
- Central light source at room top
- Ambient light component (30%)
- Diffuse lighting (70%)
- Per-vertex normals for proper shading
- World-space lighting calculations

### Geometry
- Vertex format:
  - Position (vec3)
  - Normal (vec3)
  - Surface type (float)
- Double-sided triangles for all surfaces
- Proper normal orientation
- Optimized vertex reuse

## Methods

### Constructor
```typescript
constructor(device: GPUDevice, config: RoomConfig)
```
Creates room with specified dimensions and materials.

### Dimension Updates
```typescript
updateDimensions(dimensions: RoomDimensions): void
```
Updates room size and rebuilds geometry.

### Material Updates
```typescript
updateMaterials(materials: RoomMaterials): void
```
Updates acoustic properties of surfaces.

### Rendering
```typescript
render(pass: GPURenderPassEncoder): void
```
Renders room with current configuration.

## Integration with Ray Tracing

### Surface Identification
- Each surface has a unique type identifier
- Enables material-specific ray interactions
- Supports acoustic simulation

### Normal Vectors
- Accurate surface normals for reflection
- Consistent orientation for ray calculations
- Used in both rendering and acoustics

## Usage Example
```typescript
// Create room with initial config
const room = new Room(device, {
    dimensions: { width: 10, height: 5, depth: 8 },
    materials: {
        walls: { absorption: 0.3 },
        ceiling: { absorption: 0.25 },
        floor: { absorption: 0.4 }
    }
});

// Update room in render loop
room.updateViewProjection(viewProjectionMatrix);
room.render(renderPass);
