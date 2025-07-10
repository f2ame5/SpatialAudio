# Sound Source Documentation

## Overview
The `SoundSource` class represents a point source of sound in the room. It handles position management, energy emission, and visualization of the sound source.

## Key Components

### SoundSource Class
Main class for managing sound source properties and behavior.

#### Constructor
```typescript
constructor(device: GPUDevice, position: vec3)
```
- `device`: WebGPU device for rendering
- `position`: Initial 3D position of the sound source

### Properties
- `position`: Current 3D position (vec3)
- `energy`: Initial energy level (default: 1.0)
- `direction`: Optional directivity vector

## Methods

### Position Management
```typescript
setPosition(position: vec3): void
getPosition(): vec3
```
Updates and retrieves the sound source position.

### Energy Control
```typescript
getInitialEnergy(): number
setInitialEnergy(energy: number): void
```
Manages the initial energy level of emitted rays.

### Visualization
```typescript
render(pass: GPURenderPassEncoder, viewProjectionMatrix: Float32Array): void
```
Renders the sound source as a visible sphere in the scene.

## Integration with Ray Tracing

### Ray Generation
- Starting point for all ray paths
- Initial energy distribution
- Random direction generation

### Energy Propagation
- Initial energy level affects ray path visibility
- Energy decay based on:
  - Distance traveled
  - Surface absorption
  - Number of reflections

## Usage Example
```typescript
// Create sound source
const soundSource = new SoundSource(device, vec3.fromValues(0, 1.7, 0));

// Update position
soundSource.setPosition(vec3.fromValues(1, 1.7, 2));

// Get current position for ray tracing
const rayOrigin = soundSource.getPosition();
```

## Visualization Details
1. Rendered as a sphere
2. Color indicates active state
3. Size scales with room dimensions
4. Always visible through walls

## Position Constraints
- Must stay within room boundaries
- Height limited by room height
- Updates trigger ray path recalculation
- Smooth interpolation during movement

## Future Enhancements
1. Directional sound emission
2. Frequency-dependent properties
3. Multiple source support
4. Real-time audio integration
5. Advanced visualization options

## Integration with Controls
- UI sliders for position
- Real-time position updates
- Boundary checking
- Visual feedback

## Performance Considerations
1. Efficient position updates
2. Optimized rendering
3. Minimal state changes
4. Cached calculations where possible
