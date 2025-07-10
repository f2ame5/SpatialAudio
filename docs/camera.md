# Camera Documentation

## Overview
The `Camera` class provides first-person navigation in the 3D scene, handling user input for movement and rotation.

## Key Components

### Camera Class
Main class for managing view and movement.

#### Constructor
```typescript
constructor(position: vec3, yaw: number = 0, pitch: number = 0)
```
- `position`: Initial 3D position
- `yaw`: Horizontal rotation (default: 0)
- `pitch`: Vertical rotation (default: 0)

### Properties
- `position`: Current camera position
- `yaw`: Horizontal rotation angle
- `pitch`: Vertical rotation angle
- `front`: View direction vector
- `up`: Up vector (0, 1, 0)

## Methods

### Movement
```typescript
moveForward(distance: number): void
moveRight(distance: number): void
moveUp(distance: number): void
```
Moves camera in specified direction relative to view.

### Rotation
```typescript
rotate(deltaX: number, deltaY: number): void
```
Updates camera rotation based on mouse movement.

### View Matrix
```typescript
getViewMatrix(): mat4
getViewProjection(aspect: number): mat4
```
Gets current view and projection matrices.

## Controls

### Keyboard Movement
- `W`: Move forward
- `S`: Move backward
- `A`: Strafe left
- `D`: Strafe right
- `Space`: Move up
- `Shift`: Move down

### Mouse Look
- Mouse movement controls view direction
- Vertical rotation limited to prevent flipping
- Smooth interpolation for natural feel

## Implementation Details

### View Calculation
1. Update rotation angles
2. Calculate front vector:
   ```typescript
   front.x = cos(yaw) * cos(pitch)
   front.y = sin(pitch)
   front.z = sin(yaw) * cos(pitch)
   ```
3. Normalize vectors
4. Generate view matrix

### Movement System
1. Get movement direction
2. Scale by delta time
3. Apply to position
4. Update view matrix

## Integration

### With Ray Tracing
- Camera position affects ray visualization
- View matrix used for ray path rendering
- Proper depth testing with rays

### With Room
- Camera starts inside room
- Movement can be constrained to room
- Proper occlusion with room geometry

## Performance Considerations

### Matrix Calculations
- Cache view matrix until needed
- Use gl-matrix for optimized math
- Minimize matrix operations

### State Updates
- Only update when necessary
- Batch position/rotation updates
- Efficient uniform updates

## Usage Example
```typescript
// Create camera
const camera = new Camera(
    vec3.fromValues(0, 1.7, 3), // Eye level
    -90,  // Looking forward
    0     // Level view
);

// Handle mouse movement
canvas.addEventListener('mousemove', (e) => {
    camera.rotate(e.movementX, e.movementY);
});

// Update movement
function update(deltaTime: number) {
    if (keys.W) camera.moveForward(deltaTime);
    if (keys.S) camera.moveForward(-deltaTime);
    if (keys.A) camera.moveRight(-deltaTime);
    if (keys.D) camera.moveRight(deltaTime);
}
```

## Future Enhancements
1. Smooth acceleration/deceleration
2. Collision detection with room
3. Camera path recording
4. VR/AR support
5. Multiple camera views

## Debug Features
1. Position display
2. Rotation angles
3. Movement speed control
4. View frustum visualization
