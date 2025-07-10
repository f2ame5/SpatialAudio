# Sound Source Controls Documentation

## Overview
The Sound Source Controls system provides an interactive UI for positioning the sound source within the 3D room space. It automatically synchronizes with room dimensions and ensures the source always remains within valid bounds.

## Features

### Position Control
- Real-time 3D positioning using sliders
- Automatic range adjustment based on room dimensions
- Position clamping to keep source within room bounds
- Synchronized visual feedback

### Room Integration
- Slider ranges automatically update with room dimensions
- X-axis: [-width/2, width/2]
- Y-axis: [-height/2, height/2]
- Z-axis: [-depth/2, depth/2]

### UI Components
- Position sliders for X, Y, Z axes
- Real-time value display
- Clean, modern interface
- Semi-transparent overlay

## Implementation

### Class: SourceControls

#### Constructor
```typescript
constructor(
    initialPosition: vec3,
    roomDimensions: RoomDimensions,
    events: SourceEvents
)
```

#### Properties
```typescript
interface SourceEvents {
    onPositionChange: (position: vec3) => void;
}
```

#### Methods

##### updateRoomDimensions
```typescript
updateRoomDimensions(dimensions: RoomDimensions): void
```
Updates slider ranges when room dimensions change and ensures source position remains valid.

##### setPosition
```typescript
private setPosition(position: vec3): void
```
Updates source position and UI elements, triggers position change event.

##### clampPosition
```typescript
private clampPosition(position: vec3): void
```
Ensures position stays within room boundaries.

## Usage Example

```typescript
// Create source controls
const sourceControls = new SourceControls(
    initialPosition,
    roomDimensions,
    {
        onPositionChange: (position) => {
            soundSource.setPosition(position);
        }
    }
);

// Update when room changes
sourceControls.updateRoomDimensions(newDimensions);
```

## Integration with Room

The source controls automatically integrate with the room system:
1. Initial slider ranges are set based on room dimensions
2. When room dimensions change:
   - Slider ranges update automatically
   - Source position is clamped if needed
   - UI updates to reflect new valid ranges

## UI Layout

```
┌─────────────────────────┐
│   Sound Source Position │
│                        │
│   X: [-9.0 ─╥─ 9.0]   │
│              ║         │
│   Y: [-4.5 ─╨─ 4.5]   │
│              ▲         │
│   Z: [-9.0 ──┘ 9.0]   │
└─────────────────────────┘
```

## Best Practices
1. Always initialize with valid room dimensions
2. Connect to sound source update events
3. Update dimensions when room changes
4. Use vec3 for position management
