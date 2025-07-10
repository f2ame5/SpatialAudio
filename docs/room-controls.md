# Room Controls Documentation

## Overview
The `RoomControls` class provides a user interface for modifying room dimensions, material properties, and triggering ray tracing calculations.

## Key Components

### RoomControls Class
Main class for managing room configuration UI.

#### Constructor
```typescript
constructor(config: RoomConfig, events: RoomEvents)
```
- `config`: Initial room configuration
- `events`: Event handlers for UI interactions

### Configuration Interface
```typescript
interface RoomConfig {
    dimensions: RoomDimensions;
    materials: RoomMaterials;
}

interface RoomDimensions {
    width: number;
    height: number;
    depth: number;
}

interface RoomMaterials {
    walls: { absorption: number };
    ceiling: { absorption: number };
    floor: { absorption: number };
}
```

### Event Interface
```typescript
interface RoomEvents {
    onDimensionsChange: (dimensions: RoomDimensions) => void;
    onMaterialsChange: (materials: RoomMaterials) => void;
    calculateIR: () => void;
}
```

## UI Elements

### Dimension Controls
1. Width slider (1-20m)
2. Height slider (1-10m)
3. Depth slider (1-20m)

### Material Controls
1. Wall absorption (0-1)
2. Ceiling absorption (0-1)
3. Floor absorption (0-1)

### Action Buttons
1. Calculate IR (triggers ray tracing)
2. Reset to defaults

## Implementation Details

### Dimension Updates
1. Validate input ranges
2. Update room geometry
3. Update ray tracing boundaries
4. Clear previous ray paths
5. Refresh visualization

### Material Updates
1. Update absorption coefficients
2. Recalculate energy decay
3. Update visualization

### Ray Tracing Trigger
1. Clear previous paths
2. Calculate new ray paths
3. Update visualization
4. Show progress indicator

## Event Handling

### Dimension Changes
```typescript
private handleDimensionChange(dimension: keyof RoomDimensions, value: number) {
    this.config.dimensions[dimension] = value;
    this.events.onDimensionsChange(this.config.dimensions);
}
```

### Material Changes
```typescript
private handleMaterialChange(surface: keyof RoomMaterials, value: number) {
    this.config.materials[surface].absorption = value;
    this.events.onMaterialsChange(this.config.materials);
}
```

## Usage Example
```typescript
const roomControls = new RoomControls(
    {
        dimensions: { width: 10, height: 5, depth: 8 },
        materials: {
            walls: { absorption: 0.1 },
            ceiling: { absorption: 0.12 },
            floor: { absorption: 0.15 }
        }
    },
    {
        onDimensionsChange: (dimensions) => {
            room.updateDimensions(dimensions);
            rayTracer.updateBoundaries();
        },
        onMaterialsChange: (materials) => {
            room.updateMaterials(materials);
        },
        calculateIR: () => {
            rayTracer.calculateRayPaths();
        }
    }
);
```

## Performance Considerations

### Input Debouncing
- Delay updates for rapid changes
- Batch dimension updates
- Throttle ray tracing triggers

### State Management
- Cache current values
- Update only changed properties
- Minimize redraws

## UI/UX Features
1. Real-time feedback
2. Input validation
3. Visual feedback for changes
4. Progress indicators
5. Error handling

## Future Enhancements
1. Material presets
2. Advanced material properties
3. Room templates
4. Undo/redo support
5. Configuration save/load
