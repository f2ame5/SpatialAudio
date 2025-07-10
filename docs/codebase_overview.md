# WebGPU Spatial Audio Simulation

## Project Overview
A real-time 3D acoustic simulation system that demonstrates sound propagation in virtual environments using WebGPU. The project combines ray tracing techniques with edge diffraction modeling to create accurate spatial audio representations.

## Key Features
- Real-time 3D sound propagation visualization
- Physical modeling of sound reflection and diffraction
- Material-based acoustic absorption simulation
- Interactive room configuration
- Real-time waveform visualization
- Support for multiple sound sources and receivers

## File Structure
```
spatialAudio/
├── src/
│   ├── main.ts                 # Application entry point
│   ├── camera/
│   │   └── camera.ts          # Camera controls and view management
│   ├── raytracer/
│   │   ├── ray-renderer.ts    # WebGPU ray visualization
│   │   ├── raytracer.ts       # Core ray tracing logic
│   │   └── shaders.wgsl       # Ray rendering shaders
│   ├── room/
│   │   ├── room.ts           # Room geometry and properties
│   │   ├── wall.ts           # Wall surface implementations
│   │   └── edge.ts           # Edge diffraction handling
│   └── sound/
│       ├── audio-processor.ts # Audio processing and IR generation
│       └── waveform.ts       # Audio waveform visualization
└── public/
    └── assets/               # Audio samples and textures
```

## Core Components Explained

### Main Application (main.ts)
- Initializes WebGPU context and render pipeline
- Manages simulation loop and component coordination
- Handles user input and GUI interactions
```typescript
// Key initialization
const renderer = new RayRenderer(device, canvas);
const raytracer = new Raytracer(room);
const audioProcessor = new AudioProcessor(context);
```

### Ray Tracing System (raytracer/raytracer.ts)
- Implements ray casting and reflection algorithms
- Calculates energy attenuation and absorption
- Handles ray-surface intersections
```typescript
// Example ray tracing configuration
interface RayConfig {
    maxBounces: number;    // Maximum reflection bounces
    raysPerSource: number; // Ray density
    energyThreshold: number; // Minimum energy for propagation
}
```

### Room Acoustics (room/room.ts)
- Manages room geometry and material properties
- Implements acoustic surface behaviors
- Handles real-time room modifications
```typescript
// Material acoustic properties
interface MaterialProperties {
    absorption: {
        low: number;   // Low frequency absorption (0-1)
        mid: number;   // Mid frequency absorption (0-1)
        high: number;  // High frequency absorption (0-1)
    }
    diffusion: number; // Surface diffusion coefficient
}
```

## Usage Example

```typescript
// Basic setup
const simulation = new SpatialAudioSimulation({
    roomDimensions: [10, 8, 6],
    materials: predefinedMaterials,
    sourcePosition: [2, 1.7, 3]
});

// Run simulation
simulation.start();

// Add sound source
simulation.addSource({
    position: [5, 1, 2],
    audioBuffer: audioData,
    gain: 0.8
});
```

## Performance Considerations
- Ray count vs. performance tradeoffs
- WebGPU buffer management strategies
- Audio processing optimizations

## Debug Features
- Real-time parameter adjustment
- Ray visualization options
- Energy distribution display
- Room configuration tools

## Future Enhancements
- [ ] Multi-threaded ray computation
- [ ] Advanced diffraction models
- [ ] HRTF integration
- [ ] GPU-accelerated audio processing

## Build Instructions
```bash
# Install dependencies
npm install

# Development
npm run dev

# Production build
npm run build
```