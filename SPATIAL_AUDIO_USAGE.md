# Spatial Audio System Usage Guide

## Overview

This WebGPU-based spatial audio system uses ray tracing to generate realistic impulse responses for 3D audio in a virtual room environment. The system simulates acoustic reflections, frequency-dependent absorption, and creates convolution reverb in real-time.

## Getting Started

### Running the Application

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser to `http://localhost:5173` (or the port shown in the console)

## Using the Spatial Audio System

### GUI Controls

The application includes a dat.GUI interface with the following controls:

#### Room Dimensions
- **Width**: Adjust room width (2-20 meters)
- **Height**: Adjust room height (2-10 meters)  
- **Depth**: Adjust room depth (2-20 meters)

#### Sphere Position
- **X, Y, Z**: Position the sound source (sphere) in 3D space

#### Spatial Audio Panel

##### Enable/Disable
- **Enable Spatial Audio**: Toggle the spatial audio system on/off

##### Ray Tracing Parameters
- **Ray Count**: Number of acoustic rays to trace (512-8192)
- **Max Bounces**: Maximum reflections per ray (5-50)
- **IR Length**: Impulse response duration in seconds (0.5-4.0)

##### Audio Parameters
- **Update Rate**: How often to update audio positions (10-60 Hz)
- **Dry/Wet Mix**: Balance between direct and reverberated sound
- **Master Volume**: Overall audio volume

##### Actions
- **Generate IR**: Click to generate a new impulse response based on current positions
- **Play Test Sound**: Play the loaded audio file
- **Stop Sound**: Stop audio playback
- **Export IR as WAV**: Save the generated impulse response

##### Room Acoustics Info
- **RT60**: Reverberation time (60dB decay)
- **Volume**: Room volume in cubic meters
- **Surface Area**: Total surface area in square meters

### Workflow

1. **Position the Camera**: Use WASD keys to move, arrow keys to look around
2. **Position the Sound Source**: Use the Sphere Position controls
3. **Enable Spatial Audio**: Toggle the checkbox
4. **Generate Impulse Response**: Click "Generate IR" button
5. **Play Test Sound**: Click "Play Test Sound" to hear the result

### Acoustic Materials

The system includes realistic acoustic materials with frequency-dependent properties:

- **Hard Surfaces**: Concrete, brick, plaster
- **Wood**: Hardwood floor, wood paneling
- **Fabric**: Thick carpet, heavy curtains
- **Acoustic Treatment**: Acoustic foam, bass traps, diffusers
- **Special**: Glass, metal, water

Default room configuration uses:
- Floor: Wood floor
- Ceiling: Plaster
- Walls: Plaster

### Technical Details

#### Ray Tracing Process
1. Rays are emitted spherically from the sound source
2. Each ray bounces off room surfaces based on material properties
3. Energy is absorbed according to frequency-dependent coefficients
4. Rays reaching the listener position contribute to the impulse response

#### Audio Processing
1. Impulse response is generated from collected ray data
2. Web Audio API ConvolverNode applies the IR to the audio
3. Real-time position updates adjust spatial parameters

### Performance Considerations

- **Ray Count**: More rays = better quality but slower generation
- **Max Bounces**: Higher values simulate longer reverb tails
- **Update Rate**: Higher rates provide smoother movement but use more CPU

### Troubleshooting

1. **No sound**: 
   - Check master volume
   - Ensure "Enable Spatial Audio" is checked
   - Click "Generate IR" after positioning changes

2. **Poor performance**:
   - Reduce ray count
   - Lower update rate
   - Decrease max bounces

3. **Browser compatibility**:
   - Requires WebGPU support (Chrome/Edge 113+)
   - Web Audio API support required

## Development

### Architecture

The system consists of:

1. **Visualization Layer**: WebGPU-based 3D room rendering
2. **Audio Layer**: Web Audio API integration
3. **Ray Tracing Engine**: GPU-accelerated acoustic simulation
4. **Material System**: Frequency-dependent acoustic properties
5. **UI Layer**: dat.GUI controls

### Key Components

- `SpatialAudioController`: Main integration class
- `AcousticRaytracer`: GPU compute pipeline for ray tracing
- `ImpulseResponseGenerator`: Converts ray data to audio
- `RoomAcoustics`: Material and acoustic property management
- `WebAudioManager`: Audio context and playback control

### Future Enhancements

- [ ] Multiple sound sources
- [ ] Dynamic obstacle support
- [ ] Binaural (HRTF) rendering
- [ ] Real-time ray visualization
- [ ] Custom material editor
- [ ] Preset room configurations
- [ ] Audio file drag-and-drop

## Credits

Developed as part of a spatial audio research project using WebGPU compute shaders for realistic acoustic simulation.
