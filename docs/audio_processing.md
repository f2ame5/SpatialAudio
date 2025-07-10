# Audio Processing System

## Overview
The audio processing system converts ray tracing data into an audio impulse response using GPU-accelerated computation. The system consists of three main components:

1. Audio Processing Shader (`audio_processing.wgsl`)
2. Audio Processor (`audio-processor.ts`)
3. Histogram Visualizer (`histogram-visualizer.ts`)

## Components

### Audio Processing Shader
Located in `src/shaders/audio_processing.wgsl`

The shader accumulates ray hit energy into time-frequency bins to create a room impulse response.

#### Key Structures
- `AudioHistogram`: Configuration for the histogram generation
- `RayHit`: Structure containing ray hit information including position, energy, and time

#### Bindings
- `@binding(0)`: Ray hits buffer (read)
- `@binding(1)`: Energy histogram buffer (read/write)
- `@binding(2)`: Histogram configuration (uniform)

### Audio Processor
Located in `src/sound/audio-processor.ts`

Manages the audio processing pipeline, coordinating between ray tracing data and audio output.

#### Key Features
- GPU-accelerated histogram generation
- WebAudio API integration
- Real-time visualization updates
- Poisson noise-based audio generation

#### Methods
- `processRayHits`: Process ray intersection data into an energy histogram
- `generateAudioFromHistogram`: Convert energy histogram to audio samples
- `playAudio`: Play the generated audio through WebAudio API

### Histogram Visualizer
Located in `src/visualization/histogram-visualizer.ts`

Provides real-time visualization of the room impulse response.

#### Features
- Multi-band frequency visualization
- Time-domain energy display
- Interactive grid with time and amplitude labels
- High DPI support

#### Color Scheme
- Low frequency: Dark blue (#2C3E50)
- Mid frequency: Red (#E74C3C)
- High frequency: Light blue (#3498DB)
- Combined energy: Green (#2ECC71)

## Data Flow
1. Ray tracer generates hit information (position, energy, time)
2. Audio processor accumulates hits into time-frequency histogram
3. Histogram data is used for both visualization and audio generation
4. Audio is generated using Poisson noise modulated by energy values
5. Visualization updates in real-time showing energy distribution

## Configuration
The audio processing system can be configured through the `AudioProcessorConfig` interface:
- `sampleRate`: Audio sample rate (default: 44100 Hz)
- `histogramTimeStep`: Time resolution for histogram bins
- `maxTime`: Maximum simulation time
- `numFreqBands`: Number of frequency bands (default: 3)
