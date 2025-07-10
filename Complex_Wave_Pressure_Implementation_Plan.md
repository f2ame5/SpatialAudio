# Complex Wave Pressure Implementation Plan

## Overview
Enhance the spatial audio system by representing sound wave pressure as complex numbers instead of simple phase floats. This will enable accurate modeling of constructive and destructive interference when multiple rays are summed at the listener position.

## Current Implementation Analysis

### Existing Phase Tracking
Currently, the system tracks phase as a simple float value:
- **Ray.ts**: `private phase: number` - stores phase as radians
- **RayHit interfaces**: `phase: number` - simple phase value
- **GPU shaders**: Basic phase calculations without interference modeling
- **Impulse response**: Phase used for sine wave generation but no interference

### Limitations of Current Approach
- **No interference modeling**: Rays are summed by amplitude only
- **Missing wave physics**: No representation of wave pressure magnitude and phase
- **Inaccurate spatial audio**: Constructive/destructive interference not modeled
- **Limited realism**: Missing key acoustic phenomena

## Complex Number Wave Representation

### Mathematical Foundation
**Complex Pressure Representation:**
```
P(t) = A * e^(i*φ) = A * (cos(φ) + i*sin(φ))
```
Where:
- `A` = amplitude (magnitude)
- `φ` = phase (angle)
- `i` = imaginary unit

**Benefits:**
- **Automatic interference**: Adding complex numbers naturally models interference
- **Phase relationships**: Preserved through all calculations
- **Wave physics**: Accurate representation of acoustic waves
- **Spatial accuracy**: Proper modeling of wave interactions

## Task Breakdown

### 1. Create Complex Number Infrastructure
**Goal:** Establish foundation for complex number operations in the spatial audio system.

#### 1.1 Create Complex Number Utility Class
- **Description:** Implement a robust complex number class for wave calculations
- **Implementation Details:**
  ```typescript
  class Complex {
    real: number;
    imag: number;
    
    constructor(real: number, imag: number)
    static fromPolar(magnitude: number, phase: number): Complex
    static fromCartesian(real: number, imag: number): Complex
    
    add(other: Complex): Complex
    multiply(other: Complex): Complex
    magnitude(): number
    phase(): number
    conjugate(): Complex
  }
  ```
- **Features Required:**
  - Polar and Cartesian coordinate conversion
  - Basic arithmetic operations (add, multiply, subtract)
  - Magnitude and phase extraction
  - Efficient memory management
- **Files:** `src/utils/complex.ts` (new)

#### 1.2 Create Wave Pressure Class
- **Description:** Specialized class for representing acoustic wave pressure
- **Implementation Details:**
  ```typescript
  class WavePressure {
    pressure: Complex;
    frequency: number;
    
    constructor(amplitude: number, phase: number, frequency: number)
    static fromAmplitudePhase(amp: number, phase: number, freq: number): WavePressure
    
    addWave(other: WavePressure): WavePressure
    attenuate(factor: number): WavePressure
    shiftPhase(phaseShift: number): WavePressure
    calculateInterference(waves: WavePressure[]): WavePressure
  }
  ```
- **Features Required:**
  - Automatic frequency matching for interference
  - Wave superposition calculations
  - Attenuation and phase shift operations
  - Doppler shift support
- **Files:** `src/utils/wave-pressure.ts` (new)

#### 1.3 Add GPU Complex Number Support
- **Description:** Implement complex number operations in WGSL shaders
- **Implementation Details:**
  ```wgsl
  struct Complex {
    real: f32,
    imag: f32
  }
  
  fn complex_add(a: Complex, b: Complex) -> Complex
  fn complex_multiply(a: Complex, b: Complex) -> Complex
  fn complex_magnitude(c: Complex) -> f32
  fn complex_phase(c: Complex) -> f32
  fn complex_from_polar(magnitude: f32, phase: f32) -> Complex
  ```
- **Features Required:**
  - Efficient GPU complex arithmetic
  - Polar/Cartesian conversion functions
  - Memory-aligned structures
- **Files:** `src/raytracer/shaders/complex-math.wgsl` (new)

### 2. Update Ray Class for Complex Pressure
**Goal:** Modify the Ray class to use complex wave pressure instead of simple phase tracking.

#### 2.1 Replace Phase with Wave Pressure
- **Description:** Update Ray class properties to use WavePressure objects
- **Changes Required:**
  - Remove: `private phase: number`
  - Add: `private wavePressure: WavePressure`
  - Update constructor to initialize wave pressure
  - Modify all phase-related methods
- **Implementation Details:**
  ```typescript
  class Ray {
    private wavePressure: WavePressure;
    
    constructor(origin: vec3, direction: vec3, initialAmplitude: number, frequency: number) {
      this.wavePressure = new WavePressure(initialAmplitude, 0, frequency);
    }
    
    getWavePressure(): WavePressure
    updateWavePressure(newPressure: WavePressure): void
  }
  ```
- **Files:** `src/raytracer/ray.ts`

#### 2.2 Update Ray Energy Calculations
- **Description:** Modify energy calculations to work with complex pressure
- **Changes Required:**
  - Energy = |pressure|² (magnitude squared)
  - Update frequency-dependent energy calculations
  - Maintain energy conservation laws
- **Implementation Details:**
  ```typescript
  public getEnergy(): number {
    return Math.pow(this.wavePressure.pressure.magnitude(), 2);
  }
  
  public getEnergyForBand(frequency: number): number {
    // Calculate energy for specific frequency band
    return this.wavePressure.getEnergyAtFrequency(frequency);
  }
  ```
- **Files:** `src/raytracer/ray.ts`

#### 2.3 Update Ray Propagation Logic
- **Description:** Modify ray propagation to properly handle wave pressure evolution
- **Changes Required:**
  - Phase accumulation through distance
  - Amplitude attenuation with air absorption
  - Proper wave pressure updates during bounces
- **Implementation Details:**
  ```typescript
  public updateRay(newOrigin: vec3, newDirection: vec3, distance: number, 
                   materialReflection: Complex, temperature: number, humidity: number): void {
    // Calculate phase change due to propagation
    const phaseChange = 2 * Math.PI * this.frequency * distance / SPEED_OF_SOUND;
    
    // Apply air absorption and material reflection
    const airAbsorption = this.calculateAirAbsorption(distance, temperature, humidity);
    const newPressure = this.wavePressure.pressure
      .multiply(airAbsorption)
      .multiply(materialReflection)
      .multiply(Complex.fromPolar(1, phaseChange));
    
    this.wavePressure = new WavePressure(newPressure.magnitude(), newPressure.phase(), this.frequency);
  }
  ```
- **Files:** `src/raytracer/ray.ts`

### 3. Update Material Reflection Models
**Goal:** Enhance material reflection to return complex reflection coefficients instead of simple absorption values.

#### 3.1 Create Complex Material Properties
- **Description:** Extend material definitions to include complex reflection coefficients
- **Changes Required:**
  - Add complex reflection coefficients for each frequency band
  - Include phase shifts caused by material properties
  - Support frequency-dependent reflection characteristics
- **Implementation Details:**
  ```typescript
  interface WallMaterial {
    // Existing properties...
    reflectionCoefficients: {
      band63: Complex,
      band125: Complex,
      band250: Complex,
      band500: Complex,
      band1k: Complex,
      band2k: Complex,
      band4k: Complex,
      band8k: Complex
    };
    phaseShiftOnReflection: number; // Additional phase shift
  }
  ```
- **Files:** `src/room/room-materials.ts`

#### 3.2 Update Material Reflection Calculations
- **Description:** Modify reflection calculations to return complex coefficients
- **Changes Required:**
  - Calculate complex reflection based on material properties
  - Include surface impedance effects
  - Handle frequency-dependent phase shifts
- **Implementation Details:**
  ```typescript
  public getComplexReflectionCoefficient(frequency: number, incidentAngle: number): Complex {
    const band = this.getFrequencyBand(frequency);
    const baseReflection = this.reflectionCoefficients[band];
    
    // Apply angle-dependent effects
    const angleEffect = Math.cos(incidentAngle);
    const magnitude = baseReflection.magnitude() * angleEffect;
    const phase = baseReflection.phase() + this.phaseShiftOnReflection;
    
    return Complex.fromPolar(magnitude, phase);
  }
  ```
- **Files:** `src/room/room-materials.ts`

#### 3.3 Update GPU Material Shaders
- **Description:** Modify GPU shaders to handle complex material properties
- **Changes Required:**
  - Update material uniform structures
  - Implement complex reflection calculations in shaders
  - Ensure proper data transfer from CPU
- **Files:** 
  - `src/raytracer/shaders/spatial_audio.wgsl`
  - `src/raytracer/shaders/raytracer.wgsl`

### 4. Implement Wave Interference at Listener
**Goal:** Create accurate wave interference modeling when multiple rays reach the listener position.

#### 4.1 Create Wave Interference Calculator
- **Description:** Implement system to sum complex wave pressures at listener position
- **Implementation Details:**
  ```typescript
  class WaveInterferenceCalculator {
    static calculateInterference(waves: WavePressure[]): WavePressure {
      // Group waves by frequency
      const frequencyGroups = this.groupByFrequency(waves);
      
      // Sum complex pressures for each frequency
      const resultWaves = frequencyGroups.map(group => {
        const totalPressure = group.reduce((sum, wave) => 
          sum.add(wave.pressure), new Complex(0, 0));
        return new WavePressure(totalPressure.magnitude(), totalPressure.phase(), group[0].frequency);
      });
      
      return this.combineFrequencyBands(resultWaves);
    }
    
    static calculateSpatialInterference(waves: WavePressure[], listenerPos: vec3, 
                                       sourcePositions: vec3[]): WavePressure {
      // Account for path differences and resulting phase shifts
      const adjustedWaves = waves.map((wave, index) => {
        const pathDifference = vec3.distance(listenerPos, sourcePositions[index]);
        const phaseShift = 2 * Math.PI * wave.frequency * pathDifference / SPEED_OF_SOUND;
        return wave.shiftPhase(phaseShift);
      });
      
      return this.calculateInterference(adjustedWaves);
    }
  }
  ```
- **Files:** `src/sound/wave-interference-calculator.ts` (new)

#### 4.2 Update Spatial Audio Processor
- **Description:** Modify spatial audio processing to use wave interference calculations
- **Changes Required:**
  - Replace simple amplitude summing with complex wave interference
  - Handle multiple rays arriving at listener simultaneously
  - Account for path differences and phase relationships
- **Implementation Details:**
  ```typescript
  public processSpatialAudio(rayHits: RayHit[], listenerPos: vec3): [Float32Array, Float32Array] {
    // Group ray hits by arrival time windows
    const timeWindows = this.groupByArrivalTime(rayHits);
    
    // Calculate interference for each time window
    const interferenceResults = timeWindows.map(window => {
      const waves = window.map(hit => hit.wavePressure);
      const positions = window.map(hit => hit.position);
      return WaveInterferenceCalculator.calculateSpatialInterference(waves, listenerPos, positions);
    });
    
    // Generate final impulse response with interference effects
    return this.generateInterferenceBasedIR(interferenceResults);
  }
  ```
- **Files:** `src/sound/spatial-audio-processor.ts`

#### 4.3 Update GPU Interference Calculations
- **Description:** Implement wave interference calculations in GPU shaders
- **Changes Required:**
  - Add complex wave summation in shaders
  - Handle multiple ray contributions per pixel/sample
  - Optimize for parallel processing
- **Implementation Details:**
  ```wgsl
  fn calculateWaveInterference(waves: array<Complex>, count: u32) -> Complex {
    var result = Complex(0.0, 0.0);
    for (var i = 0u; i < count; i++) {
      result = complex_add(result, waves[i]);
    }
    return result;
  }
  
  fn processRayInterference(rayIndex: u32) {
    // Collect all rays arriving at similar times
    var waveSum = Complex(0.0, 0.0);
    var rayCount = 0u;
    
    for (var i = 0u; i < totalRays; i++) {
      if (abs(rayHits[i].time - rayHits[rayIndex].time) < TIME_WINDOW) {
        let wave = complex_from_polar(rayHits[i].energy, rayHits[i].phase);
        waveSum = complex_add(waveSum, wave);
        rayCount++;
      }
    }
    
    // Store interference result
    interferenceResults[rayIndex] = waveSum;
  }
  ```
- **Files:** `src/raytracer/shaders/spatial_audio.wgsl`

### 5. Update Impulse Response Generation
**Goal:** Modify impulse response generation to account for wave interference effects.

#### 5.1 Complex-Based Impulse Response
- **Description:** Generate impulse responses using complex wave pressure data
- **Changes Required:**
  - Use complex pressure values instead of simple amplitudes
  - Account for constructive/destructive interference
  - Maintain phase relationships in final audio output
- **Implementation Details:**
  ```typescript
  public generateComplexImpulseResponse(interferenceData: WavePressure[], 
                                       sampleRate: number): [Float32Array, Float32Array] {
    const leftIR = new Float32Array(this.getIRLength(sampleRate));
    const rightIR = new Float32Array(this.getIRLength(sampleRate));
    
    interferenceData.forEach((wave, index) => {
      const time = wave.arrivalTime;
      const sampleIndex = Math.floor(time * sampleRate);
      
      if (sampleIndex < leftIR.length) {
        // Convert complex pressure to real audio sample
        const realPart = wave.pressure.real;
        const imagPart = wave.pressure.imag;
        
        // Apply HRTF and spatial processing
        const [leftSample, rightSample] = this.applySpatialProcessing(realPart, imagPart, wave);
        
        leftIR[sampleIndex] += leftSample;
        rightIR[sampleIndex] += rightSample;
      }
    });
    
    return [leftIR, rightIR];
  }
  ```
- **Files:** `src/raytracer/raytracer.ts`

#### 5.2 Phase-Coherent Audio Processing
- **Description:** Ensure phase coherence is maintained throughout audio processing chain
- **Changes Required:**
  - Preserve phase relationships during filtering
  - Handle complex-to-real conversion properly
  - Maintain stereo phase coherence
- **Files:** `src/sound/audio-processor.ts`

## Implementation Phases

### Phase 1: Infrastructure (Tasks 1.1-1.3)
**Duration:** 1-2 weeks
1. Create Complex Number Utility Class
2. Create Wave Pressure Class  
3. Add GPU Complex Number Support

### Phase 2: Ray System Update (Tasks 2.1-2.3)
**Duration:** 1-2 weeks
4. Replace Phase with Wave Pressure
5. Update Ray Energy Calculations
6. Update Ray Propagation Logic

### Phase 3: Material Enhancement (Tasks 3.1-3.3)
**Duration:** 1 week
7. Create Complex Material Properties
8. Update Material Reflection Calculations
9. Update GPU Material Shaders

### Phase 4: Interference Implementation (Tasks 4.1-4.3)
**Duration:** 2-3 weeks
10. Create Wave Interference Calculator
11. Update Spatial Audio Processor
12. Update GPU Interference Calculations

### Phase 5: Audio Output (Tasks 5.1-5.2)
**Duration:** 1 week
13. Complex-Based Impulse Response
14. Phase-Coherent Audio Processing

## Expected Benefits

### Acoustic Accuracy
- **Realistic interference patterns**: Proper constructive/destructive interference
- **Accurate wave physics**: Complex pressure representation matches real acoustics
- **Spatial precision**: Better localization through phase relationships
- **Frequency response**: More accurate frequency-dependent effects

### Audio Quality Improvements
- **Natural sound**: Interference creates more realistic audio
- **Spatial immersion**: Better 3D audio positioning
- **Dynamic range**: Proper modeling of quiet zones (destructive interference)
- **Frequency clarity**: Accurate frequency-dependent spatial effects

## Technical Considerations

### Performance Impact
- **Memory usage**: Complex numbers require 2x storage (real + imaginary)
- **Computation**: More complex arithmetic operations
- **GPU efficiency**: Need to optimize complex number operations
- **Caching**: Implement efficient caching for repeated calculations

### Validation Strategy
- **Unit tests**: Test complex number operations against known results
- **Wave interference tests**: Verify constructive/destructive interference
- **Audio comparison**: A/B testing with and without complex pressure
- **Acoustic validation**: Compare with real-world acoustic measurements

### Migration Strategy
- **Backward compatibility**: Maintain option to use simple phase tracking
- **Gradual rollout**: Implement and test one component at a time
- **Performance monitoring**: Track performance impact during implementation
- **Fallback options**: Provide simpler alternatives for low-performance devices
