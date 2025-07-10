# ISO 9613-1 Air Absorption Model Integration Plan

## Overview
Comprehensive plan to fully integrate the ISO 9613-1 standard air absorption model across all components of the spatial audio system to ensure consistent and accurate frequency-dependent air absorption calculations.

## Current Status
✅ **Already Implemented:**
- ISO 9613-1 calculation in Ray.ts for 3 frequency bands
- Basic air absorption in ray bounces
- GPU shader with 8-band structure
- Temperature and humidity consideration in Ray class

⚠️ **Issues to Address:**
- Inconsistent frequency band coverage (3 vs 8 bands)
- Simplified impulse response air absorption
- Missing GPU-CPU synchronization
- No real-time environmental parameter control

## Task Breakdown

### 1. Expand Ray.ts Model to 8 Frequency Bands
**Goal:** Modify the Ray class to handle all 8 frequency bands (63Hz, 125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz, 8kHz) instead of the current 3 bands.

#### 1.1 Update Ray Class Properties
- **Description:** Add energy properties for all 8 frequency bands
- **Changes Required:**
  - Add properties: `energy63`, `energy125`, `energy250`, `energy500`, `energy1k`, `energy2k`, `energy4k`, `energy8k`
  - Update constructor to initialize all 8 bands
  - Remove old `energyLow`, `energyMid`, `energyHigh` properties
- **Files:** `src/raytracer/ray.ts`

#### 1.2 Modify calculateAirAbsorption Method
- **Description:** Update ISO 9613-1 calculation to return coefficients for all 8 frequency bands
- **Changes Required:**
  - Extend return type to include all 8 bands
  - Use proper center frequencies: 63, 125, 250, 500, 1000, 2000, 4000, 8000 Hz
  - Maintain existing ISO 9613-1 formula accuracy
- **Files:** `src/raytracer/ray.ts`

#### 1.3 Update Ray Energy Update Logic
- **Description:** Modify updateRay method to apply air absorption to all 8 frequency bands
- **Changes Required:**
  - Update `updateRay()` method to handle 8-band energy loss
  - Add getter methods: `getEnergy63()`, `getEnergy125()`, etc.
  - Update `getEnergy()` to return average of all 8 bands
- **Files:** `src/raytracer/ray.ts`

#### 1.4 Update Ray Hit Interfaces
- **Description:** Modify RayHit interfaces across the codebase to include all 8 frequency band energy values
- **Changes Required:**
  - Update RayHit interface definitions
  - Modify ray hit data collection in raytracer
  - Update audio processor to handle 8-band data
- **Files:** 
  - `src/raytracer/raytracer.ts`
  - `src/sound/spatial-audio-processor.ts`
  - `src/sound/audio-processor.ts`

### 2. Synchronize GPU Shader Coefficients
**Goal:** Ensure GPU shader air absorption coefficients are calculated using the same ISO 9613-1 formulas as the CPU implementation.

#### 2.1 Create Air Absorption Coefficient Calculator
- **Description:** Develop a shared utility function for ISO 9613-1 calculations
- **Changes Required:**
  - Create new module: `src/utils/air-absorption-calculator.ts`
  - Implement ISO 9613-1 formula for all 8 frequency bands
  - Export functions usable by both CPU and GPU code
  - Include temperature and humidity parameters
- **Files:** `src/utils/air-absorption-calculator.ts` (new)

#### 2.2 Update GPU Shader Uniforms
- **Description:** Modify RoomAcoustics uniform structure to include dynamically calculated coefficients
- **Changes Required:**
  - Update `RoomAcoustics` struct in shader
  - Replace static absorption values with dynamic ones
  - Ensure proper data alignment for GPU
- **Files:** `src/raytracer/shaders/spatial_audio.wgsl`

#### 2.3 Implement Coefficient Transfer to GPU
- **Description:** Create mechanism to calculate and transfer coefficients to GPU uniforms
- **Changes Required:**
  - Add coefficient calculation in render loop
  - Update uniform buffer with calculated values
  - Ensure efficient transfer (only when parameters change)
- **Files:** 
  - `src/main.ts`
  - `src/raytracer/raytracer.ts`

### 3. Improve Impulse Response Generation
**Goal:** Replace simplified air absorption model with full ISO 9613-1 implementation in impulse response generation.

#### 3.1 Update Impulse Response Air Absorption
- **Description:** Replace simplified exponential decay with proper ISO 9613-1 calculations
- **Changes Required:**
  - Remove: `Math.exp(-0.1 * dopplerFrequency * point.time)`
  - Add: Proper ISO 9613-1 calculation using shared utility
  - Consider distance, temperature, and humidity
- **Files:** `src/raytracer/raytracer.ts`

#### 3.2 Implement Frequency-Dependent Processing
- **Description:** Process each frequency band separately with specific air absorption characteristics
- **Changes Required:**
  - Create separate impulse responses for each frequency band
  - Apply band-specific air absorption coefficients
  - Combine bands for final stereo output
- **Files:** `src/raytracer/raytracer.ts`

#### 3.3 Update Audio Processor Integration
- **Description:** Ensure audio processor handles 8-band frequency data correctly
- **Changes Required:**
  - Update convolution to process 8 frequency bands
  - Apply proper frequency-dependent filtering
  - Maintain phase relationships between bands
- **Files:** `src/sound/audio-processor.ts`

### 4. Add Environmental Parameters to GPU Pipeline
**Goal:** Extend GPU processing pipeline to accept and use temperature and humidity parameters for real-time calculations.

#### 4.1 Extend Room Class Environmental Data
- **Description:** Add methods for setting/getting temperature and humidity with validation
- **Changes Required:**
  - Add `setTemperature(temp: number)` with validation (-40°C to 50°C)
  - Add `setHumidity(humidity: number)` with validation (0% to 100%)
  - Add change event notifications for real-time updates
  - Set reasonable defaults (20°C, 50% RH)
- **Files:** `src/room/room.ts`

#### 4.2 Update GPU Uniform Structures
- **Description:** Modify GPU uniform structures to include environmental parameters
- **Changes Required:**
  - Add temperature and humidity to uniform structs
  - Update all shader bindings
  - Ensure proper data layout and alignment
- **Files:** 
  - `src/raytracer/shaders/spatial_audio.wgsl`
  - `src/raytracer/shaders/raytracer.wgsl`

#### 4.3 Implement Real-time Coefficient Updates
- **Description:** Create system to recalculate coefficients when environmental parameters change
- **Changes Required:**
  - Add change detection for temperature/humidity
  - Implement efficient caching to avoid unnecessary recalculations
  - Update GPU uniforms only when needed
- **Files:** 
  - `src/main.ts`
  - `src/room/room.ts`

#### 4.4 Add Environmental Controls to UI
- **Description:** Implement user interface controls for adjusting environmental parameters
- **Changes Required:**
  - Add temperature slider/input field
  - Add humidity slider/input field
  - Show real-time feedback on air absorption effects
  - Add visual indicators for parameter ranges
- **Files:** 
  - `index.html`
  - `src/main.ts`
  - Add CSS styling for controls

## Implementation Order

### Phase 1: Foundation (Tasks 1.1-1.4)
1. Update Ray Class Properties
2. Modify calculateAirAbsorption Method
3. Update Ray Energy Update Logic
4. Update Ray Hit Interfaces

### Phase 2: Synchronization (Tasks 2.1-2.3)
5. Create Air Absorption Coefficient Calculator
6. Update GPU Shader Uniforms
7. Implement Coefficient Transfer to GPU

### Phase 3: Impulse Response (Tasks 3.1-3.3)
8. Update Impulse Response Air Absorption
9. Implement Frequency-Dependent Processing
10. Update Audio Processor Integration

### Phase 4: Real-time Control (Tasks 4.1-4.4)
11. Extend Room Class Environmental Data
12. Update GPU Uniform Structures
13. Implement Real-time Coefficient Updates
14. Add Environmental Controls to UI

## Expected Benefits

### Technical Improvements
- **Consistent 8-band frequency processing** across CPU and GPU
- **Accurate ISO 9613-1 calculations** throughout the pipeline
- **Proper frequency-dependent attenuation** in all components
- **Real-time environmental parameter control**

### Audio Quality Improvements
- **More realistic spatial audio** with proper air absorption
- **Better frequency response** across the audible spectrum
- **Improved realism** for different environmental conditions
- **Enhanced immersion** through accurate acoustic modeling

## Testing Strategy

### Unit Tests
- Test ISO 9613-1 calculations against reference values
- Verify frequency band energy conservation
- Test environmental parameter validation

### Integration Tests
- Compare CPU and GPU air absorption results
- Test impulse response generation with different parameters
- Verify real-time parameter updates

### Audio Tests
- A/B testing with and without improved air absorption
- Frequency response analysis
- Subjective listening tests in different virtual environments

## Dependencies

### External
- WebGPU support
- Web Audio API
- gl-matrix library

### Internal
- Existing ray tracing system
- Current spatial audio processor
- Room acoustic modeling
- GPU shader pipeline

## Risk Mitigation

### Performance Concerns
- Implement efficient caching for coefficient calculations
- Use GPU parallel processing where possible
- Profile before and after implementation

### Compatibility Issues
- Maintain backward compatibility during transition
- Provide fallback for systems without WebGPU
- Test across different browsers and devices

### Audio Quality Regression
- Implement comprehensive testing suite
- Maintain reference audio samples
- Allow toggling between old and new implementations during development
