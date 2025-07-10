# Hybrid Ray Tracing Implementation Plan

## Overview
Implement a hybrid approach combining the image-source method for early reflections (first 2-3 bounces) with stochastic ray tracing for late reverberation. This provides the accuracy of image-source method for perceptually important early reflections and the efficiency of stochastic ray tracing for diffuse late-reverberation tail.

## Current System Analysis

### Existing Ray Tracing Implementation
- **Pure stochastic ray tracing**: All reflections handled by random ray bouncing
- **Limited early reflection accuracy**: Early reflections not precisely modeled
- **Computational inefficiency**: High ray count needed for accurate early reflections
- **Missing image sources**: No systematic early reflection calculation

### Benefits of Hybrid Approach
- **Accurate early reflections**: Image-source method provides exact early reflection paths
- **Efficient late reverberation**: Stochastic rays handle diffuse tail efficiently
- **Perceptual optimization**: Focus accuracy where human hearing is most sensitive
- **Computational balance**: Optimal use of processing resources

## Image-Source Method Foundation

### Mathematical Principles
**Image Source Calculation:**
```
For each wall surface:
1. Create mirror image of source across wall plane
2. Calculate direct path from image source to listener
3. Verify path doesn't intersect other walls (visibility check)
4. Calculate reflection properties (time, amplitude, phase)
```

**Recursive Image Sources:**
```
For higher-order reflections:
1. Create image sources of existing image sources
2. Track reflection order (1st, 2nd, 3rd order)
3. Apply cumulative material effects
4. Stop at maximum order (typically 2-3)
```

## Task Breakdown

### 1. Implement Image-Source Method Core
**Goal:** Create the fundamental image-source calculation system for early reflections.

#### 1.1 Create Image Source Calculator
- **Description:** Implement core image-source method calculations
- **Implementation Details:**
  ```typescript
  class ImageSourceCalculator {
    private room: Room;
    private maxOrder: number = 3;
    
    calculateImageSources(sourcePos: vec3, listenerPos: vec3): ImageSource[] {
      const imageSources: ImageSource[] = [];
      
      // First-order reflections (direct wall reflections)
      this.calculateFirstOrderSources(sourcePos, imageSources);
      
      // Higher-order reflections (up to maxOrder)
      for (let order = 2; order <= this.maxOrder; order++) {
        this.calculateHigherOrderSources(imageSources, order);
      }
      
      // Visibility check - ensure paths don't intersect walls
      return this.filterVisibleSources(imageSources, listenerPos);
    }
    
    private calculateFirstOrderSources(sourcePos: vec3, imageSources: ImageSource[]): void
    private calculateHigherOrderSources(imageSources: ImageSource[], order: number): void
    private filterVisibleSources(sources: ImageSource[], listenerPos: vec3): ImageSource[]
  }
  ```
- **Features Required:**
  - Mirror image calculation across wall planes
  - Recursive image source generation
  - Visibility checking with ray-wall intersection
  - Reflection order tracking
- **Files:** `src/acoustics/image-source-calculator.ts` (new)

#### 1.2 Create Image Source Data Structure
- **Description:** Define data structures for storing image source information
- **Implementation Details:**
  ```typescript
  interface ImageSource {
    position: vec3;           // Position of image source
    originalSource: vec3;     // Original source position
    reflectionOrder: number;  // Number of reflections (1, 2, 3)
    reflectionPath: Wall[];   // Sequence of walls in reflection path
    distance: number;         // Total path length to listener
    arrivalTime: number;      // Time of arrival at listener
    amplitude: number;        // Amplitude after all reflections
    phase: number;           // Phase shift from reflections
    isVisible: boolean;      // Whether path is unobstructed
  }
  
  interface ReflectionPath {
    walls: Wall[];
    totalDistance: number;
    cumulativeReflection: Complex;  // Complex reflection coefficient
    pathPoints: vec3[];            // Points along reflection path
  }
  ```
- **Features Required:**
  - Complete path information storage
  - Cumulative reflection properties
  - Visibility status tracking
  - Integration with complex wave pressure system
- **Files:** `src/acoustics/image-source-types.ts` (new)

#### 1.3 Implement Visibility Checking
- **Description:** Create robust visibility checking to ensure image source paths are valid
- **Implementation Details:**
  ```typescript
  class VisibilityChecker {
    static checkVisibility(imageSource: ImageSource, listenerPos: vec3, room: Room): boolean {
      const path = this.constructReflectionPath(imageSource, listenerPos);
      
      // Check each segment of the path for wall intersections
      for (let i = 0; i < path.segments.length; i++) {
        const segment = path.segments[i];
        if (this.segmentIntersectsWalls(segment, room, path.reflectionWalls)) {
          return false;
        }
      }
      
      return true;
    }
    
    private static constructReflectionPath(source: ImageSource, listener: vec3): ReflectionPath
    private static segmentIntersectsWalls(segment: LineSegment, room: Room, excludeWalls: Wall[]): boolean
    private static calculateReflectionPoint(source: vec3, listener: vec3, wall: Wall): vec3
  }
  ```
- **Features Required:**
  - Ray-wall intersection testing
  - Path segment validation
  - Reflection point calculation
  - Obstruction detection
- **Files:** `src/acoustics/visibility-checker.ts` (new)

### 2. Integrate Image Sources with Ray Tracing
**Goal:** Combine image-source early reflections with existing stochastic ray tracing system.

#### 2.1 Create Hybrid Ray Tracer
- **Description:** Develop main controller that coordinates image-source and stochastic methods
- **Implementation Details:**
  ```typescript
  class HybridRayTracer {
    private imageSourceCalculator: ImageSourceCalculator;
    private stochasticRayTracer: RayTracer;
    private transitionTime: number = 0.08; // 80ms transition point
    
    public calculateHybridAcoustics(sourcePos: vec3, listenerPos: vec3): HybridAcousticResult {
      // Calculate early reflections using image-source method
      const earlyReflections = this.calculateEarlyReflections(sourcePos, listenerPos);
      
      // Calculate late reverberation using stochastic ray tracing
      const lateReverberation = this.calculateLateReverberation(sourcePos, listenerPos);
      
      // Combine results with smooth transition
      return this.combineAcousticResults(earlyReflections, lateReverberation);
    }
    
    private calculateEarlyReflections(sourcePos: vec3, listenerPos: vec3): EarlyReflectionResult
    private calculateLateReverberation(sourcePos: vec3, listenerPos: vec3): LateReverberationResult
    private combineAcousticResults(early: EarlyReflectionResult, late: LateReverberationResult): HybridAcousticResult
  }
  ```
- **Features Required:**
  - Coordinated execution of both methods
  - Smooth transition between early and late reflections
  - Result combination and merging
  - Performance optimization
- **Files:** `src/raytracer/hybrid-ray-tracer.ts` (new)

#### 2.2 Implement Early Reflection Processing
- **Description:** Process image sources to generate early reflection impulse response
- **Implementation Details:**
  ```typescript
  class EarlyReflectionProcessor {
    public processImageSources(imageSources: ImageSource[], listenerPos: vec3): EarlyReflectionResult {
      const earlyIR = new Float32Array(this.getEarlyIRLength());
      const sampleRate = 44100;
      
      imageSources.forEach(source => {
        if (source.arrivalTime <= this.transitionTime && source.isVisible) {
          const sampleIndex = Math.floor(source.arrivalTime * sampleRate);
          
          if (sampleIndex < earlyIR.length) {
            // Apply HRTF and spatial processing
            const [leftSample, rightSample] = this.calculateSpatialSample(source, listenerPos);
            
            // Add to impulse response with proper amplitude and phase
            earlyIR[sampleIndex] += this.calculateComplexContribution(source, leftSample, rightSample);
          }
        }
      });
      
      return new EarlyReflectionResult(earlyIR, imageSources);
    }
    
    private calculateSpatialSample(source: ImageSource, listenerPos: vec3): [number, number]
    private calculateComplexContribution(source: ImageSource, left: number, right: number): number
  }
  ```
- **Features Required:**
  - Precise timing calculation
  - Spatial audio processing (HRTF)
  - Complex wave pressure integration
  - Frequency-dependent processing
- **Files:** `src/acoustics/early-reflection-processor.ts` (new)

#### 2.3 Modify Stochastic Ray Tracer for Late Reverberation
- **Description:** Adapt existing ray tracer to focus on late reverberation only
- **Changes Required:**
  - Start rays after transition time (skip early reflection period)
  - Increase ray count for better late reverberation density
  - Focus on diffuse scattering for realistic tail
  - Optimize for computational efficiency
- **Implementation Details:**
  ```typescript
  // In existing RayTracer class
  public calculateLateReverberation(sourcePos: vec3, listenerPos: vec3, 
                                   startTime: number = 0.08): LateReverberationResult {
    // Generate rays with initial time offset
    const rays = this.generateRaysForLateReverberation(sourcePos, startTime);
    
    // Trace rays with focus on diffuse scattering
    const lateHits = this.traceRaysForDiffuseField(rays);
    
    // Generate late reverberation impulse response
    return this.generateLateReverberationIR(lateHits, startTime);
  }
  
  private generateRaysForLateReverberation(sourcePos: vec3, startTime: number): Ray[]
  private traceRaysForDiffuseField(rays: Ray[]): RayHit[]
  ```
- **Files:** `src/raytracer/raytracer.ts`

### 3. Implement Smooth Transition Between Methods
**Goal:** Create seamless transition between image-source early reflections and stochastic late reverberation.

#### 3.1 Create Transition Controller
- **Description:** Manage the crossover between early and late reflection methods
- **Implementation Details:**
  ```typescript
  class TransitionController {
    private transitionTime: number = 0.08;      // 80ms transition point
    private transitionWindow: number = 0.02;    // 20ms transition window
    
    public combineEarlyAndLate(early: EarlyReflectionResult, 
                              late: LateReverberationResult): CombinedImpulseResponse {
      const combinedIR = new Float32Array(this.getTotalIRLength());
      
      // Add early reflections (0 to transitionTime)
      this.addEarlyReflections(combinedIR, early);
      
      // Add late reverberation with smooth transition
      this.addLateReverberationWithTransition(combinedIR, late);
      
      // Apply crossfade in transition window
      this.applyCrossfade(combinedIR, early, late);
      
      return new CombinedImpulseResponse(combinedIR);
    }
    
    private addEarlyReflections(ir: Float32Array, early: EarlyReflectionResult): void
    private addLateReverberationWithTransition(ir: Float32Array, late: LateReverberationResult): void
    private applyCrossfade(ir: Float32Array, early: EarlyReflectionResult, late: LateReverberationResult): void
  }
  ```
- **Features Required:**
  - Smooth amplitude transition
  - Phase coherence maintenance
  - Overlap handling in transition window
  - Energy conservation
- **Files:** `src/acoustics/transition-controller.ts` (new)

#### 3.2 Implement Energy Matching
- **Description:** Ensure energy continuity between early and late reflection periods
- **Implementation Details:**
  ```typescript
  class EnergyMatcher {
    public matchEnergyLevels(early: EarlyReflectionResult, late: LateReverberationResult): void {
      // Calculate energy density at transition point
      const earlyEnergyAtTransition = this.calculateEnergyAtTime(early, this.transitionTime);
      const lateEnergyAtStart = this.calculateEnergyAtTime(late, this.transitionTime);
      
      // Calculate scaling factor for energy matching
      const energyScaleFactor = earlyEnergyAtTransition / lateEnergyAtStart;
      
      // Apply scaling to late reverberation
      this.scaleEnergyLevels(late, energyScaleFactor);
    }
    
    private calculateEnergyAtTime(result: AcousticResult, time: number): number
    private scaleEnergyLevels(result: LateReverberationResult, factor: number): void
  }
  ```
- **Features Required:**
  - Energy density calculation
  - Smooth energy scaling
  - Frequency-dependent energy matching
  - Perceptual energy weighting
- **Files:** `src/acoustics/energy-matcher.ts` (new)

### 4. Optimize Performance and Quality
**Goal:** Ensure the hybrid system performs efficiently while maintaining high audio quality.

#### 4.1 Implement Adaptive Quality Control
- **Description:** Dynamically adjust quality settings based on performance requirements
- **Implementation Details:**
  ```typescript
  class AdaptiveQualityController {
    private performanceMonitor: PerformanceMonitor;
    
    public adjustQualitySettings(targetFrameTime: number): QualitySettings {
      const currentPerformance = this.performanceMonitor.getCurrentMetrics();
      
      if (currentPerformance.frameTime > targetFrameTime) {
        // Reduce quality to maintain performance
        return this.reduceQuality(currentPerformance);
      } else if (currentPerformance.frameTime < targetFrameTime * 0.8) {
        // Increase quality if performance allows
        return this.increaseQuality(currentPerformance);
      }
      
      return this.getCurrentSettings();
    }
    
    private reduceQuality(metrics: PerformanceMetrics): QualitySettings
    private increaseQuality(metrics: PerformanceMetrics): QualitySettings
  }
  
  interface QualitySettings {
    maxImageSourceOrder: number;    // 1-3
    stochasticRayCount: number;     // 500-5000
    transitionTime: number;         // 0.05-0.1s
    spatialResolution: number;      // HRTF quality
  }
  ```
- **Features Required:**
  - Real-time performance monitoring
  - Dynamic quality adjustment
  - User preference consideration
  - Graceful degradation
- **Files:** `src/acoustics/adaptive-quality-controller.ts` (new)

#### 4.2 Implement Caching System
- **Description:** Cache image source calculations for static room configurations
- **Implementation Details:**
  ```typescript
  class ImageSourceCache {
    private cache: Map<string, ImageSource[]> = new Map();
    
    public getCachedImageSources(sourcePos: vec3, roomConfig: RoomConfig): ImageSource[] | null {
      const cacheKey = this.generateCacheKey(sourcePos, roomConfig);
      return this.cache.get(cacheKey) || null;
    }
    
    public cacheImageSources(sourcePos: vec3, roomConfig: RoomConfig, sources: ImageSource[]): void {
      const cacheKey = this.generateCacheKey(sourcePos, roomConfig);
      this.cache.set(cacheKey, sources);
      
      // Implement LRU eviction if cache gets too large
      this.evictOldEntries();
    }
    
    private generateCacheKey(sourcePos: vec3, roomConfig: RoomConfig): string
    private evictOldEntries(): void
  }
  ```
- **Features Required:**
  - Efficient cache key generation
  - LRU cache eviction
  - Memory usage monitoring
  - Cache invalidation on room changes
- **Files:** `src/acoustics/image-source-cache.ts` (new)

## Implementation Phases

### Phase 1: Image-Source Foundation (Tasks 1.1-1.3)
**Duration:** 2-3 weeks
1. Create Image Source Calculator
2. Create Image Source Data Structure
3. Implement Visibility Checking

### Phase 2: Integration (Tasks 2.1-2.3)
**Duration:** 2-3 weeks
4. Create Hybrid Ray Tracer
5. Implement Early Reflection Processing
6. Modify Stochastic Ray Tracer for Late Reverberation

### Phase 3: Transition System (Tasks 3.1-3.2)
**Duration:** 1-2 weeks
7. Create Transition Controller
8. Implement Energy Matching

### Phase 4: Optimization (Tasks 4.1-4.2)
**Duration:** 1-2 weeks
9. Implement Adaptive Quality Control
10. Implement Caching System

## Expected Benefits

### Audio Quality Improvements
- **Accurate early reflections**: Precise modeling of first 2-3 reflections
- **Natural late reverberation**: Realistic diffuse tail
- **Better spatial imaging**: Improved localization and spaciousness
- **Perceptual optimization**: Focus accuracy where hearing is most sensitive

### Performance Benefits
- **Computational efficiency**: Optimal use of processing resources
- **Scalable quality**: Adaptive performance based on system capabilities
- **Reduced ray count**: Fewer rays needed for same quality
- **Caching benefits**: Reuse calculations for static configurations

## Technical Considerations

### Accuracy vs Performance Trade-offs
- **Image source order**: Higher order = more accuracy but more computation
- **Transition timing**: Earlier transition = better performance but less accuracy
- **Ray count**: More rays = better late reverberation but higher cost
- **Spatial resolution**: Higher resolution = better HRTF but more processing

### Integration Challenges
- **Smooth transitions**: Avoiding audible artifacts at crossover point
- **Energy conservation**: Maintaining consistent energy levels
- **Phase coherence**: Preserving wave relationships
- **Real-time constraints**: Meeting frame time requirements

### Validation Strategy
- **Acoustic measurements**: Compare with real room measurements
- **Perceptual testing**: A/B testing with listeners
- **Performance benchmarking**: Frame time and memory usage analysis
- **Quality metrics**: Objective audio quality measurements
