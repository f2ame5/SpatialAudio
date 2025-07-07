# Buffer Alignment Fix for Collection Parameters

## Problem Identified

The core issue was a **buffer layout mismatch** between TypeScript and WGSL due to incorrect alignment handling for `vec3<f32>` types.

### The Issue
1. **WGSL Struct**: `vec3<f32>` requires 16-byte alignment (12 bytes data + 4 bytes padding)
2. **TypeScript Layout**: Was placing `listener_radius` at index 3 (byte 12), violating alignment
3. **Result**: GPU shader reading incorrect data from misaligned buffer positions

### WGSL Struct Definition
```wgsl
struct CollectionParams {
    listener_position: vec3<f32>,    // 12 bytes + 4 bytes padding = 16 bytes
    listener_radius: f32,            // Starts at byte 16
    sample_rate: f32,                // Starts at byte 20
    ir_length: f32,                  // Starts at byte 24
    time_bin_size: f32,              // Starts at byte 28
    max_bins: u32,                   // Starts at byte 32
    energy_threshold: f32,           // Starts at byte 36
    padding: f32,                    // Starts at byte 40
}
```

### Incorrect TypeScript Layout (Before)
```typescript
// WRONG: Violates vec3 alignment rules
params[0] = listenerConfig.position[0];  // Byte 0
params[1] = listenerConfig.position[1];  // Byte 4
params[2] = listenerConfig.position[2];  // Byte 8
params[3] = listenerConfig.radius;       // Byte 12 ❌ Should be padding!
params[4] = this.config.sampleRate;      // Byte 16 ❌ Should be radius!
// ... rest misaligned
```

## Solution Implemented

### Corrected TypeScript Layout (After)
```typescript
// CORRECT: Respects 16-byte alignment for vec3<f32>
params[0] = listenerConfig.position[0];     // Byte 0  - listener_position.x
params[1] = listenerConfig.position[1];     // Byte 4  - listener_position.y
params[2] = listenerConfig.position[2];     // Byte 8  - listener_position.z
// params[3] is intentionally left empty     // Byte 12 - padding for vec3 alignment

params[4] = listenerConfig.radius;          // Byte 16 - listener_radius
params[5] = this.config.sampleRate;         // Byte 20 - sample_rate
params[6] = this.config.impulseResponseLength; // Byte 24 - ir_length
params[7] = timeBinSize;                     // Byte 28 - time_bin_size
params[8] = maxBins;                         // Byte 32 - max_bins (as u32)
params[9] = this.config.minEnergy;          // Byte 36 - energy_threshold
params[10] = 0.0;                           // Byte 40 - padding
```

### Memory Layout Comparison

#### Before (Incorrect)
```
Byte:  0   4   8  12  16  20  24  28  32  36  40  44
Field: [listener_position ] R   S   I   T   M   E   P   P
Index: 0   1   2   3   4   5   6   7   8   9  10  11
```

#### After (Correct)
```
Byte:  0   4   8  12  16  20  24  28  32  36  40  44
Field: [listener_position ][P] R   S   I   T   M   E   P
Index: 0   1   2   3   4   5   6   7   8   9  10  11
```

Where:
- `listener_position` = vec3<f32> (12 bytes)
- `[P]` = Padding (4 bytes)
- `R` = listener_radius
- `S` = sample_rate
- `I` = ir_length
- `T` = time_bin_size
- `M` = max_bins
- `E` = energy_threshold

## Key Changes Made

### 1. Updated Buffer Layout
```typescript
// OLD: Incorrect alignment
params[3] = listenerConfig.radius;

// NEW: Correct alignment with padding
// params[3] is intentionally left empty for padding
params[4] = listenerConfig.radius;
```

### 2. Shifted All Subsequent Fields
All fields after `listener_position` were shifted by one index to account for the padding:
- `listener_radius`: index 3 → 4
- `sample_rate`: index 4 → 5
- `ir_length`: index 5 → 6
- `time_bin_size`: index 6 → 7
- `max_bins`: index 7 → 8
- `energy_threshold`: index 8 → 9

### 3. Enhanced Debug Logging
```typescript
console.log('Collection params (corrected alignment):', {
    listenerPos: [params[0], params[1], params[2]],
    padding_after_vec3: params[3], // Should be 0 (padding)
    radius: params[4],
    sampleRate: params[5],
    irLength: params[6],
    timeBinSize: params[7],
    maxBins: params[8],
    energyThreshold: params[9],
    padding: params[10],
    timeBinSizeMs: params[7] * 1000,
});
```

## Impact of the Fix

### Before Fix
- **Listener Position**: Correct (vec3 data read properly)
- **Listener Radius**: Reading padding bytes (garbage data)
- **Sample Rate**: Reading radius value (wrong data type)
- **Time Bin Size**: Reading sample rate (wrong scale)
- **Max Bins**: Reading IR length (wrong data type)
- **Energy Threshold**: Reading time bin size (wrong scale)

### After Fix
- **All Fields**: Reading correct data from proper aligned positions
- **Ray Collection**: Proper listener radius for spatial filtering
- **Time Binning**: Correct time bin size for accurate temporal resolution
- **Energy Filtering**: Proper energy threshold for ray collection

## Expected Results

With this fix, the ray collection should now:

1. **Proper Spatial Filtering**: Correct listener radius for ray collection
2. **Accurate Time Binning**: Proper time bin size calculation
3. **Correct Energy Thresholding**: Appropriate energy filtering
4. **Valid Impulse Response**: Non-zero values in correct time bins
5. **Realistic Spatial Audio**: Proper directional and distance effects

## Validation

### Debug Output
The corrected debug output should show:
```
Collection params (corrected alignment): {
  listenerPos: [0, 1.5, 0],
  padding_after_vec3: 0,        // Should always be 0
  radius: 0.1,                  // Correct listener radius
  sampleRate: 48000,            // Correct sample rate
  irLength: 2,                  // Correct IR length in seconds
  timeBinSize: 0.0000208,       // Correct: 1/48000
  maxBins: 96000,               // Correct: 2 * 48000
  energyThreshold: 0.001,       // Correct energy threshold
  timeBinSizeMs: 0.0208         // Correct: ~0.021ms per sample
}
```

### GPU Memory Layout Verification
The GPU shader will now read:
- Correct listener position for spatial calculations
- Proper radius for ray collection filtering
- Accurate time bin size for temporal resolution
- Valid energy threshold for ray filtering

## Files Modified

1. **src/audio/acoustic-raytracer.ts**
   - `updateCollectionParams()` function
   - Corrected buffer layout with proper vec3 alignment
   - Updated debug logging to show alignment

## WebGPU Alignment Rules

This fix addresses the fundamental WebGPU/WGSL alignment requirement:

> **vec3<f32> Alignment**: In WGSL, `vec3<f32>` types are aligned to 16-byte boundaries, meaning they occupy 16 bytes in memory (12 bytes data + 4 bytes padding), not just 12 bytes.

This is a common source of bugs in WebGPU applications where CPU-side buffer layouts don't match GPU-side struct layouts.

## Conclusion

This buffer alignment fix resolves a critical data interpretation issue that was causing incorrect ray collection parameters to be passed to the GPU shader. With proper alignment, the spatial audio raytracing should now function correctly with accurate listener positioning, time binning, and energy filtering.
