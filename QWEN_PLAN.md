Plan Overview: Step 2 - Integrate HRTF (Head-Related Transfer Function)
This step involves integrating HRTF filtering to simulate how sound interacts with the human head and ears, enabling accurate 3D spatial audio perception. We will:

Load HRTF datasets 

Map ray hit directions to HRTF indices.

Apply HRTF convolution in the GPU compute shader and audio processor.

Subtasks
1. Load HRTF Dataset
Goal: Load precomputed HRTF data (src/hrtf folder) into GPU buffers.

Code Location:
src/audio/hrtf-loader.ts (new file)
src/raytracer/raytracer.ts (to integrate HRTF data).

Steps:

Fetch HRTF Data
Use a precomputed dataset (e.g., in /src/hrtf folder):

// src/hrtf/RIEC_hrir_subject_080.sofa
export async function loadHRTFData(): Promise<Float32Array> {
  const response = await fetch('hrtf dataset');
  return new Float32Array(await response.arrayBuffer());
}

Upload HRTF Data to GPU
Create a GPU buffer to store HRTF coefficients:

// In RayTracer class
private hrtfBuffer!: GPUBuffer;
async initializeHRTF() {
  const hrtfData = await loadHRTFData();
  this.hrtfBuffer = this.device.createBuffer({
    size: hrtfData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'HRTF Coefficients'
  });
  this.device.queue.writeBuffer(this.hrtfBuffer, 0, hrtfData);
}

Why: HRTF data must be accessible to the GPU for real-time filtering.

2. Map Ray Direction to HRTF Indices
Goal: Convert ray hit direction to HRTF lookup indices (azimuth, elevation).

Code Location:
src/raytracer/raytracer.ts
Function: calculateRayHits() in RayTracer.

Steps:

Compute Direction to Listener
For each ray hit, calculate the direction vector from the hit position to the listener:

const listenerPos = this.listener.getPosition();
const toListener = vec3.subtract(vec3.create(), listenerPos, hit.position);
vec3.normalize(toListener, toListener);

Convert to Spherical Coordinates
Map the direction to azimuth/elevation angles:

const azimuth = Math.atan2(toListener[2], toListener[0]); // Azimuth (radians)
const elevation = Math.asin(toListener[1]); // Elevation (radians)

Map to HRTF Indices
Convert angles to dataset indices (e.g., MIT KEMAR uses 360° azimuth, 0° elevation):

const hrtfAzimuthIndex = Math.floor((azimuth + Math.PI) / (2 * Math.PI) * 360);
const hrtfElevationIndex = 0; // Simplified for 0° elevation

Why: HRTF datasets are indexed by spatial direction (azimuth/elevation).

3. Apply HRTF in GPU Compute Shader
Goal: Use the HRTF buffer in the compute shader to apply directional filtering.

Code Location:
src/raytracer/shaders/spatial_audio.wgsl
Function: main() compute shader.

Steps:

Add HRTF Buffer Binding
Modify the bind group layout to include the HRTF buffer:

@group(0) @binding(5) var<storage, read> hrtfCoefficients: array<f32>;

Fetch HRTF Coefficients
Retrieve coefficients based on azimuth/elevation:

fn getHRTFCoefficient(azimuthIndex: u32, elevationIndex: u32) -> f32 {
  let index = elevationIndex * 360 + azimuthIndex;
  return hrtfCoefficients[index];
}

Apply HRTF to Ray Hits
Multiply the ray's energy by HRTF coefficients:

let hrtfFactor = getHRTFCoefficient(hrtfAzimuthIndex, hrtfElevationIndex);
hit.energy *= hrtfFactor;

Why: This offloads HRTF calculations to the GPU for performance.

4. Update Audio Processor for HRTF Convolution
Goal: Apply HRTF convolution during impulse response generation.

Code Location:
src/audio/audio-processor.ts
Function: processRayHits() method.

Steps:

Load HRTF Data
Fetch and store HRTF coefficients for CPU-side processing:

private hrtfData: Float32Array = new Float32Array();
async loadHRTF() {
  this.hrtfData = await loadHRTFData();
}

Convolve with HRTF
For each ray hit, apply HRTF convolution to left/right channels:

const hrtfIndex = calculateHRTFIndex(toListener);
const hrtfLeft = this.hrtfData.subarray(hrtfIndex * 2, hrtfIndex * 2 + 1);
const hrtfRight = this.hrtfData.subarray(hrtfIndex * 2 + 1, hrtfIndex * 2 + 2);

// Convolve with impulse response
convolve(leftIR, hrtfLeft);
convolve(rightIR, hrtfRight);

Implement Convolution
Use a fast convolution method (e.g., FFT-based):

function convolve(signal: Float32Array, kernel: Float32Array) {
  const fft = new FFT(signal.length + kernel.length - 1);
  const signalFFT = fft.transform(signal);
  const kernelFFT = fft.transform(kernel);
  // Multiply in frequency domain
  for (let i = 0; i < signalFFT.length; i++) {
    signalFFT[i] *= kernelFFT[i];
  }
  // Inverse FFT
  return fft.inverseTransform(signalFFT);
}

Why: HRTF convolution is necessary for accurate spatial audio rendering.

5. Update Material Definitions for Directional Filtering
Goal: Adjust material absorption based on HRTF direction.

Code Location:
src/room/room-materials.ts
Function: MATERIAL_PRESETS.

Steps:

Add Directional Absorption
Define absorption coefficients for different directions:

CONCRETE: {
  absorptionLow: 0.02,
  absorptionMid: 0.03,
  absorptionHigh: 0.04,
  directionalAbsorption: {
    front: 1.0,
    side: 0.8,
    back: 0.6
  }
}

Apply Directional Absorption
In the ray tracer, adjust energy based on hit direction:

const material = closestPlane.material;
const dirFactor = material.directionalAbsorption[hitDirectionCategory];
ray.setEnergyLow(ray.getEnergyLow() * dirFactor);

Why: Directional absorption improves realism by simulating how materials interact with sound from different angles.

Tips for Implementation
Use Precomputed HRTF Kernels: Use compressed HRTF datasets (e.g., SOFA format) to reduce memory usage.

Profile HRTF Performance: Use performance.now() to measure HRTF convolution time:

const start = performance.now();
convolve(leftIR, hrtfLeft);
console.log(`Convolution took ${performance.now() - start}ms`);

Add HRTF Debug Visualization: Visualize HRTF coefficients in the UI:

private visualizeHRTF(hrtfCoeff: Float32Array): void {
  const canvas = document.getElementById('hrtf-visualizer') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < hrtfCoeff.length; i++) {
    ctx.fillRect(i, canvas.height / 2 - hrtfCoeff[i] * 100, 1, 2);
  }
}

Validate HRTF with Headphone Testing: Test with headphones to ensure directional cues (e.g., sound should appear to come from the correct direction).

Validation
Test Case 1: Place a sound source directly in front of the listener. Verify that the left/right channels are balanced.

Test Case 2: Move the sound source to the side. Confirm that the left/right channels show asymmetry due to HRTF filtering.