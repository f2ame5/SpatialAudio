import { Camera } from '../camera/camera';
import { SpatialAudioProcessor } from './spatial-audio-processor';
import { Room } from '../room/room';
import { WaveformRenderer } from '../visualization/waveform-renderer';
import { vec3 } from 'gl-matrix';
import { RayHit } from '../raytracer/raytracer';

interface RoomMode {
  frequency: number;
  rt60: number;
}

const NUM_BANDS = 8;
const BAND_CENTERS = [63, 125, 250, 500, 1000, 2000, 4000, 8000];

// Helper to get total energy from a RayHit object
const getTotalEnergy = (hit: RayHit): number => {
    return (
        (hit.energy63  || 0) + (hit.energy125 || 0) + (hit.energy250 || 0) +
        (hit.energy500 || 0) + (hit.energy1k  || 0) + (hit.energy2k  || 0) +
        (hit.energy4k  || 0) + (hit.energy8k  || 0)
    );
};

// Helper to get average energy from a RayHit object
const getAverageEnergy = (hit: RayHit): number => {
    return getTotalEnergy(hit) / NUM_BANDS;
};

export class AudioProcessor {
  private audioCtx: AudioContext;
  private sampleRate: number;
  private impulseResponseBuffer: AudioBuffer | null;
  private lastRayHits: RayHit[] | null = null;
  private lastImpulseData: Float32Array | null;
  private spatialProcessor: SpatialAudioProcessor;
  private room: Room;
  private currentSoundSource: AudioBufferSourceNode | null = null;
  private currentGainNode: GainNode | null = null;
  private masterVolume: number = 0.5; // Master volume control (0.0 to 1.0)

  constructor(device: GPUDevice, room: Room, sampleRate: number = 44100) {
    // Create an Audio Context (uses webkitAudioContext as fallback)
    this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    this.sampleRate = this.audioCtx.sampleRate || sampleRate;
    this.impulseResponseBuffer = null;
    this.lastImpulseData = null;
    this.spatialProcessor = new SpatialAudioProcessor(device, this.sampleRate);
    this.room = room;
  }

  /**
   * Processes an array of ray hit data to generate an impulse response.
   *
   * @param rayHits - Array of objects representing ray hits, each with a time (in seconds) and energy.
   * @param camera - The camera object to use for spatial audio processing.
   * @param maxTime - Maximum duration of the impulse response in seconds (default: 0.5).
   * @param params - Additional parameters for spatial audio processing.
   */
  async processRayHits(
    rayHits: RayHit[],
    camera: Camera,
    maxTime: number = 2.0, // Increased default IR time for more reverb
    params = {
        speedOfSound: 343,
        maxDistance: 20,
        minDistance: 1,
        temperature: 20,
        humidity: 50,
        sourcePower: 0
    }
  ): Promise<void> {
    try {
        if (!rayHits || rayHits.length === 0) {
            console.warn("No ray hits to process, skipping IR generation.");
            return;
        }
        this.lastRayHits = rayHits;

        const sortedHits = [...rayHits].sort((a, b) => a.time - b.time);
        const irDuration = Math.min(maxTime, (sortedHits[sortedHits.length - 1]?.time || 0) + 0.5);
        const sampleCount = Math.ceil(irDuration * this.sampleRate);

        // STEP 1: Generate ONE spectrally-rich broadband impulse response from reflections.
        const [reflectionL, reflectionR] = this.createReflectionImpulse(sortedHits, sampleCount, camera);

        // STEP 2: Generate a SEPARATE signal for the room modes.
        const [modesL, modesR] = this.createRoomModesImpulse(this.room.config.dimensions, sampleCount);

        // Apply a global decay envelope to the reflections. Modes handle their own decay.
        const envelope = this.generateEnvelope(sampleCount, sortedHits);
        this.applyEnvelope(reflectionL, reflectionR, envelope);

        // STEP 3 & 4: Synthesize the final IR by mixing the parts and normalizing.
        this.impulseResponseBuffer = this._synthesizeFinalIR(
            reflectionL, reflectionR, modesL, modesR
        );

        if (this.impulseResponseBuffer) {
            this.lastImpulseData = this.impulseResponseBuffer.getChannelData(0);
            this.debugImpulseResponse(this.impulseResponseBuffer.getChannelData(0), this.impulseResponseBuffer.getChannelData(1));
        }

        console.log("Impulse response processed successfully.");
    } catch (error) {
        console.error("Error processing ray hits:", error);
        throw error;
    }
  }
  
  private createReflectionImpulse(rayHits: RayHit[], sampleCount: number, camera: Camera): [Float32Array, Float32Array] {
    const leftIR = new Float32Array(sampleCount).fill(0);
    const rightIR = new Float32Array(sampleCount).fill(0);

    const sinc = (x: number) => x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    const impulseWidth = 8;
    const impulse = new Float32Array(impulseWidth * 2 + 1);
    for (let i = -impulseWidth; i <= impulseWidth; i++) {
        impulse[i + impulseWidth] = sinc(i / 2) * (0.54 - 0.46 * Math.cos(2 * Math.PI * (i + impulseWidth) / (impulseWidth * 2))); // Sinc with Hamming
    }

    for (const hit of rayHits) {
        const time = hit.time || 0;
        const sampleIndex = Math.floor(time * this.sampleRate);
        if (sampleIndex >= sampleCount - impulseWidth) continue;

        // FIX: Use the helper function to correctly calculate total energy from the 8 bands
        const totalEnergy = getTotalEnergy(hit);
        
        const amplitude = Math.sqrt(Math.max(totalEnergy, 0));
        if (!isFinite(amplitude) || amplitude === 0) continue;

        const [leftGain, rightGain] = this.calculateSpatialGains(hit.position || vec3.fromValues(0, 0, 0), camera);

        // Create a simple filter based on energy distribution to add spectral color
        const lowEnergy = (hit.energy63 + hit.energy125 + hit.energy250) / totalEnergy;
        const midEnergy = (hit.energy500 + hit.energy1k + hit.energy2k) / totalEnergy;
        const highEnergy = (hit.energy4k + hit.energy8k) / totalEnergy;
        
        // Color the impulse - this is a simplification but better than nothing
        const coloredAmplitude = amplitude * (lowEnergy * 0.5 + midEnergy * 1.0 + highEnergy * 1.2);

        for (let i = 0; i < impulse.length; i++) {
            const idx = sampleIndex + i - impulseWidth;
            if (idx >= 0 && idx < sampleCount) {
                const impulseValue = impulse[i] * coloredAmplitude;
                leftIR[idx] += impulseValue * leftGain;
                rightIR[idx] += impulseValue * rightGain;
            }
        }
    }
    return [leftIR, rightIR];
  }

  private createRoomModesImpulse(dimensions: { width: number; height: number; depth: number; }, sampleCount: number): [Float32Array, Float32Array] {
    const modesL = new Float32Array(sampleCount).fill(0);
    const modesR = new Float32Array(sampleCount).fill(0);
    const modes = this.calculateRoomModes(dimensions);

    modes.forEach(mode => {
        const freq = mode.frequency;
        const amplitude = 0.05; // Modes have a small, fixed amplitude.
        const k = 6.907 / mode.rt60;
        const phase = Math.random() * 2 * Math.PI;

        for (let i = 0; i < sampleCount; i++) {
            const t = i / this.sampleRate;
            if (t > mode.rt60 * 1.5) break;

            const decay = Math.exp(-k * t);
            const sample = amplitude * decay * Math.sin(2 * Math.PI * freq * t + phase);
            modesL[i] += sample;
            modesR[i] += sample;
        }
    });
    return [modesL, modesR];
  }

  private _synthesizeFinalIR(
    reflectionL: Float32Array, reflectionR: Float32Array,
    modesL: Float32Array, modesR: Float32Array
  ): AudioBuffer | null {
    const sampleCount = reflectionL.length;
    if (sampleCount === 0) return null;

    // --- MANUAL MIXING (Replaces OfflineAudioContext) ---
    // This synchronous mixing is significantly faster than using an OfflineAudioContext,
    // which introduces latency by scheduling the rendering asynchronously.
    const finalL = new Float32Array(sampleCount);
    const finalR = new Float32Array(sampleCount);

    for (let i = 0; i < sampleCount; i++) {
        finalL[i] = reflectionL[i] + modesL[i];
        finalR[i] = reflectionR[i] + modesR[i];
    }

    // --- MASTER NORMALIZATION ---
    let max = 0.0001; // Avoid division by zero
    for (let i = 0; i < sampleCount; i++) {
        max = Math.max(max, Math.abs(finalL[i]), Math.abs(finalR[i]));
    }

    const gain = 0.98 / max;
    if (gain < 1.0) { // Only apply gain if it's clipping
        for (let i = 0; i < sampleCount; i++) {
            finalL[i] *= gain;
            finalR[i] *= gain;
        }
    }

    // --- CREATE FINAL AUDIOBUFFER ---
    const finalBuffer = this.audioCtx.createBuffer(2, sampleCount, this.sampleRate);
    finalBuffer.copyToChannel(finalL, 0);
    finalBuffer.copyToChannel(finalR, 1);

    return finalBuffer;
  }

  // Simplified envelope application
  private applyEnvelope(leftIR: Float32Array, rightIR: Float32Array, envelope: Float32Array): void {
      for (let i = 0; i < leftIR.length; i++) {
          leftIR[i] *= envelope[i];
          rightIR[i] *= envelope[i];
      }
  }

  private applyWaveInterference(leftIR: Float32Array, rightIR: Float32Array, rayHits: RayHit[], camera: Camera): void {
    const timeStep = 1 / this.sampleRate;
    
    // Process each sample
    for (let i = 0; i < leftIR.length; i++) {
        const currentTime = i * timeStep;
        let leftSum = 0;
        let rightSum = 0;

        // Sum contributions from all rays that have arrived by this time
        for (const hit of rayHits) {
            if (hit.time <= currentTime) {
                // Validate inputs to prevent NaN
                const frequency = 440; // Simplified
                const dopplerShift = Math.max(hit.dopplerShift || 1, 0.1);
                const phase = (hit.phase63 || 0); // Simplified
                
                // Calculate phase at current time
                const timeSinceArrival = Math.max(currentTime - hit.time, 0);
                const instantPhase = phase + 
                    2 * Math.PI * frequency * (1 + dopplerShift) * timeSinceArrival;

                // FIX: Use helper function to get average energy
                const avgEnergy = getAverageEnergy(hit);
                const amplitude = Math.sqrt(avgEnergy);

                // Add wave contribution with proper phase
                const contribution = amplitude * Math.sin(instantPhase);
                
                // Validate position for spatial gains
                const position = hit.position || vec3.fromValues(0, 0, 0);
                const [leftGain, rightGain] = this.calculateSpatialGains(position, camera);
                
                // Add validated contributions
                if (!isNaN(contribution) && isFinite(contribution)) {
                    leftSum += contribution * leftGain;
                    rightSum += contribution * rightGain;
                }
            }
        }

        // Update impulse response with validated sums
        leftIR[i] = isFinite(leftSum) ? leftSum : 0;
        rightIR[i] = isFinite(rightSum) ? rightSum : 0;
    }
  }

  // --- FIX: Corrected spatialization logic ---
  private calculateSpatialGains(hitPosition: vec3, camera: Camera): [number, number] {
    const listenerPos = camera.getPosition();
    const listenerFwd = camera.getFront();
    const listenerUp = camera.getUp();

    // Calculate the listener's right-hand vector
    const listenerRight = vec3.cross(vec3.create(), listenerFwd, listenerUp);
    vec3.normalize(listenerRight, listenerRight);

    // --- THE FIX IS HERE ---
    // Calculate the direction vector FROM the listener TOWARDS the sound hit.
    // This is the correct vector to determine if something is to the left or right.
    const fromListenerToHitDir = vec3.subtract(vec3.create(), hitPosition, listenerPos);
    vec3.normalize(fromListenerToHitDir, fromListenerToHitDir);

    // Project the direction to the sound onto the listener's right vector.
    // This gives a value from -1 (fully to the listener's left) to +1 (fully to the right).
    const pan = vec3.dot(fromListenerToHitDir, listenerRight);

    // Use a constant power panning law to calculate gain for left and right channels.
    // Angle ranges from 0 (fully left) to PI/2 (fully right).
    const angle = (pan + 1.0) * Math.PI / 4.0; // Map pan range [-1, 1] to angle [0, PI/2]
    const leftGain = Math.cos(angle);
    const rightGain = Math.sin(angle);

    return [leftGain, rightGain];
  }

  private calculateRT60(rayHits: RayHit[]): number {
    if (rayHits.length === 0) return 1.0;
    // Sort hits by time
    const sortedHits = [...rayHits].sort((a, b) => a.time - b.time);

    // Schröder integration method for a more robust RT60 calculation from impulse data
    const energyCurve = new Float32Array(sortedHits.length);
    let totalEnergySum = 0;
    for (let i = sortedHits.length - 1; i >= 0; i--) {
        totalEnergySum += getTotalEnergy(sortedHits[i]);
        energyCurve[i] = totalEnergySum;
    }
    
    if (totalEnergySum === 0) return 1.0; // No energy, can't calculate

    // Normalize to dB
    const energyDB = energyCurve.map(e => 10 * Math.log10(e / totalEnergySum));
    
    // Find T20 or T30 using linear regression, which is more robust than finding a single point
    const startIndex = energyDB.findIndex(db => db < -5);
    const endIndex = energyDB.findIndex(db => db < -25);
    
    if (startIndex === -1 || endIndex === -1 || startIndex >= endIndex) return sortedHits[sortedHits.length - 1].time;

    const times: number[] = sortedHits.slice(startIndex, endIndex).map(hit => hit.time);
    const dbs: number[] = energyDB.slice(startIndex, endIndex);

    // Simple linear regression to find the slope
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (let i = 0; i < times.length; i++) {
        sumX += times[i];
        sumY += dbs[i];
        sumXY += times[i] * dbs[i];
        sumX2 += times[i] * times[i];
    }
    const n = times.length;
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    
    if (slope >= 0) return sortedHits[sortedHits.length - 1].time; // No decay

    const rt60 = -60 / slope;
    
    return isFinite(rt60) ? rt60 : 2.0;
  }

  private calculateAverageAbsorption(): number {
    const materials = this.room.config.materials;
    const { walls, ceiling, floor } = materials;
    let totalAbsorption = 0;
    const numBands = 8;

    const getAvg = (mat: any) => {
        let sum = 0;
        for (let i = 0; i < numBands; i++) {
            sum += mat[`absorption${BAND_CENTERS[i]}`] || 0;
        }
        return sum / numBands;
    };
    
    totalAbsorption += getAvg(walls);
    totalAbsorption += getAvg(ceiling);
    totalAbsorption += getAvg(floor);

    return totalAbsorption / 3;
  }

  private generateEnvelope(sampleCount: number, rayHits: RayHit[]): Float32Array {
    const envelope = new Float32Array(sampleCount);

    // Calculate RT60 from ray hits
    const rt60 = this.calculateRT60(rayHits);

    // Generate more realistic decay curve using Schroeder integration
    for (let i = 0; i < sampleCount; i++) {
        const t = i / this.sampleRate;

        // Early reflections (first 50ms)
        if (t < 0.05) {
            envelope[i] = Math.exp(-3 * t); // Faster initial decay
        }
        // Late reverberation
        else {
            // Use Schroeder decay curve instead of simple exponential
            const decay = Math.exp(-6.91 * t / rt60);
            // Blend from early to late
            const blendFactor = Math.min(1.0, (t - 0.05) / 0.1);
            envelope[i] = (1 - blendFactor) * Math.exp(-3 * t) + blendFactor * decay;
        }
    }

    return envelope;
  }

  private _calculateFrequencyDependentRT60(frequency: number): number {
      const { materials, dimensions } = this.room.config;
      const volume = dimensions.width * dimensions.height * dimensions.depth;
      const surfaceArea = 2 * (dimensions.width * dimensions.height + dimensions.height * dimensions.depth + dimensions.width * dimensions.depth);

      if (surfaceArea === 0) return 1.0; // Avoid division by zero

      const getAbsorptionForFreq = (mat: any, freq: number) => {
          // Logarithmic interpolation between bands
          const log_f = Math.log(freq);
          const logCenters = BAND_CENTERS.map(f => Math.log(f));
          const absorptions = BAND_CENTERS.map(band => mat[`absorption${band}`] || 0);

          if (log_f <= logCenters[0]) return absorptions[0];
          if (log_f >= logCenters[NUM_BANDS - 1]) return absorptions[NUM_BANDS - 1];
          for (let i = 1; i < NUM_BANDS; i++) {
              if (log_f < logCenters[i]) {
                  const ratio = (log_f - logCenters[i - 1]) / (logCenters[i] - logCenters[i - 1]);
                  return absorptions[i - 1] + ratio * (absorptions[i] - absorptions[i - 1]);
              }
          }
          return absorptions[NUM_BANDS-1];
      }

      const avgAlpha = (
          getAbsorptionForFreq(materials.walls, frequency) +
          getAbsorptionForFreq(materials.ceiling, frequency) +
          getAbsorptionForFreq(materials.floor, frequency)
      ) / 3;

      return (avgAlpha > 0) ? (0.161 * volume) / (surfaceArea * avgAlpha) : 10.0;
  }

  private calculateRoomModes(dimensions: { width: number, height: number, depth: number }): RoomMode[] {
    const modes: RoomMode[] = [];
    const speedOfSound = 343;
    const { width, height, depth } = dimensions;

    if (width <= 0 || height <= 0 || depth <= 0) return []; // Prevent division by zero

    const maxN = 5; // Check modes up to this index
    for (let nx = 0; nx <= maxN; nx++) {
        for (let ny = 0; ny <= maxN; ny++) {
            for (let nz = 0; nz <= maxN; nz++) {
                if (nx === 0 && ny === 0 && nz === 0) continue;

                const freq = (speedOfSound / 2) * Math.sqrt(
                    (nx / width) ** 2 + (ny / height) ** 2 + (nz / depth) ** 2
                );

                if (freq < 300 && freq > 20) { // Only consider dominant, audible low-frequency modes
                    modes.push({ 
                        frequency: freq, 
                        rt60: this._calculateFrequencyDependentRT60(freq) 
                    });
                }
            }
        }
    }
    return modes;
  }

  /**
   * Plays the generated impulse response.
   * If no impulse response has been generated, logs a warning.
   */
  playAudio(): void {
    if (!this.impulseResponseBuffer) {
      console.warn("AudioProcessor: No impulse response buffer available.  Process ray hits first.");
      return;
    }
    const source = this.audioCtx.createBufferSource();
    source.buffer = this.impulseResponseBuffer;

    // Connect to destination (you might want to connect to an analyzer node first)
    source.connect(this.audioCtx.destination);
    source.start();
    console.log("AudioProcessor: Playing impulse response audio.");
  }

  /**
   * Convenience method to process ray hits and immediately play the resulting audio.
   *
   * @param rayHits - Array of ray hit data.
   * @param camera - The camera object to use for spatial audio processing.
   * @param maxTime - Maximum duration of the impulse response.
   */
  async generateAndPlay(rayHits: RayHit[], camera: Camera, maxTime: number = 2.0): Promise<void> {
    await this.processRayHits(
        rayHits,
        camera, 
        maxTime
    );
    this.playAudio();
  }



  /**
   * Creates a short click sound and processes it with the impulse response
   * using a Convolver Node.
   *
   * @param duration Duration of the input sound in seconds (default: 0.1 second).
   */
  async playSoundWithImpulseResponse(duration: number = 0.1): Promise<void> {
    if (!this.impulseResponseBuffer) {
        console.warn("AudioProcessor: No impulse response available forconvolution.");
        return;
    }

    try {
        // Create a short click sound (exponentially decaying sine wave)
        const sampleCount = Math.ceil(duration * this.sampleRate);
        const clickBuffer = this.audioCtx.createBuffer(1, sampleCount, this.sampleRate);
        const clickData = clickBuffer.getChannelData(0);

        // Generate a decaying sine wave at 440Hz
        const frequency = 440;
        const decayRate = 10;

        for (let i = 0; i < sampleCount; i++) {
            const t = i / this.sampleRate;
            clickData[i] = Math.sin(2 * Math.PI * frequency * t) *
                          Math.exp(-decayRate * t);
        }

        // Create nodes
        const clickSource = this.audioCtx.createBufferSource();
        clickSource.buffer = clickBuffer;

        const gainNode = this.audioCtx.createGain();
        gainNode.gain.value = this.masterVolume * 0.6; // Scale for click sound

        const convolver = this.audioCtx.createConvolver();
        convolver.buffer = this.impulseResponseBuffer;

        // Connect nodes
        clickSource.connect(convolver);
        convolver.connect(gainNode);
        gainNode.connect(this.audioCtx.destination);

        // Start playback
        clickSource.start();
        console.log("AudioProcessor: Playing convolved click sound");
    } catch (error) {
        console.error("Error playing convolved sound:", error);
    }
  }

  /**
   * Returns the last impulse response data as a Float32Array.
   */
  getImpulseResponseData(): Float32Array | null {
    return this.lastImpulseData;
  }

  public async visualizeImpulseResponse(renderer: WaveformRenderer): Promise<void> {
    if (this.lastImpulseData) {
        await renderer.drawWaveformWithFFT(this.lastImpulseData);
    }
  }



  /**
   * Plays a test sound with white noise to better hear the room effect.
   */
  async playNoiseWithIR(): Promise<void> {
    try {
        if (!this.impulseResponseBuffer) {
            console.warn("No impulse response available. Generate one first.");
            return;
        }

        // Create white noise buffer
        const noiseDuration = 1.0; // 1 second
        const noiseBuffer = this.audioCtx.createBuffer(
            1,
            this.sampleRate * noiseDuration,
            this.sampleRate
        );
        const noiseData = noiseBuffer.getChannelData(0);

        // Generate white noise
        for (let i = 0; i < noiseData.length; i++) {
            noiseData[i] = Math.random() * 2 - 1;
        }

        // Create audio nodes
        const sourceNode = this.audioCtx.createBufferSource();
        sourceNode.buffer = noiseBuffer;

        const convolverNode = this.audioCtx.createConvolver();
        convolverNode.buffer = this.impulseResponseBuffer;

        const gainNode = this.audioCtx.createGain();
        gainNode.gain.value = this.masterVolume * 0.4; // Scale for noise (quieter)

        // Connect nodes
        sourceNode.connect(convolverNode);
        convolverNode.connect(gainNode);
        gainNode.connect(this.audioCtx.destination);

        // Resume audio context if suspended
        if (this.audioCtx.state === 'suspended') {
            await this.audioCtx.resume();
        }

        // Start playback
        sourceNode.start();
        console.log("Playing noise with impulse response");

    } catch (error) {
        console.error("Error playing noise with IR:", error);
    }
  }

  /**
   * Sets up the impulse response buffer from the processed left and right channels.
   */
  private setupImpulseResponseBuffer(leftChannel: Float32Array, rightChannel: Float32Array): void {
    try {
        // Ensure we have valid data
        if (!leftChannel || !rightChannel || leftChannel.length === 0 || rightChannel.length === 0) {
            console.error('Invalid channel data provided to setupImpulseResponseBuffer');
            return;
        }

        // Ensure both channels have the same length
        if (leftChannel.length !== rightChannel.length) {
            console.error('Channel length mismatch in setupImpulseResponseBuffer');
            return;
        }

        // Ensure minimum buffer size (at least 1ms at current sample rate)
        const minSamples = Math.max(Math.ceil(this.sampleRate / 1000), 1);
        const numSamples = Math.max(leftChannel.length, minSamples);

        // Create the buffer
        this.impulseResponseBuffer = this.audioCtx.createBuffer(
            2,                  // Number of channels (stereo)
            numSamples,        // Buffer length
            this.sampleRate    // Sample rate
        );

        // Copy the channel data
        this.impulseResponseBuffer.copyToChannel(leftChannel, 0);
        this.impulseResponseBuffer.copyToChannel(rightChannel, 1);

        console.log(`Created impulse response buffer with ${numSamples} samples`);
    } catch (error) {
        console.error('Error setting up impulse response buffer:', error);
        throw error;
    }
  }





  /**
   * Debug method to analyze impulse response characteristics
   */
  private debugImpulseResponse(leftIR: Float32Array, rightIR: Float32Array): void {
    if (!leftIR || leftIR.length === 0) {
        console.log("=== Impulse Response Debug: No data to analyze. ===");
        return;
    }
    // Calculate basic statistics
    const leftMax = Math.max(...leftIR);
    const leftMin = Math.min(...leftIR);
    const leftRMS = Math.sqrt(leftIR.reduce((sum, val) => sum + val * val, 0) / leftIR.length);

    // Check for NaN or infinite values
    const hasNaN = leftIR.some(val => !isFinite(val));

    // Check for very high frequency content (potential noise)
    let highFreqEnergy = 0;
    for (let i = 1; i < leftIR.length; i++) {
      const diff = Math.abs(leftIR[i] - leftIR[i-1]);
      highFreqEnergy += diff;
    }
    highFreqEnergy /= leftIR.length;

    console.log("=== Impulse Response Debug ===");
    console.log(`Length: ${leftIR.length} samples (${(leftIR.length / this.sampleRate).toFixed(3)}s)`);
    console.log(`Peak amplitude: ${leftMax.toFixed(6)}`);
    console.log(`Min amplitude: ${leftMin.toFixed(6)}`);
    console.log(`RMS level: ${leftRMS.toFixed(6)}`);
    console.log(`Has NaN/Infinite: ${hasNaN}`);
    console.log(`High freq energy: ${highFreqEnergy.toFixed(6)}`);
    console.log(`Dynamic range: ${(20 * Math.log10(leftMax / Math.max(leftRMS, 1e-10))).toFixed(1)} dB`);

    // Sample first few values
    const firstSamples = Array.from(leftIR.slice(0, 10)).map(v => v.toFixed(4)).join(', ');
    console.log(`First 10 samples: [${firstSamples}]`);
  }



  /**
   * Set the master volume for all audio playback
   * @param volume - Volume level from 0.0 (silent) to 1.0 (full volume)
   */
  public setMasterVolume(volume: number): void {
    this.masterVolume = Math.max(0, Math.min(1, volume)); // Clamp between 0 and 1

    // Update current playing sound if any
    if (this.currentGainNode) {
      this.currentGainNode.gain.value = this.masterVolume;
    }
  }

  /**
   * Get the current master volume
   * @returns Current master volume (0.0 to 1.0)
   */
  public getMasterVolume(): number {
    return this.masterVolume;
  }

  /**
   * Loads an audio file from the given path and plays it convolved with the current impulse response.
   *
   * @param filePath - Path to the audio file to load
   */
  public async loadAndPlaySoundFile(filePath: string): Promise<void> {
    try {
        if (!this.impulseResponseBuffer) {
            console.warn("No impulse response available. Generate one first.");
            return;
        }

        // Stop any currently playing sound
        this.stopCurrentSound();

        // Fetch the audio file
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`Failed to fetch audio file: ${response.statusText}`);
        }

        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);

        // Create audio nodes
        const sourceNode = this.audioCtx.createBufferSource();
        sourceNode.buffer = audioBuffer;

        const convolverNode = this.audioCtx.createConvolver();
        convolverNode.buffer = this.impulseResponseBuffer;

        const gainNode = this.audioCtx.createGain();
        gainNode.gain.value = this.masterVolume; // Use master volume

        // Connect nodes: Source → Convolver → Gain → Destination
        sourceNode.connect(convolverNode);
        convolverNode.connect(gainNode);
        gainNode.connect(this.audioCtx.destination);

        // Store references for stopping later
        this.currentSoundSource = sourceNode;
        this.currentGainNode = gainNode;

        // Set up event listener for when sound ends
        sourceNode.onended = () => {
            this.currentSoundSource = null;
            this.currentGainNode = null;
        };

        // Resume audio context if suspended
        if (this.audioCtx.state === 'suspended') {
            await this.audioCtx.resume();
        }

        // Start playback
        sourceNode.start();
        console.log(`Playing convolved sound file: ${filePath}`);

    } catch (error) {
        console.error("Error loading and playing sound file:", error);
        throw error;
    }
  }

  /**
   * Stops the currently playing sound file.
   */
  public stopCurrentSound(): void {
    try {
        if (this.currentSoundSource) {
            this.currentSoundSource.stop();
            this.currentSoundSource = null;
        }
        if (this.currentGainNode) {
            this.currentGainNode.disconnect();
            this.currentGainNode = null;
        }
        console.log("Stopped current sound playback");
    } catch (error) {
        console.error("Error stopping current sound:", error);
    }
  }

  /**
   * Checks if a sound is currently playing.
   */
  public isPlaying(): boolean {
    return this.currentSoundSource !== null;
  }
}