/**
 * AudioProcessor
 *
 * This class integrates real-time audio processing using the Web Audio API.
 * It converts ray hit data (each with a time and energy value) into an impulse response
 * and plays that sound through the device's audio output.
 *
 * Methods:
 *   - processRayHits: Converts ray hit data into an impulse response.
 *   - playAudio: Plays the generated impulse response.
 *   - generateAndPlay: Convenience method combining the above two.
 *   - debugPlaySineWave: Plays a test sine wave for debug purposes.
 *   - playSoundWithImpulseResponse: Convolves a dry white noise signal with the impulse response and plays it.
 */
import { Camera } from '../camera/camera';
import { SpatialAudioProcessor } from './spatial-audio-processor';
import { Room } from '../room/room';
import { WaveformRenderer } from '../visualization/waveform-renderer';
import { vec3 } from 'gl-matrix/vec3';

export class AudioProcessor {
  private audioCtx: AudioContext;
  private sampleRate: number;
  private impulseResponseBuffer: AudioBuffer | null;
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
    rayHits: Array<{
        position: vec3,
        energyLow: number,
        energyMid: number,
        energyHigh: number,
        time: number,
        phase: number,
        frequency: number,
        dopplerShift: number
    }>,
    camera: Camera,
    maxTime: number = 0.5,
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
        const sortedHits = [...rayHits].sort((a, b) => a.time - b.time);

        const [leftIR, rightIR] = this.createImpulseResponseFromHits(sortedHits, maxTime);

        // The rest of the function remains the same...
        const sampleCount = leftIR.length;
        const envelope = this.generateEnvelope(sampleCount, sortedHits);

        // (Optional) You can still add room modes if you wish
        const roomModes = this.calculateRoomModes(this.room.config.dimensions);
        this.addRoomModes(leftIR, rightIR, roomModes);

        this.normalizeAndApplyEnvelope(leftIR, rightIR, envelope);
        this.setupImpulseResponseBuffer(leftIR, rightIR);
        this.lastImpulseData = leftIR;

        // Debug: Log impulse response characteristics
        this.debugImpulseResponse(leftIR, rightIR);

        console.log("Impulse response processed successfully.");
    } catch (error) {
        console.error("Error processing ray hits:", error);
        throw error;
    }
  }

  // Add this new function to the AudioProcessor class
  private createImpulseResponseFromHits(
      rayHits: any[],
      maxTime: number
  ): [Float32Array, Float32Array] {
    const sampleCount = Math.ceil(maxTime * this.sampleRate);
    const leftIR = new Float32Array(sampleCount).fill(0);
    const rightIR = new Float32Array(sampleCount).fill(0);

    for (const hit of rayHits) {
        const time = Math.max(hit.time || 0, 0);
        if (time < maxTime) {
            const sampleIndex = Math.floor(time * this.sampleRate);

            const energy = ( (hit.energyLow || 0) + (hit.energyMid || 0) + (hit.energyHigh || 0) ) / 3;
            const amplitude = Math.sqrt(Math.max(energy, 0));

            const [leftGain, rightGain] = this.calculateSpatialGains(hit.position || [0, 0, 0]);

            if (sampleIndex < sampleCount && isFinite(amplitude)) {
                leftIR[sampleIndex] += amplitude * leftGain;
                rightIR[sampleIndex] += amplitude * rightGain;
            }
        }
    }

    return [leftIR, rightIR];
  }

  private applyWaveInterference(leftIR: Float32Array, rightIR: Float32Array, rayHits: any[]): void {
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
                const frequency = Math.max(hit.frequency || 440, 20); // Minimum 20Hz
                const dopplerShift = Math.max(hit.dopplerShift || 1, 0.1); // Minimum 0.1
                const phase = hit.phase || 0;
                
                // Calculate phase at current time
                const timeSinceArrival = Math.max(currentTime - hit.time, 0);
                const instantPhase = phase + 
                    2 * Math.PI * frequency * (1 + dopplerShift) * timeSinceArrival;

                // Calculate amplitude with validation
                const energyLow = Math.max(hit.energyLow || 0, 0);
                const energyMid = Math.max(hit.energyMid || 0, 0);
                const energyHigh = Math.max(hit.energyHigh || 0, 0);
                const amplitude = Math.sqrt((energyLow + energyMid + energyHigh) / 3);

                // Add wave contribution with proper phase
                const contribution = amplitude * Math.sin(instantPhase);
                
                // Validate position for spatial gains
                const position = hit.position || [0, 0, 0];
                const [leftGain, rightGain] = this.calculateSpatialGains(position);
                
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

  private calculateSpatialGains(position: vec3): [number, number] {
    // Simple stereo panning based on x-position
    // Could be enhanced with HRTF in the future
    const x = position[0];
    const maxPan = 0.8; // Maximum panning amount (0.8 = 80% to either side)
    const pan = Math.max(-maxPan, Math.min(maxPan, x / 5)); // Normalize position to pan range
        
    // Convert pan to gains using constant power panning
    const leftGain = Math.cos((pan + 1) * Math.PI / 4);
    const rightGain = Math.sin((pan + 1) * Math.PI / 4);
        
    return [leftGain, rightGain];
  }

  private calculateRT60(rayHits: Array<{ time: number, energyLow: number, energyMid: number, energyHigh: number }>): number {
    if (rayHits.length === 0) {
        return 1.0; // Default RT60 if no hits
    }

    // Sort hits by time
    const sortedHits = [...rayHits].sort((a, b) => a.time - b.time);

    // Calculate energy decay curve
    const times: number[] = [];
    const energies: number[] = [];
    let totalEnergy = 0;

    sortedHits.forEach(hit => {
        times.push(hit.time);
        // Average energy across frequency bands
        const avgEnergy = (hit.energyLow + hit.energyMid + hit.energyHigh) / 3;
        totalEnergy += avgEnergy;
        energies.push(totalEnergy);
    });

    // Normalize energies
    const maxEnergy = Math.max(...energies);
    const normalizedEnergies = energies.map(e => e / maxEnergy);

    // Find -60dB point (energy = 0.001)
    let rt60Time = times[times.length - 1]; // Default to last time
    for (let i = 0; i < normalizedEnergies.length; i++) {
        if (normalizedEnergies[i] <= 0.001) { // -60dB
            rt60Time = times[i];
            break;
        }
    }

    // Apply Sabine's formula correction based on room volume and surface area
    const volume = this.room.getVolume();
    const surfaceArea = this.room.getSurfaceArea();

    // Get average absorption coefficient
    const avgAbsorption = this.calculateAverageAbsorption();

    // Sabine's formula: RT60 = 0.161 * V / (A * S)
    // where V is volume, A is average absorption, S is surface area
    const sabineRT60 = 0.161 * volume / (avgAbsorption * surfaceArea);

    // Blend measured and theoretical RT60
    return (rt60Time + sabineRT60) / 2;
  }

  private calculateAverageAbsorption(): number {
    const materials = this.room.config.materials;
    let totalAbsorption = 0;
    let count = 0;

    // Calculate average absorption across all surfaces and frequency bands
    Object.values(materials).forEach(material => {
        totalAbsorption += material.absorptionLow;
        totalAbsorption += material.absorptionMid;
        totalAbsorption += material.absorptionHigh;
        count += 3; // Three frequency bands
    });

    return totalAbsorption / count;
  }

  private generateEnvelope(sampleCount: number, rayHits: any[]): Float32Array {
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
            envelope[i] = Math.exp(-6.91 * t / rt60);
        }
    }

    return envelope;
  }

  private normalizeAndApplyEnvelope(leftIR: Float32Array, rightIR: Float32Array, envelope: Float32Array): void {
    // Find maximum amplitude
    let maxAmplitude = 0;
    for (let i = 0; i < leftIR.length; i++) {
        maxAmplitude = Math.max(maxAmplitude, Math.abs(leftIR[i]), Math.abs(rightIR[i]));
    }

    // Normalize and apply envelope
    if (maxAmplitude > 0) {
        for (let i = 0; i < leftIR.length; i++) {
            leftIR[i] = (leftIR[i] / maxAmplitude) * envelope[i];
            rightIR[i] = (rightIR[i] / maxAmplitude) * envelope[i];
        }
    }
  }

  private calculateRoomModes(dimensions: { width: number, height: number, depth: number }): RoomMode[] {
    const modes: RoomMode[] = [];
    // Calculate axial, tangential, and oblique modes
    // Add to array with frequencies and decay times
    return modes;
  }

  private addRoomModes(leftIR: Float32Array, rightIR: Float32Array, modes: RoomMode[]): void {
    // Add modal resonances to the impulse response
    modes.forEach(mode => {
        const freq = mode.frequency;
        const decay = Math.exp(-3 * mode.decayTime / mode.rt60);

        for (let t = 0; t < leftIR.length; t++) {
            const sample = decay * Math.sin(2 * Math.PI * freq * t / this.sampleRate);
            leftIR[t] += sample;
            rightIR[t] += sample;
        }
    });
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
  async generateAndPlay(rayHits: Array<{ time: number; energyLow: number; energyMid: number; energyHigh: number }>, camera: Camera, maxTime: number = 0.5): Promise<void> {
    await this.processRayHits(rayHits, camera, maxTime);
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
        console.warn("AudioProcessor: No impulse response available for convolution.");
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