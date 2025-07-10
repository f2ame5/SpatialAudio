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
        // Sort ray hits by arrival time for proper wave superposition
        const sortedHits = [...rayHits].sort((a, b) => a.time - b.time);

        // Get spatial audio data with improved frequency response
        const [leftIR, rightIR] = await this.spatialProcessor.processSpatialAudio(
            camera,
            sortedHits,
            params,
            this.room
        );

        // Create stereo impulse response
        const sampleCount = leftIR.length;

        // Generate envelope considering wave properties
        const envelope = this.generateEnvelope(sampleCount, sortedHits);

        // Add room modes for more realistic low frequency response
        const roomModes = this.calculateRoomModes(this.room.config.dimensions);
        this.addRoomModes(leftIR, rightIR, roomModes);

        // Apply wave interference patterns based on phase relationships
        this.applyWaveInterference(leftIR, rightIR, sortedHits);

        // Normalize and apply envelope
        this.normalizeAndApplyEnvelope(leftIR, rightIR, envelope);

        // Set up the impulse response buffer
        this.setupImpulseResponseBuffer(leftIR, rightIR);

        // Store for visualization
        this.lastImpulseData = leftIR;

        console.log("Impulse response processed successfully with wave properties");
    } catch (error) {
        console.error("Error processing ray hits:", error);
        throw error;
    }
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
   * Debug method: Plays a sine wave for testing audio output.
   *
   * @param frequency The frequency of the sine wave in Hz (default: 440 Hz).
   * @param duration The duration of the sine wave in seconds (default: 1 second).
   */
  debugPlaySineWave(frequency: number = 440, duration: number = 1): void {
    const sampleCount = Math.ceil(duration * this.sampleRate);
    const sineData = new Float32Array(sampleCount);
    for (let i = 0; i < sampleCount; i++) {
      const t = i / this.sampleRate;
      sineData[i] = Math.sin(2 * Math.PI * frequency * t);
    }

    const sineBuffer = this.audioCtx.createBuffer(1, sampleCount, this.sampleRate);
    sineBuffer.copyToChannel(sineData, 0, 0);

    const source = this.audioCtx.createBufferSource();
    source.buffer = sineBuffer;
    source.connect(this.audioCtx.destination);
    source.start();

    console.log(`AudioProcessor: Playing debug sine wave with frequency ${frequency}Hz for ${duration} seconds.`);
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
        gainNode.gain.value = 0.3;

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
   * Plays a convolved sound with the current impulse response.
   * Uses a sine wave as input for testing the room acoustics.
   */
  public async playConvolvedSound(): Promise<void> {
    try {
        if (!this.impulseResponseBuffer) {
            console.warn("No impulse response available. Generate one first.");
            return;
        }

        // Create sine wave buffer
        const duration = 2.0; // 2 seconds
        const frequency = 440; // 440 Hz (A4 note)
        const sineBuffer = this.audioCtx.createBuffer(
            1,
            this.sampleRate * duration,
            this.sampleRate
        );
        const sineData = sineBuffer.getChannelData(0);

        // Generate sine wave
        for (let i = 0; i < sineData.length; i++) {
            const t = i / this.sampleRate;
            sineData[i] = Math.sin(2 * Math.PI * frequency * t);

            // Apply envelope to avoid clicks
            const attack = 0.1; // 100ms attack
            const release = 0.1; // 100ms release
            if (t < attack) {
                sineData[i] *= t / attack;
            } else if (t > duration - release) {
                sineData[i] *= (duration - t) / release;
            }
        }

        // Create audio nodes
        const sourceNode = this.audioCtx.createBufferSource();
        sourceNode.buffer = sineBuffer;

        const convolverNode = this.audioCtx.createConvolver();
        convolverNode.buffer = this.impulseResponseBuffer;

        const gainNode = this.audioCtx.createGain();
        gainNode.gain.value = 0.3; // Lower volume for sine wave

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
        console.log("Playing convolved sine wave");

    } catch (error) {
        console.error("Error playing convolved sound:", error);
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
        gainNode.gain.value = 0.2; // Lower volume for noise

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
   * Plays a sine wave through the impulse response.
   */
  async playConvolvedSineWave(): Promise<void> {
    try {
        if (!this.impulseResponseBuffer) {
            console.warn("No impulse response available. Generate one first.");
            return;
        }

        // Create sine wave buffer
        const duration = 2.0; // 2 seconds
        const frequency = 440; // 440 Hz (A4 note)
        const sineBuffer = this.audioCtx.createBuffer(
            1,
            this.sampleRate * duration,
            this.sampleRate
        );
        const sineData = sineBuffer.getChannelData(0);

        // Generate sine wave
        for (let i = 0; i < sineData.length; i++) {
            const t = i / this.sampleRate;
            sineData[i] = Math.sin(2 * Math.PI * frequency * t);

            // Apply envelope to avoid clicks
            const attack = 0.1; // 100ms attack
            const release = 0.1; // 100ms release
            if (t < attack) {
                sineData[i] *= t / attack;
            } else if (t > duration - release) {
                sineData[i] *= (duration - t) / release;
            }
        }

        // Create and connect audio nodes
        const sourceNode = this.audioCtx.createBufferSource();
        sourceNode.buffer = sineBuffer;

        const convolverNode = this.audioCtx.createConvolver();
        convolverNode.buffer = this.impulseResponseBuffer;

        const gainNode = this.audioCtx.createGain();
        gainNode.gain.value = 0.3; // Lower volume for sine wave

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
        console.log("Playing convolved sine wave");

    } catch (error) {
        console.error("Error playing convolved sine wave:", error);
    }
  }

  /**
   * Debug method to play a simple sine wave without convolution.
   */
  async debugPlaySineWave(): Promise<void> {
    try {
        const oscillator = this.audioCtx.createOscillator();
        const gainNode = this.audioCtx.createGain();

        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(440, this.audioCtx.currentTime);

        gainNode.gain.setValueAtTime(0, this.audioCtx.currentTime);
        gainNode.gain.linearRampToValueAtTime(0.3, this.audioCtx.currentTime + 0.1);
        gainNode.gain.linearRampToValueAtTime(0, this.audioCtx.currentTime + 2);

        oscillator.connect(gainNode);
        gainNode.connect(this.audioCtx.destination);

        // Resume audio context if suspended
        if (this.audioCtx.state === 'suspended') {
            await this.audioCtx.resume();
        }

        oscillator.start();
        oscillator.stop(this.audioCtx.currentTime + 2);
        console.log("Playing debug sine wave");

    } catch (error) {
        console.error("Error playing debug sine wave:", error);
    }
  }
}