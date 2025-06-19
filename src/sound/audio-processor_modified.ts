// src/sound/audio-processor_modified.ts
import { Camera } from '../camera/camera';
import { Room } from '../room/room';
import { WaveformRenderer } from '../visualization/waveform-renderer';
import { RayHit } from '../raytracer/raytracer'; // Import RayHit
import { FrequencyBands } from '../raytracer/ray'; // Import FrequencyBands
import { FeedbackDelayNetwork } from './feedback-delay-network';
import { vec3 } from 'gl-matrix';

// Helper to calculate RMS of a Float32Array
function calculateRMS_AP(data: Float32Array): number {
    if (!data || data.length === 0) return 0;
    const sumOfSquares = data.reduce((sum, val) => sum + val * val, 0);
    return Math.sqrt(sumOfSquares / data.length);
}

// Helper to find max absolute value in a Float32Array
function findMaxAbsValue(data: Float32Array): number {
    if (!data || data.length === 0) return 0;
    return data.reduce((max, val) => Math.max(max, Math.abs(val)), 0);
}
// (Helper function to calculate RMS - can be outside the class or a static method if preferred)
function calculateRMS(arr: Float32Array, countNonZeroThreshold: number = 10): number {
    let sumSq = 0;
    let count = 0;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] !== 0) {
            sumSq += arr[i] * arr[i];
            count++;
        }
    }
    if (count < countNonZeroThreshold) return 1e-9; // Return a very small number if not enough content
    return Math.sqrt(sumSq / count);
}

export class AudioProcessorModified {
    private audioCtx: AudioContext;
    private room: Room;
    private camera: Camera;
    private impulseResponseBuffer: AudioBuffer | null = null;
    private lastImpulseData: Float32Array | null = null;
    private sampleRate: number;
    private lastRayHits: RayHit[] = [];
    private currentSourceNode: AudioBufferSourceNode | null = null;

    constructor(audioCtx: AudioContext, room: Room, camera: Camera, sampleRate: number) {
        this.audioCtx = audioCtx;
        this.room = room;
        this.camera = camera;
        this.sampleRate = sampleRate;

    }

    async processRayHits(
        rayHits: [RayHit[], RayHit[]],
    ): Promise<void> {
        console.log(`[AP processRayHits] Received ray hits for left ear: ${rayHits[0].length}, right ear: ${rayHits[1].length}.`);
        try {
            if (!rayHits || !Array.isArray(rayHits) || rayHits.length !== 2 || !Array.isArray(rayHits[0]) || !Array.isArray(rayHits[1])) {
                console.warn('[AP processRayHits] Invalid ray hits format: expected [leftHits[], rightHits[]]'); return;
            }
            const [leftEarHits, rightEarHits] = rayHits;

            // Note: diffuseFieldModel is no longer used, replaced by FeedbackDelayNetwork in playAudioWithIR

            // Combine hits for backward compatibility with methods expecting a single list (e.g., getAverageRT60)
            const combinedHits = [...leftEarHits, ...rightEarHits].filter(hit => hit && hit.position && hit.energies && isFinite(hit.time));
            if (combinedHits.length === 0) {
                console.warn('[AP processRayHits] No valid ray hits after filtering');
                return;
            }
            this.lastRayHits = combinedHits; // Store combined hits for RT60 calculation
            console.log(`[AP processRayHits] Processing combined valid hits: ${combinedHits.length}.`);

            const [leftIR, rightIR] = this.processRayHitsInternal(leftEarHits, rightEarHits);

            const stereoData = new Float32Array(leftIR.length * 2);
            for (let i = 0; i < leftIR.length; i++) {
                stereoData[i * 2] = leftIR[i];
                stereoData[i * 2 + 1] = rightIR[i];
            }
            this.lastImpulseData = stereoData;

            await this.setupImpulseResponseBuffer(leftIR, rightIR);
        } catch (error) {
            console.error('[AP processRayHits] Error processing ray hits:', error);
            throw error;
        }
    }
    private getAverageRT60(rayHitsForRT60: RayHit[]): number {
        // Calculate an estimated RT60 based on room properties
        const V = this.room.config.dimensions.width * this.room.config.dimensions.height * this.room.config.dimensions.depth;
        const S = 2 * (
            this.room.config.dimensions.width * this.room.config.dimensions.height +
            this.room.config.dimensions.width * this.room.config.dimensions.depth +
            this.room.config.dimensions.height * this.room.config.dimensions.depth
        );
        // Simplified average absorption (e.g., at 1kHz)
        const avgAbs = (
            this.room.config.materials.walls.absorption1kHz +
            this.room.config.materials.ceiling.absorption1kHz +
            this.room.config.materials.floor.absorption1kHz
        ) / 3;
        const effectiveAvgAbs = Math.max(0.01, avgAbs);
        let estimatedAvgRT60 = (0.161 * V) / (S * effectiveAvgAbs);
        estimatedAvgRT60 = Math.max(0.1, Math.min(5.0, estimatedAvgRT60)); // Clamp
        return estimatedAvgRT60;
    }

// src/sound/audio-processor_modified.ts

private processRayHitsInternal(leftEarHits: RayHit[], rightEarHits: RayHit[]): [Float32Array, Float32Array] {
    const irLength = Math.max(Math.ceil(this.sampleRate * 2.5), 1000);
    const leftIR = new Float32Array(irLength);
    const rightIR = new Float32Array(irLength);

    try {
        const earlyReflectionCutoffTime = 0.08; // 80ms for early part
        const SPEED_OF_SOUND = 343.0; // m/s
        
        const allEarlyHits = [...leftEarHits, ...rightEarHits].filter(hit => hit.time < earlyReflectionCutoffTime);
        const uniqueHits = Array.from(new Map(allEarlyHits.map(hit => [hit.time.toString() + hit.position.toString(), hit])).values());

        const headPos = this.camera.getPosition();
        const headRight = this.camera.getRight();
        const headFront = this.camera.getFront();
        const headRadius = 0.0875; // Approx. radius of the head in meters

        for (const hit of uniqueHits) {
            const toHeadDir = vec3.subtract(vec3.create(), hit.position, headPos);
            vec3.normalize(toHeadDir, toHeadDir);

            const lateralness = vec3.dot(toHeadDir, headRight);
            
            const rightGain = Math.pow(0.5 * (1 + lateralness), 2);
            const leftGain = Math.pow(0.5 * (1 - lateralness), 2);

            const earLeftPos = vec3.scaleAndAdd(vec3.create(), headPos, headRight, -headRadius);
            const earRightPos = vec3.scaleAndAdd(vec3.create(), headPos, headRight, headRadius);

            const distToLeftEar = vec3.distance(hit.position, earLeftPos);
            const distToRightEar = vec3.distance(hit.position, earRightPos);
            
            const timeToLeftEar = hit.time + (distToLeftEar / SPEED_OF_SOUND);
            const timeToRightEar = hit.time + (distToRightEar / SPEED_OF_SOUND);

            const leftSampleIndex = Math.floor(timeToLeftEar * this.sampleRate);
            const rightSampleIndex = Math.floor(timeToRightEar * this.sampleRate);

            const totalEnergy = Object.values(hit.energies).reduce((sum: number, e) => sum + (typeof e === 'number' ? e : 0), 0);
            const amplitude = Math.sqrt(Math.max(0, totalEnergy));

            if (isFinite(amplitude) && amplitude > 1e-6) {
                const spreadSamples = 40;
                const spreadDecay = 20;

                for (let j = 0; j < spreadSamples; j++) {
                    const idx = leftSampleIndex + j;
                    if (idx >= 0 && idx < irLength) {
                        const spreadEnvelope = Math.exp(-j / spreadDecay);
                        leftIR[idx] += amplitude * leftGain * spreadEnvelope;
                    }
                }

                for (let j = 0; j < spreadSamples; j++) {
                    const idx = rightSampleIndex + j;
                    if (idx >= 0 && idx < irLength) {
                        const spreadEnvelope = Math.exp(-j / spreadDecay);
                        rightIR[idx] += amplitude * rightGain * spreadEnvelope;
                    }
                }
            }
        }

        // Late reverberation is now handled by FeedbackDelayNetwork in playAudioWithIR, not in the impulse response
        
        this.sanitizeIRBuffers(leftIR, rightIR);
        return [leftIR, rightIR];
    } catch (error) {
        console.error('Error in processRayHitsInternal:', error);
        return [new Float32Array(irLength), new Float32Array(irLength)];
    }
}

    public async playAudioWithIR(audioBuffer: AudioBuffer): Promise<void> {
        this.stopAllSounds();

        if (!this.impulseResponseBuffer) {
            console.warn('No impulse response buffer for early reflections.');
            return;
        }

        // --- 1. Create Nodes ---
        const source = this.audioCtx.createBufferSource();
        source.buffer = audioBuffer;

        // Early reflections part
        const earlyReflections = this.audioCtx.createConvolver();
        earlyReflections.normalize = false;
        earlyReflections.buffer = this.impulseResponseBuffer;

        // Late reverb part (using the new FDN)
        const fdn = new FeedbackDelayNetwork(this.audioCtx);
        const rt60 = this.getAverageRT60(this.lastRayHits); // Assumes you have a method for this
        fdn.setRT60({ '1000': rt60, '8000': rt60 * 0.6 });
        fdn.setMix(1.0); // Use FDN as a fully wet effect

        const masterOutput = this.audioCtx.createGain();
        masterOutput.connect(this.audioCtx.destination);
        
        // --- 2. Connect the Audio Graph ---
        
        // DRY PATH: For comparison, you might want a dry signal path
        // const dryGain = this.audioCtx.createGain();
        // dryGain.gain.value = 0.5;
        // source.connect(dryGain);
        // dryGain.connect(masterOutput);

        // WET PATH
        const wetGain = this.audioCtx.createGain();
        wetGain.gain.value = 0.8; // Adjust overall reverb level

        // Route audio through early reflections AND the FDN in parallel
        source.connect(earlyReflections);
        source.connect(fdn.input); // Send audio to the FDN

        earlyReflections.connect(wetGain);
        fdn.connect(wetGain); // Connect FDN output
        
        wetGain.connect(masterOutput);

        // --- 3. Play ---
        source.start(0);
        this.currentSourceNode = source;
    }

    private sanitizeIRBuffers(leftIR: Float32Array, rightIR: Float32Array): void {
        console.log(`[AP sanitizeIRBuffers] Before sanitization. RMS L: ${calculateRMS_AP(leftIR).toExponential(3)}, R: ${calculateRMS_AP(rightIR).toExponential(3)}`);
        let preSanitizeMaxL = findMaxAbsValue(leftIR);
        let preSanitizeMaxR = findMaxAbsValue(rightIR);
        console.log(`[AP sanitizeIRBuffers] MaxAbs L (pre-sanitize): ${preSanitizeMaxL.toExponential(3)}, R (pre-sanitize): ${preSanitizeMaxR.toExponential(3)}`);


        for (let i = 0; i < leftIR.length; i++) {
            if (!isFinite(leftIR[i])) leftIR[i] = 0;
            if (!isFinite(rightIR[i])) rightIR[i] = 0;
        }

        let maxValue = 0;
        for (let i = 0; i < leftIR.length; i++) {
            maxValue = Math.max(maxValue, Math.abs(leftIR[i]), Math.abs(rightIR[i]));
        }
        console.log(`[AP sanitizeIRBuffers] Max value before normalization: ${maxValue.toExponential(3)}`);
        
        const targetPeak = 0.9; // Adjusted from 0.85 to 0.9 for slightly higher dynamic range
        if (maxValue > targetPeak) { 
            const gainFactor = (maxValue > 0) ? targetPeak / maxValue : 1.0;
            if (gainFactor < 1.0) { 
                 for (let i = 0; i < leftIR.length; i++) {
                     leftIR[i] *= gainFactor;
                     rightIR[i] *= gainFactor;
                 }
                 console.log(`[AP sanitizeIRBuffers] Applied normalization. GainFactor: ${gainFactor.toExponential(3)}`);
            }
        }

        if (leftIR.length > 0) leftIR[0] = 0;
        if (rightIR.length > 0) rightIR[0] = 0;

        const fadeSamples = Math.min(Math.floor(this.sampleRate * 0.005), 50);
        if (fadeSamples > 0 && leftIR.length > fadeSamples * 2) {
            for (let i = 0; i < fadeSamples; i++) {
                const fadeGain = i / fadeSamples;
                leftIR[i] *= fadeGain; rightIR[i] *= fadeGain;
                const endIdx = leftIR.length - 1 - i;
                leftIR[endIdx] *= fadeGain; rightIR[endIdx] *= fadeGain;
            }
             console.log(`[AP sanitizeIRBuffers] Applied ${fadeSamples} samples fade-in/out.`);
        }
        console.log(`[AP sanitizeIRBuffers] After sanitization. RMS L: ${calculateRMS_AP(leftIR).toExponential(3)}, R: ${calculateRMS_AP(rightIR).toExponential(3)}`);
        console.log(`[AP sanitizeIRBuffers] MaxAbs L (post-sanitize): ${findMaxAbsValue(leftIR).toExponential(3)}, R (post-sanitize): ${findMaxAbsValue(rightIR).toExponential(3)}`);
    }

     private async setupImpulseResponseBuffer(leftIR: Float32Array, rightIR: Float32Array): Promise<void> {
        try {
            // ... (validation and setup as before) ...
            if (!leftIR || !rightIR || leftIR.length === 0 || rightIR.length === 0) {
                this.impulseResponseBuffer = null; console.warn("[AP setupIR] IR buffers empty/null."); return;
            }
            if (leftIR.length !== rightIR.length) { 
                const maxLength = Math.max(leftIR.length, rightIR.length);
                const newLeftIR = new Float32Array(maxLength); newLeftIR.set(leftIR); leftIR = newLeftIR;
                const newRightIR = new Float32Array(maxLength); newRightIR.set(rightIR); rightIR = newRightIR;
                console.warn("[AP setupIR] Padded IR buffers to matching length:", maxLength);
            }
            let hasInvalid = false; for (let i=0; i<leftIR.length; ++i) { if(!isFinite(leftIR[i])) {leftIR[i]=0; hasInvalid=true;} if(!isFinite(rightIR[i])) {rightIR[i]=0; hasInvalid=true;} } if(hasInvalid) console.warn("[AP setupIR] Invalid values found & zeroed in IR.");
            let hasContent = false; for (let i=0; i<leftIR.length; ++i) if(Math.abs(leftIR[i]) > 1e-10 || Math.abs(rightIR[i]) > 1e-10) {hasContent=true; break;} if(!hasContent) {leftIR[0]=0.01; rightIR[0]=0.01; console.warn("[AP setupIR] IR had no content, added minimal impulse.");}


            if (this.audioCtx.state === 'suspended') await this.audioCtx.resume();
            this.impulseResponseBuffer = this.audioCtx.createBuffer(2, leftIR.length, this.audioCtx.sampleRate);
            // Ensure the buffers are standard ArrayBuffer-backed Float32Arrays for copyToChannel
            const channelLeftIR = new Float32Array(leftIR);
            const channelRightIR = new Float32Array(rightIR);
            this.impulseResponseBuffer.copyToChannel(channelLeftIR, 0);
            this.impulseResponseBuffer.copyToChannel(channelRightIR, 1);
            console.log(`[AP setupIR] Impulse response buffer created/updated, length: ${(leftIR.length / this.sampleRate).toFixed(2)}s`);
        } catch (error) {
            // ... (error handling as before) ...
            console.error('[AP setupIR] Error setting up impulse response buffer:', error);
            this.impulseResponseBuffer = null;
            try { 
                const minLength = Math.ceil(0.1 * this.sampleRate); 
                const fb = this.audioCtx.createBuffer(2, minLength, this.sampleRate);
                fb.getChannelData(0)[0] = 0.01; fb.getChannelData(1)[0] = 0.01;
                this.impulseResponseBuffer = fb; console.log('[AP setupIR] Created fallback IR buffer.');
            } catch (fbError) { console.error('[AP setupIR] Fallback buffer creation failed:', fbError); }
        }
    }

    public getImpulseResponseBuffer(): AudioBuffer | null {
        return this.impulseResponseBuffer;
    }

    public async visualizeImpulseResponse(renderer: WaveformRenderer): Promise<void> {
        if (this.lastImpulseData) {
            await renderer.drawWaveformWithFFT(this.lastImpulseData);
        } else {
            console.warn('[AP visualize] No impulse data available to visualize.');
        }
    }

    public createConvolvedSource(audioBuffer: AudioBuffer, impulseResponseBuffer: AudioBuffer): { source: AudioBufferSourceNode, convolver: ConvolverNode, wetGain: GainNode } | null {
        // Method unchanged
        if (this.audioCtx.state === 'suspended') {
             this.audioCtx.resume().catch(err => console.error("Error resuming audio context:", err));
        }
        try {
            const convolver = this.audioCtx.createConvolver();
            convolver.normalize = false;
            convolver.buffer = impulseResponseBuffer;
            const source = this.audioCtx.createBufferSource();
            source.buffer = audioBuffer;
            const wetGain = this.audioCtx.createGain();
            wetGain.gain.value = 1.0; 
            source.connect(convolver);
            convolver.connect(wetGain);
            return { source, convolver, wetGain };
        } catch (error) {
            console.error('[AP createConvolvedSource] Error creating convolved source nodes:', error);
            return null;
        }
    }

    public async loadAudioFile(url: string): Promise<AudioBuffer> {
        // Method unchanged
        try {
            const response = await fetch(url);
            const arrayBuffer = await response.arrayBuffer();
            return await this.audioCtx.decodeAudioData(arrayBuffer);
        } catch (error) {
            console.error('[AP loadAudioFile] Error loading audio file:', error);
            throw error;
        }
    }


    public stopAllSounds(): void {
        // Method unchanged
        if (this.currentSourceNode) {
            try { this.currentSourceNode.stop(); } catch (error) {} 
            this.currentSourceNode = null; 
        }
    }


    private calculateProceduralHrtfGains(
        azimuthRad: number,
        elevationRad: number,
        distance: number
    ): { left: FrequencyBands, right: FrequencyBands } {
        const headRadius = 0.0875; // meters
        const speedOfSound = 343; // m/s
        
        // Initial gains (base level for each frequency)
        const baseGains: FrequencyBands = {
            energy125Hz: 1.0, energy250Hz: 1.0, energy500Hz: 1.0, energy1kHz: 1.0,
            energy2kHz: 1.0, energy4kHz: 1.0, energy8kHz: 1.0, energy16kHz: 1.0
        };
        
        let leftGains = { ...baseGains };
        let rightGains = { ...baseGains };

        // Apply ITD and ILD based on azimuth and distance
        // Simplified ITD: time difference between ears
        const ITD = (headRadius * (azimuthRad + Math.sin(azimuthRad))) / speedOfSound;
        
        // Apply ILD: Inter-aural Level Difference, frequency-dependent
        // High frequencies are more attenuated by the head shadow
        const K_ILD = 0.5; // ILD factor
        if (azimuthRad > 0) { // Sound is from the right
            leftGains.energy4kHz *= (1 - K_ILD * Math.abs(Math.sin(azimuthRad)));
            leftGains.energy8kHz *= (1 - 2 * K_ILD * Math.abs(Math.sin(azimuthRad)));
            leftGains.energy16kHz *= (1 - 3 * K_ILD * Math.abs(Math.sin(azimuthRad)));
        } else if (azimuthRad < 0) { // Sound is from the left
            rightGains.energy4kHz *= (1 - K_ILD * Math.abs(Math.sin(azimuthRad)));
            rightGains.energy8kHz *= (1 - 2 * K_ILD * Math.abs(Math.sin(azimuthRad)));
            rightGains.energy16kHz *= (1 - 3 * K_ILD * Math.abs(Math.sin(azimuthRad)));
        }

        // Apply spectral coloration based on elevation
        // Simplified: higher elevation might introduce some comb filtering or spectral notches
        const elevationFactor = 1.0 - Math.abs(elevationRad) / (Math.PI / 2) * 0.2; // Max 20% reduction at 90 deg elevation
        for (const freq in leftGains) {
            (leftGains as any)[freq] *= elevationFactor;
            (rightGains as any)[freq] *= elevationFactor;
        }

        // Apply distance attenuation (already handled by raytracer energy, but can add a small modulation)
        const distanceAtten = 1.0 / Math.max(1, distance);
        for (const freq in leftGains) {
            (leftGains as any)[freq] *= distanceAtten;
            (rightGains as any)[freq] *= distanceAtten;
        }

        return { left: leftGains, right: rightGains };
    }
}
