// src/sound/audio-processor_modified.ts
import { Camera } from '../camera/camera';
import { Room } from '../room/room';
import { WaveformRenderer } from '../visualization/waveform-renderer';
import { RayHit } from '../raytracer/raytracer'; // Import RayHit
import { FrequencyBands } from '../raytracer/ray'; // Import FrequencyBands
import { DiffuseFieldModelModified } from './diffuse-field-model_modified';
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
    private diffuseFieldModel: DiffuseFieldModelModified;
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

        const roomConfigForModel = {
             dimensions: {
                 width: room.config.dimensions.width || 10,
                 height: room.config.dimensions.height || 3,
                 depth: room.config.dimensions.depth || 10
             },
             materials: room.config.materials
         };
        this.diffuseFieldModel = new DiffuseFieldModelModified(this.sampleRate, roomConfigForModel);
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

            if (!this.diffuseFieldModel) {
                 console.error('[AP processRayHits] Audio components not initialized'); return;
            }

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
        if (!this.diffuseFieldModel) return 1.0; // Default if DFM not ready

        // Need a roomConfig snapshot for calculateRT60Values
        // This assumes this.room.config is current.
        // DFM's calculateRT60Values is private, we'd need to expose it or replicate logic.
        // For simplicity, let's assume we can get an average RT60.
        // This part needs proper access to DFM's RT60 calculation or its results.
        // Let's simulate getting it for now, based on current room config:
         const roomConfigForRT60 = {
             dimensions: { 
                 width: this.room.config.dimensions.width, 
                 height: this.room.config.dimensions.height, 
                 depth: this.room.config.dimensions.depth 
             },
             materials: this.room.config.materials
         };
        // If DiffuseFieldModelModified.calculateRT60Values were public:
        // const rt60Values = this.diffuseFieldModel.calculateRT60Values(rayHitsForRT60, roomConfigForRT60);
        // For now, we'll use a placeholder logic if direct access isn't available,
        // or assume processRayHitsInternal has already triggered this and we can fetch from DFM.
        // To make this work cleanly, DiffuseFieldModelModified should probably store its last calculated avg RT60.
        // Or, AudioProcessorModified calls calculateRT60Values and passes results around.

        // Let's assume processRayHitsInternal will calculate and store avgRT60 for playAudioWithIR to use.
        // This requires adding a property like this.lastAverageRT60 = avgRT60;
        // For now, as a placeholder if that's not done:
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

    private processRayHitsInternal(leftEarHits: RayHit[], rightEarHits: RayHit[]): [Float32Array, Float32Array] {
        const irLength = Math.max(Math.ceil(this.sampleRate * 2.5), 1000); // Increased IR length to 2.5s
        const leftIR = new Float32Array(irLength);
        const rightIR = new Float32Array(irLength);

        try {
            const earlyReflectionCutoffTime = 0.08; // 80ms for early part

            // Process left ear hits
            const sortedLeftHits = [...leftEarHits].sort((a, b) => a.time - b.time);
            const earlyLeftHits = sortedLeftHits.filter(hit => hit.time < earlyReflectionCutoffTime);
            for (const hit of earlyLeftHits) {
                const sampleIndex = Math.floor(hit.time * this.sampleRate);
                if (sampleIndex < 0 || sampleIndex >= irLength || !isFinite(sampleIndex)) {
                    continue;
                }
                const totalEnergy = Object.values(hit.energies).reduce((sum: number, e) => sum + (typeof e === 'number' ? e : 0), 0);
                const loudnessScale = 0.05;
                let amplitude = Math.sqrt(Math.max(0, totalEnergy)) * loudnessScale;
                amplitude = Math.max(0, Math.min(1.0, amplitude));
                if (!isFinite(amplitude) || amplitude < 1e-6) {
                    continue;
                }
                leftIR[sampleIndex] += amplitude; // Add directly, ITD/ILD handled by ray tracer
            }

            // Process right ear hits
            const sortedRightHits = [...rightEarHits].sort((a, b) => a.time - b.time);
            const earlyRightHits = sortedRightHits.filter(hit => hit.time < earlyReflectionCutoffTime);
            for (const hit of earlyRightHits) {
                const sampleIndex = Math.floor(hit.time * this.sampleRate);
                if (sampleIndex < 0 || sampleIndex >= irLength || !isFinite(sampleIndex)) {
                    continue;
                }
                const totalEnergy = Object.values(hit.energies).reduce((sum: number, e) => sum + (typeof e === 'number' ? e : 0), 0);
                const loudnessScale = 0.05;
                let amplitude = Math.sqrt(Math.max(0, totalEnergy)) * loudnessScale;
                amplitude = Math.max(0, Math.min(1.0, amplitude));
                if (!isFinite(amplitude) || amplitude < 1e-6) {
                    continue;
                }
                rightIR[sampleIndex] += amplitude; // Add directly, ITD/ILD handled by ray tracer
            }

            const crossfadeStartSample = Math.floor(earlyReflectionCutoffTime * this.sampleRate);

            let sumSqEarlyL = 0, sumSqEarlyR = 0;
            let countNonZeroEarly = 0;
            for (let i = 0; i < crossfadeStartSample; i++) {
                sumSqEarlyL += leftIR[i] * leftIR[i];
                sumSqEarlyR += rightIR[i] * rightIR[i];
                if (Math.abs(leftIR[i]) > 1e-9 || Math.abs(rightIR[i]) > 1e-9) {
                    countNonZeroEarly++;
                }
            }
            const avgRmsEarly = (countNonZeroEarly > 10) ? Math.sqrt((sumSqEarlyL + sumSqEarlyR) / (2 * countNonZeroEarly)) : 1e-6;

            const lateLeftHits = sortedLeftHits.filter(hit => hit.time >= earlyReflectionCutoffTime);
            const lateRightHits = sortedRightHits.filter(hit => hit.time >= earlyReflectionCutoffTime);

            let generatedLateL = new Float32Array(0); // Initialize as empty
            let generatedLateR = new Float32Array(0);

            if ((lateLeftHits.length > 0 || lateRightHits.length > 0) && this.diffuseFieldModel) {
                 const roomConfig = {
                     dimensions: { width: this.room.config.dimensions.width, height: this.room.config.dimensions.height, depth: this.room.config.dimensions.depth },
                     materials: this.room.config.materials
                 };
                 try {
                    // Pass late hits for both ears to diffuse field model for RT60 calculation
                    // And pass combined late hits for frequency filtering
                    const combinedLateHitsForDFM = [...lateLeftHits, ...lateRightHits];

                    [generatedLateL, generatedLateR] = this.diffuseFieldModel.processLateReverberation(
                        combinedLateHitsForDFM, this.camera, roomConfig, this.sampleRate
                    );
                } catch (e) {
                    console.error("Error generating late reverberation:", e);
                }
            }
            
            const rmsDFM_L = calculateRMS(generatedLateL);
            const rmsDFM_R = calculateRMS(generatedLateR);
            const avgRmsDFM = (rmsDFM_L + rmsDFM_R) / 2;
            if (avgRmsDFM < 1e-9) {
                 console.warn("[AP processInternal] DFM output is silent or near silent.");
            }

            const desiredLateToEarlyRMS  = 0.5;
            let calculatedLateReverbGain = (avgRmsDFM > 1e-9) ? (desiredLateToEarlyRMS * avgRmsEarly) / avgRmsDFM : 0.0;

            const currentRoomVolume = this.room.config.dimensions.width * this.room.config.dimensions.height * this.room.config.dimensions.depth;
            const referenceRoomVolumeForScaling = 150;
            const volumeScaleFactor = Math.sqrt(currentRoomVolume / referenceRoomVolumeForScaling);
            const roomSizeGainModulator = Math.min(1.5, Math.max(0.3, volumeScaleFactor));

            calculatedLateReverbGain *= roomSizeGainModulator;
            
            const lateReverbGain = Math.max(0.0, Math.min(5.0, calculatedLateReverbGain));
            
            console.log(`[AP processInternal] RMS Early: ${avgRmsEarly.toExponential(3)}, RMS DFM Out: ${avgRmsDFM.toExponential(3)}, TargetRatio: ${desiredLateToEarlyRMS}, RoomMod: ${roomSizeGainModulator.toFixed(3)}, CalcGain: ${calculatedLateReverbGain.toExponential(3)}, Final LateReverbGain: ${lateReverbGain.toExponential(3)}`);


            const crossfadeEndSample = Math.floor((earlyReflectionCutoffTime + 0.04) * this.sampleRate);
            const crossfadeDuration = Math.max(1, crossfadeEndSample - crossfadeStartSample);

            if (generatedLateL.length > 0 && generatedLateR.length > 0) {
                for (let i = 0; i < irLength; i++) {
                    const lateL_contribution = (i < generatedLateL.length) ? generatedLateL[i] * lateReverbGain : 0;
                    const lateR_contribution = (i < generatedLateR.length) ? generatedLateR[i] * lateReverbGain : 0;

                    if (i < crossfadeStartSample) {
                        continue;
                    } else if (i >= crossfadeStartSample && i < crossfadeEndSample) {
                        const fadePos = (i - crossfadeStartSample) / crossfadeDuration;
                        const earlyGainFactor = 0.5 * (1 + Math.cos(fadePos * Math.PI));
                        const diffuseGainFactor = 0.5 * (1 - Math.cos(fadePos * Math.PI));
                        leftIR[i] = leftIR[i] * earlyGainFactor + lateL_contribution * diffuseGainFactor;
                        rightIR[i] = rightIR[i] * earlyGainFactor + lateR_contribution * diffuseGainFactor;
                    } else {
                        leftIR[i] = lateL_contribution;
                        rightIR[i] = lateR_contribution;
                    }
                }
            }
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
            console.warn('No impulse response buffer available for playback. Playing dry signal only.');
            const drySource = this.audioCtx.createBufferSource();
            drySource.buffer = audioBuffer;
            drySource.connect(this.audioCtx.destination);
            drySource.start(0);
            this.currentSourceNode = drySource;
            drySource.onended = () => {
                if (this.currentSourceNode === drySource) this.currentSourceNode = null;
                try { drySource.disconnect(); } catch (e) {}
            };
            return;
        }

        try {
            const nodes = this.createConvolvedSource(audioBuffer, this.impulseResponseBuffer);
            if (!nodes) return;

            const { source, convolver, wetGain } = nodes;

            const dryGainNode = this.audioCtx.createGain();
            
            const avgRT60 = this.getAverageRT60(this.lastRayHits); 

            const minRT60 = 0.2; 
            const maxRT60 = 2.5; 
            const normalizedRT60 = (avgRT60 - minRT60) / (maxRT60 - minRT60);
            const clampedNormalizedRT60 = Math.max(0.0, Math.min(1.0, normalizedRT60));

            let wetLevel = 0.4 + 0.5 * clampedNormalizedRT60; 
            let dryLevel = 1.0 - wetLevel;

            dryLevel = Math.max(0.1, Math.min(0.9, dryLevel)); 
            wetLevel = 1.0 - dryLevel;
            
            console.log(`[AP playAudioWithIR] AvgRT60: ${avgRT60.toFixed(2)}s, ClampedNormRT60: ${clampedNormalizedRT60.toFixed(2)}, Dry: ${dryLevel.toFixed(2)}, Wet: ${wetLevel.toFixed(2)}`);

            dryGainNode.gain.setValueAtTime(dryLevel, this.audioCtx.currentTime);
            wetGain.gain.setValueAtTime(wetLevel, this.audioCtx.currentTime);

            source.connect(dryGainNode);
            dryGainNode.connect(this.audioCtx.destination);
            
            wetGain.connect(this.audioCtx.destination);


            this.currentSourceNode = source;
            source.onended = () => {
                if (this.currentSourceNode === source) this.currentSourceNode = null;
                try { 
                    wetGain.disconnect(); 
                    convolver.disconnect(); 
                    source.disconnect(convolver); 
                    dryGainNode.disconnect();
                    source.disconnect(dryGainNode); 
                } catch (e) {
                    console.warn("Error during node cleanup onended:", e);
                }
            };
            source.start(0);
        } catch (error) {
            console.error('Error playing audio with IR:', error);
            this.currentSourceNode = null;
        }
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
        
        const targetPeak = 0.85; 
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
            this.impulseResponseBuffer.copyToChannel(leftIR, 0);
            this.impulseResponseBuffer.copyToChannel(rightIR, 1);
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

    const frequencies = [125, 250, 500, 1000, 2000, 4000, 8000, 16000];
    const energyKeys = Object.keys(baseGains) as Array<keyof typeof baseGains>;

    // Model Head Shadow (Interaural Level Difference - ILD)
    // This effect is more pronounced for higher frequencies
    const pathDifference = headRadius * (Math.abs(azimuthRad) + Math.sin(Math.abs(azimuthRad)));
    
    for (let i = 0; i < frequencies.length; i++) {
        const freq = frequencies[i];
        const key = energyKeys[i];
        const wavelength = speedOfSound / freq;
        
        // Attenuation is stronger when the wavelength is smaller than the path difference
        const shadowEffect = 1.0 - 0.7 * Math.min(1.0, Math.max(0, pathDifference / wavelength));
        
        if (azimuthRad > 0) { // Sound is from the right
            leftGains[key] *= shadowEffect;
        } else { // Sound is from the left
            rightGains[key] *= shadowEffect;
        }
    }
    
    // Model simple pinna effect for elevation (adds a bit of color)
    // This creates a subtle notch/peak based on elevation
    const elevationFactor = 1.0 - Math.abs(elevationRad) / (Math.PI / 2) * 0.2;
    for (const key of energyKeys) {
        leftGains[key] *= elevationFactor;
        rightGains[key] *= elevationFactor;
    }
    
    // Apply general distance attenuation
    const distanceAtten = 1.0 / Math.max(1, distance);
    for (const key of energyKeys) {
        leftGains[key] *= distanceAtten;
        rightGains[key] *= distanceAtten;
    }

    return { left: leftGains, right: rightGains };
}
}