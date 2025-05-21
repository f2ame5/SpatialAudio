// src/sound/audio-processor_modified.ts
import { Camera } from '../camera/camera';
import { Room } from '../room/room';
import { WaveformRenderer } from '../visualization/waveform-renderer';
import { RayHit, FrequencyBands } from '../raytracer/raytracer'; // Import FrequencyBands
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
        rayHits: RayHit[],
    ): Promise<void> {
        console.log(`[AP processRayHits] Received ${rayHits.length} ray hits.`);
        try {
            // ... (validation as before) ...
            if (!rayHits || !Array.isArray(rayHits) || rayHits.length === 0) {
                console.warn('[AP processRayHits] No valid ray hits to process'); return;
            }
            if (!this.diffuseFieldModel) {
                 console.error('[AP processRayHits] Audio components not initialized'); return;
            }

            const validHits = rayHits.filter(hit => hit && hit.position && hit.energies && isFinite(hit.time));
            if (validHits.length === 0) {
                console.warn('[AP processRayHits] No valid ray hits after filtering');
                return;
            }
            this.lastRayHits = validHits;
            console.log(`[AP processRayHits] Processing ${validHits.length} valid hits.`);


            const [leftIR, rightIR] = this.processRayHitsInternal(validHits);

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

    private processRayHitsInternal(hits: RayHit[]): [Float32Array, Float32Array] {
        const irLength = Math.max(Math.ceil(this.sampleRate * 2.5), 1000); // Increased IR length to 2.5s
        const leftIR = new Float32Array(irLength);
        const rightIR = new Float32Array(irLength);

        try {
            const sortedHits = [...hits].sort((a, b) => a.time - b.time);
            const earlyReflectionCutoffTime = 0.08; // 80ms for early part

            const earlyHits = sortedHits.filter(hit => hit.time < earlyReflectionCutoffTime);
            for (const hit of earlyHits) {
                const sampleIndex = Math.floor(hit.time * this.sampleRate);
                if (sampleIndex < 0 || sampleIndex >= irLength || !isFinite(sampleIndex)) {
                     continue;
                }

                const totalEnergy = Object.values(hit.energies).reduce((sum: number, e) => sum + (typeof e === 'number' ? e : 0), 0);
                const loudnessScale = 0.05; 
                let amplitude = Math.sqrt(Math.max(0, totalEnergy)) * Math.exp(-hit.bounces * 0.2) * loudnessScale;
                amplitude = Math.max(0, Math.min(1.0, amplitude));
                if (!isFinite(amplitude) || amplitude < 1e-6) {
                    continue;
                }
                
                const listenerPos = this.camera.getPosition();
                const direction = vec3.create();
                vec3.subtract(direction, hit.position, listenerPos);
                const distance = vec3.length(direction);
                vec3.normalize(direction, direction);

                const listenerRight = this.camera.getRight();
                const listenerFront = this.camera.getFront();
                const listenerUp = this.camera.getUp();

                const dotRight = vec3.dot(direction, listenerRight);
                const dotFront = vec3.dot(direction, listenerFront);
                const dotUp = vec3.dot(direction, listenerUp);

                const azimuthRad = Math.atan2(dotRight, dotFront);
                const elevationRad = Math.asin(Math.max(-1, Math.min(1, dotUp)));

                let [leftGain, rightGain] = this.calculateBalancedSpatialGains(azimuthRad, elevationRad, distance);
                if (!isFinite(leftGain) || !isFinite(rightGain)) {
                    leftGain = 0; rightGain = 0;
                }

                let itd_samples = this.calculateITDsamples(azimuthRad, this.sampleRate);
                if (!isFinite(itd_samples)) {
                    itd_samples = 0;
                }

                let leftDelaySamples = 0;
                let rightDelaySamples = 0;
                if (itd_samples > 0) {
                    rightDelaySamples = Math.round(itd_samples); 
                } else if (itd_samples < 0) {
                    leftDelaySamples = Math.round(-itd_samples); 
                }
                
                const spreadDurationSamples = Math.floor(this.sampleRate * 0.008); 
                const decayConstant = spreadDurationSamples > 0 ? spreadDurationSamples / 4 : 1; 

                let sumOfSpreadGains = 0;
                if (decayConstant > 0 && spreadDurationSamples > 0) {
                    for (let k_sum = 0; k_sum < spreadDurationSamples; k_sum++) {
                        sumOfSpreadGains += Math.exp(-k_sum / decayConstant);
                    }
                }
                if (sumOfSpreadGains < 1e-9) { 
                    sumOfSpreadGains = 1; 
                }

                const baseLeftIndex = sampleIndex + leftDelaySamples;
                for (let k = 0; k < spreadDurationSamples; k++) {
                    const fractionalIndex = baseLeftIndex + k;
                    const floorIndex = Math.floor(fractionalIndex);
                    const ceilIndex = Math.ceil(fractionalIndex);
                    const fraction = fractionalIndex - floorIndex;

                    const spreadGain = decayConstant > 0 ? Math.exp(-k / decayConstant) : 1;
                    const val = (amplitude * leftGain * spreadGain) / sumOfSpreadGains;

                    if (floorIndex >= 0 && floorIndex < irLength) {
                        leftIR[floorIndex] += val * (1 - fraction);
                    }
                    if (ceilIndex >= 0 && ceilIndex < irLength && fraction > 0 && floorIndex !== ceilIndex) { 
                        leftIR[ceilIndex] += val * fraction;
                    }
                }

                const baseRightIndex = sampleIndex + rightDelaySamples;
                 for (let k = 0; k < spreadDurationSamples; k++) {
                    const fractionalIndex = baseRightIndex + k;
                    const floorIndex = Math.floor(fractionalIndex);
                    const ceilIndex = Math.ceil(fractionalIndex);
                    const fraction = fractionalIndex - floorIndex;

                    const spreadGain = decayConstant > 0 ? Math.exp(-k / decayConstant) : 1;
                    const val = (amplitude * rightGain * spreadGain) / sumOfSpreadGains;

                    if (floorIndex >= 0 && floorIndex < irLength) {
                        rightIR[floorIndex] += val * (1 - fraction);
                    }
                    if (ceilIndex >= 0 && ceilIndex < irLength && fraction > 0 && floorIndex !== ceilIndex) { 
                        rightIR[ceilIndex] += val * fraction;
                    }
                }
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


            const lateHits = sortedHits.filter(hit => hit.time >= earlyReflectionCutoffTime);
            let generatedLateL = new Float32Array(0); // Initialize as empty
            let generatedLateR = new Float32Array(0);

            if (lateHits.length > 0 && this.diffuseFieldModel) {
                 const roomConfig = {
                     dimensions: { width: this.room.config.dimensions.width, height: this.room.config.dimensions.height, depth: this.room.config.dimensions.depth },
                     materials: this.room.config.materials
                 };
                 try {
                    [generatedLateL, generatedLateR] = this.diffuseFieldModel.processLateReverberation(
                        lateHits, this.camera, roomConfig, this.sampleRate
                    );
                } catch (e) {
                    console.error("Error generating late reverberation:", e);
                }
            }
            
            const rmsDFM_L = calculateRMS(generatedLateL);
            const rmsDFM_R = calculateRMS(generatedLateR);
            const avgRmsDFM = (rmsDFM_L + rmsDFM_R) / 2;
            if (avgRmsDFM < 1e-9) { // If DFM output is silent, no reverb gain needed
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


            const crossfadeEndSample = Math.floor((earlyReflectionCutoffTime + 0.04) * this.sampleRate); // 40ms crossfade
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

    public createConvolvedSource( /* ... */ ): { source: AudioBufferSourceNode, convolver: ConvolverNode, wetGain: GainNode } | null {
        // Method unchanged
        const audioBufferToConvolve = arguments[0];
        const impulseResponseBuffer = arguments[1];
        if (this.audioCtx.state === 'suspended') {
             this.audioCtx.resume().catch(err => console.error("Error resuming audio context:", err));
        }
        try {
            const convolver = this.audioCtx.createConvolver();
            convolver.normalize = false;
            convolver.buffer = impulseResponseBuffer;
            const source = this.audioCtx.createBufferSource();
            source.buffer = audioBufferToConvolve;
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
        dryGainNode.gain.value = 0.3; // 30% dry

        wetGain.gain.value = 0.7; // 70% wet

        source.connect(dryGainNode);
        dryGainNode.connect(this.audioCtx.destination);
        
        wetGain.connect(this.audioCtx.destination);


        this.currentSourceNode = source;
        source.onended = () => {
            if (this.currentSourceNode === source) this.currentSourceNode = null;
            try { 
                wetGain.disconnect(); 
                convolver.disconnect(); 
                source.disconnect(convolver); // Source was connected to convolver
                dryGainNode.disconnect();
                source.disconnect(dryGainNode); // Source was also connected to dryGainNode
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

    public stopAllSounds(): void {
        // Method unchanged
        if (this.currentSourceNode) {
            try { this.currentSourceNode.stop(); } catch (error) {} 
            this.currentSourceNode = null; 
        }
    }

    private calculateBalancedSpatialGains(
        azimuthRad: number, elevationRad: number, distance: number
    ): [number, number] {
        // Method unchanged
        const pi = Math.PI;
        const piOver2 = pi / 2;
        const clampedAzimuth = Math.max(-pi, Math.min(pi, azimuthRad));
        const sinAz = Math.sin(clampedAzimuth);

        const baseGain = 0.707; 
        let leftGain = baseGain * (1 - sinAz * 0.8); 
        let rightGain = baseGain * (1 + sinAz * 0.8);

        const elevationFactor = 1.0 - Math.abs(elevationRad) / piOver2 * 0.3;
        leftGain *= elevationFactor;
        rightGain *= elevationFactor;

        const distanceAtten = 1.0 / Math.max(1, distance);
        leftGain *= distanceAtten;
        rightGain *= distanceAtten;

        if (Math.abs(clampedAzimuth) > piOver2) {
            const backFactor = 0.8;
            leftGain *= backFactor;
            rightGain *= backFactor;
        }
        return [Math.max(0, Math.min(1.5, leftGain)), Math.max(0, Math.min(1.5, rightGain))];
    }

    private calculateITDsamples(azimuthRad: number, sampleRate: number): number {
        // Method unchanged
        const headRadius = 0.0875; 
        const speedOfSound = 343; 
        const clampedAzimuth = Math.max(-Math.PI, Math.min(Math.PI, azimuthRad));
        const maxITDSeconds = 0.00065; 
        const itdSeconds = maxITDSeconds * Math.sin(clampedAzimuth); 
        return Math.round(itdSeconds * sampleRate);
    }
}