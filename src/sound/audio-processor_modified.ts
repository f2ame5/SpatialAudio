// src/sound/audio-processor_modified.ts
import { Camera } from '../camera/camera';
import { Room } from '../room/room';
import { WaveformRenderer } from '../visualization/waveform-renderer';
import { RayHit, FrequencyBands } from '../raytracer/raytracer'; // Import FrequencyBands
import { FeedbackDelayNetwork } from './feedback-delay-network';
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
    private fdn: FeedbackDelayNetwork;
    private lastImpulseData: Float32Array | null = null;
    private sampleRate: number;
    private lastRayHits: RayHit[] = [];
    private lastRt60Values: { [freq: string]: number } | null = null;
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
        this.fdn = new FeedbackDelayNetwork(this.audioCtx, 16); // Initialize FDN
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
            if (!this.fdn) { // Check FDN instead of diffuseFieldModel for late reverb
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

    // Add these methods to the AudioProcessorModified class

    private getMeanAbsorptionForRoom(materials: any): { [freq: string]: number } {
        const result: { [freq: string]: number } = {
            '125': 0, '250': 0, '500': 0, '1000': 0,
            '2000': 0, '4000': 0, '8000': 0, '16000': 0
        };
        // let totalSurfaceAreaFactor = 0; // Used to weight material contributions if areas were different

        // Simplified: assumes materials object has 'walls', 'ceiling', 'floor' keys
        // and each contributes somewhat equally. A more advanced model might take actual surface areas.
        const surfaceTypes = ['walls', 'ceiling', 'floor'];
        let surfacesProcessed = 0;

        for (const type of surfaceTypes) {
            const material = materials[type];
            if (material) {
                surfacesProcessed++;
                result['125'] += (material as any).absorption125Hz || 0.1;
                result['250'] += (material as any).absorption250Hz || 0.1;
                result['500'] += (material as any).absorption500Hz || 0.1;
                result['1000'] += (material as any).absorption1kHz || 0.1;
                result['2000'] += (material as any).absorption2kHz || 0.1;
                result['4000'] += (material as any).absorption4kHz || 0.1;
                result['8000'] += (material as any).absorption8kHz || 0.1;
                result['16000'] += (material as any).absorption16kHz || 0.1;
            }
        }

        if (surfacesProcessed > 0) {
            for (const freq in result) {
                result[freq] /= surfacesProcessed;
            }
        } else { // Fallback if no materials defined
            for (const freq in result) result[freq] = 0.2; // Default average absorption
        }
        return result;
    }

    private calculateRt60ValuesForFDN(roomConfig: any): { [freq: string]: number } {
        const frequencies = ['125', '250', '500', '1000', '2000', '4000', '8000', '16000'];
        const rt60Values: { [freq: string]: number } = {};

        const meanAbsorption = this.getMeanAbsorptionForRoom(roomConfig.materials);

        const V = Math.max(roomConfig.dimensions.width * roomConfig.dimensions.height * roomConfig.dimensions.depth, 1.0);
        const S = Math.max(2 * (roomConfig.dimensions.width * roomConfig.dimensions.height +
                            roomConfig.dimensions.width * roomConfig.dimensions.depth +
                            roomConfig.dimensions.height * roomConfig.dimensions.depth), 6.0);

        for (const freq of frequencies) {
            const absorption = meanAbsorption[freq] || 0.2; // Default absorption if not found
            const effectiveAbsorption = Math.max(absorption, 0.01);
            let rt60 = 0.161 * V / (S * effectiveAbsorption); // Sabine's formula

            // Empirical adjustments (can be fine-tuned)
            if (parseInt(freq) < 500) rt60 *= 1.05; // Slightly longer decay for low frequencies
            else if (parseInt(freq) > 2000) rt60 *= 0.9; // Slightly shorter decay for high frequencies
            
            // tempRT60Log[`${freq}Hz_adjusted`] = rt60.toFixed(2);
            rt60Values[freq] = Math.min(Math.max(rt60, 0.05), 5.0); // Clamp RT60 (e.g., 0.05s to 5s)
            // tempRT60Log[`${freq}Hz_final`] = rt60Values[freq].toFixed(2);
        }
        // console.log("[AP FDN RT60] Calculated RT60s (s):", tempRT60Log);
        return rt60Values;
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
            // Late reverberation using FDN
            // Determine the length of the late reverberation part
            const lateIrLength = Math.max(1, irLength - crossfadeStartSample);
            let generatedLateL = new Float32Array(lateIrLength);
            let generatedLateR = new Float32Array(lateIrLength);

            if (lateHits.length > 0 && lateIrLength > 0 && this.fdn) { // Check if FDN is available
                const roomConfigForRT60 = {
                     dimensions: { width: this.room.config.dimensions.width, height: this.room.config.dimensions.height, depth: this.room.config.dimensions.depth },
                     materials: this.room.config.materials
                };
                
                try {
                    // Calculate RT60 values for the FDN
                    const rt60Values = this.calculateRt60ValuesForFDN(roomConfigForRT60);
                    this.lastRt60Values = rt60Values; // Store for dry/wet mix calculation
                    this.fdn.setRT60(rt60Values);
                    this.fdn.setDryWetMix(0.0, 1.0); // We want fully wet signal for reverb tail

                    // Create a mono impulse input signal for the FDN
                    const fdnInputMono = new Float32Array(lateIrLength);
                    if (fdnInputMono.length > 0) {
                        fdnInputMono[0] = 0.5; // Impulse, scaled down to avoid overly loud FDN output initially
                                               // This may need tuning.
                    }

                    // Process the mono impulse with FDN's stereo processor
                    // This relies on the FDN's internal structure to create a stereo output
                    [generatedLateL, generatedLateR] = this.fdn.processStereo(fdnInputMono, fdnInputMono);
                    
                    console.log(`[AP processInternal] FDN generated late reverb. Length: ${generatedLateL.length}`);

                } catch (e) {
                    console.error("Error generating late reverberation with FDN:", e);
                    // Fallback to empty if FDN fails
                    generatedLateL = new Float32Array(lateIrLength);
                    generatedLateR = new Float32Array(lateIrLength);
                }
            } else {
                // Fallback if no late hits, insufficient length, or FDN not present
                if (lateIrLength > 0) {
                    generatedLateL = new Float32Array(lateIrLength);
                    generatedLateR = new Float32Array(lateIrLength);
                }
                 console.warn("[AP processInternal] FDN not used for late reverb (no late hits, insufficient length, or FDN missing).");
            }
            
            const rmsFdnL = calculateRMS(generatedLateL);
            const rmsFdnR = calculateRMS(generatedLateR);
            const avgRmsLate = (rmsFdnL + rmsFdnR) / 2;
            
            if (avgRmsLate < 1e-9 && generatedLateL.length > 0) {
                 console.warn("[AP processInternal] FDN output is silent or near silent. Adding a small impulse to prevent zero division.");
                 generatedLateL[0] = 1e-9; // Add tiny impulse to avoid NaN/Infinity gain
                 generatedLateR[0] = 1e-9;
            }

            const desiredLateToEarlyRMS  = 0.5; 
            let calculatedLateReverbGain = (avgRmsLate > 1e-9) ? (desiredLateToEarlyRMS * avgRmsEarly) / avgRmsLate : 0.0;

            const currentRoomVolume = this.room.config.dimensions.width * this.room.config.dimensions.height * this.room.config.dimensions.depth;
            const referenceRoomVolumeForScaling = 150; 
            const volumeScaleFactor = Math.sqrt(currentRoomVolume / referenceRoomVolumeForScaling);
            const roomSizeGainModulator = Math.min(1.5, Math.max(0.3, volumeScaleFactor)); 

            calculatedLateReverbGain *= roomSizeGainModulator;
            
            // IMPORTANT: FDN output might be louder or softer than DFM.
            // This gain might need adjustment, or the FDN impulse input (fdnInputMono[0])
            // needs to be scaled appropriately. Start with a modest gain.
            const MAX_FDN_GAIN = 2.0; // Cap the gain to prevent excessive loudness
            const lateReverbGain = Math.max(0.0, Math.min(MAX_FDN_GAIN, calculatedLateReverbGain)); 
            
            console.log(`[AP processInternal] RMS Early: ${avgRmsEarly.toExponential(3)}, RMS FDN Out (pre-gain): ${avgRmsLate.toExponential(3)}, TargetRatio: ${desiredLateToEarlyRMS}, RoomMod: ${roomSizeGainModulator.toFixed(3)}, CalcGain: ${calculatedLateReverbGain.toExponential(3)}, Final LateReverbGain (FDN): ${lateReverbGain.toExponential(3)}`);


            const crossfadeEndSample = Math.floor((earlyReflectionCutoffTime + 0.04) * this.sampleRate); // 40ms crossfade
            const crossfadeDuration = Math.max(1, crossfadeEndSample - crossfadeStartSample);

            if (generatedLateL.length > 0 && generatedLateR.length > 0) {
                for (let i = 0; i < irLength; i++) {
                    // Note: The original code applies late reverb from index 0 of generatedLateL/R
                    // This needs to map to the correct position in the main IR (leftIR/rightIR)
                    const mainIrIndex = i; // Current index in the full IR
                    const lateIrBufferIndex = mainIrIndex - crossfadeStartSample; // Index for generatedLateL/R

                    if (mainIrIndex < crossfadeStartSample) {
                        // Only early reflections
                        continue;
                    } else if (mainIrIndex >= crossfadeStartSample && mainIrIndex < crossfadeEndSample) {
                        // Crossfade region
                        const fadePos = (mainIrIndex - crossfadeStartSample) / crossfadeDuration;
                        const earlyGainFactor = 0.5 * (1 + Math.cos(fadePos * Math.PI));
                        const diffuseGainFactor = 0.5 * (1 - Math.cos(fadePos * Math.PI));

                        const currentLateL = (lateIrBufferIndex >= 0 && lateIrBufferIndex < generatedLateL.length) ? generatedLateL[lateIrBufferIndex] * lateReverbGain : 0;
                        const currentLateR = (lateIrBufferIndex >= 0 && lateIrBufferIndex < generatedLateR.length) ? generatedLateR[lateIrBufferIndex] * lateReverbGain : 0;

                        leftIR[mainIrIndex] = leftIR[mainIrIndex] * earlyGainFactor + currentLateL * diffuseGainFactor;
                        rightIR[mainIrIndex] = rightIR[mainIrIndex] * earlyGainFactor + currentLateR * diffuseGainFactor;
                    } else {
                        // Only late reverberation
                         const currentLateL = (lateIrBufferIndex >= 0 && lateIrBufferIndex < generatedLateL.length) ? generatedLateL[lateIrBufferIndex] * lateReverbGain : 0;
                        const currentLateR = (lateIrBufferIndex >= 0 && lateIrBufferIndex < generatedLateR.length) ? generatedLateR[lateIrBufferIndex] * lateReverbGain : 0;
                        leftIR[mainIrIndex] = currentLateL;
                        rightIR[mainIrIndex] = currentLateR;
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

    private getAverageRT60(): number {
        if (!this.lastRt60Values || Object.keys(this.lastRt60Values).length === 0) {
            console.warn("[AP getAverageRT60] lastRt60Values not available, using fallback calculation.");
            const roomDims = this.room.config.dimensions;
            const V = (roomDims.width || 10) * (roomDims.height || 3) * (roomDims.depth || 10);
            const S = 2 * (
                (roomDims.width || 10) * (roomDims.height || 3) +
                (roomDims.width || 10) * (roomDims.depth || 10) +
                (roomDims.height || 3) * (roomDims.depth || 10)
            );
            if (V <= 0 || S <= 0) return 1.0;
            const avgAbs = 0.2; // Generic placeholder average absorption
            let estimatedAvgRT60 = (0.161 * V) / (S * avgAbs);
            return Math.max(0.1, Math.min(3.0, estimatedAvgRT60)); // Clamp
        }

        const values = Object.values(this.lastRt60Values).filter(v => typeof v === 'number' && isFinite(v));
        if (values.length === 0) return 1.0;

        // Prefer average of mid-frequencies for a more representative single RT60 value
        const midFreqRT60 = (this.lastRt60Values['500'] + this.lastRt60Values['1000'] + this.lastRt60Values['2000']) / 3;
        if (isFinite(midFreqRT60) && midFreqRT60 > 0.05) return Math.min(3.0, midFreqRT60);

        const sum = values.reduce((acc, val) => acc + val, 0);
        return Math.min(3.0, Math.max(0.1, sum / values.length));
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
            
            const avgRT60 = this.getAverageRT60(); 

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