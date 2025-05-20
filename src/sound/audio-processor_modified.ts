// src/sound/audio-processor_modified.ts
// Integrates HRTFProcessor and DiffuseFieldModelModified for improved realism

import { Camera } from '../camera/camera';
import { Room } from '../room/room';
import { WaveformRenderer } from '../visualization/waveform-renderer';
import { RayHit } from '../raytracer/raytracer'; // Assume RayHit is here
import { DiffuseFieldModelModified } from './diffuse-field-model_modified';
import { vec3 } from 'gl-matrix';

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
             materials: room.config.materials || {
                 walls: { absorption125Hz: 0.1, absorption250Hz: 0.1, absorption500Hz: 0.1, absorption1kHz: 0.1, absorption2kHz: 0.1, absorption4kHz: 0.1, absorption8kHz: 0.1, absorption16kHz: 0.1, scattering125Hz: 0.1, scattering250Hz: 0.2, scattering500Hz: 0.3, scattering1kHz: 0.4, scattering2kHz: 0.5, scattering4kHz: 0.6, scattering8kHz: 0.6, scattering16kHz: 0.7, roughness: 0.5, phaseShift: 0, phaseRandomization: 0 },
                 ceiling: { absorption125Hz: 0.15, absorption250Hz: 0.15, absorption500Hz: 0.15, absorption1kHz: 0.15, absorption2kHz: 0.15, absorption4kHz: 0.15, absorption8kHz: 0.15, absorption16kHz: 0.15, scattering125Hz: 0.1, scattering250Hz: 0.2, scattering500Hz: 0.3, scattering1kHz: 0.4, scattering2kHz: 0.5, scattering4kHz: 0.6, scattering8kHz: 0.6, scattering16kHz: 0.7, roughness: 0.5, phaseShift: 0, phaseRandomization: 0 },
                 floor: { absorption125Hz: 0.05, absorption250Hz: 0.05, absorption500Hz: 0.05, absorption1kHz: 0.05, absorption2kHz: 0.05, absorption4kHz: 0.05, absorption8kHz: 0.05, absorption16kHz: 0.05, scattering125Hz: 0.1, scattering250Hz: 0.2, scattering500Hz: 0.3, scattering1kHz: 0.4, scattering2kHz: 0.5, scattering4kHz: 0.6, scattering8kHz: 0.6, scattering16kHz: 0.7, roughness: 0.5, phaseShift: 0, phaseRandomization: 0 }
             }
         };
        this.diffuseFieldModel = new DiffuseFieldModelModified(this.sampleRate, roomConfigForModel);
    }

    async processRayHits(
        rayHits: RayHit[],
    ): Promise<void> {
        try {
            if (!rayHits || !Array.isArray(rayHits) || rayHits.length === 0) {
                console.warn('No valid ray hits to process'); return;
            }
            if (!this.diffuseFieldModel) {
                 console.error('Audio components not initialized'); return;
            }

            const validHits = rayHits.filter(hit => hit && hit.position && hit.energies && isFinite(hit.time));
            if (validHits.length === 0) {
                console.warn('No valid ray hits after filtering');
                return;
            }

            this.lastRayHits = validHits;

            const [leftIR, rightIR] = this.processRayHitsInternal(validHits);

            const stereoData = new Float32Array(leftIR.length * 2);
            for (let i = 0; i < leftIR.length; i++) {
                stereoData[i * 2] = leftIR[i];
                stereoData[i * 2 + 1] = rightIR[i];
            }
            this.lastImpulseData = stereoData;

            await this.setupImpulseResponseBuffer(leftIR, rightIR);
        } catch (error) {
            console.error('Error processing ray hits:', error);
            throw error;
        }
    }

    private processRayHitsInternal(hits: RayHit[]): [Float32Array, Float32Array] {
        const irLength = Math.max(Math.ceil(this.sampleRate * 2), 1000);
        const leftIR = new Float32Array(irLength);
        const rightIR = new Float32Array(irLength);

        try {
            const sortedHits = [...hits].sort((a, b) => a.time - b.time);
            const earlyReflectionCutoff = 0.1; // 100ms

            const earlyHits = sortedHits.filter(hit => hit.time < earlyReflectionCutoff);
            for (const hit of earlyHits) {
                const sampleIndex = Math.floor(hit.time * this.sampleRate);
                if (sampleIndex < 0 || sampleIndex >= irLength || !isFinite(sampleIndex)) {
                     continue;
                }

                const totalEnergy = Object.values(hit.energies).reduce((sum: number, e) => sum + (typeof e === 'number' ? e : 0), 0);
                const loudnessScale = 0.05; // Reduced from 0.15
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
                    rightDelaySamples = Math.round(itd_samples); // Ensure integer
                } else if (itd_samples < 0) {
                    leftDelaySamples = Math.round(-itd_samples); // Ensure integer
                }
                
                // Temporal spreading (e.g., 20ms, 5ms caused issues before)
                const spreadDurationSamples = Math.floor(this.sampleRate * 0.008); // 8ms spread
                const decayConstant = spreadDurationSamples / 4; // Faster decay for shorter spread

                // Apply to Left Channel with fractional delay
                const baseLeftIndex = sampleIndex + leftDelaySamples;
                for (let k = 0; k < spreadDurationSamples; k++) {
                    const fractionalIndex = baseLeftIndex + k;
                    const floorIndex = Math.floor(fractionalIndex);
                    const ceilIndex = Math.ceil(fractionalIndex);
                    const fraction = fractionalIndex - floorIndex;

                    const spreadGain = Math.exp(-k / decayConstant);
                    const val = amplitude * leftGain * spreadGain;

                    if (floorIndex >= 0 && floorIndex < irLength) {
                        leftIR[floorIndex] += val * (1 - fraction);
                    }
                    if (ceilIndex >= 0 && ceilIndex < irLength && fraction > 0) {
                        leftIR[ceilIndex] += val * fraction;
                    }
                }

                // Apply to Right Channel with fractional delay
                const baseRightIndex = sampleIndex + rightDelaySamples;
                 for (let k = 0; k < spreadDurationSamples; k++) {
                    const fractionalIndex = baseRightIndex + k;
                    const floorIndex = Math.floor(fractionalIndex);
                    const ceilIndex = Math.ceil(fractionalIndex);
                    const fraction = fractionalIndex - floorIndex;

                    const spreadGain = Math.exp(-k / decayConstant);
                    const val = amplitude * rightGain * spreadGain;

                    if (floorIndex >= 0 && floorIndex < irLength) {
                        rightIR[floorIndex] += val * (1 - fraction);
                    }
                    if (ceilIndex >= 0 && ceilIndex < irLength && fraction > 0) {
                        rightIR[ceilIndex] += val * fraction;
                    }
                }
            }

            const lateHits = sortedHits.filter(hit => hit.time >= earlyReflectionCutoff);
            let lateDiffuseL = new Float32Array(irLength);
            let lateDiffuseR = new Float32Array(irLength);

            if (lateHits.length > 0 && this.diffuseFieldModel) {
                 const roomConfig = {
                     dimensions: { width: this.room.config.dimensions.width, height: this.room.config.dimensions.height, depth: this.room.config.dimensions.depth },
                     materials: this.room.config.materials
                 };
                 try {
                    const [generatedLateL, generatedLateR] = this.diffuseFieldModel.processLateReverberation(
                        lateHits, this.camera, roomConfig, this.sampleRate
                    );
                    const copyLength = Math.min(irLength, generatedLateL.length);
                    lateDiffuseL.set(generatedLateL.slice(0, copyLength));
                    lateDiffuseR.set(generatedLateR.slice(0, copyLength));
                } catch (e) {
                    console.error("Error generating late reverberation:", e);
                }
            }

            const crossfadeStartSample = Math.floor(0.08 * this.sampleRate);
            const crossfadeEndSample = Math.floor(0.12 * this.sampleRate);
            const crossfadeDuration = Math.max(1, crossfadeEndSample - crossfadeStartSample);
            
            // --- Adaptive Late Reverb Gain ---
            const baseLateReverbGain = 0.008; // Base gain, was 0.005
            const currentRoomVolume = this.room.config.dimensions.width * this.room.config.dimensions.height * this.room.config.dimensions.depth;
            const referenceRoomVolume = 10 * 5 * 3; // Approx 150 m^3 for a medium room
            // Scale gain with square root of volume ratio, clamp to prevent excessive gain in large rooms or too little in tiny ones
            const volumeRatio = currentRoomVolume / referenceRoomVolume;
            // Let gain scale down for smaller rooms, but not up beyond 1.0 for larger rooms
            // And ensure it doesn't become too small for very tiny rooms
            const adaptiveGainScale = Math.min(1.0, Math.sqrt(volumeRatio)); 
            const lateReverbGain = baseLateReverbGain * Math.max(0.2, adaptiveGainScale); // Ensure gain is at least 20% of base


            if (lateDiffuseL.length > 0 && lateDiffuseR.length > 0) {
                for (let i = 0; i < irLength; i++) {
                    const lateL = (i < lateDiffuseL.length) ? lateDiffuseL[i] * lateReverbGain : 0;
                    const lateR = (i < lateDiffuseR.length) ? lateDiffuseR[i] * lateReverbGain : 0;

                    if (i < crossfadeStartSample) {
                        continue;
                    } else if (i >= crossfadeStartSample && i < crossfadeEndSample) {
                        const fadePos = (i - crossfadeStartSample) / crossfadeDuration;
                        const earlyGain = 0.5 * (1 + Math.cos(fadePos * Math.PI));
                        const diffuseGain = 0.5 * (1 - Math.cos(fadePos * Math.PI));
                        leftIR[i] = leftIR[i] * earlyGain + lateL * diffuseGain;
                        rightIR[i] = rightIR[i] * earlyGain + lateR * diffuseGain;
                    } else {
                        leftIR[i] = lateL;
                        rightIR[i] = lateR;
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

    private sanitizeIRBuffers(leftIR: Float32Array, rightIR: Float32Array): void {
        for (let i = 0; i < leftIR.length; i++) {
            if (!isFinite(leftIR[i])) leftIR[i] = 0;
            if (!isFinite(rightIR[i])) rightIR[i] = 0;
        }

        let maxValue = 0;
        for (let i = 0; i < leftIR.length; i++) {
            maxValue = Math.max(maxValue, Math.abs(leftIR[i]), Math.abs(rightIR[i]));
        }
        
        const targetPeak = 0.85; // Slightly reduced target peak from 0.9
        if (maxValue > targetPeak) { // Normalize only if exceeding target peak
            const gainFactor = (maxValue > 0) ? targetPeak / maxValue : 1.0;
            if (gainFactor < 1.0) { // Only apply if reducing gain
                 for (let i = 0; i < leftIR.length; i++) {
                     leftIR[i] *= gainFactor;
                     rightIR[i] *= gainFactor;
                 }
            }
        }

        if (leftIR.length > 0) leftIR[0] = 0;
        if (rightIR.length > 0) rightIR[0] = 0;

        const fadeSamples = Math.min(Math.floor(this.sampleRate * 0.005), 50);
        if (fadeSamples > 0 && leftIR.length > fadeSamples * 2) {
            for (let i = 0; i < fadeSamples; i++) {
                const fadeGain = i / fadeSamples;
                leftIR[i] *= fadeGain;
                rightIR[i] *= fadeGain;
                const endIdx = leftIR.length - 1 - i;
                leftIR[endIdx] *= fadeGain;
                rightIR[endIdx] *= fadeGain;
            }
        }
    }

     private async setupImpulseResponseBuffer(leftIR: Float32Array, rightIR: Float32Array): Promise<void> {
        try {
            if (!leftIR || !rightIR || leftIR.length === 0 || rightIR.length === 0) {
                this.impulseResponseBuffer = null; return;
            }
            if (leftIR.length !== rightIR.length) {
                const maxLength = Math.max(leftIR.length, rightIR.length);
                const newLeftIR = new Float32Array(maxLength); newLeftIR.set(leftIR); leftIR = newLeftIR;
                const newRightIR = new Float32Array(maxLength); newRightIR.set(rightIR); rightIR = newRightIR;
            }

            let hasInvalidValues = false;
            for (let i = 0; i < leftIR.length; i++) {
                if (!isFinite(leftIR[i])) { leftIR[i] = 0; hasInvalidValues = true; }
                if (!isFinite(rightIR[i])) { rightIR[i] = 0; hasInvalidValues = true; }
            }

            let hasContent = false;
            for (let i = 0; i < leftIR.length; i++) {
                if (Math.abs(leftIR[i]) > 1e-10 || Math.abs(rightIR[i]) > 1e-10) {
                    hasContent = true; break;
                }
            }
            if (!hasContent) { leftIR[0] = 0.01; rightIR[0] = 0.01; } // Minimal impulse if empty

            if (this.audioCtx.state === 'suspended') await this.audioCtx.resume();
            this.impulseResponseBuffer = this.audioCtx.createBuffer(2, leftIR.length, this.audioCtx.sampleRate);
            this.impulseResponseBuffer.copyToChannel(leftIR, 0);
            this.impulseResponseBuffer.copyToChannel(rightIR, 1);
        } catch (error) {
            console.error('Error setting up impulse response buffer:', error);
            this.impulseResponseBuffer = null;
            try { // Fallback buffer
                const minLength = Math.ceil(0.1 * this.sampleRate); // Shorter fallback
                const fb = this.audioCtx.createBuffer(2, minLength, this.audioCtx.sampleRate);
                fb.getChannelData(0)[0] = 0.01; fb.getChannelData(1)[0] = 0.01;
                this.impulseResponseBuffer = fb;
            } catch (fbError) { console.error('Fallback buffer creation failed:', fbError); }
        }
    }

    public getImpulseResponseBuffer(): AudioBuffer | null {
        return this.impulseResponseBuffer;
    }

    public async visualizeImpulseResponse(renderer: WaveformRenderer): Promise<void> {
        if (this.lastImpulseData) {
            await renderer.drawWaveformWithFFT(this.lastImpulseData);
        }
    }

    public createConvolvedSource(
        audioBufferToConvolve: AudioBuffer,
        impulseResponseBuffer: AudioBuffer
    ): { source: AudioBufferSourceNode, convolver: ConvolverNode, wetGain: GainNode } | null {
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
            console.error('Error creating convolved source nodes:', error);
            return null;
        }
    }

    public async loadAudioFile(url: string): Promise<AudioBuffer> {
        try {
            const response = await fetch(url);
            const arrayBuffer = await response.arrayBuffer();
            return await this.audioCtx.decodeAudioData(arrayBuffer);
        } catch (error) {
            console.error('Error loading audio file:', error);
            throw error;
        }
    }

    public async playAudioWithIR(audioBuffer: AudioBuffer): Promise<void> {
        this.stopAllSounds();
        if (!this.impulseResponseBuffer) {
            console.warn('No impulse response buffer available for playback.');
            // Optionally play dry sound as fallback
            // const source = this.audioCtx.createBufferSource();
            // source.buffer = audioBuffer;
            // source.connect(this.audioCtx.destination);
            // source.start(0);
            // this.currentSourceNode = source;
            return;
        }

        try {
            const nodes = this.createConvolvedSource(audioBuffer, this.impulseResponseBuffer);
            if (!nodes) return;

            const { source, wetGain } = nodes;
            wetGain.connect(this.audioCtx.destination);
            this.currentSourceNode = source;
            source.onended = () => {
                if (this.currentSourceNode === source) this.currentSourceNode = null;
                try { wetGain.disconnect(); nodes.convolver.disconnect(); source.disconnect(); } catch (e) {}
            };
            source.start(0);
        } catch (error) {
            console.error('Error playing audio with IR:', error);
            this.currentSourceNode = null;
        }
    }

    public stopAllSounds(): void {
        if (this.currentSourceNode) {
            try { this.currentSourceNode.stop(); } catch (error) {} // onended will handle cleanup
            this.currentSourceNode = null; // Clear immediately
        }
    }

    private calculateBalancedSpatialGains(
        azimuthRad: number, elevationRad: number, distance: number
    ): [number, number] {
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
        const headRadius = 0.0875; 
        const speedOfSound = 343; 
        const clampedAzimuth = Math.max(-Math.PI, Math.min(Math.PI, azimuthRad));
        // Woodworth's formula: ITD = (r/c) * (theta + sin(theta)) - simplified to r/c * sin(theta) for many models or direct r/c * theta for extremes
        // Using (r/c) * (clampedAzimuth + sin(clampedAzimuth)) can lead to overly large ITDs for side angles.
        // A common simplification: ITD approx (headRadius / speedOfSound) * sin(clampedAzimuth). Let's use a more direct approach related to path difference.
        // Max ITD is around 0.6-0.7ms. Path difference = 2 * headRadius * sin(angle_from_front_if_small) or more complex.
        // Max ITD (direct side) = (PI/2 * headRadius - (-headRadius)) / speedOfSound vs (PI/2 * headRadius + headRadius) / speedOfSound.
        // Or more simply, max path diff is diameter + some ear effect = slightly more than 2*r.
        // A common simplification is ITD = (headRadius / speedOfSound) * (clampedAzimuth + Math.sin(clampedAzimuth));
        // However, a simpler and often effective model for ITD is just proportional to sin(azimuth) up to a max.
        const maxITDSeconds = 0.00065; // Max ITD around 0.65 ms
        const itdSeconds = maxITDSeconds * Math.sin(clampedAzimuth); // Simpler model based on sine of angle

        return Math.round(itdSeconds * sampleRate);
    }
}