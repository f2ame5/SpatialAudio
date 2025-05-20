// src/sound/diffuse-field-model_modified.ts
// Enhanced diffuse field modeling with improved stereo decorrelation

import { vec3 } from 'gl-matrix';
import { Camera } from '../camera/camera';


export class DiffuseFieldModelModified {
    private sampleRate: number;
    private roomVolume: number;
    private surfaceArea: number;
    private meanAbsorption: { [freq: string]: number };
    private diffusionCoefficients: { [freq: string]: number };

    constructor(sampleRate: number, roomConfig: any) {
        this.sampleRate = sampleRate;
        const { width, height, depth } = roomConfig.dimensions;
        this.roomVolume = width * height * depth;
        this.surfaceArea = 2 * (width * height + width * depth + height * depth);
        this.meanAbsorption = this.calculateMeanAbsorption(roomConfig.materials);
        this.diffusionCoefficients = {
            '125': 0.1, '250': 0.2, '500': 0.3, '1000': 0.4,
            '2000': 0.5, '4000': 0.6, '8000': 0.6, '16000': 0.7
        };
    }

    private calculateMeanAbsorption(materials: any): { [freq: string]: number } {
        const result: { [freq: string]: number } = {
            '125': 0, '250': 0, '500': 0, '1000': 0,
            '2000': 0, '4000': 0, '8000': 0, '16000': 0
        };
        let surfaceCount = 0;
        for (const surfaceType of Object.keys(materials)) {
            const material = materials[surfaceType];
            if (material) {
                result['125'] += (material as any).absorption125Hz || 0.1;
                result['250'] += (material as any).absorption250Hz || 0.1;
                result['500'] += (material as any).absorption500Hz || 0.1;
                result['1000'] += (material as any).absorption1kHz || 0.1;
                result['2000'] += (material as any).absorption2kHz || 0.1;
                result['4000'] += (material as any).absorption4kHz || 0.1;
                result['8000'] += (material as any).absorption8kHz || 0.1;
                result['16000'] += (material as any).absorption16kHz || 0.1;
                surfaceCount++;
            }
        }
        if (surfaceCount > 0) {
            for (const freq in result) {
                result[freq] /= surfaceCount;
            }
        }
        return result;
    }

    public generateDiffuseField(
        duration: number,
        rt60Values: { [freq: string]: number }
    ): Map<string, Float32Array> {
        const result = new Map<string, Float32Array>();
        const frequencies = ['125', '250', '500', '1000', '2000', '4000', '8000', '16000'];
        const safeDuration = Math.max(duration, 0.2);

        for (const freq of frequencies) {
            const sampleCount = Math.max(Math.ceil(safeDuration * this.sampleRate), 512);
            const buffer = new Float32Array(sampleCount);
            const rt60 = rt60Values[freq] || 1.0;
            const diffusion = this.diffusionCoefficients[freq] || 0.3;
            const safeRoomVolume = Math.max(this.roomVolume, 1);
            const safeSurfaceArea = Math.max(this.surfaceArea, 6);
            const echoDensity = Math.max(500, 1000 * (safeRoomVolume / 100) * (1 + diffusion));
            const meanFreePath = 4 * safeRoomVolume / safeSurfaceArea;
            const speedOfSound = 343;
            const meanTimeGap = meanFreePath / speedOfSound;
            this.generateVelvetNoise(buffer, rt60, echoDensity, meanTimeGap, diffusion);
            result.set(freq, buffer);
        }
        return result;
    }

    private generateVelvetNoise(
        buffer: Float32Array, rt60: number, echoDensity: number,
        meanTimeGap: number, diffusion: number
    ): void {
        const safeEchoDensity = Math.max(echoDensity, 200);
        const td = 1 / safeEchoDensity;
        let totalPulses = Math.floor(buffer.length / (td * this.sampleRate));
        totalPulses = Math.max(totalPulses, 50);

        for (let i = 0; i < totalPulses; i++) {
            const basePosition = i * buffer.length / totalPulses;
            const jitter = (Math.random() - 0.5) * 2 * diffusion * buffer.length / totalPulses;
            const position = Math.floor(basePosition + jitter);

            if (position < 0 || position >= buffer.length) continue;
            const polarity = Math.random() > 0.5 ? 1 : -1;
            const time = position / this.sampleRate;
            const safeRt60 = Math.max(rt60, 0.01);
            const decayTerm = -6.91 * time / safeRt60;
            const amplitude = Math.exp(decayTerm);

            if (isNaN(amplitude)) {
                buffer[position] += polarity * 0;
            } else {
                buffer[position] += polarity * amplitude;
            }
        }
    }

    public applyFrequencyFiltering(
        impulseResponses: Map<string, Float32Array>
    ): Float32Array {
        const anyIR = impulseResponses.values().next().value;
        const totalLength = anyIR ? anyIR.length : 0;
        if (totalLength === 0) return new Float32Array(0);

        const outputIR = new Float32Array(totalLength);
        for (const [freq, ir] of impulseResponses.entries()) {
            let bandGain = 1.0;
            // Adjust gains to reduce low-end energy more significantly
            switch (freq) {
                case '125': bandGain = 0.5; break;  // Was 0.9, then 0.6
                case '250': bandGain = 0.65; break; // Was 0.95, then 0.7
                case '500': bandGain = 0.8; break; // Was 1.0, then 0.85
                case '1000': bandGain = 1.0; break; // Mid frequency reference
                case '2000': bandGain = 0.95; break; // Slightly reduced from 0.9 / previous 0.95
                case '4000': bandGain = 0.85; break; // Slightly reduced from 0.8 / previous 0.85
                case '8000': bandGain = 0.75; break; // Slightly reduced from 0.7 / previous 0.75
                case '16000': bandGain = 0.65; break;// Slightly reduced from 0.6 / previous 0.65
            }
            for (let i = 0; i < Math.min(ir.length, totalLength); i++) {
                outputIR[i] += ir[i] * bandGain;
            }
        }

        let maxAmp = 0;
        for (let i = 0; i < outputIR.length; i++) maxAmp = Math.max(maxAmp, Math.abs(outputIR[i]));
        if (maxAmp > 0) {
            const scale = 1.0 / maxAmp;
            for (let i = 0; i < outputIR.length; i++) outputIR[i] *= scale;
        }
        return outputIR;
    }

    public processLateReverberation(
        lateHits: any[], camera: Camera, roomConfig: any, sampleRate: number
    ): [Float32Array, Float32Array] {

        const { width, height, depth } = roomConfig.dimensions;
        this.roomVolume = width * height * depth;
        this.surfaceArea = 2 * (width * height + width * depth + height * depth);
        this.meanAbsorption = this.calculateMeanAbsorption(roomConfig.materials);

        if (!lateHits || lateHits.length === 0) {
            const minLength = Math.ceil(0.2 * sampleRate);
            const defaultIR = new Float32Array(minLength);
            return [defaultIR.slice(), defaultIR.slice()];
        }

        const rt60Values = this.calculateRT60Values(lateHits, roomConfig);
        const rt60NumericValues = Object.values(rt60Values).filter(val => typeof val === 'number' && isFinite(val));
        const maxRT60 = rt60NumericValues.length > 0 ? Math.max(...rt60NumericValues) : 1.0;
        const reverbGenerationDuration = Math.min(2.5, Math.max(maxRT60 * 1.2 + 0.1, 0.3));

        const diffuseResponses = this.generateDiffuseField(reverbGenerationDuration, rt60Values);
        const monoIR = this.applyFrequencyFiltering(diffuseResponses);

        if (!monoIR || monoIR.length === 0) {
            const minLength = Math.ceil(0.2 * sampleRate);
            const defaultIR = new Float32Array(minLength); defaultIR[0] = 0.01;
            return [defaultIR.slice(), defaultIR.slice()];
        }
        let hasContent = false;
        for (let i = 0; i < monoIR.length; i++) if (Math.abs(monoIR[i]) > 1e-10) { hasContent = true; break; }
        if (!hasContent) { monoIR[0] = 0.01; }

        const leftIR = new Float32Array(monoIR.length);
        const rightIR = new Float32Array(monoIR.length);
        const alpha = 0.98; 
        let prevL = 0; let prevR = 0; 
        const allPassG_L = 0.5; 
        const allPassD_L = Math.max(1, Math.floor(this.sampleRate * 0.0003));
        let allPassX_L: number[] = new Array(allPassD_L + 1).fill(0);
        let allPassY_L: number[] = new Array(allPassD_L + 1).fill(0);
        const allPassG_R = 0.45; 
        const allPassD_R = Math.max(1, Math.floor(this.sampleRate * 0.0005));
        let allPassX_R: number[] = new Array(allPassD_R + 1).fill(0);
        let allPassY_R: number[] = new Array(allPassD_R + 1).fill(0);

        for (let i = 0; i < monoIR.length; i++) {
            const currentSample = monoIR[i];
            const filteredSampleL = alpha * currentSample + (1 - alpha) * prevL;
            prevL = filteredSampleL;
            for (let k = allPassD_L; k > 0; k--) { allPassX_L[k] = allPassX_L[k-1]; allPassY_L[k] = allPassY_L[k-1]; }
            allPassX_L[0] = filteredSampleL;
            const x_n_minus_D_L = allPassX_L[allPassD_L];
            const y_n_minus_D_L = allPassY_L[allPassD_L];
            const allPassOutputL = allPassG_L * filteredSampleL + x_n_minus_D_L - allPassG_L * y_n_minus_D_L;
            allPassY_L[0] = allPassOutputL;
            leftIR[i] = allPassOutputL;

            const delayedSample = currentSample; 
            const filteredDelayedSample = alpha * delayedSample + (1 - alpha) * prevR;
            prevR = filteredDelayedSample;
            for (let k = allPassD_R; k > 0; k--) { allPassX_R[k] = allPassX_R[k-1]; allPassY_R[k] = allPassY_R[k-1]; }
            allPassX_R[0] = filteredDelayedSample;
            const x_n_minus_D_R = allPassX_R[allPassD_R];
            const y_n_minus_D_R = allPassY_R[allPassD_R];
            const allPassOutputR = allPassG_R * filteredDelayedSample + x_n_minus_D_R - allPassG_R * y_n_minus_D_R;
            allPassY_R[0] = allPassOutputR;
            rightIR[i] = allPassOutputR;
        }
        return [leftIR, rightIR];
    }

    private calculateRT60Values(lateHits: any[], roomConfig: any): { [freq: string]: number } {
        const frequencies = ['125', '250', '500', '1000', '2000', '4000', '8000', '16000'];
        const rt60Values: { [freq: string]: number } = {};
        
        for (const freq of frequencies) {
            const absorption = this.meanAbsorption[freq] || 0.1;
            const effectiveAbsorption = Math.max(absorption, 0.01);
            const V = Math.max(this.roomVolume, 1.0);
            const S = Math.max(this.surfaceArea, 6.0);
            let rt60 = 0.161 * V / (S * effectiveAbsorption);

            // Reduced the low-frequency RT60 multiplier slightly
            if (parseInt(freq) < 500) rt60 *= 1.05; // Was 1.1
            else if (parseInt(freq) > 2000) rt60 *= 0.9; 
            rt60Values[freq] = Math.min(Math.max(rt60, 0.05), 3.5); 
        }
        return rt60Values;
    }

    public combineWithEarlyReflections(
        earlyReflections: Float32Array, diffuseField: Float32Array, crossoverTime: number
    ): Float32Array {
        // This method is part of DiffuseFieldModelModified but typically called by AudioProcessorModified
        // For brevity, assuming it's correctly implemented as before.
        const earlyLength = earlyReflections.length;
        const diffuseLength = diffuseField.length;
        const totalLength = Math.max(earlyLength, diffuseLength);
        const output = new Float32Array(totalLength);
        const crossoverSample = Math.floor(crossoverTime * this.sampleRate);
        const fadeLength = Math.floor(0.01 * this.sampleRate); // 10ms fade

        for (let i = 0; i < totalLength; i++) {
            const earlyVal = (i < earlyLength) ? earlyReflections[i] : 0;
            const diffuseVal = (i < diffuseLength) ? diffuseField[i] : 0;

            if (i < crossoverSample - fadeLength) {
                output[i] = earlyVal;
            } else if (i < crossoverSample + fadeLength) {
                const fadePos = (i - (crossoverSample - fadeLength)) / (fadeLength * 2);
                const earlyGain = 0.5 * (1 + Math.cos(fadePos * Math.PI)); 
                const diffuseGain = 0.5 * (1 - Math.cos(fadePos * Math.PI)); 
                output[i] = earlyVal * earlyGain + diffuseVal * diffuseGain;
            } else {
                output[i] = diffuseVal;
            }
        }
        return output;
    }
}