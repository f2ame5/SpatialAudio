/**
 * Impulse Response Generator - Converts ray tracing results to audio impulse responses
 * Implements stochastic ray tracing with Poisson random process for dense, realistic IRs
 */

import { vec3 } from 'gl-matrix';
import { ImpulseResponseSample } from './ray-types';
import { FREQUENCY_BANDS, SAMPLE_RATES } from './audio-utils';
import { EnergyHistogram, EnergyHistogramConfig } from './energy-histogram';

/**
 * IR generation configuration
 */
export interface IRGeneratorConfig {
    sampleRate: number;
    maxLength: number;          // Maximum IR length in seconds
    binSize: number;            // Time bin size in seconds
    smoothingFactor: number;    // Temporal smoothing
    normalizeOutput: boolean;   // Normalize to prevent clipping
    frequencyBands: number;     // Number of frequency bands
    roomVolume: number;         // Room volume in cubic meters (for Poisson process)
    usePoissonProcess: boolean; // Enable Poisson random process generation
    histogramResolution: number; // Time resolution for energy histogram (seconds)
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: IRGeneratorConfig = {
    sampleRate: SAMPLE_RATES.DAT_QUALITY,
    maxLength: 2.0,
    binSize: 1.0 / SAMPLE_RATES.DAT_QUALITY, // One sample
    smoothingFactor: 0.0,
    normalizeOutput: true,
    frequencyBands: 8,
    roomVolume: 120.0, // Default room volume (8x3x5 meters)
    usePoissonProcess: true,
    histogramResolution: 0.004 // 4ms bins for energy histogram
};

/**
 * Impulse response statistics
 */
export interface IRStatistics {
    peakAmplitude: number;
    rmsLevel: number;
    rt60: number;               // Reverberation time
    clarity: number;            // C80 clarity index
    definition: number;         // D50 definition
    centerTime: number;         // TS center time
    edt: number;               // Early decay time
}

export class ImpulseResponseGenerator {
    private config: IRGeneratorConfig;
    private sampleBuffer: Float32Array;
    private frequencyBuffers: Float32Array[];
    private timeBins: number;
    private currentBin: number = 0;

    // Simple energy histogram for accurate energy collection
    private energyHistogram: EnergyHistogram;

    // Legacy energy histogram for Poisson process (keeping for compatibility)
    private legacyEnergyHistogram: Float32Array[];
    private histogramBins: number;

    // Frequency band center frequencies (Hz)
    private readonly centerFrequencies = [125, 250, 500, 1000, 2000, 4000, 8000, 16000];
    
    constructor(config: Partial<IRGeneratorConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };

        // Calculate number of time bins
        this.timeBins = Math.ceil(this.config.maxLength / this.config.binSize);

        // Initialize buffers
        const sampleCount = Math.floor(this.config.maxLength * this.config.sampleRate);
        this.sampleBuffer = new Float32Array(sampleCount);

        // Initialize frequency band buffers
        this.frequencyBuffers = [];
        for (let i = 0; i < this.config.frequencyBands; i++) {
            this.frequencyBuffers.push(new Float32Array(sampleCount));
        }

        // Initialize simple energy histogram
        this.energyHistogram = new EnergyHistogram({
            maxTime: this.config.maxLength,
            timeBinSize: this.config.histogramResolution,
            frequencyBands: this.config.frequencyBands,
            sampleRate: this.config.sampleRate
        });

        // Initialize legacy energy histogram for Poisson process
        this.histogramBins = Math.ceil(this.config.maxLength / this.config.histogramResolution);
        this.legacyEnergyHistogram = [];
        for (let i = 0; i < this.config.frequencyBands; i++) {
            this.legacyEnergyHistogram.push(new Float32Array(this.histogramBins));
        }
    }
    
    /**
     * Set room volume for Poisson process calculation
     */
    setRoomVolume(volume: number): void {
        this.config.roomVolume = volume;
    }

    /**
     * Clear all buffers
     */
    clear(): void {
        this.sampleBuffer.fill(0);
        this.frequencyBuffers.forEach(buffer => buffer.fill(0));
        this.energyHistogram.clear();
        this.legacyEnergyHistogram.forEach(histogram => histogram.fill(0));
        this.currentBin = 0;
    }
    
    /**
     * Add ray contribution to impulse response
     */
    addRayContribution(sample: ImpulseResponseSample): void {
        if (this.config.usePoissonProcess) {
            // Add to energy histogram for Poisson process generation
            this.addToEnergyHistogram(sample);
        } else {
            // Legacy direct accumulation method
            this.addDirectContribution(sample);
        }
    }

    /**
     * Add ray contribution to energy histogram (for Poisson process)
     */
    private addToEnergyHistogram(sample: ImpulseResponseSample): void {
        // Add to simple energy histogram
        this.energyHistogram.addEnergy(sample.timeBin, sample.energy, sample.frequencyEnergy);

        // Also add to legacy histogram for compatibility
        const histogramBin = Math.floor(sample.timeBin / this.config.histogramResolution);
        if (histogramBin >= 0 && histogramBin < this.histogramBins) {
            // Add energy to each frequency band
            for (let i = 0; i < this.config.frequencyBands && i < sample.frequencyEnergy.length; i++) {
                this.legacyEnergyHistogram[i][histogramBin] += sample.frequencyEnergy[i];
            }
        }
    }

    /**
     * Direct contribution method (legacy)
     */
    private addDirectContribution(sample: ImpulseResponseSample): void {
        const sampleIndex = Math.floor(sample.timeBin * this.config.sampleRate);

        if (sampleIndex >= this.sampleBuffer.length || sampleIndex < 0) {
            return; // Beyond IR length or negative time
        }

        // Add energy contribution with phase
        const contribution = sample.energy * Math.cos(sample.phase);
        this.sampleBuffer[sampleIndex] += contribution;

        // Add frequency-dependent contributions
        for (let i = 0; i < this.config.frequencyBands && i < sample.frequencyEnergy.length; i++) {
            this.frequencyBuffers[i][sampleIndex] += sample.frequencyEnergy[i] * Math.cos(sample.phase);
        }
    }
    
    /**
     * Add multiple ray contributions
     */
    addRayContributions(samples: ImpulseResponseSample[]): void {
        for (const sample of samples) {
            this.addRayContribution(sample);
        }
    }
    
    /**
     * Process ray arrival with spatial information
     */
    processRayArrival(
        energy: number,
        phase: number,
        arrivalTime: number,
        frequencyEnergy: Float32Array,
        direction: vec3,
        listenerForward: vec3
    ): void {
        // Apply directional weighting (simple cardioid pattern)
        const dotProduct = vec3.dot(direction, listenerForward);
        const directionalWeight = 0.5 + 0.5 * dotProduct; // Cardioid pattern

        // Create sample (timeBin is now arrival time in seconds)
        const sample: ImpulseResponseSample = {
            timeBin: arrivalTime,
            energy: energy * directionalWeight,
            phase,
            frequencyEnergy,
            direction
        };

        this.addRayContribution(sample);
    }
    
    /**
     * Apply temporal smoothing to reduce artifacts
     */
    private applySmoothing(buffer: Float32Array): void {
        if (this.config.smoothingFactor <= 0) return;
        
        const alpha = this.config.smoothingFactor;
        const beta = 1 - alpha;
        
        // Forward pass
        for (let i = 1; i < buffer.length; i++) {
            buffer[i] = beta * buffer[i] + alpha * buffer[i - 1];
        }
        
        // Backward pass
        for (let i = buffer.length - 2; i >= 0; i--) {
            buffer[i] = beta * buffer[i] + alpha * buffer[i + 1];
        }
    }
    
    /**
     * Generate final impulse response
     */
    generateImpulseResponse(): AudioBuffer {
        if (this.config.usePoissonProcess) {
            // Generate using Poisson random process
            this.generatePoissonImpulseResponse();
        }

        const context = new OfflineAudioContext(
            1,
            this.sampleBuffer.length,
            this.config.sampleRate
        );

        const buffer = context.createBuffer(
            1,
            this.sampleBuffer.length,
            this.config.sampleRate
        );

        // Apply smoothing
        this.applySmoothing(this.sampleBuffer);

        // Normalize if requested
        if (this.config.normalizeOutput) {
            this.normalizeBuffer(this.sampleBuffer);
        }

        // Copy to audio buffer
        buffer.copyToChannel(this.sampleBuffer, 0);

        return buffer;
    }
    
    /**
     * Generate frequency-dependent impulse responses
     */
    generateFrequencyImpulseResponses(): AudioBuffer[] {
        const buffers: AudioBuffer[] = [];
        
        for (let i = 0; i < this.config.frequencyBands; i++) {
            const context = new OfflineAudioContext(
                1,
                this.frequencyBuffers[i].length,
                this.config.sampleRate
            );
            
            const buffer = context.createBuffer(
                1,
                this.frequencyBuffers[i].length,
                this.config.sampleRate
            );
            
            // Apply smoothing
            this.applySmoothing(this.frequencyBuffers[i]);
            
            // Normalize if requested
            if (this.config.normalizeOutput) {
                this.normalizeBuffer(this.frequencyBuffers[i]);
            }
            
            // Copy to audio buffer
            buffer.copyToChannel(this.frequencyBuffers[i], 0);
            buffers.push(buffer);
        }
        
        return buffers;
    }
    
    /**
     * Normalize buffer to prevent clipping
     */
    private normalizeBuffer(buffer: Float32Array): void {
        let maxValue = 0;
        
        for (let i = 0; i < buffer.length; i++) {
            maxValue = Math.max(maxValue, Math.abs(buffer[i]));
        }
        
        if (maxValue > 0.95) {
            const scale = 0.95 / maxValue;
            for (let i = 0; i < buffer.length; i++) {
                buffer[i] *= scale;
            }
        }
    }
    
    /**
     * Generate impulse response using Poisson random process
     * Based on the MATLAB stochastic ray tracing implementation
     */
    private generatePoissonImpulseResponse(): void {
        // Clear the main sample buffer
        this.sampleBuffer.fill(0);

        // Generate Poisson random process
        const poissonProcess = this.generatePoissonRandomProcess();

        // Process each frequency band
        for (let bandIndex = 0; bandIndex < this.config.frequencyBands; bandIndex++) {
            // Create bandpass filter for this frequency band
            const filteredSignal = this.applyBandpassFilter(poissonProcess, bandIndex);

            // Weight by energy histogram
            const weightedSignal = this.applyEnergyWeighting(filteredSignal, bandIndex);

            // Store in frequency buffer first
            if (bandIndex < this.frequencyBuffers.length) {
                for (let i = 0; i < Math.min(this.frequencyBuffers[bandIndex].length, weightedSignal.length); i++) {
                    this.frequencyBuffers[bandIndex][i] = weightedSignal[i];
                }
            }
        }

        // Combine frequency bands following MATLAB approach
        this.combineFrequencyBands();
    }

    /**
     * Generate Poisson random process
     */
    private generatePoissonRandomProcess(): Float32Array {
        const c = 343; // Speed of sound (m/s)
        const V = this.config.roomVolume;

        // Calculate start time (equation 5.45 from MATLAB implementation)
        const t0 = Math.pow((2 * V * Math.log(2)) / (4 * Math.PI * Math.pow(c, 3)), 1/3);

        const timeValues: number[] = [];
        const processValues: number[] = [];

        let t = t0;
        while (t < this.config.maxLength) {
            timeValues.push(t);

            // Determine polarity randomly
            const polarity = (Math.round(t * this.config.sampleRate) - t * this.config.sampleRate) < 0 ? 1 : -1;
            processValues.push(polarity);

            // Determine mean event occurrence (equation 5.44)
            const mu = Math.min(1e4, 4 * Math.PI * Math.pow(c, 3) * Math.pow(t, 2) / V);

            // Determine interval size (equation 5.43)
            const deltaTA = (1 / mu) * Math.log(1 / Math.random());
            t += deltaTA;
        }

        // Convert to sampled signal
        const sampleCount = Math.floor(this.config.maxLength * this.config.sampleRate);
        const poissonSignal = new Float32Array(sampleCount);

        for (let i = 0; i < timeValues.length; i++) {
            const sampleIndex = Math.round(timeValues[i] * this.config.sampleRate);
            if (sampleIndex < sampleCount) {
                poissonSignal[sampleIndex] = processValues[i];
            }
        }

        return poissonSignal;
    }

    /**
     * Apply bandpass filter for specific frequency band using MATLAB-style approach
     */
    private applyBandpassFilter(signal: Float32Array, bandIndex: number): Float32Array {
        if (bandIndex >= this.centerFrequencies.length) {
            return new Float32Array(signal.length);
        }

        const centerFreq = this.centerFrequencies[bandIndex];

        // Calculate band edge frequencies following MATLAB approach
        const bandEdges = this.getBandEdgeFrequencies(centerFreq);
        const lowFreq = bandEdges.low;
        const highFreq = bandEdges.high;

        // Use FFT-based filtering for better frequency response
        return this.applyFFTBandpassFilter(signal, lowFreq, highFreq);
    }

    /**
     * Calculate band edge frequencies following MATLAB implementation
     */
    private getBandEdgeFrequencies(centerFreq: number): { low: number, high: number } {
        // MATLAB approach: G = 2, BandsPerOctave = 1
        const G = 2;
        const bandsPerOctave = 1;
        const fbpo = 0.5 / bandsPerOctave;

        const lowFreq = centerFreq * Math.pow(G, -fbpo);
        const highFreq = Math.min(centerFreq * Math.pow(G, fbpo), this.config.sampleRate / 2);

        return { low: lowFreq, high: highFreq };
    }

    /**
     * FFT-based bandpass filter for better frequency response
     */
    private applyFFTBandpassFilter(signal: Float32Array, lowFreq: number, highFreq: number): Float32Array {
        const N = signal.length;
        const nyquist = this.config.sampleRate / 2;

        // For simplicity, use a time-domain implementation
        // In a production system, you'd use proper FFT
        const filtered = new Float32Array(N);

        // Create filter response following MATLAB equation 5.46
        for (let i = 0; i < N; i++) {
            const freq = (i / N) * this.config.sampleRate;
            let response = 0;

            if (freq >= lowFreq && freq < highFreq) {
                if (freq < (lowFreq + highFreq) / 2) {
                    // Rising edge
                    response = 0.5 * (1 + Math.cos(2 * Math.PI * freq / (highFreq - lowFreq)));
                } else {
                    // Falling edge
                    response = 0.5 * (1 - Math.cos(2 * Math.PI * freq / (highFreq - lowFreq)));
                }
            }

            // Apply simple convolution (approximation)
            filtered[i] = signal[i] * response;
        }

        return filtered;
    }

    /**
     * Simple lowpass filter
     */
    private applyLowpassFilter(signal: Float32Array, cutoffNorm: number): Float32Array {
        const filtered = new Float32Array(signal.length);
        const alpha = Math.exp(-2 * Math.PI * cutoffNorm);

        filtered[0] = signal[0] * (1 - alpha);
        for (let i = 1; i < signal.length; i++) {
            filtered[i] = signal[i] * (1 - alpha) + filtered[i - 1] * alpha;
        }

        return filtered;
    }

    /**
     * Apply energy weighting from histogram following MATLAB equation 5.47
     */
    private applyEnergyWeighting(signal: Float32Array, bandIndex: number): Float32Array {
        const weighted = new Float32Array(signal.length);
        const histogram = this.energyHistogram[bandIndex];

        if (!histogram) {
            return signal;
        }

        // Calculate weighting factors following MATLAB implementation
        const samplesPerBin = Math.floor(this.config.histogramResolution * this.config.sampleRate);

        // Calculate bandwidth for this frequency band
        const bandEdges = this.getBandEdgeFrequencies(this.centerFrequencies[bandIndex]);
        const bandwidth = bandEdges.high - bandEdges.low;

        for (let k = 0; k < histogram.length; k++) {
            const gk0 = Math.floor((k) * this.config.sampleRate * this.config.histogramResolution);
            const gk1 = Math.floor((k + 1) * this.config.sampleRate * this.config.histogramResolution);

            if (gk0 >= signal.length) break;

            // Calculate energy in this time bin (MATLAB: sum(yy, 1))
            let binEnergy = 0;
            for (let i = gk0; i < Math.min(gk1, signal.length); i++) {
                binEnergy += signal[i] * signal[i];
            }

            // Calculate weighting factor (MATLAB equation 5.47)
            let weightingFactor = 0;
            if (binEnergy > 0 && histogram[k] > 0) {
                weightingFactor = Math.sqrt(histogram[k] / binEnergy) * Math.sqrt(bandwidth / (this.config.sampleRate / 2));
            }

            // Apply weighting to samples in this bin
            for (let i = gk0; i < Math.min(gk1, signal.length); i++) {
                weighted[i] = signal[i] * weightingFactor;
            }
        }

        return weighted;
    }

    /**
     * Combine frequency bands into final impulse response following MATLAB approach
     */
    private combineFrequencyBands(): void {
        // Clear the main buffer
        this.sampleBuffer.fill(0);

        // Sum all frequency bands with proper weighting
        for (let bandIndex = 0; bandIndex < this.config.frequencyBands; bandIndex++) {
            if (bandIndex < this.frequencyBuffers.length) {
                const bandBuffer = this.frequencyBuffers[bandIndex];

                // Calculate band weight based on frequency content
                const centerFreq = this.centerFrequencies[bandIndex];
                const bandWeight = this.calculateBandWeight(centerFreq);

                // Add weighted band contribution to main buffer
                for (let i = 0; i < Math.min(this.sampleBuffer.length, bandBuffer.length); i++) {
                    this.sampleBuffer[i] += bandBuffer[i] * bandWeight;
                }
            }
        }
    }

    /**
     * Calculate frequency band weight for combination
     */
    private calculateBandWeight(centerFreq: number): number {
        // Apply A-weighting-like curve for perceptual accuracy
        // This gives more weight to frequencies where human hearing is most sensitive

        if (centerFreq < 100) {
            return 0.3; // Low frequencies - reduced weight
        } else if (centerFreq < 1000) {
            return 0.8; // Mid-low frequencies
        } else if (centerFreq < 4000) {
            return 1.0; // Mid frequencies - full weight (most important for speech)
        } else if (centerFreq < 8000) {
            return 0.9; // High frequencies
        } else {
            return 0.6; // Very high frequencies - reduced weight
        }
    }

    /**
     * Calculate IR statistics
     */
    calculateStatistics(): IRStatistics {
        const stats: IRStatistics = {
            peakAmplitude: 0,
            rmsLevel: 0,
            rt60: 0,
            clarity: 0,
            definition: 0,
            centerTime: 0,
            edt: 0
        };
        
        // Peak amplitude
        for (let i = 0; i < this.sampleBuffer.length; i++) {
            stats.peakAmplitude = Math.max(stats.peakAmplitude, Math.abs(this.sampleBuffer[i]));
        }
        
        // RMS level
        let sumSquares = 0;
        for (let i = 0; i < this.sampleBuffer.length; i++) {
            sumSquares += this.sampleBuffer[i] * this.sampleBuffer[i];
        }
        stats.rmsLevel = Math.sqrt(sumSquares / this.sampleBuffer.length);
        
        // RT60 calculation
        stats.rt60 = this.calculateRT60();
        
        // C80 clarity index (ratio of early to late energy)
        stats.clarity = this.calculateClarity(0.08); // 80ms
        
        // D50 definition (ratio of early energy to total)
        stats.definition = this.calculateDefinition(0.05); // 50ms
        
        // Center time
        stats.centerTime = this.calculateCenterTime();
        
        // Early decay time
        stats.edt = this.calculateEDT();
        
        return stats;
    }
    
    /**
     * Calculate RT60 (time for 60dB decay)
     */
    private calculateRT60(): number {
        // Find peak
        let peakIndex = 0;
        let peakValue = 0;
        
        for (let i = 0; i < this.sampleBuffer.length; i++) {
            const value = Math.abs(this.sampleBuffer[i]);
            if (value > peakValue) {
                peakValue = value;
                peakIndex = i;
            }
        }
        
        // Find -60dB point
        const threshold = peakValue * 0.001; // -60dB
        let decayIndex = peakIndex;
        
        for (let i = peakIndex; i < this.sampleBuffer.length; i++) {
            if (Math.abs(this.sampleBuffer[i]) < threshold) {
                decayIndex = i;
                break;
            }
        }
        
        return (decayIndex - peakIndex) / this.config.sampleRate;
    }
    
    /**
     * Calculate clarity index (C80)
     */
    private calculateClarity(timeThreshold: number): number {
        const thresholdSample = Math.floor(timeThreshold * this.config.sampleRate);
        
        let earlyEnergy = 0;
        let lateEnergy = 0;
        
        for (let i = 0; i < this.sampleBuffer.length; i++) {
            const energy = this.sampleBuffer[i] * this.sampleBuffer[i];
            if (i < thresholdSample) {
                earlyEnergy += energy;
            } else {
                lateEnergy += energy;
            }
        }
        
        if (lateEnergy === 0) return 0;
        return 10 * Math.log10(earlyEnergy / lateEnergy);
    }
    
    /**
     * Calculate definition (D50)
     */
    private calculateDefinition(timeThreshold: number): number {
        const thresholdSample = Math.floor(timeThreshold * this.config.sampleRate);
        
        let earlyEnergy = 0;
        let totalEnergy = 0;
        
        for (let i = 0; i < this.sampleBuffer.length; i++) {
            const energy = this.sampleBuffer[i] * this.sampleBuffer[i];
            totalEnergy += energy;
            if (i < thresholdSample) {
                earlyEnergy += energy;
            }
        }
        
        if (totalEnergy === 0) return 0;
        return earlyEnergy / totalEnergy;
    }
    
    /**
     * Calculate center time (TS)
     */
    private calculateCenterTime(): number {
        let numerator = 0;
        let denominator = 0;
        
        for (let i = 0; i < this.sampleBuffer.length; i++) {
            const energy = this.sampleBuffer[i] * this.sampleBuffer[i];
            const time = i / this.config.sampleRate;
            numerator += time * energy;
            denominator += energy;
        }
        
        if (denominator === 0) return 0;
        return numerator / denominator;
    }
    
    /**
     * Calculate early decay time
     */
    private calculateEDT(): number {
        // Find initial 10dB decay
        let peakValue = 0;
        let peakIndex = 0;
        
        for (let i = 0; i < Math.min(1000, this.sampleBuffer.length); i++) {
            const value = Math.abs(this.sampleBuffer[i]);
            if (value > peakValue) {
                peakValue = value;
                peakIndex = i;
            }
        }
        
        // Find -10dB point
        const threshold = peakValue * 0.316; // -10dB
        let decayIndex = peakIndex;
        
        for (let i = peakIndex; i < this.sampleBuffer.length; i++) {
            if (Math.abs(this.sampleBuffer[i]) < threshold) {
                decayIndex = i;
                break;
            }
        }
        
        // Extrapolate to 60dB
        const decay10dB = (decayIndex - peakIndex) / this.config.sampleRate;
        return decay10dB * 6; // 10dB to 60dB
    }
    
    /**
     * Get current buffer data
     */
    getBufferData(): Float32Array {
        return new Float32Array(this.sampleBuffer);
    }
    
    /**
     * Get frequency band buffer data
     */
    getFrequencyBufferData(bandIndex: number): Float32Array | null {
        if (bandIndex < 0 || bandIndex >= this.frequencyBuffers.length) {
            return null;
        }
        return new Float32Array(this.frequencyBuffers[bandIndex]);
    }
    
    /**
     * Get energy histogram statistics for debugging
     */
    getEnergyHistogramStatistics(): any {
        return this.energyHistogram.getStatistics();
    }

    /**
     * Get energy histogram data for visualization
     */
    getEnergyHistogramData(): Array<{ time: number; energy: number; sampleCount: number }> {
        return this.energyHistogram.exportForVisualization();
    }

    /**
     * Generate simple impulse response from energy histogram
     */
    generateSimpleImpulseResponse(): Float32Array {
        return this.energyHistogram.toImpulseResponse();
    }

    /**
     * Add test data to energy histogram for debugging
     */
    addTestEnergyData(): void {
        console.log('🧪 Adding test energy data to histogram...');

        // Add some test energy at different times
        this.energyHistogram.addEnergy(0.0, 1.0);      // Direct sound at t=0
        this.energyHistogram.addEnergy(0.01, 0.8);     // Early reflection at 10ms
        this.energyHistogram.addEnergy(0.025, 0.6);    // Another reflection at 25ms
        this.energyHistogram.addEnergy(0.05, 0.4);     // Later reflection at 50ms
        this.energyHistogram.addEnergy(0.1, 0.2);      // Late reflection at 100ms
        this.energyHistogram.addEnergy(0.2, 0.1);      // Very late reflection at 200ms

        const stats = this.energyHistogram.getStatistics();
        console.log('📊 Test energy histogram stats:', stats);
    }

    /**
     * Export impulse response as WAV data
     */
    exportAsWAV(): ArrayBuffer {
        const length = this.sampleBuffer.length;
        const arrayBuffer = new ArrayBuffer(44 + length * 2);
        const view = new DataView(arrayBuffer);
        
        // WAV header
        const writeString = (offset: number, string: string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true); // fmt chunk size
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // Mono
        view.setUint32(24, this.config.sampleRate, true);
        view.setUint32(28, this.config.sampleRate * 2, true); // byte rate
        view.setUint16(32, 2, true); // block align
        view.setUint16(34, 16, true); // bits per sample
        writeString(36, 'data');
        view.setUint32(40, length * 2, true);
        
        // Convert float samples to 16-bit PCM
        let offset = 44;
        for (let i = 0; i < length; i++) {
            const sample = Math.max(-1, Math.min(1, this.sampleBuffer[i]));
            view.setInt16(offset, sample * 0x7FFF, true);
            offset += 2;
        }
        
        return arrayBuffer;
    }
}
