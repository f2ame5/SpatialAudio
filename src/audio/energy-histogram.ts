/**
 * Energy Histogram - Simple energy accumulation for impulse response generation
 * Focuses on accurate energy collection and time binning
 */

export interface EnergyBin {
    energy: number;
    sampleCount: number;
    arrivalTime: number;
    frequencyEnergy: Float32Array; // 8 frequency bands
}

export interface EnergyHistogramConfig {
    maxTime: number;        // Maximum time in seconds
    timeBinSize: number;    // Time bin size in seconds
    frequencyBands: number; // Number of frequency bands
    sampleRate: number;     // Sample rate for time calculations
}

export class EnergyHistogram {
    private config: EnergyHistogramConfig;
    private bins: EnergyBin[];
    private maxBins: number;

    constructor(config: EnergyHistogramConfig) {
        this.config = config;
        this.maxBins = Math.ceil(config.maxTime / config.timeBinSize);
        this.bins = [];
        this.clear();
    }

    /**
     * Clear all energy bins
     */
    clear(): void {
        this.bins = [];
        for (let i = 0; i < this.maxBins; i++) {
            this.bins.push({
                energy: 0,
                sampleCount: 0,
                arrivalTime: i * this.config.timeBinSize,
                frequencyEnergy: new Float32Array(this.config.frequencyBands)
            });
        }
    }

    /**
     * Add energy to a specific time bin
     */
    addEnergy(arrivalTime: number, energy: number, frequencyEnergy?: Float32Array): void {
        const binIndex = Math.floor(arrivalTime / this.config.timeBinSize);
        
        if (binIndex >= 0 && binIndex < this.maxBins) {
            this.bins[binIndex].energy += energy;
            this.bins[binIndex].sampleCount++;
            
            if (frequencyEnergy) {
                for (let i = 0; i < Math.min(this.config.frequencyBands, frequencyEnergy.length); i++) {
                    this.bins[binIndex].frequencyEnergy[i] += frequencyEnergy[i];
                }
            }
        }
    }

    /**
     * Get energy bin at specific index
     */
    getBin(index: number): EnergyBin | null {
        if (index >= 0 && index < this.bins.length) {
            return this.bins[index];
        }
        return null;
    }

    /**
     * Get all bins
     */
    getAllBins(): EnergyBin[] {
        return this.bins;
    }

    /**
     * Get total energy across all bins
     */
    getTotalEnergy(): number {
        return this.bins.reduce((total, bin) => total + bin.energy, 0);
    }

    /**
     * Get number of non-empty bins
     */
    getNonEmptyBinCount(): number {
        return this.bins.filter(bin => bin.energy > 0).length;
    }

    /**
     * Get peak energy and its time
     */
    getPeakEnergy(): { energy: number; time: number; binIndex: number } {
        let maxEnergy = 0;
        let maxTime = 0;
        let maxIndex = 0;

        for (let i = 0; i < this.bins.length; i++) {
            if (this.bins[i].energy > maxEnergy) {
                maxEnergy = this.bins[i].energy;
                maxTime = this.bins[i].arrivalTime;
                maxIndex = i;
            }
        }

        return { energy: maxEnergy, time: maxTime, binIndex: maxIndex };
    }

    /**
     * Get energy decay statistics
     */
    getDecayStatistics(): { rt60: number; edt: number; totalEnergy: number } {
        const totalEnergy = this.getTotalEnergy();
        if (totalEnergy === 0) {
            return { rt60: 0, edt: 0, totalEnergy: 0 };
        }

        // Find peak
        const peak = this.getPeakEnergy();
        const peakEnergy = peak.energy;
        const peakIndex = peak.binIndex;

        // Calculate RT60 (time for 60dB decay)
        const rt60Threshold = peakEnergy * 0.001; // -60dB
        let rt60 = 0;
        for (let i = peakIndex; i < this.bins.length; i++) {
            if (this.bins[i].energy < rt60Threshold) {
                rt60 = this.bins[i].arrivalTime - peak.time;
                break;
            }
        }

        // Calculate EDT (early decay time, first 10dB)
        const edtThreshold = peakEnergy * 0.316; // -10dB
        let edt = 0;
        for (let i = peakIndex; i < this.bins.length; i++) {
            if (this.bins[i].energy < edtThreshold) {
                edt = (this.bins[i].arrivalTime - peak.time) * 6; // Extrapolate to 60dB
                break;
            }
        }

        return { rt60, edt, totalEnergy };
    }

    /**
     * Convert to impulse response array
     */
    toImpulseResponse(): Float32Array {
        const sampleCount = Math.floor(this.config.maxTime * this.config.sampleRate);
        const impulseResponse = new Float32Array(sampleCount);

        for (let i = 0; i < this.bins.length; i++) {
            const bin = this.bins[i];
            if (bin.energy > 0) {
                const sampleIndex = Math.floor(bin.arrivalTime * this.config.sampleRate);
                if (sampleIndex < sampleCount) {
                    impulseResponse[sampleIndex] += bin.energy;
                }
            }
        }

        return impulseResponse;
    }

    /**
     * Get histogram statistics for debugging
     */
    getStatistics(): {
        totalBins: number;
        nonEmptyBins: number;
        totalEnergy: number;
        peakEnergy: number;
        peakTime: number;
        averageEnergy: number;
        energyRange: { min: number; max: number };
    } {
        const totalEnergy = this.getTotalEnergy();
        const nonEmptyBins = this.getNonEmptyBinCount();
        const peak = this.getPeakEnergy();

        let minEnergy = Infinity;
        let maxEnergy = 0;
        for (const bin of this.bins) {
            if (bin.energy > 0) {
                minEnergy = Math.min(minEnergy, bin.energy);
                maxEnergy = Math.max(maxEnergy, bin.energy);
            }
        }

        if (minEnergy === Infinity) minEnergy = 0;

        return {
            totalBins: this.bins.length,
            nonEmptyBins,
            totalEnergy,
            peakEnergy: peak.energy,
            peakTime: peak.time,
            averageEnergy: nonEmptyBins > 0 ? totalEnergy / nonEmptyBins : 0,
            energyRange: { min: minEnergy, max: maxEnergy }
        };
    }

    /**
     * Export histogram data for visualization
     */
    exportForVisualization(): Array<{ time: number; energy: number; sampleCount: number }> {
        return this.bins
            .filter(bin => bin.energy > 0)
            .map(bin => ({
                time: bin.arrivalTime,
                energy: bin.energy,
                sampleCount: bin.sampleCount
            }));
    }
}
