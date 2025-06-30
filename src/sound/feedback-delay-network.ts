// src/sound/feedback-delay-network.ts

/**
 * A robust Feedback Delay Network (FDN) reverb implementation using the Web Audio API.
 * This version uses native nodes for performance and high-quality audio processing.
 * Reworked to improve stability and reduce audio artifacts.
 */
export class FeedbackDelayNetwork {
    private audioCtx: AudioContext;
    private _input: GainNode;
    private _output: GainNode;
    private wetGain: GainNode;
    private dryGain: GainNode;
    private inputFilter: BiquadFilterNode;

    private delayNodes: DelayNode[] = [];
    private feedbackGains: GainNode[] = [];
    private filters: BiquadFilterNode[] = [];
    private allpassFilters: BiquadFilterNode[][] = []; // Array of arrays to hold multiple all-pass filter nodes per channel
    private merger: ChannelMergerNode;
    private splitter: ChannelSplitterNode;
    private readonly numChannels: number;
    private readonly timeConstant = 0.05; // Increased smoothing time in seconds for parameter changes

    constructor(audioCtx: AudioContext, channels: number = 8) {
        this.audioCtx = audioCtx;
        this.numChannels = channels;
        console.log(`[FDN CONSTRUCTOR] Creating FDN with ${channels} channels.`);

        // --- Core Nodes ---
        this._input = this.audioCtx.createGain();
        this._output = this.audioCtx.createGain();
        this.wetGain = this.audioCtx.createGain();
        this.dryGain = this.audioCtx.createGain();

        // --- Input Filtering ---
        this.inputFilter = this.audioCtx.createBiquadFilter();
        this.inputFilter.type = 'highpass';
        this.inputFilter.frequency.value = 20; // Cut sub-bass to prevent mud
        this._input.connect(this.inputFilter);

        // --- Routing ---
        this.splitter = this.audioCtx.createChannelSplitter(this.numChannels);
        this.merger = this.audioCtx.createChannelMerger(this.numChannels);

        // --- Create Delay Lines and Feedback Loop ---
        const maxDelayTime = 2.0; // Increased max delay time
        const delayTimes = this.generateDelayTimes(this.numChannels);
        console.log('[FDN CONSTRUCTOR] Generated Delay Times (s):', delayTimes);

        for (let i = 0; i < this.numChannels; i++) {
            const delay = this.audioCtx.createDelay(maxDelayTime);
            delay.delayTime.value = delayTimes[i];
            this.delayNodes.push(delay);

            const gain = this.audioCtx.createGain();
            gain.gain.value = 0.58; // Initial feedback gain, slightly reduced further
            this.feedbackGains.push(gain);

            const filter = this.audioCtx.createBiquadFilter();
            filter.type = 'lowpass';
            filter.frequency.value = 3000; // Initial cutoff, lowered for more damping
            this.filters.push(filter);

            // Create three all-pass filters for maximum diffusion
            const allpass1 = this.audioCtx.createBiquadFilter();
            allpass1.type = 'allpass';
            allpass1.frequency.value = 100 + i * 250; // Wide frequency spread
            allpass1.Q.value = 0.1 + (i % 2) * 0.02; // Very low Q

            const allpass2 = this.audioCtx.createBiquadFilter();
            allpass2.type = 'allpass';
            allpass2.frequency.value = 800 + i * 300; // Wide frequency spread
            allpass2.Q.value = 0.15 + (i % 3) * 0.03; // Very low Q

            const allpass3 = this.audioCtx.createBiquadFilter();
            allpass3.type = 'allpass';
            allpass3.frequency.value = 2000 + i * 400; // Wide frequency spread
            allpass3.Q.value = 0.2 + (i % 4) * 0.04; // Very low Q

            this.allpassFilters.push([allpass1, allpass2, allpass3]);

            // --- Internal Connections ---
            // Input filter output goes to each delay line
            this.inputFilter.connect(delay);

            // Delay output goes through filter, then all-passes, then gain, then mixes back
            delay.connect(filter);
            filter.connect(allpass1); // Connect filter to first all-pass
            allpass1.connect(allpass2); // Connect first all-pass to second
            allpass2.connect(allpass3); // Connect second all-pass to third
            allpass3.connect(gain); // Connect third all-pass to feedback gain
            gain.connect(this.merger, 0, i);
        }

        // --- Feedback Loop with Mixing Matrix ---
        const hadamardMatrix = this.createHadamardMatrix(this.numChannels);
        console.log('[FDN CONSTRUCTOR] Created Hadamard Matrix:', hadamardMatrix);

        this.merger.connect(this.splitter);

        for (let i = 0; i < this.numChannels; i++) {
            for (let j = 0; j < this.numChannels; j++) {
                const mixGain = hadamardMatrix[i][j];
                if (mixGain !== 0) {
                    const gainNode = this.audioCtx.createGain();
                    gainNode.gain.value = mixGain;
                    this.splitter.connect(gainNode, j);
                    gainNode.connect(this.delayNodes[i]);
                }
            }
        }
        
        // --- Wet/Dry Mix ---
        this.merger.connect(this.wetGain);
        this.wetGain.connect(this._output);

        this._input.connect(this.dryGain);
        this.dryGain.connect(this._output);

        this.setMix(1.0); // Default to fully wet
    }

    // --- Public Methods ---

    public connect(destination: AudioNode): void {
        this._output.connect(destination);
    }

    public disconnect(): void {
        this._output.disconnect();
    }

    public get input(): GainNode {
        return this._input;
    }
    
    public setMix(wetLevel: number): void {
        const clampedWet = Math.max(0, Math.min(1, wetLevel));
        console.log(`[FDN setMix] Setting wet level to: ${clampedWet}`);
        const now = this.audioCtx.currentTime;
        this.wetGain.gain.setTargetAtTime(clampedWet, now, this.timeConstant);
        this.dryGain.gain.setTargetAtTime(1 - clampedWet, now, this.timeConstant);
    }

    public setRT60(rt60Values: { [frequency: string]: number }): void {
        console.log('[FDN setRT60] Received RT60 values:', rt60Values);
        const rt60_1k = rt60Values['1000'] || 1.5;
        const rt60_high = rt60Values['8000'] || rt60_1k * 0.7;
        const now = this.audioCtx.currentTime;

        for (let i = 0; i < this.numChannels; i++) {
            const delayTime = this.delayNodes[i].delayTime.value;
            let feedbackGain = 0;
            
            if (rt60_1k > delayTime && delayTime > 0) {
                feedbackGain = Math.pow(10, (-3 * delayTime) / rt60_1k);
            }

            // Ensure gain is a finite number and safely below 1.0
            const safeGain = isFinite(feedbackGain) ? Math.min(feedbackGain, 0.998) : 0;
            this.feedbackGains[i].gain.setTargetAtTime(safeGain, now, this.timeConstant);

            const highFreqRatio = Math.max(0.1, rt60_high / rt60_1k);
            const cutoff = 20000 * Math.pow(highFreqRatio, 2);
            const safeCutoff = Math.max(400, Math.min(20000, cutoff));
            this.filters[i].frequency.setTargetAtTime(safeCutoff, now, this.timeConstant);
        }
        console.log(`[FDN setRT60] Configured feedback gains and filters based on RT60_1k=${rt60_1k}.`);
    }

    // --- Private Helper Methods ---

    private generateDelayTimes(count: number): number[] {
        const times = [];
        // Using a set of well-distributed, mutually prime-like delay lengths
        // These values are chosen to be non-harmonically related to avoid metallic ringing
        const baseDelays = [
            0.0173, 0.0211, 0.0269, 0.0317, 0.0373, 0.0419, 0.0467, 0.0509,
            0.0557, 0.0601, 0.0647, 0.0691, 0.0739, 0.0787, 0.0827, 0.0871
        ];

        for (let i = 0; i < count; i++) {
            // Add a small random perturbation to further break up periodicity
            times.push(baseDelays[i % baseDelays.length] * (1 + (Math.random() - 0.5) * 0.02));
        }
        return times;
    }
    
    private createHadamardMatrix(size: number): number[][] {
        if (size === 1) return [[1]];
        if (size <= 0 || size % 2 !== 0) {
            console.warn("Matrix size must be a power of 2. Using a fallback Householder matrix.");
            const norm = 1 / Math.sqrt(size);
            const matrix = Array.from({ length: size }, () => Array(size).fill(norm * -1));
            for(let i = 0; i < size; i++) {
                matrix[i][i] = norm * (Math.sqrt(size) - 1);
            }
            return matrix;
        }

        const half = this.createHadamardMatrix(size / 2);
        const halfSize = half.length;
        const matrix = Array.from({ length: size }, () => Array(size).fill(0));
        const norm = 1 / Math.sqrt(2);

        for (let i = 0; i < halfSize; i++) {
            for (let j = 0; j < halfSize; j++) {
                const val = half[i][j];
                matrix[i][j] = val;
                matrix[i][j + halfSize] = val;
                matrix[i + halfSize][j] = val;
                matrix[i + halfSize][j + halfSize] = -val;
            }
        }
        // Final normalization should be done once outside the recursion
        const finalNorm = 1 / Math.sqrt(size);
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                matrix[i][j] *= finalNorm;
            }
        }
        return matrix;
    }
}
''