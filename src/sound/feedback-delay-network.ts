// src/sound/feedback-delay-network.ts

/**
 * A robust Feedback Delay Network (FDN) reverb implementation using the Web Audio API.
 * This version uses native nodes for performance and high-quality audio processing.
 */
export class FeedbackDelayNetwork {
    private audioCtx: AudioContext;
    private _input: GainNode;
    private _output: GainNode;
    private wetGain: GainNode;
    private dryGain: GainNode;

    private delayNodes: DelayNode[] = [];
    private feedbackGains: GainNode[] = [];
    private filters: BiquadFilterNode[] = [];
    private merger: ChannelMergerNode;
    private splitter: ChannelSplitterNode;
    private readonly numChannels: number;

    constructor(audioCtx: AudioContext, channels: number = 8) {
        this.audioCtx = audioCtx;
        this.numChannels = channels;
        console.log(`[FDN CONSTRUCTOR] Creating FDN with ${channels} channels.`);

        // --- Core Nodes ---
        this._input = this.audioCtx.createGain();
        this._output = this.audioCtx.createGain();
        this.wetGain = this.audioCtx.createGain();
        this.dryGain = this.audioCtx.createGain();

        // --- Routing ---
        this.splitter = this.audioCtx.createChannelSplitter(this.numChannels);
        this.merger = this.audioCtx.createChannelMerger(this.numChannels);

        // --- Create Delay Lines and Feedback Loop ---
        const maxDelayTime = 1.0; // Max delay of 1 second
        const delayTimes = this.generateDelayTimes(this.numChannels);
        console.log('[FDN CONSTRUCTOR] Generated Delay Times (s):', delayTimes);


        for (let i = 0; i < this.numChannels; i++) {
            // 1. Delay Line
            const delay = this.audioCtx.createDelay(maxDelayTime);
            delay.delayTime.value = delayTimes[i];
            this.delayNodes.push(delay);

            // 2. Feedback Gain (for decay)
            const gain = this.audioCtx.createGain();
            gain.gain.value = 0.7; // Initial feedback gain
            this.feedbackGains.push(gain);

            // 3. Damping Filter (frequency-dependent decay)
            const filter = this.audioCtx.createBiquadFilter();
            filter.type = 'lowpass';
            filter.frequency.value = 5000; // Initial cutoff
            this.filters.push(filter);

            // --- Internal Connections ---
            // Input goes to each delay line
            this._input.connect(delay);

            // Delay output goes through filter, then gain, then mixes back
            delay.connect(filter);
            filter.connect(gain);
            gain.connect(this.merger, 0, i);
        }

        // --- Feedback Loop with Mixing Matrix ---
        // A Hadamard matrix is great for mixing signals efficiently and without coloration.
        const hadamardMatrix = this.createHadamardMatrix(this.numChannels);
        console.log('[FDN CONSTRUCTOR] Created Hadamard Matrix:', hadamardMatrix);


        // The output of the merged feedback is split and sent back to the delays
        this.merger.connect(this.splitter);

        // Apply the mixing matrix using GainNodes
        for (let i = 0; i < this.numChannels; i++) {
            for (let j = 0; j < this.numChannels; j++) {
                const mixGain = hadamardMatrix[i][j];
                if (mixGain !== 0) {
                    const gainNode = this.audioCtx.createGain();
                    gainNode.gain.value = mixGain;
                    // Connect the mixed signal back to the input of the delay line
                    this.splitter.connect(gainNode, j);
                    gainNode.connect(this.delayNodes[i]);
                }
            }
        }
        
        // --- Wet/Dry Mix ---
        // Connect the final reverb output (wet signal)
        this.merger.connect(this.wetGain);
        this.wetGain.connect(this._output);

        // Connect the original signal (dry signal)
        this._input.connect(this.dryGain);
        this.dryGain.connect(this._output);

        this.setMix(1.0); // Default to fully wet for use as a send effect
    }

    // --- Public Methods ---

    /**
     * Connect the FDN's output to another Web Audio node.
     */
    public connect(destination: AudioNode): void {
        this._output.connect(destination);
    }

    /**
     * The input node to which the audio source should be connected.
     */
    public get input(): GainNode {
        return this._input;
    }
    
    /**
     * Sets the Dry/Wet mix of the reverb.
     * @param wetLevel - A value from 0 (fully dry) to 1 (fully wet).
     */
    public setMix(wetLevel: number): void {
        const clampedWet = Math.max(0, Math.min(1, wetLevel));
        console.log(`[FDN setMix] Setting wet level to: ${clampedWet}`);
        this.wetGain.gain.setValueAtTime(clampedWet, this.audioCtx.currentTime);
        this.dryGain.gain.setValueAtTime(1 - clampedWet, this.audioCtx.currentTime);
    }

    /**
     * Configures the reverb's decay time (RT60) for different frequency bands.
     * @param rt60Values - An object with frequencies as keys and decay times in seconds as values.
     */
    public setRT60(rt60Values: { [frequency: string]: number }): void {
        console.log('[FDN setRT60] Received RT60 values:', rt60Values);
        const rt60_1k = rt60Values['1000'] || 1.5;
        const rt60_high = rt60Values['8000'] || rt60_1k * 0.7;

        for (let i = 0; i < this.numChannels; i++) {
            const delayTime = this.delayNodes[i].delayTime.value;
            let feedbackGain = 0;
            
            // Set feedback gain based on mid-frequency decay
            if (rt60_1k > delayTime) {
                feedbackGain = Math.pow(10, (-3 * delayTime) / rt60_1k);
            }

            // --- MODIFICATION START ---
            // Add a safety clamp to prevent instability from gains >= 1.0
            this.feedbackGains[i].gain.value = Math.min(feedbackGain, 0.999);
            // --- MODIFICATION END ---

            // Set filter frequency for high-frequency damping
            // A higher ratio of high-freq RT60 means less damping (higher cutoff)
            const highFreqRatio = Math.max(0.1, rt60_high / rt60_1k);
            const cutoff = 20000 * Math.pow(highFreqRatio, 2); // Exponential scaling
            this.filters[i].frequency.value = Math.max(400, Math.min(20000, cutoff));
        }
        console.log(`[FDN setRT60] Configured feedback gains and filter cutoffs based on RT60_1k=${rt60_1k}. Max gain clamped to 0.999.`);
    }

    // --- Private Helper Methods ---

    private generateDelayTimes(count: number): number[] {
        // Use prime numbers for delay lengths to minimize ringing and coloration
        const primes = [113, 131, 157, 181, 199, 233, 271, 311, 347, 383, 421, 449, 487, 523, 569, 607];
        const times = [];
        for (let i = 0; i < count; i++) {
            times.push(primes[i % primes.length] / 1000); // Convert to seconds
        }
        return times;
    }
    
    private createHadamardMatrix(size: number): number[][] {
        if (size === 1) return [[1]];
        if (size % 2 !== 0) {
            console.error("Matrix size must be a power of 2 for Hadamard. Using fallback.");
            // Fallback to a simple mixing matrix if not a power of 2
            const fallbackMatrix = Array.from({ length: size }, () => Array(size).fill(1/Math.sqrt(size)));
            return fallbackMatrix;
        }

        const half = this.createHadamardMatrix(size / 2);
        const halfSize = half.length;
        const matrix = Array.from({ length: size }, () => Array(size).fill(0));
        const norm = 1 / Math.sqrt(2);

        for (let i = 0; i < halfSize; i++) {
            for (let j = 0; j < halfSize; j++) {
                matrix[i][j] = half[i][j] * norm;
                matrix[i][j + halfSize] = half[i][j] * norm;
                matrix[i + halfSize][j] = half[i][j] * norm;
                matrix[i + halfSize][j + halfSize] = -half[i][j] * norm;
            }
        }
        return matrix;
    }
}