/**
 * Audio helper utilities for spatial audio processing
 */

import { FREQUENCY_BANDS } from '../audio/audio-utils';

/**
 * Generate white noise buffer
 */
export function generateWhiteNoise(
    context: AudioContext,
    duration: number
): AudioBuffer {
    const sampleRate = context.sampleRate;
    const samples = duration * sampleRate;
    const buffer = context.createBuffer(1, samples, sampleRate);
    const channel = buffer.getChannelData(0);

    for (let i = 0; i < samples; i++) {
        channel[i] = (Math.random() * 2 - 1) * 0.2; // Scale to avoid clipping
    }

    return buffer;
}

/**
 * Generate sine sweep buffer for impulse response measurement
 */
export function generateSineSweep(
    context: AudioContext,
    duration: number,
    startFreq: number = 20,
    endFreq: number = 20000
): AudioBuffer {
    const sampleRate = context.sampleRate;
    const samples = duration * sampleRate;
    const buffer = context.createBuffer(1, samples, sampleRate);
    const channel = buffer.getChannelData(0);

    const startW = 2 * Math.PI * startFreq / sampleRate;
    const endW = 2 * Math.PI * endFreq / sampleRate;
    const K = duration * sampleRate / Math.log(endW / startW);

    for (let i = 0; i < samples; i++) {
        const t = i / sampleRate;
        const phase = K * (Math.exp(t * Math.log(endW / startW) / duration) - 1);
        channel[i] = Math.sin(startW * phase) * 0.5;
    }

    // Apply fade in/out
    const fadeLength = Math.floor(sampleRate * 0.01); // 10ms fade
    for (let i = 0; i < fadeLength; i++) {
        const gain = i / fadeLength;
        channel[i] *= gain;
        channel[samples - 1 - i] *= gain;
    }

    return buffer;
}

/**
 * Generate click/impulse for testing
 */
export function generateImpulse(
    context: AudioContext,
    amplitude: number = 1.0
): AudioBuffer {
    const sampleRate = context.sampleRate;
    const buffer = context.createBuffer(1, sampleRate, sampleRate); // 1 second buffer
    const channel = buffer.getChannelData(0);
    
    // Single sample impulse at the beginning
    channel[0] = amplitude;
    
    return buffer;
}

/**
 * Apply exponential decay to a buffer
 */
export function applyExponentialDecay(
    buffer: AudioBuffer,
    decayTime: number
): void {
    const sampleRate = buffer.sampleRate;
    const decayRate = Math.log(0.001) / (decayTime * sampleRate); // -60dB decay

    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
        const channel = buffer.getChannelData(ch);
        for (let i = 0; i < channel.length; i++) {
            channel[i] *= Math.exp(decayRate * i);
        }
    }
}

/**
 * Calculate RT60 (reverberation time) from impulse response
 */
export function calculateRT60(
    impulseResponse: Float32Array,
    sampleRate: number
): number {
    // Find peak
    let peakIndex = 0;
    let peakValue = 0;
    for (let i = 0; i < impulseResponse.length; i++) {
        const absValue = Math.abs(impulseResponse[i]);
        if (absValue > peakValue) {
            peakValue = absValue;
            peakIndex = i;
        }
    }

    // Find -60dB point (0.001 of peak)
    const threshold = peakValue * 0.001;
    let decayIndex = peakIndex;
    
    for (let i = peakIndex; i < impulseResponse.length; i++) {
        if (Math.abs(impulseResponse[i]) < threshold) {
            decayIndex = i;
            break;
        }
    }

    // Calculate RT60
    const decayTime = (decayIndex - peakIndex) / sampleRate;
    return decayTime;
}

/**
 * Apply frequency-dependent gain to audio buffer
 */
export function applyFrequencyGains(
    context: AudioContext,
    buffer: AudioBuffer,
    gains: number[] // One gain per frequency band
): AudioBuffer {
    if (gains.length !== FREQUENCY_BANDS.length) {
        throw new Error('Gains array must match frequency bands length');
    }

    // Create offline context for processing
    const offlineContext = new OfflineAudioContext(
        buffer.numberOfChannels,
        buffer.length,
        buffer.sampleRate
    );

    // Create source
    const source = offlineContext.createBufferSource();
    source.buffer = buffer;

    // Create filter chain
    let lastNode: AudioNode = source;
    const filters: BiquadFilterNode[] = [];

    for (let i = 0; i < FREQUENCY_BANDS.length; i++) {
        const filter = offlineContext.createBiquadFilter();
        
        if (i === 0) {
            // Low shelf for lowest band
            filter.type = 'lowshelf';
            filter.frequency.value = FREQUENCY_BANDS[i];
        } else if (i === FREQUENCY_BANDS.length - 1) {
            // High shelf for highest band
            filter.type = 'highshelf';
            filter.frequency.value = FREQUENCY_BANDS[i];
        } else {
            // Peaking filter for middle bands
            filter.type = 'peaking';
            filter.frequency.value = FREQUENCY_BANDS[i];
            filter.Q.value = 1.0;
        }
        
        filter.gain.value = 20 * Math.log10(gains[i]); // Convert to dB
        
        lastNode.connect(filter);
        lastNode = filter;
        filters.push(filter);
    }

    // Connect to destination
    lastNode.connect(offlineContext.destination);

    // Start and render
    source.start(0);
    return offlineContext.startRendering() as any; // Returns Promise<AudioBuffer>
}

/**
 * Mix two audio buffers
 */
export function mixBuffers(
    context: AudioContext,
    buffer1: AudioBuffer,
    buffer2: AudioBuffer,
    gain1: number = 0.5,
    gain2: number = 0.5
): AudioBuffer {
    const length = Math.max(buffer1.length, buffer2.length);
    const channels = Math.max(buffer1.numberOfChannels, buffer2.numberOfChannels);
    const sampleRate = buffer1.sampleRate;
    
    const mixedBuffer = context.createBuffer(channels, length, sampleRate);
    
    for (let ch = 0; ch < channels; ch++) {
        const mixedChannel = mixedBuffer.getChannelData(ch);
        
        // Get channel data or create empty if not available
        const channel1 = ch < buffer1.numberOfChannels ? 
            buffer1.getChannelData(ch) : new Float32Array(buffer1.length);
        const channel2 = ch < buffer2.numberOfChannels ? 
            buffer2.getChannelData(ch) : new Float32Array(buffer2.length);
        
        // Mix samples
        for (let i = 0; i < length; i++) {
            const sample1 = i < channel1.length ? channel1[i] * gain1 : 0;
            const sample2 = i < channel2.length ? channel2[i] * gain2 : 0;
            mixedChannel[i] = sample1 + sample2;
        }
    }
    
    return mixedBuffer;
}

/**
 * Convert mono buffer to stereo
 */
export function monoToStereo(
    context: AudioContext,
    monoBuffer: AudioBuffer
): AudioBuffer {
    if (monoBuffer.numberOfChannels !== 1) {
        throw new Error('Input must be mono');
    }
    
    const stereoBuffer = context.createBuffer(
        2,
        monoBuffer.length,
        monoBuffer.sampleRate
    );
    
    const monoData = monoBuffer.getChannelData(0);
    stereoBuffer.copyToChannel(monoData, 0);
    stereoBuffer.copyToChannel(monoData, 1);
    
    return stereoBuffer;
}

/**
 * Normalize audio buffer to prevent clipping
 */
export function normalizeBuffer(buffer: AudioBuffer): void {
    let maxValue = 0;
    
    // Find maximum value across all channels
    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
        const channel = buffer.getChannelData(ch);
        for (let i = 0; i < channel.length; i++) {
            maxValue = Math.max(maxValue, Math.abs(channel[i]));
        }
    }
    
    // Normalize if needed
    if (maxValue > 0.95) {
        const scale = 0.95 / maxValue;
        for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
            const channel = buffer.getChannelData(ch);
            for (let i = 0; i < channel.length; i++) {
                channel[i] *= scale;
            }
        }
    }
}
