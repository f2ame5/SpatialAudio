/**
 * Audio utilities for spatial audio processing
 */

/**
 * Creates and initializes a Web Audio API context
 * Handles browser compatibility and user gesture requirements
 */
export async function createAudioContext(): Promise<AudioContext> {
    // Check for Web Audio API support
    if (!window.AudioContext && !(window as any).webkitAudioContext) {
        throw new Error('Web Audio API is not supported in this browser');
    }

    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const context = new AudioContextClass();

    // Resume context if it's suspended (due to autoplay policy)
    if (context.state === 'suspended') {
        await context.resume();
    }

    return context;
}

/**
 * Loads an audio file and decodes it into an AudioBuffer
 */
export async function loadAudioFile(
    context: AudioContext, 
    url: string
): Promise<AudioBuffer> {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load audio file: ${response.statusText}`);
        }
        
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await context.decodeAudioData(arrayBuffer);
        
        return audioBuffer;
    } catch (error) {
        throw new Error(`Error loading audio file ${url}: ${error}`);
    }
}

/**
 * Creates a gain node with specified initial gain value
 */
export function createGainNode(
    context: AudioContext, 
    gainValue: number = 1.0
): GainNode {
    const gainNode = context.createGain();
    gainNode.gain.value = gainValue;
    return gainNode;
}

/**
 * Creates a panner node for 3D spatial positioning
 */
export function createPannerNode(
    context: AudioContext,
    position: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 }
): PannerNode {
    const panner = context.createPanner();
    
    // Set panning model
    panner.panningModel = 'HRTF';
    panner.distanceModel = 'inverse';
    
    // Set position
    panner.positionX.value = position.x;
    panner.positionY.value = position.y;
    panner.positionZ.value = position.z;
    
    // Set default cone parameters
    panner.coneInnerAngle = 360;
    panner.coneOuterAngle = 360;
    panner.coneOuterGain = 0;
    
    // Set distance parameters
    panner.refDistance = 1;
    panner.maxDistance = 10000;
    panner.rolloffFactor = 1;
    
    return panner;
}

/**
 * Updates the listener position and orientation in 3D space
 */
export function updateListener(
    context: AudioContext,
    position: { x: number; y: number; z: number },
    forward: { x: number; y: number; z: number },
    up: { x: number; y: number; z: number }
): void {
    const listener = context.listener;
    
    // Update position
    if (listener.positionX) {
        // New API
        listener.positionX.value = position.x;
        listener.positionY.value = position.y;
        listener.positionZ.value = position.z;
        
        listener.forwardX.value = forward.x;
        listener.forwardY.value = forward.y;
        listener.forwardZ.value = forward.z;
        
        listener.upX.value = up.x;
        listener.upY.value = up.y;
        listener.upZ.value = up.z;
    } else {
        // Legacy API fallback
        (listener as any).setPosition(position.x, position.y, position.z);
        (listener as any).setOrientation(
            forward.x, forward.y, forward.z,
            up.x, up.y, up.z
        );
    }
}

/**
 * Calculates the distance between two 3D points
 */
export function calculateDistance(
    point1: { x: number; y: number; z: number },
    point2: { x: number; y: number; z: number }
): number {
    const dx = point2.x - point1.x;
    const dy = point2.y - point1.y;
    const dz = point2.z - point1.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Converts decibels to linear gain value
 */
export function dbToLinear(db: number): number {
    return Math.pow(10, db / 20);
}

/**
 * Converts linear gain value to decibels
 */
export function linearToDb(linear: number): number {
    return 20 * Math.log10(Math.max(0.00001, linear));
}

/**
 * Creates a simple oscillator for testing
 */
export function createTestOscillator(
    context: AudioContext,
    frequency: number = 440,
    type: OscillatorType = 'sine'
): OscillatorNode {
    const oscillator = context.createOscillator();
    oscillator.frequency.value = frequency;
    oscillator.type = type;
    return oscillator;
}

/**
 * Frequency bands for acoustic analysis (in Hz)
 */
export const FREQUENCY_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000, 16000];

/**
 * Common sample rates for audio processing
 */
export const SAMPLE_RATES = {
    CD_QUALITY: 44100,
    DAT_QUALITY: 48000,
    HIGH_QUALITY: 96000,
    ULTRA_QUALITY: 192000
};

/**
 * Audio format support detection
 */
export function getSupportedAudioFormats(): {
    mp3: boolean;
    wav: boolean;
    ogg: boolean;
    aac: boolean;
    flac: boolean;
} {
    const audio = new Audio();
    
    return {
        mp3: audio.canPlayType('audio/mpeg') !== '',
        wav: audio.canPlayType('audio/wav') !== '',
        ogg: audio.canPlayType('audio/ogg') !== '',
        aac: audio.canPlayType('audio/aac') !== '',
        flac: audio.canPlayType('audio/flac') !== ''
    };
}

/**
 * Creates an analyser node for frequency analysis
 */
export function createAnalyser(
    context: AudioContext,
    fftSize: number = 2048
): AnalyserNode {
    const analyser = context.createAnalyser();
    analyser.fftSize = fftSize;
    analyser.smoothingTimeConstant = 0.8;
    return analyser;
}

/**
 * Gets frequency data from an analyser node
 */
export function getFrequencyData(analyser: AnalyserNode): Uint8Array {
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);
    return dataArray;
}

/**
 * Simple fade in/out utility
 */
export function fade(
    gainNode: GainNode,
    targetValue: number,
    duration: number,
    startTime?: number
): void {
    const context = gainNode.context;
    const time = startTime || context.currentTime;
    
    gainNode.gain.cancelScheduledValues(time);
    gainNode.gain.setValueAtTime(gainNode.gain.value, time);
    gainNode.gain.linearRampToValueAtTime(targetValue, time + duration);
}
