import { vec3 } from 'gl-matrix';
import { Camera } from '../camera/camera';

export class SpatialAudioProcessor {
    private sampleRate: number;
    private hrtfEnabled = false;
    private hrtfFilters: Map<string, Float32Array[]> = new Map();

    constructor(sampleRate: number = 44100) {
        this.sampleRate = sampleRate;
        this.initializeHRTF();
    }

    private initializeHRTF(): void {
        try {
            // Generate HRTF filters for key directions
            const directions = [
                { azimuth: 0, elevation: 0 },    // Front
                { azimuth: 90, elevation: 0 },   // Right
                { azimuth: 180, elevation: 0 },  // Back
                { azimuth: 270, elevation: 0 },  // Left
                { azimuth: 0, elevation: 45 },   // Above
                { azimuth: 0, elevation: -45 }   // Below
            ];
            
            for (const dir of directions) {
                const [leftFilter, rightFilter] = this.generateHRTFFilters(dir.azimuth, dir.elevation);
                this.hrtfFilters.set(`${dir.azimuth}_${dir.elevation}`, [leftFilter, rightFilter]);
            }
            
            console.log("Initialized HRTF filters");
            this.hrtfEnabled = true;
        } catch (error) {
            console.error("Failed to initialize HRTF filters:", error);
            this.hrtfEnabled = false;
        }
    }
    
    private generateHRTFFilters(azimuth: number, elevation: number): [Float32Array, Float32Array] {
        const filterLength = 512; // Increased filter length for better resolution
        const leftFilter = new Float32Array(filterLength).fill(0);
        const rightFilter = new Float32Array(filterLength).fill(0);
        
        // Convert angles to radians
        const azimuthRad = azimuth * Math.PI / 180;
        const elevationRad = elevation * Math.PI / 180;
        
        const SPEED_OF_SOUND = 343.0; // m/s
        const HEAD_RADIUS = 0.0875; // Approx. radius of the head in meters

        // 1. Interaural Time Difference (ITD)
        // More accurate ITD based on KEMAR dummy head measurements (simplified)
        const itd = (HEAD_RADIUS / SPEED_OF_SOUND) * (azimuthRad + 0.5 * Math.sin(2 * azimuthRad));
        const leftDelaySamples = -itd * this.sampleRate; 
        const rightDelaySamples = itd * this.sampleRate; 

        // 2. Interaural Level Difference (ILD) and Pinna Effects
        const applySpatialFilter = (filter: Float32Array, delaySamples: number, isLeftEar: boolean) => {
            const impulsePos = Math.round(delaySamples); // Position of the main impulse

            // Apply a windowed impulse for smoother response
            const windowSize = 50; // samples
            for (let i = 0; i < windowSize; i++) {
                const idx = impulsePos + i - Math.floor(windowSize / 2);
                if (idx >= 0 && idx < filterLength) {
                    const windowVal = 0.5 * (1 - Math.cos(2 * Math.PI * i / (windowSize - 1))); // Hanning window
                    filter[idx] += windowVal; // Base impulse
                }
            }

            // Apply ILD and Pinna effects by shaping the impulse
            for (let i = 0; i < filterLength; i++) {
                const time = i / this.sampleRate;
                let gain = 1.0;

                // Head shadow (ILD): more attenuation for higher frequencies on the far ear
                const shadowFactor = Math.abs(azimuthRad) / Math.PI; // 0 at front/back, 1 at sides
                const highFreqAttenuation = 1.0 - 0.8 * shadowFactor; // Increased attenuation
                const lowFreqAttenuation = 1.0 - 0.3 * shadowFactor; // Increased attenuation

                // Simple frequency-dependent shaping (simulating a low-pass for far ear)
                if ((isLeftEar && azimuthRad > 0) || (!isLeftEar && azimuthRad < 0)) { // Far ear
                    gain *= (i < filterLength / 8) ? lowFreqAttenuation : highFreqAttenuation; // Apply more to high freq part of impulse
                }

                // Pinna effects (simplified: subtle high-frequency boost/cut based on elevation)
                const elevationEffect = Math.sin(elevationRad); // -1 to 1
                if (elevationEffect > 0) { // Sound from above
                    gain *= (1.0 + 0.15 * elevationEffect); // Increased boost
                } else { // Sound from below
                    gain *= (1.0 + 0.08 * elevationEffect); // Increased cut
                }

                filter[i] *= gain;
            }
        };

        applySpatialFilter(leftFilter, leftDelaySamples, true);
        applySpatialFilter(rightFilter, rightDelaySamples, false);

        // Normalize filters to prevent clipping
        const maxLeft = Math.max(...Array.from(leftFilter).map(Math.abs));
        const maxRight = Math.max(...Array.from(rightFilter).map(Math.abs));
        const overallMax = Math.max(maxLeft, maxRight);

        if (overallMax > 0) {
            for (let i = 0; i < filterLength; i++) {
                leftFilter[i] /= overallMax;
                rightFilter[i] /= overallMax;
            }
        }
        
        return [leftFilter, rightFilter];
    }

    public calculateImprovedHRTF(
        sourcePos: vec3,
        listenerPos: vec3,
        listenerFront: vec3,
        listenerRight: vec3,
        listenerUp: vec3
    ): [Float32Array, Float32Array] {
        // Calculate direction vector from listener to source
        const direction = vec3.create();
        vec3.subtract(direction, sourcePos, listenerPos);
        vec3.normalize(direction, direction);
        
        // Calculate azimuth (horizontal angle)
        const dotRight = vec3.dot(direction, listenerRight);
        const dotFront = vec3.dot(direction, listenerFront);
        const azimuth = Math.atan2(dotRight, dotFront);
        
        // Calculate elevation (vertical angle)
        const dotUp = vec3.dot(direction, listenerUp);
        const elevation = Math.asin(Math.max(-1, Math.min(1, dotUp)));
        
        // Convert radians to degrees for generateHRTFFilters
        const azimuthDeg = azimuth * 180 / Math.PI;
        const elevationDeg = elevation * 180 / Math.PI;

        // Generate HRTF filters for this specific direction
        return this.generateHRTFFilters(azimuthDeg, elevationDeg);
    }
}