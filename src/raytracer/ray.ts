import { vec3 } from 'gl-matrix';

export class Ray {
    private origin: vec3;
    private direction: vec3;
    // Energy for each of the 8 frequency bands
    private energy63: number;
    private energy125: number;
    private energy250: number;
    private energy500: number;
    private energy1k: number;
    private energy2k: number;
    private energy4k: number;
    private energy8k: number;
    private pathLength: number;
    private bounces: number;
    private isActive: boolean;
    private time: number;
    // Phase for each of the 8 frequency bands
    private phase63: number;
    private phase125: number;
    private phase250: number;
    private phase500: number;
    private phase1k: number;
    private phase2k: number;
    private phase4k: number;
    private phase8k: number;
    // Frequency is now implicit in the band

    // Define the 8 frequency bands
    private static readonly FREQUENCY_BANDS = {
        63: 63,
        125: 125,
        250: 250,
        500: 500,
        1000: 1000,
        2000: 2000,
        4000: 4000,
        8000: 8000
    };

    constructor(origin: vec3, direction: vec3, initialEnergy: number = 1.0) {
        this.origin = vec3.clone(origin);
        this.direction = vec3.normalize(vec3.create(), direction);
        // Initialize energy for all bands
        this.energy63 = initialEnergy;
        this.energy125 = initialEnergy;
        this.energy250 = initialEnergy;
        this.energy500 = initialEnergy;
        this.energy1k = initialEnergy;
        this.energy2k = initialEnergy;
        this.energy4k = initialEnergy;
        this.energy8k = initialEnergy;
        this.pathLength = 0;
        this.bounces = 0;
        this.isActive = true;
        this.time = 0;
        // Initialize phase for all bands
        this.phase63 = 0;
        this.phase125 = 0;
        this.phase250 = 0;
        this.phase500 = 0;
        this.phase1k = 0;
        this.phase2k = 0;
        this.phase4k = 0;
        this.phase8k = 0;
    }

    public getOrigin(): vec3 {
        return vec3.clone(this.origin);
    }

    public getDirection(): vec3 {
        return vec3.clone(this.direction);
    }

    // Get average energy across all bands
    public getEnergy(): number {
        return (this.energy63 + this.energy125 + this.energy250 + this.energy500 +
                this.energy1k + this.energy2k + this.energy4k + this.energy8k) / 8;
    }

    // Get energy for specific bands
    public getEnergy63(): number {
        return this.energy63;
    }

    public getEnergy125(): number {
        return this.energy125;
    }

    public getEnergy250(): number {
        return this.energy250;
    }

    public getEnergy500(): number {
        return this.energy500;
    }

    public getEnergy1k(): number {
        return this.energy1k;
    }

    public getEnergy2k(): number {
        return this.energy2k;
    }

    public getEnergy4k(): number {
        return this.energy4k;
    }

    public getEnergy8k(): number {
        return this.energy8k;
    }

    public getBounces(): number {
        return this.bounces;
    }

    public isRayActive(): boolean {
        return this.isActive;
    }

    public getTime(): number {
        return this.time;
    }

    // Get phase for specific bands
    public getPhase63(): number {
        return this.phase63;
    }

    public getPhase125(): number {
        return this.phase125;
    }

    public getPhase250(): number {
        return this.phase250;
    }

    public getPhase500(): number {
        return this.phase500;
    }

    public getPhase1k(): number {
        return this.phase1k;
    }

    public getPhase2k(): number {
        return this.phase2k;
    }

    public getPhase4k(): number {
        return this.phase4k;
    }

    public getPhase8k(): number {
        return this.phase8k;
    }

    public updateTime(newTime: number): void {
        this.time = newTime;
    }

    // Update phase for all bands
    public updatePhase(
        phase63: number,
        phase125: number,
        phase250: number,
        phase500: number,
        phase1k: number,
        phase2k: number,
        phase4k: number,
        phase8k: number
    ): void {
        this.phase63 = phase63;
        this.phase125 = phase125;
        this.phase250 = phase250;
        this.phase500 = phase500;
        this.phase1k = phase1k;
        this.phase2k = phase2k;
        this.phase4k = phase4k;
        this.phase8k = phase8k;
    }

    public updateRay(
        newOrigin: vec3,
        newDirection: vec3,
        energyLoss: {
            band63: number,
            band125: number,
            band250: number,
            band500: number,
            band1k: number,
            band2k: number,
            band4k: number,
            band8k: number
        },
        distance: number,
        temperature: number = 20,
        humidity: number = 50
    ): void {
        vec3.copy(this.origin, newOrigin);
        vec3.normalize(this.direction, newDirection);

        // Apply frequency-dependent air absorption based on the ISO 9613-1 standard.
        const airAbsorption = this.calculateAirAbsorption(distance, temperature, humidity);

        // Update energy levels with both material absorption and air absorption.
        this.energy63  *= (1.0 - energyLoss.band63)  * airAbsorption.band63;
        this.energy125 *= (1.0 - energyLoss.band125) * airAbsorption.band125;
        this.energy250 *= (1.0 - energyLoss.band250) * airAbsorption.band250;
        this.energy500 *= (1.0 - energyLoss.band500) * airAbsorption.band500;
        this.energy1k  *= (1.0 - energyLoss.band1k)  * airAbsorption.band1k;
        this.energy2k  *= (1.0 - energyLoss.band2k)  * airAbsorption.band2k;
        this.energy4k  *= (1.0 - energyLoss.band4k)  * airAbsorption.band4k;
        this.energy8k  *= (1.0 - energyLoss.band8k)  * airAbsorption.band8k;

        this.pathLength += distance;
        this.bounces++;

        // Calculate speed of sound (simplified formula)
        const speedOfSound = 331.3 + 0.6 * temperature;

        // Update time
        const travelTime = distance / speedOfSound;
        this.time += travelTime;

        // Update phase for each band (2π * frequency * time)
        const phaseChange63 = 2 * Math.PI * Ray.FREQUENCY_BANDS[63] * travelTime;
        const phaseChange125 = 2 * Math.PI * Ray.FREQUENCY_BANDS[125] * travelTime;
        const phaseChange250 = 2 * Math.PI * Ray.FREQUENCY_BANDS[250] * travelTime;
        const phaseChange500 = 2 * Math.PI * Ray.FREQUENCY_BANDS[500] * travelTime;
        const phaseChange1k = 2 * Math.PI * Ray.FREQUENCY_BANDS[1000] * travelTime;
        const phaseChange2k = 2 * Math.PI * Ray.FREQUENCY_BANDS[2000] * travelTime;
        const phaseChange4k = 2 * Math.PI * Ray.FREQUENCY_BANDS[4000] * travelTime;
        const phaseChange8k = 2 * Math.PI * Ray.FREQUENCY_BANDS[8000] * travelTime;

        this.phase63 = (this.phase63 + phaseChange63) % (2 * Math.PI);
        this.phase125 = (this.phase125 + phaseChange125) % (2 * Math.PI);
        this.phase250 = (this.phase250 + phaseChange250) % (2 * Math.PI);
        this.phase500 = (this.phase500 + phaseChange500) % (2 * Math.PI);
        this.phase1k = (this.phase1k + phaseChange1k) % (2 * Math.PI);
        this.phase2k = (this.phase2k + phaseChange2k) % (2 * Math.PI);
        this.phase4k = (this.phase4k + phaseChange4k) % (2 * Math.PI);
        this.phase8k = (this.phase8k + phaseChange8k) % (2 * Math.PI);
    }

    /**
     * Calculates frequency-dependent air absorption based on the ISO 9613-1 standard.
     * Returns the transmission coefficient (0-1) for each frequency band.
     */
    private calculateAirAbsorption(distance: number, temperature: number, humidity: number): {
        band63: number,
        band125: number,
        band250: number,
        band500: number,
        band1k: number,
        band2k: number,
        band4k: number,
        band8k: number
    } {
        // Simplified ISO 9613-1 model for atmospheric absorption.
        const T_celsius = temperature;
        const T_kelvin = T_celsius + 273.15;
        const T_ref = 293.15; // 20°C in Kelvin
 
        // Calculate absorption for each frequency band
        const calculateBandAbsorption = (freq: number): number => {
            // Relaxation frequency for oxygen and nitrogen (simplified)
            const f_r_O = 24 + 4.04e4 * humidity * (0.02 + humidity) / (0.391 + humidity);
            const f_r_N = (T_kelvin / T_ref)**(-0.5) * (9 + 280 * humidity * Math.exp(-4.17 * ((T_kelvin / T_ref)**(-1/3) - 1)));

            // Absorption coefficient in dB/m
            const alpha_dB_per_m = freq**2 * (
                1.84e-11 * (T_kelvin / T_ref)**0.5 +
                (T_kelvin / T_ref)**(-2.5) * (
                    0.1068 * Math.exp(-3352 / T_kelvin) / (f_r_N + freq**2 / f_r_N) + 
                    0.01278 * Math.exp(-2239.1 / T_kelvin) / (f_r_O + freq**2 / f_r_O)
                )
            );

            // Total attenuation in dB
            const total_attenuation_dB = alpha_dB_per_m * distance;
            // Convert dB attenuation to a linear transmission factor (0 to 1)
            return Math.pow(10, -total_attenuation_dB / 20);
        };

        return {
            band63: calculateBandAbsorption(Ray.FREQUENCY_BANDS[63]),
            band125: calculateBandAbsorption(Ray.FREQUENCY_BANDS[125]),
            band250: calculateBandAbsorption(Ray.FREQUENCY_BANDS[250]),
            band500: calculateBandAbsorption(Ray.FREQUENCY_BANDS[500]),
            band1k: calculateBandAbsorption(Ray.FREQUENCY_BANDS[1000]),
            band2k: calculateBandAbsorption(Ray.FREQUENCY_BANDS[2000]),
            band4k: calculateBandAbsorption(Ray.FREQUENCY_BANDS[4000]),
            band8k: calculateBandAbsorption(Ray.FREQUENCY_BANDS[8000])
        };
    }

    public deactivate(): void {
        this.isActive = false;
    }
}