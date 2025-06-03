import { vec3 } from 'gl-matrix';

export interface FrequencyBands {
    energy125Hz: number;
    energy250Hz: number;
    energy500Hz: number;
    energy1kHz: number;
    energy2kHz: number;
    energy4kHz: number;
    energy8kHz: number;
    energy16kHz: number;
}

export class Ray {
    private origin: vec3;
    private direction: vec3;
    private energies: FrequencyBands;
    private pathLength: number;
    private bounces: number;
    private isActive: boolean;
    private time: number;
    private phase: number;
    private frequency: number;

    constructor(origin: vec3, direction: vec3, initialEnergy: number = 1.0, frequency: number = 1000) {
        this.origin = vec3.clone(origin);
        this.direction = vec3.normalize(vec3.create(), direction);

        // Initialize all frequency bands with the same initial energy
        this.energies = {
            energy125Hz: initialEnergy,
            energy250Hz: initialEnergy,
            energy500Hz: initialEnergy,
            energy1kHz: initialEnergy,
            energy2kHz: initialEnergy,
            energy4kHz: initialEnergy,
            energy8kHz: initialEnergy,
            energy16kHz: initialEnergy
        };

        this.pathLength = 0;
        this.bounces = 0;
        this.isActive = true;
        this.time = 0;
        this.phase = 0;
        this.frequency = frequency;
    }

    public getOrigin(): vec3 {
        return vec3.clone(this.origin);
    }

    public getDirection(): vec3 {
        return vec3.clone(this.direction);
    }

    public getEnergies(): FrequencyBands {
        return { ...this.energies };
    }

    public getAverageEnergy(): number {
        const values = Object.values(this.energies);
        return values.reduce((sum, energy) => sum + energy, 0) / values.length;
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

    public getPhase(): number {
        return this.phase;
    }

    public getFrequency(): number {
        return this.frequency;
    }

    public updateTime(newTime: number): void {
        this.time = newTime;
    }

    public updatePhase(newPhase: number): void {
        this.phase = newPhase;
    }

    public updateRay(
    newOrigin: vec3,
    newDirection: vec3,
    energyLoss: {
        absorption125Hz: number,
        absorption250Hz: number,
        absorption500Hz: number,
        absorption1kHz: number,
        absorption2kHz: number,
        absorption4kHz: number,
        absorption8kHz: number,
        absorption16kHz: number
    },
    distance: number,
    temperature: number = 20,
    humidity: number = 50
): void {
    vec3.copy(this.origin, newOrigin);
    vec3.normalize(this.direction, newDirection);

    this.energies.energy125Hz *= (1 - energyLoss.absorption125Hz);
    this.energies.energy250Hz *= (1 - energyLoss.absorption250Hz);
    this.energies.energy500Hz *= (1 - energyLoss.absorption500Hz);
    this.energies.energy1kHz *= (1 - energyLoss.absorption1kHz);
    this.energies.energy2kHz *= (1 - energyLoss.absorption2kHz);
    this.energies.energy4kHz *= (1 - energyLoss.absorption4kHz);
    this.energies.energy8kHz *= (1 - energyLoss.absorption8kHz);
    this.energies.energy16kHz *= (1 - energyLoss.absorption16kHz);

    const airAmpFactors = this.calculateAirAbsorption(distance, temperature, humidity);

    this.energies.energy125Hz *= Math.pow(airAmpFactors.absorption125Hz, 2);
    this.energies.energy250Hz *= Math.pow(airAmpFactors.absorption250Hz, 2);
    this.energies.energy500Hz *= Math.pow(airAmpFactors.absorption500Hz, 2);
    this.energies.energy1kHz *= Math.pow(airAmpFactors.absorption1kHz, 2);
    this.energies.energy2kHz *= Math.pow(airAmpFactors.absorption2kHz, 2);
    this.energies.energy4kHz *= Math.pow(airAmpFactors.absorption4kHz, 2);
    this.energies.energy8kHz *= Math.pow(airAmpFactors.absorption8kHz, 2);
    this.energies.energy16kHz *= Math.pow(airAmpFactors.absorption16kHz, 2);

    this.pathLength += distance;
    this.bounces++;
    const speedOfSound = 331.3 + 0.6 * temperature;
    const travelTime = distance / speedOfSound;
    this.time += travelTime;
    const phaseChange = 2 * Math.PI * this.frequency * travelTime;
    this.phase = (this.phase + phaseChange) % (2 * Math.PI);
}

    public calculateAirAbsorption(distance: number, temperature: number, humidity: number): {
        absorption125Hz: number,
        absorption250Hz: number,
        absorption500Hz: number,
        absorption1kHz: number,
        absorption2kHz: number,
        absorption4kHz: number,
        absorption8kHz: number,
        absorption16kHz: number
    } {
        // ISO 9613-1 standard air absorption calculation
        const T = temperature + 273.15;
        const T0 = 293.15;
        const T01 = T / T0;
        const hr = humidity * Math.pow(T01, -4.17);

        // Calculate absorption for each frequency band
        const calculateBandAbsorption = (freq: number): number => {
            const fr = freq * T01;
            const alpha = 1.84e-11 * (1 / T01) * Math.sqrt(T01) +
                Math.pow(fr, 2.5) * (0.10680 * Math.exp(-3352 / T) * 1 / (fr + 3352 / T)) +
                Math.pow(fr, 2.5) * (0.01278 * Math.exp(-2239.1 / T) * 1 / (fr + 2239.1 / T));

            return Math.exp(-alpha * distance);
        };

        return {
            absorption125Hz: calculateBandAbsorption(125),
            absorption250Hz: calculateBandAbsorption(250),
            absorption500Hz: calculateBandAbsorption(500),
            absorption1kHz: calculateBandAbsorption(1000),
            absorption2kHz: calculateBandAbsorption(2000),
            absorption4kHz: calculateBandAbsorption(4000),
            absorption8kHz: calculateBandAbsorption(8000),
            absorption16kHz: calculateBandAbsorption(16000)
        };
    }

    public deactivate(): void {
        this.isActive = false;
    }

    public setEnergies(energies: FrequencyBands): void {
        this.energies = { ...energies };
    }
}