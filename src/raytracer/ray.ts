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

    const airAmpFactors = this.calculateAirAbsorption(distance);

    this.energies.energy125Hz *= airAmpFactors.absorption125Hz;
    this.energies.energy250Hz *= airAmpFactors.absorption250Hz;
    this.energies.energy500Hz *= airAmpFactors.absorption500Hz;
    this.energies.energy1kHz *= airAmpFactors.absorption1kHz;
    this.energies.energy2kHz *= airAmpFactors.absorption2kHz;
    this.energies.energy4kHz *= airAmpFactors.absorption4kHz;
    this.energies.energy8kHz *= airAmpFactors.absorption8kHz;
    this.energies.energy16kHz *= airAmpFactors.absorption16kHz;

    this.pathLength += distance;
    this.bounces++;
    const speedOfSound = 331.3 + 0.6 * temperature;
    const travelTime = distance / speedOfSound;
    this.time += travelTime;
    const phaseChange = 2 * Math.PI * this.frequency * travelTime;
    this.phase = (this.phase + phaseChange) % (2 * Math.PI);
}

    public calculateAirAbsorption(distance: number): {
        absorption125Hz: number,
        absorption250Hz: number,
        absorption500Hz: number,
        absorption1kHz: number,
        absorption2kHz: number,
        absorption4kHz: number,
        absorption8kHz: number,
        absorption16kHz: number
    } {
        // Simplified exponential decay model for air absorption
        const m = (freq: number) => 0.00005 * Math.pow(freq / 1000, 1.5);

        return {
            absorption125Hz: Math.exp(-m(125) * distance),
            absorption250Hz: Math.exp(-m(250) * distance),
            absorption500Hz: Math.exp(-m(500) * distance),
            absorption1kHz: Math.exp(-m(1000) * distance),
            absorption2kHz: Math.exp(-m(2000) * distance),
            absorption4kHz: Math.exp(-m(4000) * distance),
            absorption8kHz: Math.exp(-m(8000) * distance),
            absorption16kHz: Math.exp(-m(16000) * distance)
        };
    }

    public deactivate(): void {
        this.isActive = false;
    }

    public setEnergies(energies: FrequencyBands): void {
        this.energies = { ...energies };
    }
}