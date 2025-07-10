import { vec3 } from 'gl-matrix';

export class Ray {
    private origin: vec3;
    private direction: vec3;
    private energyLow: number;
    private energyMid: number;
    private energyHigh: number;
    private pathLength: number;
    private bounces: number;
    private isActive: boolean;
    private time: number;
    private phase: number;
    private frequency: number;

    constructor(origin: vec3, direction: vec3, initialEnergy: number = 1.0, frequency: number = 1000) {
        this.origin = vec3.clone(origin);
        this.direction = vec3.normalize(vec3.create(), direction);
        this.energyLow = initialEnergy;
        this.energyMid = initialEnergy;
        this.energyHigh = initialEnergy;
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

    public getEnergy(): number {
        return (this.energyLow + this.energyMid + this.energyHigh) / 3;
    }

    public getEnergyLow(): number {
        return this.energyLow;
    }

    public getEnergyMid(): number {
        return this.energyMid;
    }

    public getEnergyHigh(): number {
        return this.energyHigh;
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
        energyLoss: {low: number, mid: number, high: number},
        distance: number,
        temperature: number = 20,
        humidity: number = 50
    ): void {
        vec3.copy(this.origin, newOrigin);
        vec3.normalize(this.direction, newDirection);

        // Apply frequency-dependent air absorption
        const airAbsorption = this.calculateAirAbsorption(distance, temperature, humidity);

        // Update energy levels with both material absorption and air absorption
        this.energyLow *= (1 - energyLoss.low) * airAbsorption.low;
        this.energyMid *= (1 - energyLoss.mid) * airAbsorption.mid;
        this.energyHigh *= (1 - energyLoss.high) * airAbsorption.high;

        this.pathLength += distance;
        this.bounces++;

        // Calculate speed of sound (simplified formula)
        const speedOfSound = 331.3 + 0.6 * temperature;

        // Update time
        const travelTime = distance / speedOfSound;
        this.time += travelTime;

        // Update phase (2π * frequency * time)
        const phaseChange = 2 * Math.PI * this.frequency * travelTime;
        this.phase = (this.phase + phaseChange) % (2 * Math.PI); // Keep phase between 0 and 2π
    }

    private calculateAirAbsorption(distance: number, temperature: number, humidity: number): {
        low: number,
        mid: number,
        high: number
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
            low: calculateBandAbsorption(250),   // Center frequency for low band
            mid: calculateBandAbsorption(1000),  // Center frequency for mid band
            high: calculateBandAbsorption(4000)  // Center frequency for high band
        };
    }

    public deactivate(): void {
        this.isActive = false;
    }
}