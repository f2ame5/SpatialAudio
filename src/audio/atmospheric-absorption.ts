/**
 * Atmospheric Absorption Models - ISO 9613-1 Implementation
 * 
 * This module implements the ISO 9613-1 standard for calculating
 * frequency-dependent atmospheric absorption of sound.
 */

/**
 * Atmospheric conditions for absorption calculations
 */
export interface AtmosphericConditions {
    temperature: number;    // Temperature in Celsius
    humidity: number;       // Relative humidity (0-100%)
    pressure: number;       // Atmospheric pressure in kPa (default: 101.325)
}

/**
 * Standard atmospheric conditions (20°C, 50% RH, 101.325 kPa)
 */
export const STANDARD_CONDITIONS: AtmosphericConditions = {
    temperature: 20.0,
    humidity: 50.0,
    pressure: 101.325
};

/**
 * Standard frequency bands for acoustic analysis (Hz)
 */
export const FREQUENCY_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000, 16000];

/**
 * Calculate atmospheric absorption coefficient according to ISO 9613-1
 * 
 * @param frequency Frequency in Hz
 * @param conditions Atmospheric conditions
 * @returns Absorption coefficient in dB/km
 */
export function calculateISO9613Absorption(
    frequency: number, 
    conditions: AtmosphericConditions = STANDARD_CONDITIONS
): number {
    const { temperature, humidity, pressure } = conditions;
    
    // Convert temperature to Kelvin
    const T = temperature + 273.15;
    const T0 = 293.15; // Reference temperature (20°C)
    
    // Relative temperature
    const Tr = T / T0;
    
    // Molar concentration of water vapor
    const h = humidity * Math.pow(10, -6.8346 * Math.pow(273.16 / T, 1.261) + 4.6151);
    
    // Relaxation frequencies for oxygen and nitrogen
    const frO = (pressure / 101.325) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
    const frN = (pressure / 101.325) * Math.pow(Tr, -0.5) * (9 + 280 * h * Math.exp(-4.170 * (Math.pow(Tr, -1/3) - 1)));
    
    // Frequency in kHz
    const f = frequency / 1000;
    
    // Classical absorption
    const alpha_classical = 1.84e-11 * (pressure / 101.325) * Math.pow(Tr, -0.5) * f * f;
    
    // Vibrational absorption due to oxygen
    const alpha_oxygen = 0.01275 * Math.exp(-2239.1 / T) * (frO + f * f / frO) / (frO * frO + f * f) * f * f;
    
    // Vibrational absorption due to nitrogen
    const alpha_nitrogen = 0.1068 * Math.exp(-3352.0 / T) * (frN + f * f / frN) / (frN * frN + f * f) * f * f;
    
    // Total absorption coefficient in dB/km
    const alpha_total = alpha_classical + alpha_oxygen + alpha_nitrogen;
    
    return alpha_total;
}

/**
 * Calculate absorption coefficients for all standard frequency bands
 * 
 * @param conditions Atmospheric conditions
 * @returns Array of absorption coefficients in dB/km for each frequency band
 */
export function calculateFrequencyBandAbsorption(
    conditions: AtmosphericConditions = STANDARD_CONDITIONS
): Float32Array {
    const absorptions = new Float32Array(8);
    
    for (let i = 0; i < FREQUENCY_BANDS.length; i++) {
        absorptions[i] = calculateISO9613Absorption(FREQUENCY_BANDS[i], conditions);
    }
    
    return absorptions;
}

/**
 * Convert absorption coefficient from dB/km to linear attenuation per meter
 * 
 * @param absorptionDbKm Absorption coefficient in dB/km
 * @returns Linear attenuation coefficient per meter
 */
export function dbKmToLinearPerMeter(absorptionDbKm: number): number {
    // Convert dB/km to Neper/m: dB/km * ln(10)/20 / 1000
    return absorptionDbKm * Math.LN10 / 20000;
}

/**
 * Calculate atmospheric absorption for raytracing
 * Returns linear attenuation coefficients suitable for GPU shaders
 * 
 * @param conditions Atmospheric conditions
 * @returns Object with low and high frequency absorption arrays
 */
export function calculateRaytracingAbsorption(
    conditions: AtmosphericConditions = STANDARD_CONDITIONS
): { low: Float32Array; high: Float32Array } {
    const absorptionsDbKm = calculateFrequencyBandAbsorption(conditions);
    
    // Convert to linear attenuation per meter
    const absorptionsLinear = new Float32Array(8);
    for (let i = 0; i < 8; i++) {
        absorptionsLinear[i] = dbKmToLinearPerMeter(absorptionsDbKm[i]);
    }
    
    // Split into low and high frequency groups for GPU vec4 alignment
    const low = new Float32Array(4);  // 125, 250, 500, 1k Hz
    const high = new Float32Array(4); // 2k, 4k, 8k, 16k Hz
    
    for (let i = 0; i < 4; i++) {
        low[i] = absorptionsLinear[i];
        high[i] = absorptionsLinear[i + 4];
    }
    
    return { low, high };
}

/**
 * Create atmospheric conditions from common presets
 */
export const ATMOSPHERIC_PRESETS = {
    STANDARD: STANDARD_CONDITIONS,
    
    DRY_COLD: {
        temperature: 5.0,
        humidity: 20.0,
        pressure: 101.325
    } as AtmosphericConditions,
    
    HUMID_WARM: {
        temperature: 30.0,
        humidity: 80.0,
        pressure: 101.325
    } as AtmosphericConditions,
    
    MOUNTAIN: {
        temperature: 10.0,
        humidity: 40.0,
        pressure: 85.0  // ~1500m altitude
    } as AtmosphericConditions,
    
    DESERT: {
        temperature: 35.0,
        humidity: 10.0,
        pressure: 101.325
    } as AtmosphericConditions
};

/**
 * Validate atmospheric conditions
 */
export function validateAtmosphericConditions(conditions: AtmosphericConditions): boolean {
    return (
        conditions.temperature >= -40 && conditions.temperature <= 60 &&
        conditions.humidity >= 0 && conditions.humidity <= 100 &&
        conditions.pressure >= 50 && conditions.pressure <= 120
    );
}

/**
 * Get atmospheric absorption summary for debugging
 */
export function getAbsorptionSummary(conditions: AtmosphericConditions): string {
    const absorptions = calculateFrequencyBandAbsorption(conditions);
    const linear = calculateRaytracingAbsorption(conditions);
    
    let summary = `Atmospheric Absorption (${conditions.temperature}°C, ${conditions.humidity}% RH):\n`;
    
    for (let i = 0; i < FREQUENCY_BANDS.length; i++) {
        const freq = FREQUENCY_BANDS[i];
        const dbKm = absorptions[i].toFixed(3);
        const linearCoeff = (i < 4 ? linear.low[i] : linear.high[i - 4]).toFixed(6);
        summary += `  ${freq}Hz: ${dbKm} dB/km (${linearCoeff}/m)\n`;
    }
    
    return summary;
}
