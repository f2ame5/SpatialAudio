struct ListenerData {
    position: vec3f,
    forward: vec3f,
    up: vec3f,
    right: vec3f,
};

// This struct MUST MATCH the one in raytracer.wgsl
struct RayHit {
    position: vec3f,
    time: f32,
    normal: vec3f,
    // Energy for each frequency band
    energy63: f32,
    energy125: f32,
    energy250: f32,
    energy500: f32,
    energy1k: f32,
    energy2k: f32,
    energy4k: f32,
    energy8k: f32,
    incomingDirection: vec3f,
    // Phase for each frequency band
    phase63: f32,
    phase125: f32,
    phase250: f32,
    phase500: f32,
    phase1k: f32,
    phase2k: f32,
    phase4k: f32,
    phase8k: f32,
    hrtfIndex: u32,
    _padding: f32 // Maintain alignment
};


struct RoomAcoustics {
    // RT60 for each frequency band
    rt60_63: f32,
    rt60_125: f32,
    rt60_250: f32,
    rt60_500: f32,
    rt60_1k: f32,
    rt60_2k: f32,
    rt60_4k: f32,
    rt60_8k: f32,

    // Air absorption coefficients (increases with frequency)
    absorption_63: f32,
    absorption_125: f32,
    absorption_250: f32,
    absorption_500: f32,
    absorption_1k: f32,
    absorption_2k: f32,
    absorption_4k: f32,
    absorption_8k: f32,

    // Scattering coefficients (frequency dependent)
    scattering_63: f32,
    scattering_125: f32,
    scattering_250: f32,
    scattering_500: f32,
    scattering_1k: f32,
    scattering_2k: f32,
    scattering_4k: f32,
    scattering_8k: f32,

    // Room characteristics
    earlyReflectionTime: f32,
    roomVolume: f32,
    totalSurfaceArea: f32,
};

struct SpatialAudioParams {
    speedOfSound: f32,
    maxDistance: f32,
    minDistance: f32,
    temperature: f32,    // For air absorption calculation
    humidity: f32,       // For air absorption calculation
    sourcePower: f32,    // Source power in dB
    _padding1: f32,
    _padding2: f32,
};

@group(0) @binding(0) var<uniform> listener: ListenerData;
@group(0) @binding(1) var<storage, read> rayHits: array<RayHit>;
@group(0) @binding(2) var<storage, read_write> spatialIR: array<vec4f>;
@group(0) @binding(3) var<uniform> params: SpatialAudioParams;
@group(0) @binding(4) var<uniform> acoustics: RoomAcoustics;
@group(0) @binding(7) var<storage, read> hrtfCoefficients: array<f32>;

// Simplified HRTF lookup
fn getHRTFCoefficient(index: u32) -> vec2f {
    let safeIndex = min(index, arrayLength(&hrtfCoefficients) / 2u - 1u);
    return vec2f(
        hrtfCoefficients[safeIndex * 2u],
        hrtfCoefficients[safeIndex * 2u + 1u]
    );
}

// Simplified air absorption for path from hit to listener
fn calculateAirAbsorption(distance: f32) -> array<f32, 8> {
    var absorption: array<f32, 8>;
    absorption[0] = exp(-acoustics.absorption_63 * distance);
    absorption[1] = exp(-acoustics.absorption_125 * distance);
    absorption[2] = exp(-acoustics.absorption_250 * distance);
    absorption[3] = exp(-acoustics.absorption_500 * distance);
    absorption[4] = exp(-acoustics.absorption_1k * distance);
    absorption[5] = exp(-acoustics.absorption_2k * distance);
    absorption[6] = exp(-acoustics.absorption_4k * distance);
    absorption[7] = exp(-acoustics.absorption_8k * distance);
    return absorption;
}


@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let hitIndex = global_id.x;
    if (hitIndex >= arrayLength(&rayHits)) {
        return;
    }

    let hit = rayHits[hitIndex];
    
    // Vector from the hit point to the listener
    let toListenerDir = normalize(listener.position - hit.position);
    let distance = length(listener.position - hit.position);

    // Calculate total energy from all bands at the hit point
    let totalEnergy = hit.energy63 + hit.energy125 + hit.energy250 + hit.energy500 +
                      hit.energy1k + hit.energy2k + hit.energy4k + hit.energy8k;

    if (totalEnergy <= 0.0) {
        spatialIR[hitIndex] = vec4f(0.0);
        return;
    }

    // 1. Distance Attenuation (Inverse Square Law)
    let distAttenuation = 1.0 / (1.0 + distance * distance);
    
    // 2. Air Absorption from hit to listener
    let airAbsorption = calculateAirAbsorption(distance);

    // 3. HRTF for spatialization
    let hrtf = getHRTFCoefficient(hit.hrtfIndex);

    // 4. Combine effects for each band and sum them up for a final amplitude
    var totalAmplitude = 0.0;
    let energies = array<f32, 8>(hit.energy63, hit.energy125, hit.energy250, hit.energy500, hit.energy1k, hit.energy2k, hit.energy4k, hit.energy8k);
    
    for (var i = 0u; i < 8u; i = i + 1u) {
        // Amplitude is sqrt of energy
        let bandAmplitude = sqrt(max(0.0, energies[i]));
        // Apply air absorption for this band
        let absorbedAmplitude = bandAmplitude * airAbsorption[i];
        totalAmplitude = totalAmplitude + absorbedAmplitude;
    }

    // Apply distance attenuation to the summed amplitude
    totalAmplitude = totalAmplitude * distAttenuation;
    
    // Final contribution for left and right channels
    let leftContribution = totalAmplitude * hrtf.x;
    let rightContribution = totalAmplitude * hrtf.y;

    // Store result: Left, Right, Frequency (placeholder), Time
    spatialIR[hitIndex] = vec4f(
        leftContribution,
        rightContribution,
        1000.0, // Frequency is now a mix, 1k is a placeholder
        hit.time
    );
}