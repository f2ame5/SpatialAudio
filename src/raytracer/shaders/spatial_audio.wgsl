struct ListenerData {
    position: vec3f,
    forward: vec3f,
    up: vec3f,
    right: vec3f,
}

// Add directional absorption struct
struct DirectionalAbsorption {
    front: vec3f, // [low, mid, high]
    side: vec3f,
    back: vec3f,
};

struct WallMaterial {
    directional: DirectionalAbsorption,
    scatteringLow: f32,
    scatteringMid: f32,
    scatteringHigh: f32,
    roughness: f32,
    phaseShift: f32,
    phaseRandomization: f32
}

struct RayHit {
    position: vec3f,
    time: f32,
    normal: vec3f,
    energy: f32,
    incomingDirection: vec3f,
    // Eight frequency bands
    energy63: f32,
    hrtfIndex: u32,
    energy125: f32,
    energy250: f32,
    energy500: f32,
    energy1k: f32,
    energy2k: f32,
    energy4k: f32,
    energy8k: f32,
    // Wave properties
    phase: f32,
    frequency: f32,
    dopplerShift: f32,
    _padding: f32  // Maintain alignment
}

struct RoomMaterials {
    walls: WallMaterial,
    ceiling: WallMaterial,
    floor: WallMaterial
}

struct FrequencyBands {
    band63: f32,    // Sub-bass
    band125: f32,   // Bass
    band250: f32,   // Low-mids
    band500: f32,   // Mids
    band1k: f32,    // Upper-mids
    band2k: f32,    // Presence
    band4k: f32,    // Brilliance
    band8k: f32     // Air
}

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
}

struct SpatialAudioParams {
    speedOfSound: f32,
    maxDistance: f32,
    minDistance: f32,
    temperature: f32,    // For air absorption calculation
    humidity: f32,       // For air absorption calculation
    sourcePower: f32,    // Source power in dB
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
    _padding5: f32      // Total size: 48 bytes (12 floats)
}

struct WaveProperties {
    phase: f32,
    frequency: f32,
    dopplerShift: f32,
    _padding: f32
}

@group(0) @binding(0) var<uniform> listener: ListenerData;
@group(0) @binding(1) var<storage, read> rayHits: array<RayHit>;
@group(0) @binding(2) var<storage, read_write> spatialIR: array<vec4f>;
@group(0) @binding(3) var<uniform> params: SpatialAudioParams;
@group(0) @binding(4) var<uniform> acoustics: RoomAcoustics;
@group(0) @binding(5) var<storage, read> waveProperties: array<WaveProperties>;
@group(0) @binding(6) var<uniform> roomMaterials: RoomMaterials;
@group(0) @binding(7) var<storage, read> hrtfCoefficients: array<f32>;

// Update the constants and helper functions
const SPEED_OF_SOUND = 343.0;
const AIR_DENSITY = 1.225;  // kg/m³ at room temperature
const REFERENCE_PRESSURE = 2e-5;  // 20 micropascals (threshold of hearing)

// Helper function to calculate directional attenuation
// Added directional absorption logic
fn getAbsorptionDirection(dotValue: f32) -> Direction {
    let angle = acos(dotValue);
    if (angle < 0.25 * PI) { return Front; }
    if (angle < 0.75 * PI) { return Side; }
    return Back;
}

fn applyDirectionalAbsorption(hit: RayHit, material: WallMaterial) -> vec3f {
    let dot = dot(hit.incomingDirection, hit.normal);
    let direction = getAbsorptionDirection(dot);
        direction == Front ? material.directional.front :
        direction == Side  ? material.directional.side  :
        material.directional.back;
    // or alternatively using ternary:
    // return direction == Front ? material.directional.front :
    //        direction == Side ? material.directional.side :
    //        material.directional.back;
}
// Removed misplaced code
//     let forward = normalize(listenerForward);
    let cosAngle = dot(dir, forward);
    return clamp((cosAngle + 1.0) * 0.5, 0.0, 1.0);
}

// HRTF approximation (simplified)
// Replace entire calculateHRTF function with new getHRTFCoefficient
fn getHRTFCoefficient(index: u32) -> vec2f {
    return vec2f(
        hrtfCoefficients[index * 2],
        hrtfCoefficients[index * 2 + 1]
    );
}

// Calculate frequency-dependent scattering based on acoustic research
fn calculateScattering(direction: vec3f, normal: vec3f) -> FrequencyBands {
    let incidentAngle = acos(dot(normalize(direction), normalize(normal)));

    // Scattering increases with frequency according to acoustic theory
    // Based on surface roughness relative to wavelength
    return FrequencyBands(
        mix(0.9, cos(incidentAngle), acoustics.scattering_63),   // More diffuse at low freq
        mix(0.8, cos(incidentAngle), acoustics.scattering_125),
        mix(0.7, cos(incidentAngle), acoustics.scattering_250),
        mix(0.6, cos(incidentAngle), acoustics.scattering_500),
        mix(0.5, cos(incidentAngle), acoustics.scattering_1k),
        mix(0.4, cos(incidentAngle), acoustics.scattering_2k),
        mix(0.3, cos(incidentAngle), acoustics.scattering_4k),
        mix(0.2, cos(incidentAngle), acoustics.scattering_8k)    // More specular at high freq
    );
}

// Calculate air absorption based on ISO 9613-1
fn calculateAirAbsorption(distance: f32) -> FrequencyBands {
    let d = max(distance, 0.001);
    return FrequencyBands(
        exp(-acoustics.absorption_63 * d),
        exp(-acoustics.absorption_125 * d),
        exp(-acoustics.absorption_250 * d),
        exp(-acoustics.absorption_500 * d),
        exp(-acoustics.absorption_1k * d),
        exp(-acoustics.absorption_2k * d),
        exp(-acoustics.absorption_4k * d),
        exp(-acoustics.absorption_8k * d)
    );
}

// Calculate energy decay with Sabine's formula for each band
fn calculateEnergyDecay(time: f32, distance: f32, direction: vec3f, normal: vec3f) -> FrequencyBands {
    let t = max(time, 0.0);

    // Basic decay based on RT60
    let timeDecay = FrequencyBands(
        exp(-3.0 * t / acoustics.rt60_63),
        exp(-3.0 * t / acoustics.rt60_125),
        exp(-3.0 * t / acoustics.rt60_250),
        exp(-3.0 * t / acoustics.rt60_500),
        exp(-3.0 * t / acoustics.rt60_1k),
        exp(-3.0 * t / acoustics.rt60_2k),
        exp(-3.0 * t / acoustics.rt60_4k),
        exp(-3.0 * t / acoustics.rt60_8k)
    );

    // Combine with scattering
    let scattering = calculateScattering(direction, normal);

    return FrequencyBands(
        timeDecay.band63 * scattering.band63,
        timeDecay.band125 * scattering.band125,
        timeDecay.band250 * scattering.band250,
        timeDecay.band500 * scattering.band500,
        timeDecay.band1k * scattering.band1k,
        timeDecay.band2k * scattering.band2k,
        timeDecay.band4k * scattering.band4k,
        timeDecay.band8k * scattering.band8k
    );
}

// Improved early reflection detection
fn isEarlyReflection(time: f32, distance: f32) -> bool {
    // Consider both time and order of reflection
    let directSound = distance / SPEED_OF_SOUND;
    let normalizedTime = (time - directSound) / acoustics.earlyReflectionTime;
    return normalizedTime < 1.0;
}

// Add helper function for dB to linear conversion
fn dbToLinear(db: f32) -> f32 {
    return pow(10.0, db / 20.0);
}

// Helper function to calculate Doppler shift
fn calculateDopplerShift(rayVelocity: vec3f, rayDirection: vec3f, speedOfSound: f32) -> f32 {
    let relativeVelocity = dot(rayVelocity, rayDirection);
    return speedOfSound / (speedOfSound - relativeVelocity);
}

// Helper function to calculate wave contribution
fn calculateWaveContribution(
    time: f32,
    phase: f32,
    frequency: f32,
    dopplerShift: f32,
    amplitude: f32,
    distance: f32
) -> f32 {
    // Validate inputs
    let validFreq = max(frequency, 20.0);  // Ensure frequency is at least 20Hz
    let validAmplitude = max(amplitude, 0.0);  // Ensure non-negative amplitude
    let validDistance = max(distance, 0.001);  // Prevent division by zero
    
    let shiftedFreq = validFreq * max(dopplerShift, 0.1);  // Limit minimum doppler shift
    let wavelength = SPEED_OF_SOUND / shiftedFreq;
    let distancePhase = 2.0 * 3.14159 * validDistance / wavelength;
    let totalPhase = phase + distancePhase;
    
    // Apply window function (Hann window)
    let windowPos = clamp(time / (validDistance / SPEED_OF_SOUND), 0.0, 1.0);
    let window = 0.5 * (1.0 - cos(2.0 * 3.14159 * windowPos));
    
    return validAmplitude * window * sin(totalPhase);
}

// Function to apply material absorption based on surface type
fn applyMaterialAbsorption(hit: RayHit, distance: f32, materials: RoomMaterials) -> f32 {
    // Determine which surface was hit based on normal direction
    // This is a simplified approach - in a real implementation, you might pass the surface type directly
    let upNormal = vec3f(0.0, 1.0, 0.0);
    let downNormal = vec3f(0.0, -1.0, 0.0);
    let floorCeilingThreshold = 0.9; // Threshold for determining floor/ceiling
    
    var material: WallMaterial;
    
    // Determine if hit is on floor or ceiling
    if (dot(hit.normal, upNormal) > floorCeilingThreshold) {
        // Hit on floor
        material = materials.floor;
    } else if (dot(hit.normal, downNormal) > floorCeilingThreshold) {
        // Hit on ceiling
        material = materials.ceiling;
    } else {
        // Hit on wall
        material = materials.walls;
    }
    
    // Calculate combined absorption factor
    // This is a simplified model - you might want to use frequency-specific absorption
    let absorptionFactor = (material.absorptionLow + material.absorptionMid + material.absorptionHigh) / 1.0;
    
    // Return the transmission factor (1 - absorption)
    return 1.0 - absorptionFactor;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let hitIndex = global_id.x;
    if (hitIndex >= arrayLength(&rayHits)) {
        return;
    }

    let hit = rayHits[hitIndex];
    let toListener = normalize(listener.position - hit.position);
    
    // Calculate basic spatial properties
    let distance = length(listener.position - hit.position);
    let dirAttenuation = calculateDirectionalAttenuation(toListener, listener.forward);
    let hrtfCoeffs = getHRTFCoefficient(hit.hrtfIndex);
    let hrtf = hrtfCoeffs;

    // Calculate wave properties
    let waveProps = waveProperties[hitIndex];
    let amplitude = sqrt(max(hit.energy, 0.0));  // Ensure non-negative energy
    
    // Calculate frequency-dependent properties
    let airAbsorption = calculateAirAbsorption(distance);
    let energyDecay = calculateEnergyDecay(hit.time, distance, toListener, hit.normal);
    
    // Apply material absorption
    // Calculate directional absorption
    let directionalAbsorption = applyDirectionalAbsorption(hit, roomMaterials.walls.directional);
    amplitude *= directionalAbsorption.x; // Low frequency absorption
    amplitude *= directionalAbsorption.y; // Mid frequency absorption
    amplitude *= directionalAbsorption.z; // High frequency absorption
    amplitude = amplitude * materialTransmission;
    
    // Calculate wave contribution with interference
    let contribution = calculateWaveContribution(
        hit.time,
        waveProps.phase,
        waveProps.frequency,
        waveProps.dopplerShift,
        amplitude,
        distance
    );

    // Apply spatial and frequency-dependent effects
    // Ensure we have non-zero contributions
    let leftContribution = max(contribution * hrtf.x * dirAttenuation * 0.5, 0.0001);
    let rightContribution = max(contribution * hrtf.y * dirAttenuation * 0.5, 0.0001);

    // Store result with frequency band information
    spatialIR[hitIndex] = vec4f(
        leftContribution,
        rightContribution,
        waveProps.frequency,  // Store frequency for later processing
        hit.time             // Store time for sorting
    );
}