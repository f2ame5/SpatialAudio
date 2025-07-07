// Ray Generation Compute Shader
// Generates acoustic rays from a spherical sound source using uniform spherical distribution

struct Ray {
    origin: vec4<f32>,          // xyz = position, w = padding
    direction: vec4<f32>,       // xyz = direction, w = padding
    energy_phase: vec4<f32>,    // x = energy, y = phase, zw = padding
    frequency_energy_low: vec4<f32>,  // 125, 250, 500, 1k Hz
    frequency_energy_high: vec4<f32>, // 2k, 4k, 8k, 16k Hz
    path_data: vec4<f32>,       // x = path_length, y = arrival_time, z = bounce_count, w = active
    material_data: vec4<f32>,   // x = last_material_id, yzw = padding
}

struct RayGenerationParams {
    source_position: vec3<f32>,
    source_radius: f32,
    ray_count: u32,
    initial_energy: f32,
    time: f32,
    seed: u32,
    frequency_weights_low: vec4<f32>,   // 125, 250, 500, 1k Hz weights
    frequency_weights_high: vec4<f32>,  // 2k, 4k, 8k, 16k Hz weights
    distribution_type: u32,     // 0 = uniform sphere, 1 = hemisphere, 2 = cone
    cone_angle: f32,           // For directional sources (radians)
    padding: vec2<f32>,        // Alignment padding
}

@group(0) @binding(0) var<storage, read_write> rays: array<Ray>;
@group(0) @binding(1) var<uniform> params: RayGenerationParams;

// Constants
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const GOLDEN_RATIO: f32 = 1.61803398875;

// High-quality pseudo-random number generator
// Based on PCG (Permuted Congruential Generator)
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate random float in [0, 1) from seed and index
fn random_float(seed: u32, index: u32) -> f32 {
    return f32(pcg_hash(seed + index)) / f32(0xFFFFFFFFu);
}

// Generate random float in [min, max)
fn random_range(seed: u32, index: u32, min_val: f32, max_val: f32) -> f32 {
    return min_val + random_float(seed, index) * (max_val - min_val);
}

// Generate uniform random point on unit sphere using Marsaglia method
fn uniform_sphere_point(seed: u32, ray_index: u32) -> vec3<f32> {
    // Use different hash offsets for x, y components to avoid correlation
    let u1 = random_float(seed, ray_index * 2u);
    let u2 = random_float(seed, ray_index * 2u + 1u);

    // Marsaglia method for uniform sphere sampling
    let z = 1.0 - 2.0 * u1;  // z in [-1, 1]
    let r = sqrt(1.0 - z * z);  // radius in xy plane
    let phi = TWO_PI * u2;  // azimuthal angle

    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// Generate point on upper hemisphere (z >= 0)
fn hemisphere_point(seed: u32, ray_index: u32) -> vec3<f32> {
    let u1 = random_float(seed, ray_index * 2u);
    let u2 = random_float(seed, ray_index * 2u + 1u);

    let z = u1;  // z in [0, 1]
    let r = sqrt(1.0 - z * z);
    let phi = TWO_PI * u2;

    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// Generate point within cone (for directional sources)
fn cone_point(seed: u32, ray_index: u32, cone_angle: f32) -> vec3<f32> {
    let u1 = random_float(seed, ray_index * 2u);
    let u2 = random_float(seed, ray_index * 2u + 1u);

    let cos_cone = cos(cone_angle);
    let z = cos_cone + u1 * (1.0 - cos_cone);  // z in [cos(cone_angle), 1]
    let r = sqrt(1.0 - z * z);
    let phi = TWO_PI * u2;

    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// Fibonacci spiral distribution (more uniform than random)
fn fibonacci_sphere_point(ray_index: u32, total_rays: u32) -> vec3<f32> {
    let i = f32(ray_index);
    let n = f32(total_rays);

    let theta = TWO_PI * i / GOLDEN_RATIO;  // Golden angle
    let y = 1.0 - (i / (n - 1.0)) * 2.0;   // y goes from 1 to -1
    let radius = sqrt(1.0 - y * y);

    let x = cos(theta) * radius;
    let z = sin(theta) * radius;

    return vec3<f32>(x, y, z);
}

// Initialize frequency energy distribution
fn initialize_frequency_energy(seed: u32, ray_index: u32) -> array<f32, 8> {
    var freq_energy: array<f32, 8>;

    // Get frequency weights from vec4 parameters
    let weights_low = params.frequency_weights_low;
    let weights_high = params.frequency_weights_high;

    // Low frequency bands (125, 250, 500, 1k Hz)
    freq_energy[0] = weights_low.x * params.initial_energy;
    freq_energy[1] = weights_low.y * params.initial_energy;
    freq_energy[2] = weights_low.z * params.initial_energy;
    freq_energy[3] = weights_low.w * params.initial_energy;

    // High frequency bands (2k, 4k, 8k, 16k Hz)
    freq_energy[4] = weights_high.x * params.initial_energy;
    freq_energy[5] = weights_high.y * params.initial_energy;
    freq_energy[6] = weights_high.z * params.initial_energy;
    freq_energy[7] = weights_high.w * params.initial_energy;

    // Add small random variation (±5%) for more realistic behavior
    for (var i = 0u; i < 8u; i++) {
        let variation = random_range(seed, ray_index * 8u + i, 0.95, 1.05);
        freq_energy[i] *= variation;
    }

    return freq_energy;
}

// Generate random phase offset
fn generate_phase(seed: u32, ray_index: u32) -> f32 {
    return random_range(seed, ray_index * 16u, 0.0, TWO_PI);
}

// Main ray generation compute shader
// Note: Workgroup size is now configurable and set at pipeline creation time
@compute @workgroup_size(64) // Default size, will be overridden by specialization constants
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.x;

    // --- DEBUG: Unconditionally write to the first ray ---
    if (ray_index == 0u) {
        rays[0].energy_phase.x = 777.0; // Unique energy value
        rays[0].path_data.w = 1.0;      // Force active
    }

    // Check array bounds first
    if (ray_index >= arrayLength(&rays)) {
        return;
    }

    // Early exit if beyond ray count (DEBUG: Hardcoded value)
    if (ray_index >= 2048u) {
        return;
    }

    // Generate ray direction based on distribution type
    var direction: vec3<f32>;
    switch (params.distribution_type) {
        case 0u: {
            // Uniform sphere distribution
            direction = uniform_sphere_point(params.seed, ray_index);
        }
        case 1u: {
            // Hemisphere distribution (upper half)
            direction = hemisphere_point(params.seed, ray_index);
        }
        case 2u: {
            // Cone distribution (directional source)
            direction = cone_point(params.seed, ray_index, params.cone_angle);
        }
        default: {
            // Fibonacci spiral (deterministic, very uniform)
            direction = fibonacci_sphere_point(ray_index, params.ray_count);
        }
    }

    // Ensure direction is normalized (should already be, but safety check)
    direction = normalize(direction);

    // Calculate ray origin on source sphere surface
    let surface_offset = direction * params.source_radius;
    let origin = params.source_position + surface_offset;

    // Initialize frequency energy distribution
    let freq_energy = initialize_frequency_energy(params.seed, ray_index);

    // Generate random phase for wave interference
    let phase = generate_phase(params.seed, ray_index);

    // Calculate total energy as sum of frequency energies
    let total_energy = freq_energy[0] + freq_energy[1] + freq_energy[2] + freq_energy[3] +
                      freq_energy[4] + freq_energy[5] + freq_energy[6] + freq_energy[7];

    // Initialize the ray structure
    rays[ray_index].origin = vec4<f32>(origin, 0.0);
    rays[ray_index].direction = vec4<f32>(direction, 0.0);
    rays[ray_index].energy_phase = vec4<f32>(total_energy, phase, 0.0, 0.0);

    // Set frequency-dependent energy (split into two vec4s for alignment)
    rays[ray_index].frequency_energy_low = vec4<f32>(
        freq_energy[0],  // 125 Hz
        freq_energy[1],  // 250 Hz
        freq_energy[2],  // 500 Hz
        freq_energy[3]   // 1 kHz
    );
    rays[ray_index].frequency_energy_high = vec4<f32>(
        freq_energy[4],  // 2 kHz
        freq_energy[5],  // 4 kHz
        freq_energy[6],  // 8 kHz
        freq_energy[7]   // 16 kHz
    );

    // Initialize path tracking data
    rays[ray_index].path_data = vec4<f32>(
        0.0,  // path_length (starts at 0)
        0.0,  // arrival_time (starts at 0)
        0.0,  // bounce_count (starts at 0)
        1.0   // active (1 = active, 0 = terminated)
    );

    // Debug: Force first few rays to have high energy for testing
    if (ray_index < 10u) {
        rays[ray_index].energy_phase.x = 10.0; // High energy for debugging
        rays[ray_index].frequency_energy_low = vec4<f32>(1.0, 1.0, 1.0, 1.0);
        rays[ray_index].frequency_energy_high = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    // Initialize material data
    rays[ray_index].material_data = vec4<f32>(
        -1.0,  // last_material_id (-1 = no previous material)
        0.0,   // padding
        0.0,   // padding
        0.0    // padding
    );
}
