// Ray Collection Compute Shader - Minimal Version
// Collects ray data at the listener position for impulse response generation

struct Ray {
    origin: vec4<f32>,          // xyz = position, w = padding
    direction: vec4<f32>,       // xyz = direction, w = padding
    energy_phase: vec4<f32>,    // x = energy, y = phase, zw = padding
    frequency_energy_low: vec4<f32>,  // 125, 250, 500, 1k Hz
    frequency_energy_high: vec4<f32>, // 2k, 4k, 8k, 16k Hz
    path_data: vec4<f32>,       // x = path_length, y = arrival_time, z = bounce_count, w = active
    material_data: vec4<f32>,   // x = last_material_id, yzw = padding
}

struct CollectionParams {
    listener_position: vec3<f32>,
    listener_radius: f32,
    sample_rate: f32,
    ir_length: f32,             // In seconds
    time_bin_size: f32,         // Size of each time bin in seconds
    max_bins: u32,              // Maximum number of time bins
    energy_threshold: f32,      // Minimum energy to collect
    padding: f32,               // Alignment
}

struct ImpulseBin {
    energy: f32,
    phase_real: f32,            // Real part of complex phase
    phase_imag: f32,            // Imaginary part of complex phase
    sample_count: u32,          // Number of rays in this bin
    frequency_energy_low: vec4<f32>,  // 125, 250, 500, 1k Hz energy
    frequency_energy_high: vec4<f32>, // 2k, 4k, 8k, 16k Hz energy
    padding: vec3<f32>,         // Alignment padding
}

// Use atomic integers for statistics instead of floats
struct Statistics {
    shader_running: atomic<u32>,     // Debug marker
    total_rays_processed: atomic<u32>,
    rays_collected: atomic<u32>,
    total_energy_x1000: atomic<u32>, // Energy * 1000 to store as integer
}

@group(0) @binding(0) var<storage, read> rays: array<Ray>;
@group(0) @binding(1) var<uniform> params: CollectionParams;
@group(0) @binding(2) var<storage, read_write> impulse_response: array<ImpulseBin>;
@group(0) @binding(3) var<storage, read_write> stats: Statistics;

// Constants
const PI: f32 = 3.14159265359;
const EPSILON: f32 = 1e-6;

// MINIMAL but functional ray collection compute shader with debugging
// Note: Workgroup size is now configurable and set at pipeline creation time
@compute @workgroup_size(64) // Default size, will be overridden by specialization constants
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.x;

    // Write debug marker to show shader is running (only first thread)
    if (ray_index == 0u) {
        atomicStore(&stats.shader_running, 999u);
    }

    // Check bounds
    if (ray_index >= arrayLength(&rays)) {
        return;
    }

    let ray = rays[ray_index];

    // Count total rays processed
    atomicAdd(&stats.total_rays_processed, 1u);

    // More lenient ray collection for debugging
    let ray_energy = ray.energy_phase.x;
    let ray_origin = ray.origin.xyz;
    let distance_to_listener = length(ray_origin - params.listener_position);

    // Collect rays with very lenient criteria for debugging
    let should_collect = (
        ray_energy > 0.0001 &&  // Very low energy threshold
        distance_to_listener <= params.listener_radius * 2.0  // Double the radius for debugging
    );

    if (should_collect) {
        // Calculate time bin based on arrival time, with fallback
        var arrival_time = ray.path_data.y; // arrival_time field
        if (arrival_time <= 0.0) {
            arrival_time = distance_to_listener / 343.0; // Calculate from distance if not set
        }

        let time_bin = u32(arrival_time / params.time_bin_size);

        // Check if time bin is valid
        if (time_bin < params.max_bins && time_bin < arrayLength(&impulse_response)) {
            // Non-atomic accumulation (single thread per bin assumption for now)
            impulse_response[time_bin].energy += ray_energy;
            impulse_response[time_bin].sample_count += 1u;

            // Accumulate frequency energy
            impulse_response[time_bin].frequency_energy_low += ray.frequency_energy_low;
            impulse_response[time_bin].frequency_energy_high += ray.frequency_energy_high;

            // Update statistics (convert energy to integer by multiplying by 1000)
            let energy_as_int = u32(ray_energy * 1000.0);
            atomicAdd(&stats.total_energy_x1000, energy_as_int);
            atomicAdd(&stats.rays_collected, 1u);
        }
    }
}

// Alternative compute shader for statistics normalization (run after main collection)
@compute @workgroup_size(1)
fn normalize_statistics(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x != 0u) {
        return;
    }

    let collected_rays = atomicLoad(&stats.rays_collected);

    if (collected_rays > 0u) {
        // Normalize impulse response bins
        for (var i = 0u; i < params.max_bins; i++) {
            if (impulse_response[i].sample_count > 0u) {
                let sample_count = f32(impulse_response[i].sample_count);

                // Normalize phase components if they exist
                if (abs(impulse_response[i].phase_real) > EPSILON || abs(impulse_response[i].phase_imag) > EPSILON) {
                    impulse_response[i].phase_real /= sample_count;
                    impulse_response[i].phase_imag /= sample_count;

                    // Calculate final phase
                    let final_phase = atan2(impulse_response[i].phase_imag, impulse_response[i].phase_real);
                    impulse_response[i].phase_real = cos(final_phase);
                    impulse_response[i].phase_imag = sin(final_phase);
                }
            }
        }
    }
}


