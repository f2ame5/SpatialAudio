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
    // Use atomic integers for thread-safe accumulation
    energy_atomic: atomic<u32>,          // Energy * 10000 as integer
    phase_real_atomic: atomic<i32>,      // Real part * 10000 as signed integer
    phase_imag_atomic: atomic<i32>,      // Imaginary part * 10000 as signed integer
    sample_count: atomic<u32>,           // Number of rays in this bin
    frequency_energy_low: vec4<f32>,     // 125, 250, 500, 1k Hz energy
    frequency_energy_high: vec4<f32>,    // 2k, 4k, 8k, 16k Hz energy
    padding: vec3<f32>,                  // Alignment padding
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
            // Get ray phase information
            let ray_phase = ray.energy_phase.y; // Phase in radians

            // Calculate complex phase components weighted by energy
            let phase_real_component = cos(ray_phase) * ray_energy;
            let phase_imag_component = sin(ray_phase) * ray_energy;

            // Convert to integers for atomic operations (multiply by 10000 for precision)
            let energy_int = u32(ray_energy * 10000.0);
            let phase_real_int = i32(phase_real_component * 10000.0);
            let phase_imag_int = i32(phase_imag_component * 10000.0);

            // Atomically accumulate values to handle multiple threads writing to same bin
            atomicAdd(&impulse_response[time_bin].energy_atomic, energy_int);
            atomicAdd(&impulse_response[time_bin].phase_real_atomic, phase_real_int);
            atomicAdd(&impulse_response[time_bin].phase_imag_atomic, phase_imag_int);
            atomicAdd(&impulse_response[time_bin].sample_count, 1u);

            // Accumulate frequency energy (non-atomic for now - could be improved)
            impulse_response[time_bin].frequency_energy_low += ray.frequency_energy_low;
            impulse_response[time_bin].frequency_energy_high += ray.frequency_energy_high;

            // Update statistics
            atomicAdd(&stats.total_energy_x1000, energy_int);
            atomicAdd(&stats.rays_collected, 1u);
        }
    }
}

// Normalization compute shader to convert atomic values back to floats (run after main collection)
@compute @workgroup_size(64)
fn normalize_impulse_response(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bin_index = global_id.x;

    if (bin_index >= params.max_bins || bin_index >= arrayLength(&impulse_response)) {
        return;
    }

    let sample_count = atomicLoad(&impulse_response[bin_index].sample_count);

    if (sample_count > 0u) {
        // Convert atomic values back to floats
        let energy_int = atomicLoad(&impulse_response[bin_index].energy_atomic);
        let phase_real_int = atomicLoad(&impulse_response[bin_index].phase_real_atomic);
        let phase_imag_int = atomicLoad(&impulse_response[bin_index].phase_imag_atomic);

        // Convert back to float values (divide by 10000)
        let energy = f32(energy_int) / 10000.0;
        let phase_real = f32(phase_real_int) / 10000.0;
        let phase_imag = f32(phase_imag_int) / 10000.0;

        // Normalize by sample count if multiple rays hit this bin
        let sample_count_f = f32(sample_count);
        let normalized_energy = energy / sample_count_f;
        let normalized_phase_real = phase_real / sample_count_f;
        let normalized_phase_imag = phase_imag / sample_count_f;

        // Store normalized values in the frequency energy fields for readback
        // We'll use the frequency_energy_low vec4 to store our normalized values
        // This avoids the bitcast issue while providing a clean readback interface
        impulse_response[bin_index].frequency_energy_low.x = normalized_energy;         // Position 0: energy
        impulse_response[bin_index].frequency_energy_low.y = normalized_phase_real;     // Position 1: phase_real
        impulse_response[bin_index].frequency_energy_low.z = normalized_phase_imag;     // Position 2: phase_imag
        impulse_response[bin_index].frequency_energy_low.w = f32(sample_count);         // Position 3: sample_count as float
    }
}


