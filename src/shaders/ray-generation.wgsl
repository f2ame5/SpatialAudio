// Simple Ray Generation Compute Shader
// Generates a single acoustic ray from a sound source

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

// Simple ray generation - just create one ray going in a fixed direction
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.x;

    // Only generate the first ray (ray_index 0)
    if (ray_index != 0u) {
        return;
    }

    // Check array bounds
    if (ray_index >= arrayLength(&rays)) {
        return;
    }

    // Simple fixed direction for testing - ray goes forward and slightly down
    let direction = normalize(vec3<f32>(1.0, -0.3, 0.5));

    // Ray starts at the source position
    let origin = params.source_position;

    // Initialize the ray with simple values
    rays[ray_index].origin = vec4<f32>(origin, 0.0);
    rays[ray_index].direction = vec4<f32>(direction, 0.0);
    rays[ray_index].energy_phase = vec4<f32>(params.initial_energy, 0.0, 0.0, 0.0);

    // Set frequency energy to equal values
    rays[ray_index].frequency_energy_low = vec4<f32>(0.125, 0.125, 0.125, 0.125);
    rays[ray_index].frequency_energy_high = vec4<f32>(0.125, 0.125, 0.125, 0.125);

    // Initialize path data - ray is active
    rays[ray_index].path_data = vec4<f32>(
        0.0,  // path_length (starts at 0)
        0.0,  // arrival_time (starts at 0)
        0.0,  // bounce_count (starts at 0)
        1.0   // active (1.0 = active)
    );

    // Initialize material data
    rays[ray_index].material_data = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
}
