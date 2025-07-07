// Simple Ray Bouncing Compute Shader
// Just moves rays forward and bounces them off walls

struct Ray {
    origin: vec4<f32>,          // xyz = position, w = padding
    direction: vec4<f32>,       // xyz = direction, w = padding
    energy_phase: vec4<f32>,    // x = energy, y = phase, zw = padding
    frequency_energy_low: vec4<f32>,  // 125, 250, 500, 1k Hz
    frequency_energy_high: vec4<f32>, // 2k, 4k, 8k, 16k Hz
    path_data: vec4<f32>,       // x = path_length, y = arrival_time, z = bounce_count, w = active
    material_data: vec4<f32>,   // x = last_material_id, yzw = padding
}

struct RayBouncingParams {
    room_min: vec3<f32>,
    room_max: vec3<f32>,
    max_bounces: f32,
    min_energy: f32,
    speed_of_sound: f32,
    time_step: f32,
    air_absorption_low: vec4<f32>,   // 125, 250, 500, 1k Hz
    air_absorption_high: vec4<f32>,  // 2k, 4k, 8k, 16k Hz
}

@group(0) @binding(0) var<storage, read_write> rays: array<Ray>;
@group(0) @binding(1) var<uniform> params: RayBouncingParams;

// Simple ray bouncing - move ray forward and bounce off walls
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.x;
    
    // Only process the first ray
    if (ray_index != 0u) {
        return;
    }
    
    // Check if ray is active
    if (rays[ray_index].path_data.w < 0.5) {
        return; // Ray is inactive
    }
    
    // Get current ray data
    var origin = rays[ray_index].origin.xyz;
    var direction = rays[ray_index].direction.xyz;
    var energy = rays[ray_index].energy_phase.x;
    var path_length = rays[ray_index].path_data.x;
    var bounce_count = rays[ray_index].path_data.z;
    
    // Move ray forward by a larger step to travel across the room
    let step_size = 0.5; // Bigger steps to see movement
    let new_origin = origin + direction * step_size;
    path_length += step_size;
    
    // Check for wall collisions and bounce
    var bounced = false;
    var final_origin = new_origin;

    // Check X walls (left/right)
    if (new_origin.x <= params.room_min.x) {
        direction.x = abs(direction.x); // Force positive direction
        final_origin.x = params.room_min.x + 0.01; // Small offset from wall
        bounced = true;
    } else if (new_origin.x >= params.room_max.x) {
        direction.x = -abs(direction.x); // Force negative direction
        final_origin.x = params.room_max.x - 0.01; // Small offset from wall
        bounced = true;
    }

    // Check Y walls (floor/ceiling)
    if (new_origin.y <= params.room_min.y) {
        direction.y = abs(direction.y); // Force positive direction (up)
        final_origin.y = params.room_min.y + 0.01;
        bounced = true;
    } else if (new_origin.y >= params.room_max.y) {
        direction.y = -abs(direction.y); // Force negative direction (down)
        final_origin.y = params.room_max.y - 0.01;
        bounced = true;
    }

    // Check Z walls (front/back)
    if (new_origin.z <= params.room_min.z) {
        direction.z = abs(direction.z); // Force positive direction
        final_origin.z = params.room_min.z + 0.01;
        bounced = true;
    } else if (new_origin.z >= params.room_max.z) {
        direction.z = -abs(direction.z); // Force negative direction
        final_origin.z = params.room_max.z - 0.01;
        bounced = true;
    }

    // If bounced, increment bounce count and reduce energy
    if (bounced) {
        bounce_count += 1.0;
        energy *= 0.7; // Lose 30% energy per bounce (more noticeable)
    }

    // Apply air absorption (small energy loss per step)
    energy *= 0.995; // Slightly more air absorption
    
    // Check termination conditions
    if (bounce_count >= params.max_bounces || energy < params.min_energy) {
        rays[ray_index].path_data.w = 0.0; // Deactivate ray
        return;
    }

    // Update ray data (can't assign to .xyz directly in WGSL)
    rays[ray_index].origin = vec4<f32>(final_origin, 0.0);
    rays[ray_index].direction = vec4<f32>(normalize(direction), 0.0);
    rays[ray_index].energy_phase.x = energy;
    rays[ray_index].path_data.x = path_length;
    rays[ray_index].path_data.z = bounce_count;
}
