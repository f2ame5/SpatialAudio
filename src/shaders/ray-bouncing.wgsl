// Ray Bouncing Compute Shader
// Handles ray-surface intersections, reflections, and energy absorption

struct Ray {
    origin: vec4<f32>,          // xyz = position, w = padding
    direction: vec4<f32>,       // xyz = direction, w = padding
    energy_phase: vec4<f32>,    // x = energy, y = phase, zw = padding
    frequency_energy_low: vec4<f32>,  // 125, 250, 500, 1k Hz
    frequency_energy_high: vec4<f32>, // 2k, 4k, 8k, 16k Hz
    path_data: vec4<f32>,       // x = path_length, y = arrival_time, z = bounce_count, w = active
    material_data: vec4<f32>,   // x = last_material_id, yzw = padding
}

struct Material {
    absorption_low: vec4<f32>,      // 125,250,500,1k Hz absorption
    absorption_high: vec4<f32>,     // 2k,4k,8k,16k Hz absorption
    scattering_low: vec4<f32>,      // 125,250,500,1k Hz scattering
    scattering_high: vec4<f32>,     // 2k,4k,8k,16k Hz scattering
    properties: vec4<f32>,          // x=impedance, y=roughness, zw=padding
}

struct RayBouncingParams {
    room_min: vec3<f32>,           // Room bounding box minimum
    room_max: vec3<f32>,           // Room bounding box maximum
    max_bounces: u32,              // Maximum allowed bounces
    min_energy: f32,               // Energy threshold for termination
    speed_of_sound: f32,           // Speed of sound (m/s)
    time_step: f32,                // Simulation time step
    air_absorption_low: vec4<f32>, // 125, 250, 500, 1k Hz air absorption
    air_absorption_high: vec4<f32>, // 2k, 4k, 8k, 16k Hz air absorption
    surface_materials: vec4<u32>,  // Material IDs for [+X, -X, +Y, -Y] faces
    surface_materials_zw: vec2<u32>, // Material IDs for [+Z, -Z] faces
    padding: vec2<f32>,            // Alignment padding
}

@group(0) @binding(0) var<storage, read_write> rays: array<Ray>;
@group(0) @binding(1) var<uniform> params: RayBouncingParams;
@group(0) @binding(2) var<storage, read> materials: array<Material>;

// Constants
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const EPSILON: f32 = 1e-6;

// Surface IDs for room faces
const FACE_POS_X: u32 = 0u; // Right wall
const FACE_NEG_X: u32 = 1u; // Left wall
const FACE_POS_Y: u32 = 2u; // Ceiling
const FACE_NEG_Y: u32 = 3u; // Floor
const FACE_POS_Z: u32 = 4u; // Front wall
const FACE_NEG_Z: u32 = 5u; // Back wall

// Intersection result
struct IntersectionResult {
    hit: bool,
    distance: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
    surface_id: u32,
}

// High-quality pseudo-random number generator
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn random_float(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / f32(0xFFFFFFFFu);
}

// Ray-box intersection using a more robust slab method with proper zero handling
fn intersect_room(ray_origin: vec3<f32>, ray_direction: vec3<f32>) -> IntersectionResult {
    var result: IntersectionResult;
    result.hit = false;
    result.distance = 1e30;

    // Handle division by zero for ray direction components
    var inv_dir: vec3<f32>;
    var t_min_vec: vec3<f32>;
    var t_max_vec: vec3<f32>;

    // X component
    if (abs(ray_direction.x) < EPSILON) {
        if (ray_origin.x < params.room_min.x || ray_origin.x > params.room_max.x) {
            return result; // Ray is parallel and outside the slab
        }
        t_min_vec.x = -1e30;
        t_max_vec.x = 1e30;
    } else {
        inv_dir.x = 1.0 / ray_direction.x;
        t_min_vec.x = (params.room_min.x - ray_origin.x) * inv_dir.x;
        t_max_vec.x = (params.room_max.x - ray_origin.x) * inv_dir.x;
    }

    // Y component
    if (abs(ray_direction.y) < EPSILON) {
        if (ray_origin.y < params.room_min.y || ray_origin.y > params.room_max.y) {
            return result; // Ray is parallel and outside the slab
        }
        t_min_vec.y = -1e30;
        t_max_vec.y = 1e30;
    } else {
        inv_dir.y = 1.0 / ray_direction.y;
        t_min_vec.y = (params.room_min.y - ray_origin.y) * inv_dir.y;
        t_max_vec.y = (params.room_max.y - ray_origin.y) * inv_dir.y;
    }

    // Z component
    if (abs(ray_direction.z) < EPSILON) {
        if (ray_origin.z < params.room_min.z || ray_origin.z > params.room_max.z) {
            return result; // Ray is parallel and outside the slab
        }
        t_min_vec.z = -1e30;
        t_max_vec.z = 1e30;
    } else {
        inv_dir.z = 1.0 / ray_direction.z;
        t_min_vec.z = (params.room_min.z - ray_origin.z) * inv_dir.z;
        t_max_vec.z = (params.room_max.z - ray_origin.z) * inv_dir.z;
    }

    let t_smaller = min(t_min_vec, t_max_vec);
    let t_larger = max(t_min_vec, t_max_vec);

    var t_near = max(t_smaller.x, max(t_smaller.y, t_smaller.z));
    var t_far = min(t_larger.x, min(t_larger.y, t_larger.z));

    if (t_near >= t_far || t_far < EPSILON) {
        return result; // No intersection or intersection is behind the ray
    }

    var hit_distance = t_near;
    if (hit_distance < EPSILON) {
        // If the origin is inside the box, we want the exit point
        hit_distance = t_far;
    }

    result.hit = true;
    result.distance = hit_distance;
    result.point = ray_origin + ray_direction * hit_distance;

    // Determine which face was hit to find the normal and surface ID
    let hit_point_centered = result.point - (params.room_min + params.room_max) * 0.5;
    let room_size = params.room_max - params.room_min;
    let abs_hit_point = abs(hit_point_centered);

    if (abs_hit_point.x > room_size.x * 0.5 - EPSILON) {
        result.normal = vec3<f32>(-sign(ray_direction.x), 0.0, 0.0);
        result.surface_id = select(FACE_POS_X, FACE_NEG_X, ray_direction.x > 0.0);
    } else if (abs_hit_point.y > room_size.y * 0.5 - EPSILON) {
        result.normal = vec3<f32>(0.0, -sign(ray_direction.y), 0.0);
        result.surface_id = select(FACE_POS_Y, FACE_NEG_Y, ray_direction.y > 0.0);
    } else {
        result.normal = vec3<f32>(0.0, 0.0, -sign(ray_direction.z));
        result.surface_id = select(FACE_POS_Z, FACE_NEG_Z, ray_direction.z > 0.0);
    }

    return result;
}

// Calculate specular reflection direction
fn reflect_direction(incident: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return incident - 2.0 * dot(incident, normal) * normal;
}

// Calculate diffuse reflection direction (cosine-weighted hemisphere)
fn diffuse_reflection(normal: vec3<f32>, seed: u32) -> vec3<f32> {
    // Generate random direction in hemisphere
    let u1 = random_float(seed);
    let u2 = random_float(seed + 1u);

    let cos_theta = sqrt(u1);
    let sin_theta = sqrt(1.0 - u1);
    let phi = TWO_PI * u2;

    // Local coordinates (normal is Z-axis)
    let local_dir = vec3<f32>(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );

    // Transform to world coordinates
    // Create orthonormal basis from normal
    var tangent: vec3<f32>;
    if (abs(normal.x) > 0.9) {
        tangent = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        tangent = vec3<f32>(1.0, 0.0, 0.0);
    }

    tangent = normalize(cross(normal, tangent));
    let bitangent = cross(normal, tangent);

    return local_dir.x * tangent + local_dir.y * bitangent + local_dir.z * normal;
}

// Apply frequency-dependent absorption and scattering
fn apply_material_interaction(
    ray_index: u32,
    material_id: u32,
    incident_dir: vec3<f32>,
    normal: vec3<f32>
) -> vec3<f32> {
    let material = materials[material_id];

    // Get current frequency energies
    var freq_energy_low = rays[ray_index].frequency_energy_low;
    var freq_energy_high = rays[ray_index].frequency_energy_high;

    // Apply absorption (energy loss)
    freq_energy_low.x *= (1.0 - material.absorption_low.x); // 125 Hz
    freq_energy_low.y *= (1.0 - material.absorption_low.y); // 250 Hz
    freq_energy_low.z *= (1.0 - material.absorption_low.z); // 500 Hz
    freq_energy_low.w *= (1.0 - material.absorption_low.w); // 1 kHz

    freq_energy_high.x *= (1.0 - material.absorption_high.x); // 2 kHz
    freq_energy_high.y *= (1.0 - material.absorption_high.y); // 4 kHz
    freq_energy_high.z *= (1.0 - material.absorption_high.z); // 8 kHz
    freq_energy_high.w *= (1.0 - material.absorption_high.w); // 16 kHz

    // Update ray energy
    rays[ray_index].frequency_energy_low = freq_energy_low;
    rays[ray_index].frequency_energy_high = freq_energy_high;

    // Calculate total energy for reflection type decision
    let total_energy = freq_energy_low.x + freq_energy_low.y + freq_energy_low.z + freq_energy_low.w +
                      freq_energy_high.x + freq_energy_high.y + freq_energy_high.z + freq_energy_high.w;

    // Update total energy in ray
    rays[ray_index].energy_phase.x = total_energy / 8.0;

    // Determine reflection type based on material properties
    let avg_scattering = (material.scattering_low.x + material.scattering_low.y +
                         material.scattering_low.z + material.scattering_low.w +
                         material.scattering_high.x + material.scattering_high.y +
                         material.scattering_high.z + material.scattering_high.w) / 8.0;

    let roughness = material.properties.y;
    let scattering_probability = avg_scattering * (1.0 + roughness);

    // Use ray index as seed for deterministic but varied behavior
    let random_val = random_float(ray_index * 1000u + u32(material_id));

    if (random_val < scattering_probability) {
        // Diffuse reflection
        return diffuse_reflection(normal, ray_index * 2000u);
    } else {
        // Specular reflection with some roughness
        let perfect_reflection = reflect_direction(incident_dir, normal);

        if (roughness > 0.01) {
            // Add some randomness based on roughness
            let perturbation = diffuse_reflection(normal, ray_index * 3000u) * roughness * 0.1;
            return normalize(perfect_reflection + perturbation);
        } else {
            return perfect_reflection;
        }
    }
}

// Apply air absorption during ray travel
fn apply_air_absorption(ray_index: u32, distance: f32) {
    let absorption_factor = exp(-distance * 0.001); // Simple air absorption model

    // Apply frequency-dependent air absorption
    var freq_low = rays[ray_index].frequency_energy_low;
    var freq_high = rays[ray_index].frequency_energy_high;

    freq_low.x *= exp(-distance * params.air_absorption_low.x); // 125 Hz
    freq_low.y *= exp(-distance * params.air_absorption_low.y); // 250 Hz
    freq_low.z *= exp(-distance * params.air_absorption_low.z); // 500 Hz
    freq_low.w *= exp(-distance * params.air_absorption_low.w); // 1 kHz

    freq_high.x *= exp(-distance * params.air_absorption_high.x); // 2 kHz
    freq_high.y *= exp(-distance * params.air_absorption_high.y); // 4 kHz
    freq_high.z *= exp(-distance * params.air_absorption_high.z); // 8 kHz
    freq_high.w *= exp(-distance * params.air_absorption_high.w); // 16 kHz

    rays[ray_index].frequency_energy_low = freq_low;
    rays[ray_index].frequency_energy_high = freq_high;

    // Update total energy
    let total_energy = (freq_low.x + freq_low.y + freq_low.z + freq_low.w +
                       freq_high.x + freq_high.y + freq_high.z + freq_high.w) / 8.0;
    rays[ray_index].energy_phase.x = total_energy;
}

// Helper function to check ray termination conditions
fn should_terminate_ray(ray_index: u32) -> i32 {
    // Return codes: 0 = continue, -1 = max bounces, -2 = min energy, -3 = inactive

    // Skip inactive rays
    if (rays[ray_index].path_data.w < 0.5) {
        return -3;
    }

    // Check if ray has exceeded maximum bounces
    if (u32(rays[ray_index].path_data.z) >= params.max_bounces) {
        return -1;
    }

    // Check if ray energy is below threshold
    if (rays[ray_index].energy_phase.x < params.min_energy) {
        return -2;
    }

    return 0; // Continue processing
}

// Helper function to get material ID for surface
fn get_surface_material_id(surface_id: u32) -> u32 {
    if (surface_id < 4u) {
        return params.surface_materials[surface_id];
    } else if (surface_id == 4u) {
        return params.surface_materials_zw.x; // +Z face
    } else {
        return params.surface_materials_zw.y; // -Z face
    }
}

// Helper function to update ray path data
fn update_ray_path(ray_index: u32, distance: f32) {
    rays[ray_index].path_data.x += distance; // path_length
    rays[ray_index].path_data.y += distance / params.speed_of_sound; // arrival_time
}

// Helper function to update ray position and direction after bounce
fn update_ray_after_bounce(ray_index: u32, intersection: IntersectionResult, new_direction: vec3<f32>, material_id: u32) {
    // Update ray position and direction
    rays[ray_index].origin = vec4<f32>(intersection.point + intersection.normal * EPSILON, 0.0);
    rays[ray_index].direction = vec4<f32>(normalize(new_direction), 0.0);

    // Increment bounce count
    rays[ray_index].path_data.z += 1.0;

    // Update material history and store surface normal for energy calculation
    rays[ray_index].material_data.x = f32(material_id);
    rays[ray_index].material_data.y = intersection.normal.x;
    rays[ray_index].material_data.z = intersection.normal.y;
    rays[ray_index].material_data.w = intersection.normal.z;

    // Update phase (simplified - could be more sophisticated)
    let phase_shift = (intersection.distance / params.speed_of_sound) * TWO_PI * 1000.0; // 1kHz reference
    rays[ray_index].energy_phase.y += phase_shift;

    // Ensure phase stays in [0, 2π] range
    if (rays[ray_index].energy_phase.y > TWO_PI) {
        rays[ray_index].energy_phase.y -= TWO_PI;
    }
}

// Main ray bouncing compute shader - optimized for register efficiency
// Note: Workgroup size is now configurable and set at pipeline creation time
@compute @workgroup_size(64) // Default size, will be overridden by specialization constants
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.x;

    // Early bounds check
    if (ray_index >= arrayLength(&rays)) {
        return;
    }

    // Check termination conditions (moved to helper function)
    let termination_code = should_terminate_ray(ray_index);
    if (termination_code != 0) {
        rays[ray_index].path_data.w = 0.0; // Deactivate ray
        rays[ray_index].material_data.x = f32(termination_code); // Store termination reason
        return;
    }

    // Get ray properties (minimize local variables)
    let ray_origin = rays[ray_index].origin.xyz;
    let ray_direction = rays[ray_index].direction.xyz;

    // Find intersection with room
    let intersection = intersect_room(ray_origin, ray_direction);

    if (!intersection.hit) {
        // Ray escaped room - deactivate
        rays[ray_index].path_data.w = 0.0;
        rays[ray_index].material_data.x = -3.0; // Escaped room
        return;
    }

    // Apply air absorption during travel (helper function call)
    apply_air_absorption(ray_index, intersection.distance);

    // Update path data (helper function call)
    update_ray_path(ray_index, intersection.distance);

    // Get material ID (helper function call)
    let material_id = get_surface_material_id(intersection.surface_id);

    // Calculate new direction after reflection
    let new_direction = apply_material_interaction(
        ray_index,
        material_id,
        ray_direction,
        intersection.normal
    );

    // Update ray after bounce (helper function call)
    update_ray_after_bounce(ray_index, intersection, new_direction, material_id);
}


