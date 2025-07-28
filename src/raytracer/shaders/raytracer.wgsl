// Assuming these constants based on spatial_audio.wgsl
const BAND_63 = 0;
const BAND_125 = 1;
const BAND_250 = 2;
const BAND_500 = 3;
const BAND_1K = 4;
const BAND_2K = 5;
const BAND_4K = 6;
const BAND_8K = 7;
const NUM_BANDS = 8;

struct Ray {
    origin: vec3f,
    direction: vec3f,
    // Energy for each frequency band
    energy63: f32,
    energy125: f32,
    energy250: f32,
    energy500: f32,
    energy1k: f32,
    energy2k: f32,
    energy4k: f32,
    energy8k: f32,
    pathLength: f32,
    bounces: u32,
    isActive: u32,
    // Phase for each frequency band
    phase63: f32,
    phase125: f32,
    phase250: f32,
    phase500: f32,
    phase1k: f32,
    phase2k: f32,
    phase4k: f32,
    phase8k: f32,
    // Time is common for all bands
    time: f32
};

struct Surface {
    normal: vec3f,
    position: vec3f,
    // Absorption for each frequency band
    absorption63: f32,
    absorption125: f32,
    absorption250: f32,
    absorption500: f32,
    absorption1k: f32,
    absorption2k: f32,
    absorption4k: f32,
    absorption8k: f32,
    // Scattering for each frequency band
    scattering63: f32,
    scattering125: f32,
    scattering250: f32,
    scattering500: f32,
    scattering1k: f32,
    scattering2k: f32,
    scattering4k: f32,
    scattering8k: f32
};

struct RayHit {
    position: vec3f,
    time: f32,
    normal: vec3f,
    // Energy for each frequency band (matching spatial_audio.wgsl)
    energy63: f32,
    energy125: f32,
    energy250: f32,
    energy500: f32,
    energy1k: f32,
    energy2k: f32,
    energy4k: f32,
    energy8k: f32,
    incomingDirection: vec3f, // Added to match spatial_audio.wgsl
    // Phase for each frequency band
    phase63: f32,
    phase125: f32,
    phase250: f32,
    phase500: f32,
    phase1k: f32,
    phase2k: f32,
    phase4k: f32,
    phase8k: f32,
    // Frequency is implied by the band
    // Add HRTF index as in spatial_audio.wgsl
    hrtfIndex: u32,
    _padding: f32 // Maintain alignment
};

struct RayIntersection {
    hit: bool,
    distance: f32,
    position: vec3f,
    normal: vec3f,
    surfaceIndex: u32
};

// Add RNG helper (simple Wang hash or similar)
fn rand(seed: u32) -> f32 {
    var s = seed ^ 2747636419u;
    s = (s * 2654435769u) ^ (s >> 16u);
    return f32(s) / 4294967295.0;
}

fn random_in_hemisphere(normal: vec3f, seed: u32) -> vec3f {
    let r1 = rand(seed);
    let r2 = rand(seed + 1u);
    let r3 = rand(seed + 2u);
    let dir = vec3f(r1 * 2.0 - 1.0, r2 * 2.0 - 1.0, r3 * 2.0 - 1.0);
    let in_unit_sphere = normalize(dir);
    if (dot(in_unit_sphere, normal) > 0.0) {
        return in_unit_sphere;
    } else {
        return -in_unit_sphere;
    }
}

// Updated reflect function with hybrid specular-diffuse reflection for a specific band
fn reflect_band(incident: vec3f, normal: vec3f, scattering: f32, seed: u32) -> vec3f {
    let specular_dir = incident - 2.0 * dot(incident, normal) * normal;
    let diffuse_dir = normal + random_in_hemisphere(normal, seed);
    return normalize(mix(specular_dir, diffuse_dir, scattering));
}

// Helper function to find closest intersection
fn findClosestIntersection(ray: Ray) -> RayIntersection {
    var closest: RayIntersection;
    closest.hit = false;
    closest.distance = 999999.0;

    for (var i = 0u; i < arrayLength(&surfaces); i++) {
        let surface = surfaces[i];

        // Calculate intersection with plane
        let denom = dot(ray.direction, surface.normal);
        if (abs(denom) > 0.0001) { // Avoid parallel rays
            let t = dot(surface.position - ray.origin, surface.normal) / denom;
            if (t > 0.0001 && t < closest.distance) {
                closest.hit = true;
                closest.distance = t;
                closest.position = ray.origin + ray.direction * t;
                closest.normal = surface.normal;
                closest.surfaceIndex = i;
            }
        }
    }

    return closest;
}

@group(0) @binding(0) var<storage, read_write> rays: array<Ray>;
@group(0) @binding(1) var<storage, read> surfaces: array<Surface>;
@group(0) @binding(2) var<storage, read_write> hits: array<RayHit>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let ray_index = global_id.x;
    if (ray_index >= arrayLength(&rays)) {
        return;
    }

    var ray = rays[ray_index];
    // Check if any band has significant energy
    let total_energy = ray.energy63 + ray.energy125 + ray.energy250 + ray.energy500 +
                       ray.energy1k + ray.energy2k + ray.energy4k + ray.energy8k;
    
    if (ray.isActive == 0u || ray.bounces >= 50u || total_energy < 0.01) {
        return;
    }

    let intersection = findClosestIntersection(ray);
    if (intersection.hit) {
        let distance = intersection.distance;
        let speed_of_sound = 343.0;  // Speed of sound in m/s
        let travel_time = distance / speed_of_sound;
        
        // Calculate phase change over distance for each band
        // Phase = 2π * frequency * time
        // Using approximate center frequencies for each band
        let phase_change_63 = 2.0 * 3.14159 * 63.0 * travel_time;
        let phase_change_125 = 2.0 * 3.14159 * 125.0 * travel_time;
        let phase_change_250 = 2.0 * 3.14159 * 250.0 * travel_time;
        let phase_change_500 = 2.0 * 3.14159 * 500.0 * travel_time;
        let phase_change_1k = 2.0 * 3.14159 * 1000.0 * travel_time;
        let phase_change_2k = 2.0 * 3.14159 * 2000.0 * travel_time;
        let phase_change_4k = 2.0 * 3.14159 * 4000.0 * travel_time;
        let phase_change_8k = 2.0 * 3.14159 * 8000.0 * travel_time;
        
        let new_phase_63 = ray.phase63 + phase_change_63;
        let new_phase_125 = ray.phase125 + phase_change_125;
        let new_phase_250 = ray.phase250 + phase_change_250;
        let new_phase_500 = ray.phase500 + phase_change_500;
        let new_phase_1k = ray.phase1k + phase_change_1k;
        let new_phase_2k = ray.phase2k + phase_change_2k;
        let new_phase_4k = ray.phase4k + phase_change_4k;
        let new_phase_8k = ray.phase8k + phase_change_8k;

        // Record hit with wave properties for all bands
        hits[ray_index] = RayHit(
            intersection.position,
            ray.time + travel_time,
            intersection.normal,
            ray.energy63,
            ray.energy125,
            ray.energy250,
            ray.energy500,
            ray.energy1k,
            ray.energy2k,
            ray.energy4k,
            ray.energy8k,
            ray.direction, // incomingDirection
            new_phase_63,
            new_phase_125,
            new_phase_250,
            new_phase_500,
            new_phase_1k,
            new_phase_2k,
            new_phase_4k,
            new_phase_8k,
            0u, // hrtfIndex - will be set elsewhere or defaulted
            0.0 // padding
        );

        // Update ray for next bounce for each band
        let surface = surfaces[intersection.surfaceIndex];
        let seed = u32(ray_index * ray.bounces + global_id.x); // Simple seed
        
        // Reflect for each band using its specific scattering coefficient
        let reflected_63 = reflect_band(ray.direction, intersection.normal, surface.scattering63, seed);
        let reflected_125 = reflect_band(ray.direction, intersection.normal, surface.scattering125, seed + 1u);
        let reflected_250 = reflect_band(ray.direction, intersection.normal, surface.scattering250, seed + 2u);
        let reflected_500 = reflect_band(ray.direction, intersection.normal, surface.scattering500, seed + 3u);
        let reflected_1k = reflect_band(ray.direction, intersection.normal, surface.scattering1k, seed + 4u);
        let reflected_2k = reflect_band(ray.direction, intersection.normal, surface.scattering2k, seed + 5u);
        let reflected_4k = reflect_band(ray.direction, intersection.normal, surface.scattering4k, seed + 6u);
        let reflected_8k = reflect_band(ray.direction, intersection.normal, surface.scattering8k, seed + 7u);

        // Average the reflected directions from all bands to get a more representative new path for the ray.
        // This prevents biasing the path to a single frequency's scattering properties.
        let avg_reflected_direction = normalize(
            reflected_63 + reflected_125 + reflected_250 + reflected_500 +
            reflected_1k + reflected_2k + reflected_4k + reflected_8k
        );
        ray.origin = intersection.position;
        ray.direction = avg_reflected_direction;
        
        // Apply absorption for each band
        ray.energy63 *= (1.0 - surface.absorption63);
        ray.energy125 *= (1.0 - surface.absorption125);
        ray.energy250 *= (1.0 - surface.absorption250);
        ray.energy500 *= (1.0 - surface.absorption500);
        ray.energy1k *= (1.0 - surface.absorption1k);
        ray.energy2k *= (1.0 - surface.absorption2k);
        ray.energy4k *= (1.0 - surface.absorption4k);
        ray.energy8k *= (1.0 - surface.absorption8k);
        
        ray.pathLength += distance;
        ray.bounces += 1u;
        ray.time += travel_time;
        
        // Update phases
        ray.phase63 = new_phase_63;
        ray.phase125 = new_phase_125;
        ray.phase250 = new_phase_250;
        ray.phase500 = new_phase_500;
        ray.phase1k = new_phase_1k;
        ray.phase2k = new_phase_2k;
        ray.phase4k = new_phase_4k;
        ray.phase8k = new_phase_8k;

        rays[ray_index] = ray;
    } else {
        ray.isActive = 0u;
        rays[ray_index] = ray;
    }
}