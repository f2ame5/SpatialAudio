struct Ray {
    origin: vec3f,
    direction: vec3f,
    energy: f32,
    pathLength: f32,
    bounces: u32,
    isActive: u32,
    frequency: f32,    // Frequency of the wave in Hz
    phase: f32,        // Current phase of the wave
    time: f32         // Time elapsed for this ray
};

struct Surface {
    normal: vec3f,
    position: vec3f,
    absorption: f32
};

struct RayHit {
    position: vec3f,
    energy: f32,
    time: f32,
    phase: f32,       // Phase at hit point
    frequency: f32    // Frequency of the wave at hit point
};

struct RayIntersection {
    hit: bool,
    distance: f32,
    position: vec3f,
    normal: vec3f,
    surfaceIndex: u32
};

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

// Calculate reflected direction
fn reflect(incident: vec3f, normal: vec3f) -> vec3f {
    return incident - 2.0 * dot(incident, normal) * normal;
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
    if (ray.isActive == 0u || ray.bounces >= 50u || ray.energy < 0.01) {
        return;
    }

    let intersection = findClosestIntersection(ray);
    if (intersection.hit) {
        let distance = intersection.distance;
        let speed_of_sound = 343.0;  // Speed of sound in m/s
        let travel_time = distance / speed_of_sound;
        
        // Calculate phase change over distance
        // Phase = 2π * frequency * time
        let phase_change = 2.0 * 3.14159 * ray.frequency * travel_time;
        let new_phase = ray.phase + phase_change;

        // Record hit with wave properties
        hits[ray_index] = RayHit(
            intersection.position,
            ray.energy,
            ray.time + travel_time,
            new_phase,
            ray.frequency
        );

        // Update ray for next bounce
        let surface = surfaces[intersection.surfaceIndex];
        let reflected = reflect(ray.direction, intersection.normal);

        ray.origin = intersection.position;
        ray.direction = reflected;
        ray.energy *= (1.0 - surface.absorption);
        ray.pathLength += distance;
        ray.bounces += 1u;
        ray.time += travel_time;
        ray.phase = new_phase;

        rays[ray_index] = ray;
    } else {
        ray.isActive = 0u;
        rays[ray_index] = ray;
    }
}