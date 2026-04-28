#version 460
#extension GL_GOOGLE_include_directive : enable

// ============================================================================
// particle.vert — point-sprite vertex shader for SPH visualization.
//
// One vertex per particle slot; gl_VertexIndex is 0-based, particle_id is
// 1-based (slot 0 unused). Reads particle state from SSBOs (same physical
// buffers the simulator owns), applies model-view-projection transform, and
// passes a per-vertex color to the fragment shader based on color_mode.
//
// Color modes:
//   0  speed         (length(velocity) → viridis colormap, normalized by scale)
//   1  acceleration  (length(acceleration) → viridis, normalized by scale)
//   2  density       ((rho - rho_0)/rho_0 → diverging blue/white/red)
//   3  voxel_id      (deterministic rainbow hash; debug neighbor-cell sort)
//   4  kernel_sum    ((kernel_sum - 1) → diverging blue/white/red; centered on 1
//                     because Σ V·W ≈ 1 for a well-sampled interior particle)
//
// Dead particles (voxel_id < 0.5) are pushed off-screen with size 0.
// ============================================================================

layout(set = 0, binding = 0) readonly buffer PositionVoxelIdBuffer {
    vec4 position_voxel_id[];
};
layout(set = 0, binding = 1) readonly buffer VelocityMassBuffer {
    vec4 velocity_mass[];
};
layout(set = 0, binding = 2) readonly buffer AccelerationBuffer {
    vec4 acceleration[];
};
layout(set = 0, binding = 3) readonly buffer DensityPressureBuffer {
    // Whichever ping-pong side is the current read-side at this frame.
    vec2 density_pressure[];
};
layout(set = 0, binding = 4) readonly buffer DensityGradientKernelSumBuffer {
    // Same physical buffer regardless of ping-pong parity. We only read .w
    // (kernel_sum) for color mode 4; .xyz holds ∇ρ which is unused here.
    vec4 density_gradient_kernel_sum[];
};

layout(push_constant) uniform PushConstants {
    mat4  view_proj;            // 64 B
    uint  color_mode;           //  4 B   0/1/2/3/4
    float velocity_scale;       //  4 B   normalize speed → [0, 1]
    float acceleration_scale;   //  4 B   normalize |a| → [0, 1]
    float density_deviation_scale; // 4 B normalize (rho-rho0)/rho0 → [-1, +1]
    float rest_density;         //  4 B   reference density for mode 2
    float point_size;           //  4 B   gl_PointSize in pixels
    float kernel_sum_scale;     //  4 B   normalize (kernel_sum - 1) → [-1, +1]
} pc;

layout(location = 0) out vec3 frag_color;

// ----- colormap helpers ------------------------------------------------------

// Viridis colormap for unsigned [0, 1] values. Always visibly colored —
// the minimum is a saturated dark purple (not black), so static particles
// remain visible against the dark background.
vec3 colormap_viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.267, 0.005, 0.329);   // dark purple   (t=0)
    vec3 c1 = vec3(0.190, 0.408, 0.557);   // blue          (t≈0.33)
    vec3 c2 = vec3(0.208, 0.718, 0.473);   // green         (t≈0.66)
    vec3 c3 = vec3(0.992, 0.906, 0.144);   // yellow        (t=1)
    if (t < 0.333) return mix(c0, c1, t / 0.333);
    if (t < 0.666) return mix(c1, c2, (t - 0.333) / 0.333);
    return mix(c2, c3, (t - 0.666) / 0.334);
}

// Diverging blue-white-red for signed [-1, +1] values (density deviation).
vec3 colormap_diverging(float t) {
    t = clamp(t, -1.0, 1.0);
    vec3 cold  = vec3(0.230, 0.299, 0.754);
    vec3 white = vec3(0.985, 0.985, 0.985);
    vec3 hot   = vec3(0.706, 0.016, 0.150);
    if (t < 0.0) return mix(white, cold, -t);
    return mix(white, hot, t);
}

// Pseudo-rainbow from uint hash (golden ratio for nice spread).
vec3 colormap_rainbow_hash(uint key) {
    // Hash key into a [0, 1] hue; HSV→RGB with full saturation/value.
    float hue = fract(float(key) * 0.6180339887);
    vec3 k = vec3(5.0, 3.0, 1.0);
    vec3 p = abs(fract(vec3(hue) + k / 6.0) * 6.0 - 3.0);
    return clamp(p - 1.0, 0.0, 1.0);
}

void main() {
    uint particle_id = gl_VertexIndex + 1u;
    vec4 pos_vid = position_voxel_id[particle_id];

    // Dead / unallocated slot → push off-screen and shrink to nothing.
    if (pos_vid.w < 0.5) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        frag_color = vec3(0.0);
        return;
    }

    gl_Position = pc.view_proj * vec4(pos_vid.xyz, 1.0);
    gl_PointSize = pc.point_size;

    // ---- per-mode colorization ---------------------------------------------
    if (pc.color_mode == 0u) {
        float speed = length(velocity_mass[particle_id].xyz);
        frag_color = colormap_viridis(speed * pc.velocity_scale);
    } else if (pc.color_mode == 1u) {
        float accel_mag = length(acceleration[particle_id].xyz);
        frag_color = colormap_viridis(accel_mag * pc.acceleration_scale);
    } else if (pc.color_mode == 2u) {
        float rho = density_pressure[particle_id].x;
        float dev = (rho - pc.rest_density) / max(pc.rest_density, 1e-6);
        frag_color = colormap_diverging(dev * pc.density_deviation_scale);
    } else if (pc.color_mode == 3u) {
        // voxel_id rainbow
        uint vid = uint(round(pos_vid.w));
        frag_color = colormap_rainbow_hash(vid);
    } else {
        // mode 4: kernel_sum, centered on 1.0
        float ks = density_gradient_kernel_sum[particle_id].w;
        float dev = (ks - 1.0) * pc.kernel_sum_scale;
        frag_color = colormap_diverging(dev);
    }
}
