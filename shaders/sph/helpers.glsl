// ============================================================================
// helpers.glsl
//
// Shared device-side helper functions for all SPH compute shaders.
// Dependency: MUST be included AFTER common.glsl (reads its spec constants
// and buffer declarations).
//
// Usage:
//     #version 460
//     #extension GL_GOOGLE_include_directive : enable
//     #include "common.glsl"
//     #include "helpers.glsl"
//
// Contents:
//   - Wendland C4 kernel evaluation (value + gradient)
//   - 1-based voxel_id ↔ 3D voxel coord conversions + grid bounds check
//   - Symmetric 3×3 correction_inverse unpacking (2 vec4 → mat3)
// ============================================================================

#ifndef SPH_HELPERS_GLSL_INCLUDED
#define SPH_HELPERS_GLSL_INCLUDED

// ============================================================================
// Wendland C4 kernel and gradient.
//
// Support radius = h, q = r / h ∈ [0, 1). Normalization baked into
// KERNEL_COEFFICIENT / KERNEL_GRADIENT_COEFFICIENT (Python-side precomputed,
// including 1/h^DIM and 1/h^(DIM+1)).
//
//   W(q)   = KERNEL_COEFFICIENT          · (1-q)^6 · (35/3·q² + 6q + 1)
//   dW/dq  = (1-q)^5 · q · (-280/3·q - 56/3)
//   ∇W     = KERNEL_GRADIENT_COEFFICIENT · dW/dq · r̂
// ============================================================================

float evaluate_kernel(float distance) {
    float normalized_distance = distance / SMOOTHING_LENGTH;
    if (normalized_distance >= 1.0) return 0.0;

    float one_minus_q    = 1.0 - normalized_distance;
    float one_minus_q_sq = one_minus_q * one_minus_q;
    float one_minus_q_6  = one_minus_q_sq * one_minus_q_sq * one_minus_q_sq;

    // (35/3) · q² + 6q + 1
    float profile_polynomial =
        normalized_distance * ((35.0 / 3.0) * normalized_distance + 6.0) + 1.0;

    return KERNEL_COEFFICIENT * one_minus_q_6 * profile_polynomial;
}

vec3 evaluate_kernel_gradient(vec3 relative_position, float distance) {
    if (distance < 1e-12) return vec3(0.0);
    float normalized_distance = distance / SMOOTHING_LENGTH;
    if (normalized_distance >= 1.0) return vec3(0.0);

    float one_minus_q    = 1.0 - normalized_distance;
    float one_minus_q_sq = one_minus_q * one_minus_q;
    float one_minus_q_4  = one_minus_q_sq * one_minus_q_sq;
    float one_minus_q_5  = one_minus_q_4 * one_minus_q;

    // q · (-280/3 · q + -56/3)
    float derivative_polynomial =
        normalized_distance * ((-280.0 / 3.0) * normalized_distance + (-56.0 / 3.0));

    float gradient_magnitude_scalar =
        KERNEL_GRADIENT_COEFFICIENT * one_minus_q_5 * derivative_polynomial;

    return gradient_magnitude_scalar * (relative_position / distance);
}

// ============================================================================
// 1-based voxel_id ↔ 3D voxel coord.
//
// Particles store 1-based voxel_id in position_voxel_id.w (0 = dead sentinel).
// Voxel coord remains 0-based spatial index (coord.x ∈ [0, GRID_DIMENSION_X-1]).
// The -1u / +1u adjustments live inside these helpers and are invisible to callers.
// ============================================================================

ivec3 own_coord_of(uint voxel_id) {
    uint zero_based = voxel_id - 1u;
    return ivec3(
        zero_based % GRID_DIMENSION_X,
        (zero_based / GRID_DIMENSION_X) % GRID_DIMENSION_Y,
        zero_based / (GRID_DIMENSION_X * GRID_DIMENSION_Y));
}

uint own_voxel_id_of(ivec3 coord) {
    return uint(coord.x)
         + uint(coord.y) * GRID_DIMENSION_X
         + uint(coord.z) * GRID_DIMENSION_X * GRID_DIMENSION_Y
         + 1u;
}

bool in_own_grid(ivec3 coord) {
    return coord.x >= 0 && coord.x < int(GRID_DIMENSION_X)
        && coord.y >= 0 && coord.y < int(GRID_DIMENSION_Y)
        && coord.z >= 0 && coord.z < int(GRID_DIMENSION_Z);
}

// ============================================================================
// Unpack symmetric 3×3 correction_inverse from 2 vec4.
//
//   Storage: [pid*2]   = (M[0][0], M[1][1], M[2][2], M[0][1])
//            [pid*2+1] = (M[0][2], M[1][2], _, _)
// By symmetry M[1][0] = M[0][1], M[2][0] = M[0][2], M[2][1] = M[1][2].
// ============================================================================

mat3 unpack_correction_inverse(uint particle_id) {
    vec4 a = correction_inverse[particle_id * 2u];
    vec4 b = correction_inverse[particle_id * 2u + 1u];
    return mat3(
        vec3(a.x, a.w, b.x),   // column 0
        vec3(a.w, a.y, b.y),   // column 1
        vec3(b.x, b.y, a.z)    // column 2
    );
}

#endif  // SPH_HELPERS_GLSL_INCLUDED
