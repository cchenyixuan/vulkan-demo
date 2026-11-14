// ============================================================================
// periodic.glsl - the periodic domain (ported from experiment/v8 of origin/v8-tgv, Stage 8b of
// docs/micropolar_design.md). One slab on one device; x and / or y periodic, z never.
//
// Included only by the kernels that need it (the neighbour sweeps of correction, correction_density, density,
// force, wall_extrapolate, wall_extrapolate_band, deep_wall_marker / deep_wall_skip.glsl, and the binning in
// predict and initialize_voxelization), after helpers.glsl, so every other module keeps its SPIR-V.
//
// With both flags off (the default) every function below folds to the expression it replaces: the kernels
// specialize to the code of thesis-ch5-rc2.
//
// PERIODIC_X / PERIODIC_Y: the own grid tiles the axis: GRID_DIMENSION * SMOOTHING_LENGTH = DOMAIN_LENGTH (the
// loader asserts it, and >= 3 voxels so the three neighbour columns are distinct). The neighbour loops wrap the
// neighbour voxel coordinate (wrap_neighbor_coord) and take the minimum-image separation (minimum_image: exactly
// antisymmetric, so pair forces stay equal and opposite); predict folds a drifted position back into
// [origin, origin + L) (wrap_position) instead of killing it, and the voxel coordinate of a wrapped position is
// clamped to the grid (clamp_periodic_coord). No ghost columns and no walls are needed.
//
// Spec constant ids 116 - 119 (common.glsl lists them; the micropolar replica layers moved to 120).
// ============================================================================

#ifndef SPH_PERIODIC_GLSL_INCLUDED
#define SPH_PERIODIC_GLSL_INCLUDED

layout(constant_id = 116) const bool  PERIODIC_X      = false;
layout(constant_id = 117) const bool  PERIODIC_Y      = false;
layout(constant_id = 118) const float DOMAIN_LENGTH_X = 1.0;
layout(constant_id = 119) const float DOMAIN_LENGTH_Y = 1.0;

// Neighbour voxel coordinate of the 3^d sweep: wrapped on a periodic axis (the sweep offsets are -1 / 0 / +1 and
// the grid has >= 3 voxels, so one wrap suffices and the three columns stay distinct); false when the coordinate
// is off the grid on a non-periodic axis.
bool wrap_neighbor_coord(inout ivec3 coord) {
    if (PERIODIC_X) {
        if (coord.x < 0) coord.x += int(GRID_DIMENSION_X);
        else if (coord.x >= int(GRID_DIMENSION_X)) coord.x -= int(GRID_DIMENSION_X);
    }
    if (PERIODIC_Y) {
        if (coord.y < 0) coord.y += int(GRID_DIMENSION_Y);
        else if (coord.y >= int(GRID_DIMENSION_Y)) coord.y -= int(GRID_DIMENSION_Y);
    }
    return in_own_grid(coord);
}

// Minimum-image separation r_i - r_j on the periodic axes. Applied to the float32 difference, so the pair (j, i)
// gets exactly the negated vector: a - b = -(b - a) in IEEE arithmetic, round() is odd away from the ties
// |d| = L / 2 (never a neighbour: |d| < h <= L / 3 after the image), and fl(d - L k) = -fl(-d + L k).
vec3 minimum_image(vec3 difference) {
    if (PERIODIC_X) difference.x -= DOMAIN_LENGTH_X * round(difference.x / DOMAIN_LENGTH_X);
    if (PERIODIC_Y) difference.y -= DOMAIN_LENGTH_Y * round(difference.y / DOMAIN_LENGTH_Y);
    return difference;
}

// Fold a position into [origin, origin + L) on each periodic axis. A position inside the box is returned bit for
// bit (floor(...) = 0). The float32 wrap of a position a rounding error below the origin can land exactly on
// origin + L; that point is the origin.
vec3 wrap_position(vec3 position) {
    if (PERIODIC_X) {
        position.x -= DOMAIN_LENGTH_X * floor((position.x - OWN_ORIGIN_X) / DOMAIN_LENGTH_X);
        if (position.x >= OWN_ORIGIN_X + DOMAIN_LENGTH_X) position.x = OWN_ORIGIN_X;
    }
    if (PERIODIC_Y) {
        position.y -= DOMAIN_LENGTH_Y * floor((position.y - OWN_ORIGIN_Y) / DOMAIN_LENGTH_Y);
        if (position.y >= OWN_ORIGIN_Y + DOMAIN_LENGTH_Y) position.y = OWN_ORIGIN_Y;
    }
    return position;
}

// The voxel coordinate floor((x - origin) / h) of a wrapped position can round to GRID_DIMENSION on a periodic
// axis (x within an ulp of origin + L): clamp it onto the grid. The separations the neighbour loops use come from
// minimum_image, so the listing voxel does not change the physics.
ivec3 clamp_periodic_coord(ivec3 coord) {
    if (PERIODIC_X) coord.x = clamp(coord.x, 0, int(GRID_DIMENSION_X) - 1);
    if (PERIODIC_Y) coord.y = clamp(coord.y, 0, int(GRID_DIMENSION_Y) - 1);
    return coord;
}

#endif // SPH_PERIODIC_GLSL_INCLUDED
