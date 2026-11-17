// ============================================================================
// wall_boundary.glsl - the wall boundary option (E37). Included by density.comp,
// force.comp and wall_extrapolate.comp only, after common.glsl / helpers.glsl,
// so the other modules' SPIR-V does not change.
//
// WALL_BOUNDARY (spec const 100) = case.yaml numerics.wall_boundary:
//   0 = simple (default): the v6 walls. A wall particle integrates its density
//       from its fluid neighbours (density.comp skips wall-wall pairs), stores
//       rho0 and the pressure EOS(rho0 + dt * drho/dt); force uses its stored
//       velocity (0, or the lid speed) in the viscous term.
//   1 = adami: the wall pressure and no-slip condition of Adami, Hu & Adams
//       (2012), walls storing rho0 (E36's adami_rho0 = WALL_BC 3 of the
//       v7-wall-bc branch). wall_extrapolate.comp, after density and before
//       every force, sets each wall particle's pressure to the Shepard average
//       of its fluid neighbours' pressure (+ the body-force term), keeps its
//       stored density at rho0 and writes a dummy velocity 2 u_w - (Shepard
//       average of the fluid velocity), which force uses for wall neighbours in
//       the viscous term. Walls do not integrate density. Continuity, delta
//       term, KCG and the wall volume see rho0 and the wall's stored,
//       prescribed velocity, as with simple.
//       E40: on a slab with a peer the wall pass is split like density (an
//       interior pass in phase B, a band pass in phase C that also computes
//       the inner ghost column's walls as self from a three-layer ghost); the
//       transport still carries no wall_dummy_velocity. See
//       wall_extrapolate.comp and partition_v7 (GHOST_LAYERS = 3).
//   2 = crossing: the crossing-fraction walls (first order, axis-aligned box),
//       in the WALL_CROSSING compile variants only: see the section at the end
//       of this file. No wall pass, no wall state: a wall particle keeps rho0
//       and its prescribed velocity and never enters a neighbour loop as self.
// ============================================================================

layout(constant_id = 100) const uint WALL_BOUNDARY = 0u;
const uint WALL_BOUNDARY_SIMPLE = 0u;
const uint WALL_BOUNDARY_ADAMI  = 1u;
// adami: walls skip the density pass and take (rho0, p_w) from the wall pass; a wall neighbour's viscous
// velocity is its dummy velocity
const bool WALL_ADAMI = (WALL_BOUNDARY == WALL_BOUNDARY_ADAMI);

layout(std430, set = 0, binding = 10) buffer WallDummyVelocityBuffer {
    // adami: per wall particle, written by wall_extrapolate.comp every step before force reads it:
    // (u_dummy = 2 u_w - u~_w, Sigma_f W_wf). Fluid slots are never written. Transient: defrag does not permute
    // it (it is recomputed before every read); not part of the restart state. simple: a 16-byte stub, never read.
    vec4 wall_dummy_velocity[];
};

#ifdef WALL_CROSSING
// ============================================================================
// WALL_BOUNDARY = 2 = crossing (case.yaml numerics.wall_boundary: crossing),
// first order, axis-aligned box. Compiled only into the WALL_CROSSING variants
// (compile_shaders_v7.py: correction_density_wall_crossing.comp.spv,
// force_wall_crossing.comp.spv), so every other module, and the plain builds of
// correction_density.comp / force.comp that simple and adami run, keep their
// bytes. Inside a variant every statement of the mode sits behind
// WALL_CROSSING_ACTIVE (spec const 100 == 2): with 0 or 1 the variant computes
// what the plain build computes.
//
// A wall particle carries no physical state, it is a quadrature point. When a
// fluid particle i meets a wall particle j in its neighbour loop, the value of
// j is the linear continuation of the field along the segment i -> j. The only
// geometry is the share of that segment on the fluid side,
//     t_ij = phi_i / (phi_i + d_j),
// phi_i = distance of i to the wall (floored at WALL_CROSSING_DISTANCE_FLOOR),
// d_j = depth of j behind the wall.
//   - a field with a prescribed wall value theta_b (the velocity: no slip,
//     theta_b = the wall particle's stored velocity):
//         theta_j - theta_i = (theta_b - theta_i) / t_ij
//                           = (theta_b - theta_i) (1 + d_j / phi_i);
//   - a field without one (pressure, density):
//         theta_j - theta_i = G_i . (r_j - r_i),
//     G_i = A_f^-1 g_f the one-sided gradient from the fluid neighbours only,
//     A_f = sum_{j fluid} V_j (r_j - r_i) (x) grad W_ij,
//     g_f = sum_{j fluid} V_j (theta_j - theta_i) grad W_ij.
// Both rules are exact for a linear field. The correction matrix L_i (every
// neighbour, real displacements) is unchanged.
//
// The two functions of the geometry are the box's; another geometry replaces
// wall_crossing_distance_to_wall / wall_crossing_depth_behind_wall only.
// ============================================================================
const uint WALL_BOUNDARY_CROSSING = 2u;
const bool WALL_CROSSING_ACTIVE = (WALL_BOUNDARY == WALL_BOUNDARY_CROSSING);

// numerics.wall_crossing.continuity: the velocity difference of a wall neighbour in the continuity equation,
// 0 = extrapolate ((v_w - v_i) (1 + d_j / phi_i), the rule above), 1 = wall_velocity (v_w - v_i, the stored
// wall velocity as it is). The viscous term always extrapolates.
layout(constant_id = 120) const uint  WALL_CROSSING_CONTINUITY = 0u;
const uint WALL_CROSSING_CONTINUITY_EXTRAPOLATE   = 0u;
const uint WALL_CROSSING_CONTINUITY_WALL_VELOCITY = 1u;
// DIAGNOSTIC, not part of the mode (numerics.wall_crossing.delta_wall, default extension = 0, which is the mode
// as specified): how a wall neighbour enters the density diffusion (delta term) of the fused pass.
//   0 = extension:    psi_ij = G_i . (r_j - r_i), the continuation of the density, applied after the loop
//                     through sum_wall V_j r^2 / (r^2 + eps^2) grad W_ij;
//   1 = rest_density: psi_ij = rho0 - rho_i pair by pair, the expression simple uses (a wall particle stores
//                     rho0), and no term after the loop;
//   2 = body_force:   psi_ij = rho_i (g - a_wall) . (r_j - r_i) / c^2 pair by pair, the density difference of a
//                     fluid at rest against the wall, and no term after the loop. g = (GRAVITY_X, GRAVITY_Y,
//                     GRAVITY_Z); a_wall = 0 (no wall of this solver accelerates: a wall is at rest or moves at
//                     a constant velocity); c^2 = dp/drho of the particle's Tait EOS at rho_i,
//                     gamma B / rho0 (rho_i / rho0)^(gamma - 1) (= c0^2 at rho0). Without gravity a wall
//                     neighbour contributes nothing to the density diffusion.
// The drift of the continuity equation, the pressure continuation and the viscous term are the same in all three.
// 1 exists to test whether the mean density drifts because the delta term has no rho0 reference, 2 to test
// whether the drift comes from the continuation term itself (neither 0 nor 2 refers to rho0).
layout(constant_id = 116) const uint  WALL_CROSSING_DELTA_WALL = 0u;
const uint WALL_CROSSING_DELTA_WALL_EXTENSION    = 0u;
const uint WALL_CROSSING_DELTA_WALL_REST_DENSITY = 1u;
const uint WALL_CROSSING_DELTA_WALL_BODY_FORCE   = 2u;
// lower bound of phi_i = numerics.wall_crossing.distance_floor_factor (default 0.25) x the particle spacing
layout(constant_id = 121) const float WALL_CROSSING_DISTANCE_FLOOR = 0.001;
// the fluid box [lower, upper] in world coordinates (2-D: x and y only)
layout(constant_id = 122) const float WALL_CROSSING_BOX_LOWER_X = 0.0;
layout(constant_id = 123) const float WALL_CROSSING_BOX_LOWER_Y = 0.0;
layout(constant_id = 124) const float WALL_CROSSING_BOX_LOWER_Z = 0.0;
layout(constant_id = 125) const float WALL_CROSSING_BOX_UPPER_X = 1.0;
layout(constant_id = 126) const float WALL_CROSSING_BOX_UPPER_Y = 1.0;
layout(constant_id = 127) const float WALL_CROSSING_BOX_UPPER_Z = 1.0;

layout(std430, set = 0, binding = 12) buffer WallCrossingFluidInverseBuffer {
    // crossing: per fluid particle A_f^-1 (symmetric, the packing of correction_inverse) and a validity flag:
    //   [pid * 2]     = (m00, m11, m22, m01)
    //   [pid * 2 + 1] = (m02, m12, 0, valid)     valid = 1, or 0 when the inversion fell back (then G = 0)
    // Written by the fused correction + density pass, read by force of the same step for the particle itself
    // only. Transient like wall_dummy_velocity: defrag does not permute it, the transport does not carry it,
    // it is not part of the restart state. Wall slots are never written. Exists only with wall_boundary
    // crossing (the plain builds do not declare the binding).
    vec4 wall_crossing_fluid_inverse[];
};

// phi(r): distance of a point of the box to its nearest wall plane (negative outside the box)
float wall_crossing_distance_to_wall(vec3 position) {
    vec3 lower_gap = position - vec3(WALL_CROSSING_BOX_LOWER_X, WALL_CROSSING_BOX_LOWER_Y, WALL_CROSSING_BOX_LOWER_Z);
    vec3 upper_gap = vec3(WALL_CROSSING_BOX_UPPER_X, WALL_CROSSING_BOX_UPPER_Y, WALL_CROSSING_BOX_UPPER_Z) - position;
    vec3 axis_gap  = min(lower_gap, upper_gap);
    float nearest  = min(axis_gap.x, axis_gap.y);
    if (DIMENSION != 2u) {
        nearest = min(nearest, axis_gap.z);
    }
    return nearest;
}

// d(r): depth of a point outside the box = its distance to the box (in an edge or corner block the distance to
// the edge or corner)
float wall_crossing_depth_behind_wall(vec3 position) {
    vec3 below_lower = vec3(WALL_CROSSING_BOX_LOWER_X, WALL_CROSSING_BOX_LOWER_Y, WALL_CROSSING_BOX_LOWER_Z) - position;
    vec3 above_upper = position - vec3(WALL_CROSSING_BOX_UPPER_X, WALL_CROSSING_BOX_UPPER_Y, WALL_CROSSING_BOX_UPPER_Z);
    vec3 outside     = max(max(below_lower, above_upper), vec3(0.0));
    if (DIMENSION == 2u) {
        outside.z = 0.0;
    }
    return length(outside);
}

// phi_i of a fluid particle with the floor
float wall_crossing_floored_distance(vec3 fluid_position) {
    return max(wall_crossing_distance_to_wall(fluid_position), WALL_CROSSING_DISTANCE_FLOOR);
}

// 1 / t_ij = 1 + d_j / phi_i
float wall_crossing_extrapolation_factor(float fluid_wall_distance, vec3 wall_position) {
    return 1.0 + wall_crossing_depth_behind_wall(wall_position) / fluid_wall_distance;
}

// A_f^-1: the correction matrix's epilogue (correction.comp: xi on the diagonal, the 2-D block, the determinant
// threshold, the Frobenius cap) on the fluid-only matrix. false = fallback (too few fluid neighbours): the
// identity is returned and the caller takes G = 0.
bool wall_crossing_invert_fluid_matrix(mat3 fluid_correction_matrix, out mat3 fluid_correction_inverse) {
    fluid_correction_matrix[0][0] += REGULARIZATION_XI;
    fluid_correction_matrix[1][1] += REGULARIZATION_XI;
    if (DIMENSION == 2u) {
        fluid_correction_matrix[0][2] = 0.0;
        fluid_correction_matrix[1][2] = 0.0;
        fluid_correction_matrix[2][0] = 0.0;
        fluid_correction_matrix[2][1] = 0.0;
        fluid_correction_matrix[2][2] = 1.0;
    } else {
        fluid_correction_matrix[2][2] += REGULARIZATION_XI;
    }
    if (abs(determinant(fluid_correction_matrix)) <= REGULARIZATION_DETERMINANT_THRESHOLD) {
        fluid_correction_inverse = mat3(1.0);
        return false;
    }
    fluid_correction_inverse = inverse(fluid_correction_matrix);
    float frobenius_squared = 0.0;
    for (int row_index = 0; row_index < 3; row_index++) {
        for (int column_index = 0; column_index < 3; column_index++) {
            float element = fluid_correction_inverse[row_index][column_index];
            frobenius_squared += element * element;
        }
    }
    float frobenius = sqrt(frobenius_squared);
    if (frobenius > REGULARIZATION_MAX_FROBENIUS_NORM) {
        fluid_correction_inverse *= (REGULARIZATION_MAX_FROBENIUS_NORM / frobenius);
    }
    return true;
}

// the symmetric matrix of the six stored entries (helpers.glsl unpack_correction_inverse's convention)
mat3 wall_crossing_symmetric_matrix(vec4 diagonal_and_xy, vec4 xz_and_yz) {
    return mat3(
        vec3(diagonal_and_xy.x, diagonal_and_xy.w, xz_and_yz.x),
        vec3(diagonal_and_xy.w, diagonal_and_xy.y, xz_and_yz.y),
        vec3(xz_and_yz.x,       xz_and_yz.y,       diagonal_and_xy.z));
}
#endif  // WALL_CROSSING
