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
//
// numerics.wall_crossing.geometry pairwise (off by default; spec const 128
// below) takes t_ij and the wall normal pair by pair from the face the segment
// i -> j leaves the box through, and gives the pair a mirror point; the
// geometry is then wall_crossing_pair_geometry alone. With it, pressure_wall
// reflection (spec const 129) and delta_wall reflection (spec const 116 = 3)
// take the pressure / the density of a wall particle from its mirror point
// plus the body force, in place of the plain continuation.
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
//   3 = reflection:   NOT a diagnostic, a variant of the mode (needs geometry pairwise, spec const 128): the
//                     density of the wall particle is the continued density at its mirror point r_j' plus the
//                     density difference of a fluid at rest between r_j' and r_j,
//                         psi_ij = G_i . (r_j' - r_i) + rho_i g . (r_j - r_j') / c^2
//                                = G_i . (r_j - r_i) - (G_i - rho_i g / c^2) . (r_j - r_j'),
//                     i.e. value 0's term after the loop and, also after the loop,
//                         - 2 delta h c0 (G_i - rho_i g / c^2) . sum_wall V_j |grad W_ij| r / (r^2 + eps^2) (r_j - r_j')
//                     (the pair's r_ij . grad W_ij = |grad W_ij| r without the correction matrix). No diffusive
//                     flux through the wall: the normal density gradient at the wall is the body force's.
//                     c^2 as with 2. Zero when A_f fell back (G_i = 0 there, as with 0).
// The drift of the continuity equation, the pressure continuation and the viscous term are the same in all four.
// 1 exists to test whether the mean density drifts because the delta term has no rho0 reference, 2 to test
// whether the drift comes from the continuation term itself (neither 0 nor 2 refers to rho0).
layout(constant_id = 116) const uint  WALL_CROSSING_DELTA_WALL = 0u;
const uint WALL_CROSSING_DELTA_WALL_EXTENSION    = 0u;
const uint WALL_CROSSING_DELTA_WALL_REST_DENSITY = 1u;
const uint WALL_CROSSING_DELTA_WALL_BODY_FORCE   = 2u;
const uint WALL_CROSSING_DELTA_WALL_REFLECTION   = 3u;
// numerics.wall_crossing.continuity_wall_component: which part of a wall neighbour's velocity difference enters
// the continuity equation.
//   1 = normal (the default since 2026-11-18): its component along the wall's inward normal only,
//               n_j (n_j . (v_w - v_i)), n_j = the unit vector from the wall particle to the nearest point of
//               the box (then the continuity rule above);
//   0 = full:   the whole vector (then the continuity rule); the mode as first specified and the default
//               until 2026-11-18, kept as a switch.
// The viscous term takes the whole velocity difference with either value. What the choice rests on: in the
// density budget of the 250^2 cavity (claude/wall_crossing_report.md, section 8) the continuity term over
// the fluid neighbours of the interior was a net sink with full and was not one with normal.
layout(constant_id = 117) const uint  WALL_CROSSING_CONTINUITY_WALL_COMPONENT = 1u;
const uint WALL_CROSSING_CONTINUITY_WALL_COMPONENT_FULL   = 0u;
const uint WALL_CROSSING_CONTINUITY_WALL_COMPONENT_NORMAL = 1u;
// numerics.wall_crossing.viscous_extension / continuity_extension (both default 0 = linear, the first-order rule
// above): the SECOND-ORDER continuation of the velocity. Along the segment i -> j the velocity is a parabola
// with the value v_i and the slope grad v_i (r_j - r_i) at i and the wall value v_w at the crossing point, which
// gives a wall neighbour the velocity difference (a_j = d_j / phi_i, so 1 / t_ij = 1 + a_j)
//     v_j - v_i = (v_w - v_i) (1 + a_j)^2 - a_j grad v_i (r_j - r_i).
// grad v_i ((grad v)_ab = d v_a / d x_b) is the corrected gradient over EVERY neighbour, the wall neighbours
// entering with this same rule, i.e. the solution of a small linear system per particle: with
//     M_0   = sum_fluid V_j (v_j - v_i) (x) grad W_ij + sum_wall V_j (v_w - v_i) (1 + a_j)^2 (x) grad W_ij
//     K_raw = sum_wall V_j (-a_j) (r_j - r_i) (x) grad W_ij
// the raw sum over all neighbours is M_0 + grad v_i K_raw, and grad v_i = (M_0 + grad v_i K_raw) L_i, so
//     grad v_i = M_0 L_i (I - K_raw L_i)^-1.
// The fused pass accumulates M_0 and K_raw beside the correction matrix (the same V_j and grad W_ij), solves and
// stores grad v_i for force of the same step (wall_crossing_velocity_gradient below).
//   viscous_extension quadratic:    force's viscous term takes the rule above for a wall neighbour;
//   continuity_extension quadratic: the wall neighbours' share of the velocity divergence in the continuity
//       equation is tr[(M_w + grad v_i K_raw) L_i], M_w = the wall part of M_0 (the whole vector:
//       WALL_CROSSING_CONTINUITY_WALL_COMPONENT does not apply), in place of the pair terms.
// A particle falls back to the first-order rule in both terms for the step (state 2 below) when
// det(I - K_raw L_i) < 0.5, when the solution is not finite or when L_i itself fell back. Exact for a quadratic
// velocity field on a lattice (L_i without regularisation).
layout(constant_id = 118) const uint  WALL_CROSSING_VISCOUS_EXTENSION = 0u;
layout(constant_id = 119) const uint  WALL_CROSSING_CONTINUITY_EXTENSION = 0u;
const uint WALL_CROSSING_EXTENSION_LINEAR    = 0u;
const uint WALL_CROSSING_EXTENSION_QUADRATIC = 1u;
// numerics.wall_crossing.geometry: where the pair's crossing fraction and wall normal come from.
//   0 = nearest_wall (the default, everything above): phi_i = the distance of i to ITS nearest wall plane, d_j =
//       the distance of j to the box, n_j = the unit vector from j to the nearest point of the box. Near an edge
//       or corner of the box the two distances belong to different faces and t_ij is not the pair's.
//   1 = pairwise: from the face the segment i -> j leaves the box through (wall_crossing_pair_geometry below):
//       phi_ij = the distance of i to that face (floored), d_ij = the depth of j behind that face, n_ij = that
//       face's inward normal, 1 / t_ij = 1 + d_ij / phi_ij; the velocity rules above with these (the viscous
//       term: the whole vector; the continuity equation: the component along n_ij with
//       WALL_CROSSING_CONTINUITY_WALL_COMPONENT normal). Not offered with a quadratic extension.
layout(constant_id = 128) const uint  WALL_CROSSING_GEOMETRY = 0u;
const uint WALL_CROSSING_GEOMETRY_NEAREST_WALL = 0u;
const uint WALL_CROSSING_GEOMETRY_PAIRWISE     = 1u;
// numerics.wall_crossing.pressure_wall: the pressure of a wall neighbour in force.
//   0 = extension (the default, the rule above): P_j - P_i = G_i . (r_j - r_i);
//   1 = reflection (needs geometry pairwise): the continued pressure at the mirror point r_j' plus the body
//       force between r_j' and r_j,
//           P_j - P_i = G_i . (r_j' - r_i) + rho_i g . (r_j - r_j')
//                     = G_i . (r_j - r_i) - (G_i - rho_i g) . (r_j - r_j'),
//       g = (GRAVITY_X, GRAVITY_Y, GRAVITY_Z): the wall carries the condition "normal pressure gradient = the body
//       force's" and so pushes on the fluid itself. With M_d = sum_wall V_j grad W_ij (x) (r_j - r_j') force adds
//           (1 / rho_i) L_i M_d (G_i - rho_i g)
//       to the pressure acceleration after the loop (the same on both TIC branches). Zero for a fluid at rest
//       (G_i = rho_i g) and when A_f fell back (G_i = 0 there, as with 0).
layout(constant_id = 129) const uint  WALL_CROSSING_PRESSURE_WALL = 0u;
const uint WALL_CROSSING_PRESSURE_WALL_EXTENSION  = 0u;
const uint WALL_CROSSING_PRESSURE_WALL_REFLECTION = 1u;
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

// the second-order continuation of the velocity (spec consts 118, 119 above): which terms take it, and whether
// the fused pass solves and stores grad v_i at all. All three are false with wall_boundary simple / adami and
// with both switches linear: every statement behind them is then removed at pipeline creation.
const bool WALL_CROSSING_VISCOUS_QUADRATIC =
    WALL_CROSSING_ACTIVE && (WALL_CROSSING_VISCOUS_EXTENSION == WALL_CROSSING_EXTENSION_QUADRATIC);
const bool WALL_CROSSING_CONTINUITY_QUADRATIC =
    WALL_CROSSING_ACTIVE && (WALL_CROSSING_CONTINUITY_EXTENSION == WALL_CROSSING_EXTENSION_QUADRATIC);
const bool WALL_CROSSING_VELOCITY_GRADIENT_ACTIVE =
    WALL_CROSSING_VISCOUS_QUADRATIC || WALL_CROSSING_CONTINUITY_QUADRATIC;
// numerics.volume_constraint.enabled (off by default; utils/volume_constraint_v7.py): the constraint on the
// fluid's total volume in a closed container. The host takes theta = (V_now - V_target) / V_target, V = the sum of
// m_i / rho_i over the fluid particles, every update interval and writes rate = theta / relaxation_time to
// volume_constraint_state.x; the fused pass adds rho_i * rate to every fluid particle's density rate. One relative
// rate for all of them: it moves the level of the pressure and makes no gradient.
layout(constant_id = 130) const uint  WALL_CROSSING_VOLUME_CONSTRAINT_ENABLED = 0u;
const bool WALL_CROSSING_VOLUME_CONSTRAINT = WALL_CROSSING_ACTIVE && (WALL_CROSSING_VOLUME_CONSTRAINT_ENABLED != 0u);

layout(std430, set = 0, binding = 15) buffer VolumeConstraintStateBuffer {
    // (rate in 1 / time, theta, 0, 0): written by the host at an update of the volume constraint, zero until the
    // first one. Read by the fused pass only with WALL_CROSSING_VOLUME_CONSTRAINT; not permuted by defrag, not
    // carried by the transport (every slab's host writes the same value).
    vec4 volume_constraint_state;
};

// the pairwise geometry and the two reflections (spec consts 128, 129, 116 = 3 above). All three are false with
// wall_boundary simple / adami and with the defaults: every statement behind them is then removed at pipeline
// creation.
const bool WALL_CROSSING_PAIRWISE =
    WALL_CROSSING_ACTIVE && (WALL_CROSSING_GEOMETRY == WALL_CROSSING_GEOMETRY_PAIRWISE);
const bool WALL_CROSSING_PRESSURE_REFLECTION =
    WALL_CROSSING_ACTIVE && (WALL_CROSSING_PRESSURE_WALL == WALL_CROSSING_PRESSURE_WALL_REFLECTION);
const bool WALL_CROSSING_DELTA_REFLECTION =
    WALL_CROSSING_ACTIVE && (WALL_CROSSING_DELTA_WALL == WALL_CROSSING_DELTA_WALL_REFLECTION);
// state of a particle's second-order continuation in a step (wall_crossing_velocity_gradient[pid * 3].w)
const float WALL_CROSSING_EXTENSION_STATE_NONE     = 0.0;   // no wall neighbour (or a wall particle): nothing solved
const float WALL_CROSSING_EXTENSION_STATE_VALID    = 1.0;   // grad v_i solved and stored
const float WALL_CROSSING_EXTENSION_STATE_FALLBACK = 2.0;   // fell back to the first-order rule for this step

layout(std430, set = 0, binding = 14) buffer WallCrossingVelocityGradientBuffer {
    // crossing with a quadratic extension: per fluid particle grad v_i, column by column, and the state:
    //   [pid * 3 + b] = (d v_x / d x_b, d v_y / d x_b, d v_z / d x_b, b == 0 ? state : 0)     b = 0, 1, 2
    // (2-D: every z entry is 0). Written by the fused correction + density pass (a wall particle writes state 0
    // to [pid * 3]), read by force of the same step for the particle itself only. Transient like
    // wall_crossing_fluid_inverse: defrag does not permute it, the transport does not carry it, it is not part
    // of the restart state. A 48-byte stub that nothing reads or writes when both extensions are linear.
    vec4 wall_crossing_velocity_gradient[];
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

// n(r): the unit vector from a point outside the box to the nearest point of the box (the inward normal of the
// wall a wall particle belongs to; towards the edge or corner in an edge or corner block); zero inside the box
vec3 wall_crossing_inward_normal(vec3 position) {
    vec3 lower  = vec3(WALL_CROSSING_BOX_LOWER_X, WALL_CROSSING_BOX_LOWER_Y, WALL_CROSSING_BOX_LOWER_Z);
    vec3 upper  = vec3(WALL_CROSSING_BOX_UPPER_X, WALL_CROSSING_BOX_UPPER_Y, WALL_CROSSING_BOX_UPPER_Z);
    vec3 to_box = clamp(position, lower, upper) - position;
    if (DIMENSION == 2u) {
        to_box.z = 0.0;
    }
    float depth = length(to_box);
    return (depth > 0.0) ? to_box / depth : vec3(0.0);
}

// phi_i of a fluid particle with the floor
float wall_crossing_floored_distance(vec3 fluid_position) {
    return max(wall_crossing_distance_to_wall(fluid_position), WALL_CROSSING_DISTANCE_FLOOR);
}

// 1 / t_ij = 1 + d_j / phi_i
float wall_crossing_extrapolation_factor(float fluid_wall_distance, vec3 wall_position) {
    return 1.0 + wall_crossing_depth_behind_wall(wall_position) / fluid_wall_distance;
}

// geometry pairwise: the geometry of one pair, a fluid particle i (inside the box) and a wall particle j (outside
// it). THE function of the pairwise geometry: another geometry than the axis-aligned box replaces it alone.
//   - j lies outside the box on one axis (a face block), two (an edge block; a corner block in 2-D) or three (a
//     corner block). Each such axis a names a face of the box, at face_a.
//   - The segment i -> j leaves the box through the face it reaches first: the axis a* with the smallest
//         s_a = (face_a - r_i,a) / (r_j,a - r_i,a) = phi_a / (phi_a + d_a),
//     phi_a = the distance of i to face a (0 for a particle that has left the box there), d_a = the depth of j
//     behind it. Equal s_a: the first axis in the order x, y, z.
//   crossed_wall_normal  n_ij: the inward unit normal of face a*
//   depth_ratio          d_ij / phi_ij, phi_ij = max(phi_a*, WALL_CROSSING_DISTANCE_FLOOR), d_ij = d_a*
//                        (1 / t_ij = 1 + depth_ratio)
//   mirror_displacement  r_j - r_j', r_j' = r_j reflected in every face it lies outside of (twice the vector from
//                        the nearest point of the box to j)
// A point that is not outside the box gets zero for all three.
void wall_crossing_pair_geometry(vec3 fluid_position, vec3 wall_position, out vec3 crossed_wall_normal,
                                 out float depth_ratio, out vec3 mirror_displacement) {
    vec3 lower = vec3(WALL_CROSSING_BOX_LOWER_X, WALL_CROSSING_BOX_LOWER_Y, WALL_CROSSING_BOX_LOWER_Z);
    vec3 upper = vec3(WALL_CROSSING_BOX_UPPER_X, WALL_CROSSING_BOX_UPPER_Y, WALL_CROSSING_BOX_UPPER_Z);
    vec3 nearest_box_point = clamp(wall_position, lower, upper);    // face_a on the axes j is outside on
    vec3 outside = wall_position - nearest_box_point;               // signed depth on those axes, 0 on the others
    if (DIMENSION == 2u) {
        outside.z = 0.0;
    }
    mirror_displacement = 2.0 * outside;
    crossed_wall_normal = vec3(0.0);
    depth_ratio = 0.0;
    float smallest_crossing = 2.0;                                  // every s_a is in [0, 1)
    for (int axis_index = 0; axis_index < 3; axis_index++) {
        float depth = abs(outside[axis_index]);
        if (depth > 0.0) {
            float outward_sign = sign(outside[axis_index]);
            // distance of i to this face, positive inside the box
            float inward_distance = outward_sign * (nearest_box_point[axis_index] - fluid_position[axis_index]);
            float clamped_distance = max(inward_distance, 0.0);
            float crossing = clamped_distance / (clamped_distance + depth);
            if (crossing < smallest_crossing) {
                smallest_crossing = crossing;
                depth_ratio = depth / max(inward_distance, WALL_CROSSING_DISTANCE_FLOOR);
                crossed_wall_normal = vec3(0.0);
                crossed_wall_normal[axis_index] = -outward_sign;
            }
        }
    }
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

// a_j = d_j / phi_i (the extrapolation factor above is 1 + a_j)
float wall_crossing_depth_ratio(float fluid_wall_distance, vec3 wall_position) {
    return wall_crossing_depth_behind_wall(wall_position) / fluid_wall_distance;
}

// The second-order continuation's system of one particle: grad v_i = M_0 L_i (I - K_raw L_i)^-1.
//   velocity_matrix          = M_0   (rows: velocity component, columns: direction; GLSL m[column][row])
//   wall_displacement_matrix = K_raw
//   correction_inverse       = L_i as force reads it back (the symmetric matrix of the six stored entries)
// system_determinant = det(I - K_raw L_i). false = fall back to the first-order rule (the determinant below
// 0.5, a NaN included, or a solution that is not finite); velocity_gradient is then zero.
bool wall_crossing_solve_velocity_gradient(mat3 velocity_matrix, mat3 wall_displacement_matrix,
                                           mat3 correction_inverse, out mat3 velocity_gradient,
                                           out float system_determinant) {
    mat3 system_matrix = mat3(1.0) - wall_displacement_matrix * correction_inverse;
    system_determinant = determinant(system_matrix);
    velocity_gradient = mat3(0.0);
    if (!(system_determinant >= 0.5)) {
        return false;
    }
    mat3 solution = velocity_matrix * correction_inverse * inverse(system_matrix);
    bool finite = true;
    for (int column_index = 0; column_index < 3; column_index++) {
        if (any(isnan(solution[column_index])) || any(isinf(solution[column_index]))) {
            finite = false;
        }
    }
    if (!finite) {
        return false;
    }
    velocity_gradient = solution;
    return true;
}

// trace of a matrix product's diagonal: tr(matrix)
float wall_crossing_trace(mat3 matrix) {
    return matrix[0][0] + matrix[1][1] + matrix[2][2];
}
#endif  // WALL_CROSSING
