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
//   2 = crossing: the crossing-fraction walls, in the WALL_CROSSING compile
//       variants only: see the section at the end of this file. The wall
//       particles are not in the particle pool (plan C): a static wall store
//       holds them, and the fused pass and force meet them in a second segment
//       of their neighbour loop.
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
// WALL_BOUNDARY = 2 = crossing (case.yaml numerics.wall_boundary: crossing;
// plan C, the static wall store, branch v7-crossing-wall-store). Compiled only
// into the WALL_CROSSING variants (compile_shaders_v7.py:
// correction_density_wall_crossing.comp.spv, force_wall_crossing.comp.spv), so
// every other module, and the plain builds of correction_density.comp /
// force.comp that simple and adami run, keep their bytes.
//
// The particle pool holds the fluid only. The wall particles live in a static
// wall store (utils/wall_store_v7.py): one 32-byte record per wall particle
// (position, volume, the planes it lies behind, body, wall material), sorted by
// voxel, with the voxel offsets, a plane table, a body table and a wall
// material table (the boundary rules). Both passes run two segments for a
// particle i near a wall:
//   - the fluid segment: the 3^d voxels' pool neighbours, the plain code (the
//     correction matrix after it is A_f, over the fluid only; force also sums
//     g_f = sum_fluid V_j (p_j - p_i) grad W_ij);
//   - between the segments: the one-sided gradients, G = A_f^-1 g_f(rho) in the
//     fused pass (A_f^-1 stored for force), G_p = A_f^-1 g_f(p) in force;
//   - the wall segment: the 3^d voxels' wall records; every wall neighbour j
//     gets its values from the ONE rule (wall_crossing_rule_scalar_difference /
//     _velocity_difference) and enters the same pair terms as a fluid
//     neighbour (correction matrix, kernel sum, continuity + density diffusion;
//     pressure (TIC), viscosity, shifting).
//
// The rule, for a field f (the wall material table gives lambda, gamma, w):
//     f_j - f_i = lambda (1 + d/phi) (f_w - f_i)
//               + (1 - lambda) (gamma G_f . (r_j' - r_i) + w q . (r_j - r_j'))
// i.e. f_j = alpha f_i + beta f_w + gamma G_f . (r_j' - r_i) + q . (r_j - r_j')
// with lambda = 1 (prescribed value): alpha = -d/phi, beta = 1 + d/phi,
// gamma = q = 0; lambda = 0 (prescribed normal derivative): alpha = 1, beta = 0.
// d/phi = the pair's depth ratio, r_j' = r_j reflected in every plane j lies
// behind (wall_crossing_pair_geometry). The velocity splits into its normal
// and tangential components (normal = the crossed plane's), each with its own
// lambda (prescribed value, f_w = v_w; lambda = 0 gives v_j = v_i). The fields:
//   velocity, continuity equation (row 0) and viscous term (row 1): v_w of the
//     wall's body, v_w = V_b + omega_b x (r_j - c_b);
//   pressure (row 2): G_f = G_p, q = rho_i (g - a_w);
//   density (row 3): G_f = G, q = rho_i (g - a_w) / c^2(rho_i), c^2 = dp/drho of
//     the Tait EOS at rho_i.
// Table rows (wall_store_v7.rule_table): "default" = prescribed value for both
// velocities, prescribed normal derivative for pressure and density (the
// particle-pool version's default scheme); "continuity_normal" = the
// continuity equation's tangential velocity v_j = v_i.
// ============================================================================
const uint WALL_BOUNDARY_CROSSING = 2u;
const bool WALL_CROSSING_ACTIVE = (WALL_BOUNDARY == WALL_BOUNDARY_CROSSING);

// lower bound of phi = numerics.wall_crossing.distance_floor_factor (default 0.25) x the particle spacing
layout(constant_id = 121) const float WALL_CROSSING_DISTANCE_FLOOR = 0.001;
// numerics.volume_constraint.enabled (off by default; utils/volume_constraint_v7.py): the constraint on the
// fluid's total volume in a closed container. The host takes theta = (V_now - V_target) / V_target, V = the sum of
// m_i / rho_i over the fluid particles, every update interval and writes rate = theta / relaxation_time to
// volume_constraint_state.x; the fused pass adds rho_i * rate to every fluid particle's density rate. One relative
// rate for all of them: it moves the level of the pressure and makes no gradient.
layout(constant_id = 130) const uint  WALL_CROSSING_VOLUME_CONSTRAINT_ENABLED = 0u;
const bool WALL_CROSSING_VOLUME_CONSTRAINT = WALL_CROSSING_ACTIVE && (WALL_CROSSING_VOLUME_CONSTRAINT_ENABLED != 0u);
// the dispatch class of a pipeline (numerics.wall_crossing.split_dispatch):
//   0 = every particle of the dispatch, the near-wall flag decides whether the wall segment runs (organisation (A);
//       also (B)'s phase C band pipeline in band mode 1, and the bootstrap's pipelines);
//   1 = (B)'s interior pipeline: the particles of the voxels that are not near a wall; no wall segment compiled in;
//   2 = (B)'s near-wall pipeline: the particles of the near-wall voxels.
layout(constant_id = 131) const uint WALL_CROSSING_DISPATCH_CLASS = 0u;
const uint WALL_CROSSING_DISPATCH_EVERY_PARTICLE = 0u;
const uint WALL_CROSSING_DISPATCH_INTERIOR       = 1u;
const uint WALL_CROSSING_DISPATCH_NEAR_WALL      = 2u;
const bool WALL_CROSSING_WALL_SEGMENT_COMPILED =
    WALL_CROSSING_ACTIVE && (WALL_CROSSING_DISPATCH_CLASS != WALL_CROSSING_DISPATCH_INTERIOR);

layout(std430, set = 0, binding = 12) buffer WallCrossingFluidInverseBuffer {
    // per fluid particle near a wall, A_f^-1 (symmetric, the packing of correction_inverse) and a validity flag:
    //   [pid * 2]     = (m00, m11, m22, m01)
    //   [pid * 2 + 1] = (m02, m12, 0, valid)     valid = 1, or 0 when the inversion fell back (then G = G_p = 0)
    // Written by the fused pass between its segments, read by force of the same step for the particle itself
    // only. Transient: defrag does not permute it, the transport does not carry it, it is not part of the restart
    // state.
    vec4 wall_crossing_fluid_inverse[];
};

layout(std430, set = 0, binding = 15) buffer VolumeConstraintStateBuffer {
    // (rate in 1 / time, theta, 0, 0): written by the host at an update of the volume constraint, zero until the
    // first one. Read by the fused pass only with WALL_CROSSING_VOLUME_CONSTRAINT; not permuted by defrag, not
    // carried by the transport (every slab's host writes the same value).
    vec4 volume_constraint_state;
};

// ---- the static wall store (set 1 bindings 9-14; uploaded once per slab, never written by a kernel) ----
layout(std430, set = 1, binding = 9) buffer WallStoreNearWallVoxelBuffer {
    // per voxel id of the slab's extended grid: 1 iff a wall particle lies closer than h (+ h / 256) to its cube
    uint wall_store_near_wall_voxel_flag[];
};
layout(std430, set = 1, binding = 10) buffer WallStoreVoxelOffsetBuffer {
    // the wall records of voxel id v are [wall_voxel_offset[v], wall_voxel_offset[v + 1])
    uint wall_voxel_offset[];
};
struct WallParticle {
    vec4 position_volume;    // position, volume (m / rho of the particle-pool version)
    // xyz: the mirror displacement r_j - r_j' (fixed per wall particle: computed by the host in float32 exactly as the
    // shader did); w: the identity word's bits (floatBitsToUint): plane ids 3 x 6 bits (63 = none), body id 7 bits
    // (bits 18-24), wall material id 7 bits (bits 25-31)
    vec4 mirror_identity;
};
layout(std430, set = 1, binding = 11) buffer WallStoreParticleBuffer {
    WallParticle wall_particles[];
};
layout(std430, set = 1, binding = 12) buffer WallStorePlaneBuffer {
    // plane p: [2 p] = (inward unit normal, 0), [2 p + 1] = (a point of the plane, 0)
    vec4 wall_planes[];
};
layout(std430, set = 1, binding = 13) buffer WallStoreBodyBuffer {
    // body b: [5 b] linear velocity, [5 b + 1] angular velocity, [5 b + 2] rotation centre, [5 b + 3] linear
    // acceleration, [5 b + 4] angular acceleration (xyz, 0)
    vec4 wall_bodies[];
};
layout(std430, set = 1, binding = 14) buffer WallStoreRuleBuffer {
    // wall material m: [4 m + WALL_CROSSING_RULE_*] (the rule of that field, see the section comment)
    vec4 wall_rules[];
};
const uint WALL_CROSSING_RULE_FIELDS              = 4u;
const uint WALL_CROSSING_RULE_CONTINUITY_VELOCITY = 0u;   // (lambda_normal, lambda_tangential, 0, 0)
const uint WALL_CROSSING_RULE_VISCOUS_VELOCITY    = 1u;   // (lambda_normal, lambda_tangential, 0, 0)
const uint WALL_CROSSING_RULE_PRESSURE            = 2u;   // (lambda, gamma, w, prescribed value)
const uint WALL_CROSSING_RULE_DENSITY             = 3u;   // (lambda, gamma, w, prescribed value)
const uint WALL_CROSSING_PLANE_ID_BITS = 6u;
const uint WALL_CROSSING_PLANE_ID_NONE = 63u;
const uint WALL_CROSSING_PLANE_ID_FIELD = 0x3FFFFu;     // the three plane ids
const uint WALL_CROSSING_BODY_ID_SHIFT = 18u;
const uint WALL_CROSSING_MATERIAL_ID_SHIFT = 25u;

uint wall_crossing_plane_ids(uint identity) { return identity & WALL_CROSSING_PLANE_ID_FIELD; }
uint wall_crossing_body_id(uint identity) { return (identity >> WALL_CROSSING_BODY_ID_SHIFT) & 127u; }
uint wall_crossing_material_id(uint identity) { return identity >> WALL_CROSSING_MATERIAL_ID_SHIFT; }

// The kind of rule of every field and operator, fixed at pipeline creation (wall_store report, section on the
// registers): 0 = prescribed value, 1 = prescribed normal derivative (gamma = w = 1: G_f and the body-force term q). The
// host takes them from the rule set's row of the wall material table (every wall of a case has the same rule set), so
// the compiler folds every lambda, gamma and w; only a prescribed value is still read from the table (wall_rules
// [..].w), and only by a value rule. A new kind of rule is one more value of these constants.
//   134 / 135 velocity in the continuity equation, normal / tangential component; 136 / 137 the viscous term's;
//   138 pressure; 139 density.
// WALL_CROSSING_REVERT_RULE_TYPES (debug build, V7_WALL_CROSSING_RULE_TYPES=table) reads lambda, gamma, w from the table.
layout(constant_id = 134) const uint WALL_CROSSING_RULE_TYPE_CONTINUITY_NORMAL     = 0u;
layout(constant_id = 135) const uint WALL_CROSSING_RULE_TYPE_CONTINUITY_TANGENTIAL = 0u;
layout(constant_id = 136) const uint WALL_CROSSING_RULE_TYPE_VISCOUS_NORMAL        = 0u;
layout(constant_id = 137) const uint WALL_CROSSING_RULE_TYPE_VISCOUS_TANGENTIAL    = 0u;
layout(constant_id = 138) const uint WALL_CROSSING_RULE_TYPE_PRESSURE              = 1u;
layout(constant_id = 139) const uint WALL_CROSSING_RULE_TYPE_DENSITY               = 1u;
const uint WALL_CROSSING_RULE_TYPE_VALUE      = 0u;
const uint WALL_CROSSING_RULE_TYPE_DERIVATIVE = 1u;
// The motion of the wall bodies, fixed at pipeline creation: 0 = every body at rest or translating at a constant
// velocity (v_w = V_b, a_w = 0: one read of the body table), 1 = general (the host passes 1 when a body rotates or
// accelerates). WALL_CROSSING_REVERT_BODY_MOTION (debug build, V7_WALL_CROSSING_BODY_MOTION=general) ignores it.
layout(constant_id = 140) const uint WALL_CROSSING_BODY_MOTION = 1u;
const uint WALL_CROSSING_BODY_MOTION_TRANSLATION = 0u;

// spec const 142 (claude/wall_store_report.md section 10.5): where the fused pass takes the wall pairs of S of a
// particle near a wall. 0 (the default since wall store round 4, 2-D and 3-D, simulator_v7
// WALL_CROSSING_WALL_RATE_LOOP_BY_DIMENSION): in the wall segment's loop, beside the wall pairs of the correction
// matrix; 1 (round 2's item S, the default until round 4; V7_WALL_CROSSING_WALL_RATE_LOOP=after_store): a loop of
// their own after the epilogue's store (wall_crossing_wall_rate_segment), so that the correction matrix and L_i are not
// held through the pair work. Either way S takes the same pairs in the same order. The DIAGNOSTIC summed density
// diffusion sums after the store.
#ifdef WALL_CROSSING_DIFFUSION_SUMMED
const bool WALL_CROSSING_WALL_RATE_AFTER_STORE = true;
#else
layout(constant_id = 142) const uint WALL_CROSSING_WALL_RATE_LOOP = 0u;
const bool WALL_CROSSING_WALL_RATE_AFTER_STORE = WALL_CROSSING_WALL_RATE_LOOP == 1u;
#endif

float wall_crossing_rule_lambda(uint rule_type) {
    return (rule_type == WALL_CROSSING_RULE_TYPE_VALUE) ? 1.0 : 0.0;
}

// The rule of one field (WALL_CROSSING_RULE_*) of a wall material, in the table's form (lambda, gamma, w, value)
vec4 wall_crossing_rule(uint rule_base, uint field) {
#ifdef WALL_CROSSING_REVERT_RULE_TYPES
    return wall_rules[rule_base + field];
#else
    if (field == WALL_CROSSING_RULE_CONTINUITY_VELOCITY) {
        return vec4(wall_crossing_rule_lambda(WALL_CROSSING_RULE_TYPE_CONTINUITY_NORMAL),
                    wall_crossing_rule_lambda(WALL_CROSSING_RULE_TYPE_CONTINUITY_TANGENTIAL), 0.0, 0.0);
    }
    if (field == WALL_CROSSING_RULE_VISCOUS_VELOCITY) {
        return vec4(wall_crossing_rule_lambda(WALL_CROSSING_RULE_TYPE_VISCOUS_NORMAL),
                    wall_crossing_rule_lambda(WALL_CROSSING_RULE_TYPE_VISCOUS_TANGENTIAL), 0.0, 0.0);
    }
    uint rule_type = (field == WALL_CROSSING_RULE_PRESSURE) ? WALL_CROSSING_RULE_TYPE_PRESSURE
                                                            : WALL_CROSSING_RULE_TYPE_DENSITY;
    if (rule_type == WALL_CROSSING_RULE_TYPE_DERIVATIVE) {
        return vec4(0.0, 1.0, 1.0, 0.0);
    }
    return vec4(1.0, 0.0, 0.0, wall_rules[rule_base + field].w);
#endif
}

bool wall_crossing_voxel_near_wall(uint voxel_id) {
    return wall_store_near_wall_voxel_flag[voxel_id] != 0u;
}

// The geometry of the pair (fluid particle i, wall particle j) from the planes j lies behind (the plane table; on a
// box the particle-pool version's analytic box geometry gives the same values):
//   for every plane a of j: phi_a = n_a . (r_i - p_a) (0 for a particle that has left the box there),
//   d_a = n_a . (p_a - r_j) > 0, s_a = phi_a / (phi_a + d_a); the segment i -> j leaves through the plane with the
//   smallest s_a (equal s_a: the first in the record, i.e. the lowest plane id = the first axis in x, y, z).
//   crossed_wall_normal  n of that plane
//   depth_ratio          d / max(phi, WALL_CROSSING_DISTANCE_FLOOR)       (1 / t_ij = 1 + depth_ratio)
//   mirror_displacement  r_j - r_j', r_j' = r_j reflected in every plane it lies behind (r_j + 2 d_a n_a each)
void wall_crossing_pair_geometry(vec3 fluid_position, vec3 wall_position, uint plane_ids,
                                 out vec3 crossed_wall_normal, out float depth_ratio, inout vec3 mirror_displacement) {
    crossed_wall_normal = vec3(0.0);
    depth_ratio = 0.0;
#ifdef WALL_CROSSING_REVERT_MIRROR
    // DEBUG (V7_WALL_CROSSING_MIRROR=geometry): the mirror displacement summed here, plane by plane, as before it was
    // stored in the record
    mirror_displacement = vec3(0.0);
#endif
    float smallest_crossing = 2.0;                                  // every s_a is in [0, 1)
    for (uint slot_index = 0u; slot_index < 3u; slot_index++) {
        uint plane_id = (plane_ids >> (WALL_CROSSING_PLANE_ID_BITS * slot_index)) & WALL_CROSSING_PLANE_ID_NONE;
        if (plane_id == WALL_CROSSING_PLANE_ID_NONE) break;
        vec3 plane_normal = wall_planes[2u * plane_id].xyz;
        vec3 plane_point  = wall_planes[2u * plane_id + 1u].xyz;
        float depth = dot(plane_normal, plane_point - wall_position);
        if (depth > 0.0) {
#ifdef WALL_CROSSING_REVERT_MIRROR
            mirror_displacement -= (2.0 * depth) * plane_normal;
#endif
            float inward_distance  = dot(plane_normal, fluid_position - plane_point);
            float clamped_distance = max(inward_distance, 0.0);
            float crossing = clamped_distance / (clamped_distance + depth);
            if (crossing < smallest_crossing) {
                smallest_crossing   = crossing;
                depth_ratio         = depth / max(inward_distance, WALL_CROSSING_DISTANCE_FLOOR);
                crossed_wall_normal = plane_normal;
            }
        }
    }
}

// v_w and a_w of a wall body at a point
vec3 wall_crossing_body_velocity(uint body_id, vec3 position) {
#ifndef WALL_CROSSING_REVERT_BODY_MOTION
    if (WALL_CROSSING_BODY_MOTION == WALL_CROSSING_BODY_MOTION_TRANSLATION) {
        return wall_bodies[5u * body_id].xyz;
    }
#endif
    vec3 arm = position - wall_bodies[5u * body_id + 2u].xyz;
    return wall_bodies[5u * body_id].xyz + cross(wall_bodies[5u * body_id + 1u].xyz, arm);
}

vec3 wall_crossing_body_acceleration(uint body_id, vec3 position) {
#ifndef WALL_CROSSING_REVERT_BODY_MOTION
    if (WALL_CROSSING_BODY_MOTION == WALL_CROSSING_BODY_MOTION_TRANSLATION) {
        return vec3(0.0);
    }
#endif
    vec3 arm = position - wall_bodies[5u * body_id + 2u].xyz;
    vec3 angular_velocity = wall_bodies[5u * body_id + 1u].xyz;
    return wall_bodies[5u * body_id + 3u].xyz + cross(wall_bodies[5u * body_id + 4u].xyz, arm)
         + cross(angular_velocity, cross(angular_velocity, arm));
}

// The rule for a scalar field (pressure, density): f_j - f_i.
//   rule                        (lambda, gamma, w, prescribed value f_w) of the wall material table
//   extrapolation_factor        1 + d/phi
//   one_sided_gradient          G_f (A_f^-1 g_f of the field)
//   mirror_point_from_self      r_j' - r_i
//   body_force_gradient         q without its weight (rho_i (g - a_w), or that over c^2 for the density)
//   mirror_displacement         r_j - r_j'
float wall_crossing_rule_scalar_difference(vec4 rule, float extrapolation_factor, float self_value,
                                           vec3 one_sided_gradient, vec3 mirror_point_from_self,
                                           vec3 body_force_gradient, vec3 mirror_displacement) {
    float derivative_part = rule.y * dot(one_sided_gradient, mirror_point_from_self)
                          + rule.z * dot(body_force_gradient, mirror_displacement);
    if (rule.x == 0.0) {
        return derivative_part;
    }
    return rule.x * extrapolation_factor * (rule.w - self_value) + (1.0 - rule.x) * derivative_part;
}

// The rule for the velocity: v_j - v_i, the normal and tangential components (normal = the crossed plane's) each with
// its own lambda of a prescribed value f_w = v_w; lambda = 0 gives that component v_j = v_i.
vec3 wall_crossing_rule_velocity_difference(vec4 rule, float extrapolation_factor, vec3 wall_velocity,
                                            vec3 self_velocity, vec3 crossed_wall_normal) {
    vec3 velocity_difference = wall_velocity - self_velocity;
    if (rule.x == rule.y) {
        return (rule.x * extrapolation_factor) * velocity_difference;
    }
    vec3 normal_part = dot(velocity_difference, crossed_wall_normal) * crossed_wall_normal;
    return extrapolation_factor * (rule.x * normal_part + rule.y * (velocity_difference - normal_part));
}

// A_f^-1: the correction matrix's epilogue (correction.comp: xi on the diagonal, the 2-D block, the determinant
// threshold, the Frobenius cap) on the fluid-only matrix. false = fallback (too few fluid neighbours): the
// identity is returned and the caller takes G = G_p = 0.
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

// dp/drho of the Tait EOS at a density
float wall_crossing_sound_speed_squared(float density, MaterialParameters parameters) {
    return (POWER_PARAMETER * parameters.eos_constant / parameters.rest_density)
         * pow(density / parameters.rest_density, POWER_PARAMETER - 1.0);
}
#endif  // WALL_CROSSING
