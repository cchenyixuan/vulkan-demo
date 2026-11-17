// ============================================================================
// density_budget.glsl - DIAGNOSTIC (V7_DENSITY_BUDGET=1), not part of any wall
// mode. Included by correction_density.comp only, and only in its DENSITY_BUDGET
// compile variants (compile_shaders_v7.py: correction_density_budget.comp.spv,
// correction_density_wall_crossing_budget.comp.spv), so every other SPIR-V file
// keeps its bytes.
//
// What it records: the density rate of a FLUID particle split into four terms,
//     .x  continuity (drift), fluid neighbours
//     .y  continuity (drift), wall neighbours
//     .z  density diffusion (delta term), fluid neighbours
//     .w  density diffusion (delta term), wall neighbours (wall_boundary
//         crossing: the continuation applied after the loop, or the pair term
//         of the diagnostic delta_wall values; simple / adami: the pair term
//         with the wall's stored rho0)
// each contracted with the stored L_i as the pass contracts the sum. It adds
// TIMESTEP x the four terms to
//     density_budget[(particle_id * DENSITY_BUDGET_REGIONS + region) * 2]
// every time the pass runs, region = where the particle is at that step, and
// to the .x of the next entry what the stored density actually changed by in
// that step: the float32 the pass stored minus rho_n (an exact difference). The
// stored density is a float32 near rho0, so rho_n + TIMESTEP x rate is rounded
// to its grid (6.1e-5 kg/m^3 at 1000 kg/m^3): applied - TIMESTEP x (sum of the
// four terms) is that rounding, accumulated.
//
// The pass itself is not touched: the four terms are taken in a neighbour
// traversal of their own (density_budget_pass), called after the pass has
// stored its results, from the pass's inputs and from what it stored. Taken
// inside the pass's loop they change how the driver contracts the loop's
// multiply-adds, and the pass's results lose their bits after a few hundred
// steps. So the state a budget build computes is the state of the build
// without it, and the four terms are the pass's pair expressions written a
// second time (their sum agrees with the stored rate to float32 rounding).
//
// The buffer is not permuted by defrag and not carried by the transport: only
// its sum over all slots means anything (per region and term), which a defrag
// does not change. The host reads it at a sample and sums in float64.
//
// Regions (2-D box, the fluid box of wall_boundary crossing: the fluid
// particles' bounding box widened by one particle radius): which wall planes
// lie within the kernel support radius h of the particle,
//     region = horizontal + 3 x vertical,
//     horizontal = 0 none, 1 lower x (left), 2 upper x (right)
//     vertical   = 0 none, 1 lower y (bottom), 2 upper y (top: the cavity's lid)
// i.e. 1 left, 2 right, 3 bottom, 4 bottom-left, 5 bottom-right, 6 top,
// 7 top-left, 8 top-right; and the interior (no wall plane within h) by the
// distance g to the nearest wall plane: 9 for h <= g < 2h, 10 for 2h <= g < 4h,
// 11 for 4h <= g < 8h, 0 for g >= 8h (the core). z is ignored (a 3-D box is
// binned by x and y).
// ============================================================================

#ifndef WALL_CROSSING
// the box in x and y, the ids wall_boundary crossing uses (wall_boundary.glsl declares them in its variants)
layout(constant_id = 122) const float WALL_CROSSING_BOX_LOWER_X = 0.0;
layout(constant_id = 123) const float WALL_CROSSING_BOX_LOWER_Y = 0.0;
layout(constant_id = 125) const float WALL_CROSSING_BOX_UPPER_X = 1.0;
layout(constant_id = 126) const float WALL_CROSSING_BOX_UPPER_Y = 1.0;
#endif

const uint DENSITY_BUDGET_REGIONS = 12u;

layout(std430, set = 0, binding = 13) buffer DensityBudgetBuffer {
    // [(pid * DENSITY_BUDGET_REGIONS + region) * 2]     = accumulated TIMESTEP x (drift fluid, drift wall,
    //                                                     diffusion fluid, diffusion wall)
    // [(pid * DENSITY_BUDGET_REGIONS + region) * 2 + 1] = (accumulated change of the stored density, 0, 0, 0)
    // zeroed by the host at start-up; exists only with V7_DENSITY_BUDGET=1
    vec4 density_budget[];
};

uint density_budget_region(vec3 position) {
    uint horizontal = (position.x - WALL_CROSSING_BOX_LOWER_X < SMOOTHING_LENGTH) ? 1u
                    : ((WALL_CROSSING_BOX_UPPER_X - position.x < SMOOTHING_LENGTH) ? 2u : 0u);
    uint vertical   = (position.y - WALL_CROSSING_BOX_LOWER_Y < SMOOTHING_LENGTH) ? 1u
                    : ((WALL_CROSSING_BOX_UPPER_Y - position.y < SMOOTHING_LENGTH) ? 2u : 0u);
    uint near_wall = horizontal + 3u * vertical;
    if (near_wall != 0u) {
        return near_wall;
    }
    float gap = min(min(position.x - WALL_CROSSING_BOX_LOWER_X, WALL_CROSSING_BOX_UPPER_X - position.x),
                    min(position.y - WALL_CROSSING_BOX_LOWER_Y, WALL_CROSSING_BOX_UPPER_Y - position.y));
    if (gap >= 8.0 * SMOOTHING_LENGTH) return 0u;
    if (gap >= 4.0 * SMOOTHING_LENGTH) return 11u;
    if (gap >= 2.0 * SMOOTHING_LENGTH) return 10u;
    return 9u;
}

// S += q (x) grad W, symmetric part (correction_density.comp's statement, on one of the budget's sums)
void density_budget_add(vec3 rate_vector, vec3 kernel_gradient_value, inout vec3 rate_diagonal,
                        inout vec3 rate_off_diagonal) {
    if (DIMENSION == 2u) {
        rate_diagonal.xy    += rate_vector.xy * kernel_gradient_value.xy;
        rate_off_diagonal.x += rate_vector.x * kernel_gradient_value.y
                             + rate_vector.y * kernel_gradient_value.x;
    } else {
        rate_diagonal     += rate_vector * kernel_gradient_value;
        rate_off_diagonal += rate_vector.xxy * kernel_gradient_value.yzz
                           + rate_vector.yzz * kernel_gradient_value.xxy;
    }
}

// L_i : S with the six stored entries of L_i, the contraction of correction_density.comp's density epilogue
float density_budget_contract(vec3 rate_diagonal, vec3 rate_off_diagonal, vec3 inverse_diagonal,
                              vec3 inverse_off_diagonal) {
    float rate = inverse_diagonal.x * rate_diagonal.x
               + inverse_diagonal.y * rate_diagonal.y
               + inverse_off_diagonal.x * rate_off_diagonal.x;
    if (DIMENSION != 2u) {
        rate += inverse_diagonal.z * rate_diagonal.z
              + inverse_off_diagonal.y * rate_off_diagonal.y
              + inverse_off_diagonal.z * rate_off_diagonal.z;
    }
    return rate;
}

// The four terms of a fluid particle's density rate in a neighbour traversal of their own, and the change its
// stored density took, into the particle's region. Called after the pass has stored its results. The pair set and
// the pair expressions are the pass's (density.comp's; with wall_boundary crossing the wall rules of
// wall_boundary.glsl, the diagnostic switches included, and the walls' continuation term from the traversal's
// own A_f and g_f: nothing the pass computed for that term is read back, so no intermediate value is shared).
// inverse_diagonal / inverse_off_diagonal: the six entries of L_i the pass contracted with.
void density_budget_pass(uint self_particle_id, vec3 self_position, ivec3 self_voxel_coord,
                         vec3 self_velocity, float self_density, float self_stored_density,
                         vec3 inverse_diagonal, vec3 inverse_off_diagonal, float applied_change) {
    float diffusion_coefficient = 2.0 * DELTA_COEFFICIENT * SMOOTHING_LENGTH * SPEED_OF_SOUND;
    vec3 drift_fluid_diagonal         = vec3(0.0);
    vec3 drift_fluid_off_diagonal     = vec3(0.0);
    vec3 drift_wall_diagonal          = vec3(0.0);
    vec3 drift_wall_off_diagonal      = vec3(0.0);
    vec3 diffusion_fluid_diagonal     = vec3(0.0);
    vec3 diffusion_fluid_off_diagonal = vec3(0.0);
    vec3 diffusion_wall_diagonal      = vec3(0.0);
    vec3 diffusion_wall_off_diagonal  = vec3(0.0);
#ifdef WALL_CROSSING
    // wall_boundary crossing: phi_i with the floor, the walls' sum of the continuation term, the density gradient
    // of the diagnostic delta_wall body_force (correction_density.comp's statements)
    float self_wall_distance = 1.0;
    vec3  wall_diffusion_vector = vec3(0.0);
    mat3  fluid_correction_matrix = mat3(0.0);         // A_f: fluid neighbours only
    vec3  fluid_density_gradient_sum = vec3(0.0);      // g_f: fluid neighbours only
    vec3  wall_body_force_density_gradient = vec3(0.0);
    if (WALL_CROSSING_ACTIVE) {
        self_wall_distance = wall_crossing_floored_distance(self_position);
        if (WALL_CROSSING_DELTA_WALL == WALL_CROSSING_DELTA_WALL_BODY_FORCE) {
            MaterialParameters self_params = material_parameters[material[self_particle_id]];
            float pressure_derivative_by_density =
                (POWER_PARAMETER * self_params.eos_constant / self_params.rest_density)
                * pow(self_density / self_params.rest_density, POWER_PARAMETER - 1.0);
            wall_body_force_density_gradient = (self_density / pressure_derivative_by_density)
                                             * vec3(GRAVITY_X, GRAVITY_Y, GRAVITY_Z);
        }
    }
#endif
    int neighbor_z_range = int(NEIGHBOR_Z_RANGE);
    for (int delta_x = -1; delta_x <= 1; delta_x++) {
        for (int delta_z = -neighbor_z_range; delta_z <= neighbor_z_range; delta_z++) {
            for (int delta_y = -1; delta_y <= 1; delta_y++) {
                ivec3 neighbor_coord = self_voxel_coord + ivec3(delta_x, delta_y, delta_z);
                if (!in_own_grid(neighbor_coord)) continue;
                uint neighbor_voxel_id = own_voxel_id_of(neighbor_coord);
                uint neighbor_voxel_particle_count = inside_particle_count[neighbor_voxel_id];
                for (uint slot_index = 0u; slot_index < neighbor_voxel_particle_count; slot_index++) {
                    uint neighbor_particle_id =
                        inside_particle_index[neighbor_voxel_id * MAX_PARTICLES_PER_VOXEL + slot_index];
                    if (neighbor_particle_id == INSIDE_SLOT_EMPTY) break;
                    if (neighbor_particle_id == self_particle_id) continue;
                    vec3 neighbor_position = position_voxel_id[neighbor_particle_id].xyz;
                    vec3 position_difference_neighbor_to_self = self_position - neighbor_position;
                    float distance = length(position_difference_neighbor_to_self);
                    if (distance >= SMOOTHING_LENGTH || distance < 1e-12) continue;
                    vec4  neighbor_velocity_mass  = velocity_mass[neighbor_particle_id];
                    float neighbor_stored_density = density_pressure[neighbor_particle_id].x;
                    float neighbor_volume = neighbor_velocity_mass.w / density_from_stored(neighbor_stored_density);
                    vec3  kernel_gradient_value = evaluate_kernel_gradient(position_difference_neighbor_to_self,
                                                                           distance);
                    vec3  position_difference_self_to_neighbor = -position_difference_neighbor_to_self;
                    bool  neighbor_is_wall =
                        material_parameters[material[neighbor_particle_id]].kind == MATERIAL_BOUNDARY;
                    vec3  drift_vector = (-self_density * neighbor_volume)
                                       * (neighbor_velocity_mass.xyz - self_velocity);
                    // the pair's diffusion: psi_ij = rho_j - rho_i (a wall of simple / adami stores rho0)
                    bool  diffuses_as_pair = true;
                    float psi_ij = neighbor_stored_density - self_stored_density;
#ifdef WALL_CROSSING
                    if (WALL_CROSSING_ACTIVE && !neighbor_is_wall) {
                        fluid_correction_matrix += neighbor_volume
                            * outerProduct(position_difference_self_to_neighbor, kernel_gradient_value);
                        if (DIMENSION == 2u) {
                            fluid_density_gradient_sum.xy += neighbor_volume * psi_ij * kernel_gradient_value.xy;
                        } else {
                            fluid_density_gradient_sum += neighbor_volume * psi_ij * kernel_gradient_value;
                        }
                    }
                    if (WALL_CROSSING_ACTIVE && neighbor_is_wall) {
                        if (WALL_CROSSING_CONTINUITY_WALL_COMPONENT
                                == WALL_CROSSING_CONTINUITY_WALL_COMPONENT_NORMAL) {
                            vec3 wall_inward_normal = wall_crossing_inward_normal(neighbor_position);
                            drift_vector = dot(drift_vector, wall_inward_normal) * wall_inward_normal;
                        }
                        if (WALL_CROSSING_CONTINUITY == WALL_CROSSING_CONTINUITY_EXTRAPOLATE) {
                            drift_vector *= wall_crossing_extrapolation_factor(self_wall_distance, neighbor_position);
                        }
                        if (WALL_CROSSING_DELTA_WALL == WALL_CROSSING_DELTA_WALL_BODY_FORCE) {
                            psi_ij = dot(wall_body_force_density_gradient, position_difference_self_to_neighbor);
                        } else if (WALL_CROSSING_DELTA_WALL == WALL_CROSSING_DELTA_WALL_EXTENSION) {
                            // the continuation: not a pair term, the walls' sum for the term after the loop
                            diffuses_as_pair = false;
                            wall_diffusion_vector += (neighbor_volume * distance * distance
                                                      / (distance * distance + EPS_H_SQUARED))
                                                   * kernel_gradient_value;
                        }
                    }
#endif
                    vec3 diffusion_vector = vec3(0.0);
                    if (USE_DENSITY_DIFFUSION && diffuses_as_pair) {
                        diffusion_vector = (diffusion_coefficient * psi_ij * neighbor_volume
                                            / (distance * distance + EPS_H_SQUARED))
                                         * position_difference_self_to_neighbor;
                    }
                    if (neighbor_is_wall) {
                        density_budget_add(drift_vector, kernel_gradient_value,
                                           drift_wall_diagonal, drift_wall_off_diagonal);
                        density_budget_add(diffusion_vector, kernel_gradient_value,
                                           diffusion_wall_diagonal, diffusion_wall_off_diagonal);
                    } else {
                        density_budget_add(drift_vector, kernel_gradient_value,
                                           drift_fluid_diagonal, drift_fluid_off_diagonal);
                        density_budget_add(diffusion_vector, kernel_gradient_value,
                                           diffusion_fluid_diagonal, diffusion_fluid_off_diagonal);
                    }
                }
            }
        }
    }
    vec4 budget_terms = vec4(
        density_budget_contract(drift_fluid_diagonal, drift_fluid_off_diagonal,
                                inverse_diagonal, inverse_off_diagonal),
        density_budget_contract(drift_wall_diagonal, drift_wall_off_diagonal,
                                inverse_diagonal, inverse_off_diagonal),
        density_budget_contract(diffusion_fluid_diagonal, diffusion_fluid_off_diagonal,
                                inverse_diagonal, inverse_off_diagonal),
        density_budget_contract(diffusion_wall_diagonal, diffusion_wall_off_diagonal,
                                inverse_diagonal, inverse_off_diagonal));
#ifdef WALL_CROSSING
    // wall_boundary crossing, delta_wall extension: 2 delta h c0 G_i . (L_i sum_wall V_j r^2 / (r^2 + eps^2) grad W),
    // G_i = A_f^-1 g_f with the stored (symmetric) entries of A_f^-1, zero on a fallback of A_f (the pass's rule)
    if (WALL_CROSSING_ACTIVE && USE_DENSITY_DIFFUSION
            && WALL_CROSSING_DELTA_WALL == WALL_CROSSING_DELTA_WALL_EXTENSION) {
        mat3 fluid_correction_inverse;
        if (wall_crossing_invert_fluid_matrix(fluid_correction_matrix, fluid_correction_inverse)) {
            vec3 fluid_density_gradient =
                wall_crossing_symmetric_matrix(
                    vec4(fluid_correction_inverse[0][0], fluid_correction_inverse[1][1],
                         fluid_correction_inverse[2][2], fluid_correction_inverse[0][1]),
                    vec4(fluid_correction_inverse[0][2], fluid_correction_inverse[1][2], 0.0, 1.0))
                * fluid_density_gradient_sum;
            mat3 stored_correction_inverse = mat3(
                vec3(inverse_diagonal.x, inverse_off_diagonal.x, inverse_off_diagonal.y),
                vec3(inverse_off_diagonal.x, inverse_diagonal.y, inverse_off_diagonal.z),
                vec3(inverse_off_diagonal.y, inverse_off_diagonal.z, inverse_diagonal.z));
            budget_terms.w += diffusion_coefficient
                            * dot(fluid_density_gradient, stored_correction_inverse * wall_diffusion_vector);
        }
    }
#endif
    uint budget_slot = (self_particle_id * DENSITY_BUDGET_REGIONS + density_budget_region(self_position)) * 2u;
    density_budget[budget_slot]        += TIMESTEP * budget_terms;
    density_budget[budget_slot + 1u].x += applied_change;
}
