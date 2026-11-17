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
// each contracted with the stored L_i exactly as the pass contracts the sum,
// and adds TIMESTEP x the four terms to
//     density_budget[particle_id * DENSITY_BUDGET_REGIONS + region]
// every time the pass runs, region = where the particle is at that step. The
// pass's results are not touched (the state a budget build computes is the
// state the build without it computes): the WALL_CROSSING build takes the four
// sums beside the pass's own in its loop; the plain build (simple / adami)
// takes them in a second neighbour traversal, density_budget_plain_pass, after
// the pass has stored its results.
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
// i.e. 0 interior, 1 left, 2 right, 3 bottom, 4 bottom-left, 5 bottom-right,
// 6 top, 7 top-left, 8 top-right. z is ignored (a 3-D box is binned by x and y).
// ============================================================================

#ifndef WALL_CROSSING
// the box in x and y, the ids wall_boundary crossing uses (wall_boundary.glsl declares them in its variants)
layout(constant_id = 122) const float WALL_CROSSING_BOX_LOWER_X = 0.0;
layout(constant_id = 123) const float WALL_CROSSING_BOX_LOWER_Y = 0.0;
layout(constant_id = 125) const float WALL_CROSSING_BOX_UPPER_X = 1.0;
layout(constant_id = 126) const float WALL_CROSSING_BOX_UPPER_Y = 1.0;
#endif

const uint DENSITY_BUDGET_REGIONS = 9u;

layout(std430, set = 0, binding = 13) buffer DensityBudgetBuffer {
    // [pid * DENSITY_BUDGET_REGIONS + region] = accumulated TIMESTEP x (drift fluid, drift wall, diffusion fluid,
    // diffusion wall); zeroed by the host at start-up; exists only with V7_DENSITY_BUDGET=1
    vec4 density_budget[];
};

uint density_budget_region(vec3 position) {
    uint horizontal = (position.x - WALL_CROSSING_BOX_LOWER_X < SMOOTHING_LENGTH) ? 1u
                    : ((WALL_CROSSING_BOX_UPPER_X - position.x < SMOOTHING_LENGTH) ? 2u : 0u);
    uint vertical   = (position.y - WALL_CROSSING_BOX_LOWER_Y < SMOOTHING_LENGTH) ? 1u
                    : ((WALL_CROSSING_BOX_UPPER_Y - position.y < SMOOTHING_LENGTH) ? 2u : 0u);
    return horizontal + 3u * vertical;
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

#ifndef WALL_CROSSING
// simple / adami: the four terms of a fluid particle in a neighbour traversal of their own (density.comp's pair
// set and pair expressions: a wall neighbour enters with its stored velocity and its stored rho0), contracted with
// the six stored entries of L_i, into the region the particle is in. Called after the pass has stored its results.
void density_budget_plain_pass(uint self_particle_id, vec3 self_position, ivec3 self_voxel_coord,
                               vec3 self_velocity, float self_density, float self_stored_density,
                               vec3 inverse_diagonal, vec3 inverse_off_diagonal) {
    float diffusion_coefficient = 2.0 * DELTA_COEFFICIENT * SMOOTHING_LENGTH * SPEED_OF_SOUND;
    vec3 drift_fluid_diagonal         = vec3(0.0);
    vec3 drift_fluid_off_diagonal     = vec3(0.0);
    vec3 drift_wall_diagonal          = vec3(0.0);
    vec3 drift_wall_off_diagonal      = vec3(0.0);
    vec3 diffusion_fluid_diagonal     = vec3(0.0);
    vec3 diffusion_fluid_off_diagonal = vec3(0.0);
    vec3 diffusion_wall_diagonal      = vec3(0.0);
    vec3 diffusion_wall_off_diagonal  = vec3(0.0);
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
                    vec3  drift_vector = (-self_density * neighbor_volume)
                                       * (neighbor_velocity_mass.xyz - self_velocity);
                    vec3  diffusion_vector = vec3(0.0);
                    if (USE_DENSITY_DIFFUSION) {
                        float psi_ij = neighbor_stored_density - self_stored_density;
                        diffusion_vector = (diffusion_coefficient * psi_ij * neighbor_volume
                                            / (distance * distance + EPS_H_SQUARED))
                                         * position_difference_self_to_neighbor;
                    }
                    if (material_parameters[material[neighbor_particle_id]].kind == MATERIAL_BOUNDARY) {
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
    density_budget[self_particle_id * DENSITY_BUDGET_REGIONS + density_budget_region(self_position)]
        += TIMESTEP * budget_terms;
}
#endif
