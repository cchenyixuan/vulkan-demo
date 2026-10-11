// ============================================================================
// deep_wall_skip.glsl - E39 B4 (audit H06): deep walls skip correction and
// the density neighbour loop (V7_DEEP_WALL_SKIP). Included by
// deep_wall_marker.comp and by the DEEP_WALL_SKIP variants of correction.comp
// and density.comp (compile_shaders_v7.py builds correction.comp /
// density.comp a second time with -DDEEP_WALL_SKIP into
// correction_deep_wall_skip.comp.spv / density_deep_wall_skip.comp.spv), after
// common.glsl / helpers.glsl, so correction.comp.spv / density.comp.spv and
// every other module stay byte-identical.
//
// Who reads a wall particle's correction output (common.glsl kinds; wall =
// MATERIAL_BOUNDARY, the lid included):
//   - its own density (density.comp, self L only, inside the neighbour loop
//     after the wall-wall `continue` and the support test; no neighbour L or
//     density gradient is read anywhere, the full psi_ij term is not
//     implemented);
//   - force.comp returns before reading it (BOUNDARY), wall_extrapolate.comp
//     (adami) never reads it;
//   - ghost_send copies it into the outbox only for the two seam columns
//     (band columns, never skipped), install_migrations only for migrants
//     (walls do not move), defrag moves it with the particle.
// So a wall whose density adds nothing this step (no neighbour that density
// counts: a listed, non-BOUNDARY particle with 1e-12 <= r < h in the 3^d voxels
// its loop visits) has a dead correction output, and its density loop is a
// no-op: drift and diffusion stay 0, and the kernel's epilogue stores
// (stored_rest_density(rho0), tait_pressure(rho_n + dt * 0)) to scratch, which
// the variant still runs (the scratch -> primary copy publishes the whole own
// range and defrag does not move scratch). Its position, mass and stored rho
// stay in the voxel lists: active walls' KCG matrix reads them.
//
// The criterion (conservative, exact on what the loops read):
//   presence[v] = 1 iff voxel v's inside list holds a particle whose kind is not
//                 MATERIAL_BOUNDARY (the kinds a wall's density counts: FLUID,
//                 ROTOR; INLET is never listed; unknown kinds count too);
//   deep[v]     = 1 iff every in-grid voxel of v's 3^d neighbourhood (the voxels
//                 the correction / density loops visit, in_own_grid) is an OWN
//                 voxel with presence 0 (ghost voxels count as present);
//   skip        = deep[voxel of self] && kind(self) == MATERIAL_BOUNDARY && self
//                 outside the band of width DEEP_WALL_SKIP_BAND_WIDTH.
// A skipped wall's density loop would visit exactly those lists, so it finds no
// counted neighbour: the skip changes no consumed output whatever the
// geometry. (The voxel edge is SMOOTHING_LENGTH = the support radius:
// predict / initialize_voxelization bin with floor((x - origin) / h), so the
// lists hold every neighbour within h, up to float rounding of the binning,
// which the full loop shares.)
//
// deep_wall_marker.comp builds presence and deep every step from that step's
// lists (after update_voxel): K = 1 in front of correction (phase B, the
// single-cmd step, the bootstrap), K >= 2 at the start of phase B. Phase B's
// interior kernels read only columns >= 1 from the seam (bands >= 2), whose
// lists are final after update_voxel (migrants land in the seam column in
// phase C, ghost lists are not read); every own voxel is rewritten each step,
// so no clear pass and no stale value after a restart or a defrag.
//
// Which pipelines skip (simulator_v7, spec constant 111 on the variant
// modules): correction_interior / density_deep_interior (phase B, every K;
// K = 1: the whole domain) and correction_all / density_all on a slab without
// peers (bootstrap, single-cmd step); with E39 B1 (V7_FUSED_CORRECTION_DENSITY)
// their fused counterparts correction_density_interior / correction_density_all
// (correction_density.comp's DEEP_WALL_SKIP variant). Band / boundary / compact
// pipelines use the plain modules and never skip. Correction and
// density take the same decision: both test the band of width
// DEEP_WALL_SKIP_BAND_WIDTH = density's band (V7_BAND_WIDTHS d >= c), so
// correction_interior never skips a particle that density_boundary computes in
// phase C (with 2,3,4 or the compact dispatch: column 2), and a particle
// density_deep_interior skips was skipped by correction_interior.
//
// Spec constants 111-114 are B4's own (local to the variants and the marker).
// ============================================================================

#ifndef SPH_DEEP_WALL_SKIP_GLSL_INCLUDED
#define SPH_DEEP_WALL_SKIP_GLSL_INCLUDED

// Band (voxel columns per side with a peer) inside which nothing is skipped:
// the density band width of the slab (simulator_v7: band_widths[1]).
layout(constant_id = 111) const uint DEEP_WALL_SKIP_BAND_WIDTH = 0u;
// V7_DEEP_WALL_CHECK: every skipped wall also runs density's neighbour test and
// counts what would have made it active into overflow_deep_wall_skip_count.
layout(constant_id = 112) const bool DEEP_WALL_CHECK = false;
// Record the decision (frame_stamp + 1 per skipped particle and kernel) in
// deep_wall_skip_record: V7_DEEP_WALL_CHECK=1, or the canonical_dump hook.
layout(constant_id = 113) const bool DEEP_WALL_RECORD = false;

const uint DEEP_WALL_KERNEL_CORRECTION = 0u;
const uint DEEP_WALL_KERNEL_DENSITY    = 1u;

layout(std430, set = 1, binding = 8) buffer DeepWallVoxelFlagBuffer {
    // [voxel_id] = presence, [TOTAL_VOXEL_COUNT + 1 + voxel_id] = deep (own
    // voxels written every step by deep_wall_marker.comp; ghost entries stay 0).
    uint deep_wall_voxel_flag[];
};

layout(std430, set = 3, binding = 10) buffer DeepWallSkipRecordBuffer {
    // [particle_id * 2 + kernel] = frame_stamp + 1 of the last step in which
    // that kernel skipped the particle (DEEP_WALL_RECORD; a 16-byte stub
    // otherwise). Not moved by defrag: the stamp makes older slots harmless.
    uint deep_wall_skip_record[];
};

uint deep_wall_deep_flag_index(uint voxel_id) {
    return TOTAL_VOXEL_COUNT + 1u + voxel_id;
}

// helpers.glsl in_boundary_band with the width DEEP_WALL_SKIP_BAND_WIDTH
bool deep_wall_in_skip_band(ivec3 coord) {
    int leading_x  = int(leading_ghost_x_thickness());
    int trailing_x = int(trailing_ghost_x_thickness());
    int range      = int(DEEP_WALL_SKIP_BAND_WIDTH);
    if (FAKE_BAND_COLUMN > 0u) {
        // diagnostic band in the interior of a single-GPU domain
        return coord.x >= int(FAKE_BAND_COLUMN) && coord.x < int(FAKE_BAND_COLUMN) + range;
    }

    bool near_leading  = leading_x  > 0 && coord.x <  leading_x + range;
    int  own_last_x    = int(GRID_DIMENSION_X) - 1 - trailing_x;
    bool near_trailing = trailing_x > 0 && coord.x >  own_last_x - range;

    return near_leading || near_trailing;
}

// The voxel half of the decision (the caller adds kind(self) == BOUNDARY).
bool deep_wall_voxel_skippable(ivec3 self_voxel_coord, uint self_voxel_id) {
    if (deep_wall_in_skip_band(self_voxel_coord)) return false;
    return deep_wall_voxel_flag[deep_wall_deep_flag_index(self_voxel_id)] != 0u;
}

// V7_DEEP_WALL_CHECK: density's neighbour test for a BOUNDARY self (the same
// voxels, list order, self / wall-wall / support conditions), counting the
// neighbours its sums would take. Must find none.
void deep_wall_check(uint self_particle_id, vec3 self_position, ivec3 self_voxel_coord) {
    uint active_neighbor_count = 0u;
    int neighbor_z_range = int(NEIGHBOR_Z_RANGE);
    for (int delta_x = -1; delta_x <= 1; delta_x++) {
        for (int delta_z = -neighbor_z_range; delta_z <= neighbor_z_range; delta_z++) {
            for (int delta_y = -1; delta_y <= 1; delta_y++) {
                ivec3 neighbor_coord = self_voxel_coord + ivec3(delta_x, delta_y, delta_z);
                if (!in_own_grid(neighbor_coord)) continue;

                uint neighbor_voxel_id = own_voxel_id_of(neighbor_coord);
                uint neighbor_voxel_particle_count = inside_particle_count[neighbor_voxel_id];

                for (uint slot_index = 0u; slot_index < neighbor_voxel_particle_count; slot_index++) {
                    uint neighbor_particle_id = inside_particle_index[neighbor_voxel_id * MAX_PARTICLES_PER_VOXEL + slot_index];
                    if (neighbor_particle_id == INSIDE_SLOT_EMPTY) break;
                    if (neighbor_particle_id == self_particle_id) continue;
                    if (material_parameters[material[neighbor_particle_id]].kind == MATERIAL_BOUNDARY) continue;

                    vec3  neighbor_position = position_voxel_id[neighbor_particle_id].xyz;
                    float distance = length(self_position - neighbor_position);
                    if (distance >= SMOOTHING_LENGTH || distance < 1e-12) continue;
                    active_neighbor_count++;
                }
            }
        }
    }
    if (active_neighbor_count > 0u) atomicAdd(overflow_deep_wall_skip_count, active_neighbor_count);
}

// Everything a skipped wall does besides skipping (nothing in production).
void deep_wall_on_skip(uint self_particle_id, vec3 self_position, ivec3 self_voxel_coord, uint kernel) {
    if (DEEP_WALL_RECORD) {
        deep_wall_skip_record[self_particle_id * 2u + kernel] = frame_stamp + 1u;
    }
    if (DEEP_WALL_CHECK) {
        deep_wall_check(self_particle_id, self_position, self_voxel_coord);
    }
}

#endif  // SPH_DEEP_WALL_SKIP_GLSL_INCLUDED
