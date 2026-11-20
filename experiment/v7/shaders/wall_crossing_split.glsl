// ============================================================================
// wall_crossing_split.glsl - numerics.wall_crossing.split_dispatch (report section 22; on by default since section 25): the fused
// correction + density pass and force run as two dispatches each on a crossing slab,
//   - the INTERIOR build (correction_density.comp / force.comp -DWALL_CROSSING_INTERIOR): the plain code, none of
//     the mode's statements (so none of its registers), for the particles of the voxels that are NOT near a wall;
//     the fused pass adds only the volume constraint's rho_i * rate;
//   - the NEAR-WALL build (-DWALL_CROSSING -DWALL_CROSSING_NEAR_WALL): the mode's code for the particles of the
//     voxels near a wall, which tells a wall neighbour by its position (outside the fluid box) instead of reading
//     its material pair by pair.
// A voxel is near a wall when a voxel of its 3^d neighbourhood (in_own_grid, the neighbour loops' test) holds a
// wall particle; the support radius is the voxel size, so a particle of any other voxel has no wall neighbour and
// the plain code computes for it what the mode's code computes. Every wall particle lies in a near-wall voxel.
// The marker (wall_crossing_near_wall_marker.comp) fills the flags once the lists hold every wall particle the
// loops can meet (the walls do not move): a slab without peers after the initial registration, a slab with a peer
// after the first replica exchange, and after a restart.
// Included by those two builds of correction_density.comp / force.comp and by the marker only; every other
// module, the plain builds and the WALL_CROSSING builds keep their SPIR-V.
// ============================================================================

layout(std430, set = 1, binding = 9) buffer WallCrossingNearWallVoxelBuffer {
    // per voxel of the extended grid (own and ghost columns, voxel ids 1 .. extended_voxel_count()):
    //   [voxel_id]                              1 iff the voxel's inside list holds a wall particle (marker pass 0)
    //   [extended_voxel_count() + 1 + voxel_id] 1 iff a voxel of its 3^d neighbourhood holds one (marker pass 1)
    // Not permuted by defrag (it holds voxels, not particles), not carried by the transport.
    uint wall_crossing_near_wall_voxel_flag[];
};

uint wall_crossing_near_wall_flag_index(uint voxel_id) {
    return extended_voxel_count() + 1u + voxel_id;
}

bool wall_crossing_voxel_is_near_wall(uint voxel_id) {
    return wall_crossing_near_wall_voxel_flag[wall_crossing_near_wall_flag_index(voxel_id)] != 0u;
}

// which particles a split build processes (per pipeline):
//   0 = every particle of its dispatch: the near-wall build over a whole band (phase C, the default band mode);
//   1 = its own class only: the interior build skips the particles of near-wall voxels, the near-wall build the
//       others.
layout(constant_id = 131) const uint WALL_CROSSING_SPLIT_SELECTION = 1u;
const uint WALL_CROSSING_SPLIT_SELECTION_EVERY_PARTICLE = 0u;
const uint WALL_CROSSING_SPLIT_SELECTION_OWN_CLASS      = 1u;

#ifdef WALL_CROSSING_NEAR_WALL
// a neighbour is a wall particle iff it lies outside the fluid box on an active axis (the loader checks that every
// wall particle starts strictly outside and every fluid particle strictly inside; walls do not move)
bool wall_crossing_position_outside_box(vec3 position) {
    bool outside = position.x < WALL_CROSSING_BOX_LOWER_X || position.x > WALL_CROSSING_BOX_UPPER_X
                || position.y < WALL_CROSSING_BOX_LOWER_Y || position.y > WALL_CROSSING_BOX_UPPER_Y;
    if (DIMENSION != 2u) {
        outside = outside || position.z < WALL_CROSSING_BOX_LOWER_Z || position.z > WALL_CROSSING_BOX_UPPER_Z;
    }
    return outside;
}
#endif

#ifdef WALL_CROSSING_INTERIOR
// numerics.volume_constraint in the interior build (the WALL_CROSSING build declares both in wall_boundary.glsl):
// a fluid self's density rate takes rho_i * volume_constraint_state.x when spec const 130 is on.
layout(constant_id = 130) const uint WALL_CROSSING_VOLUME_CONSTRAINT_ENABLED = 0u;
layout(std430, set = 0, binding = 15) buffer VolumeConstraintStateBuffer {
    vec4 volume_constraint_state;
};
#endif
