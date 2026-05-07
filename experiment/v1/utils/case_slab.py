"""
case_slab.py — build a 'sub-Case' for one GPU's slab of a Partition.

Used by the V1 simplified path: load the full case, compute the partition,
build a slab Case for the chosen slot, hand it to V0 SphSimulator unchanged.

The slab Case carries over physics / numerics / materials / time / dt
verbatim. It overrides:
  - particle_sources: each source's vertices are filtered by this slot's
    boolean mask (computed by partition.compute_partition).
  - grid: origin shifted along X; dim_x narrowed to this slot's own voxel
    columns. Y / Z untouched.

V1.0 first iteration here: NO ghost. The slab is exactly the own columns.
Particles drifting past the slab edge are killed by V0 predict (voxel_id=0
on out-of-grid). Physics at the K_split column is wrong (density
under-counts because peer's neighbours are missing). This is the BASELINE
against which frozen ghost / live ghost will be measured.
"""

from utils.sph.case import Case, ParticleSource

from experiment.v1.utils.partition import Partition


def build_slab_case(
    case: Case,
    partition: Partition,
    slot_index: int,
) -> Case:
    """Return a new Case representing one GPU's slab (own columns only).

    Args:
        case:       the full, loaded Case.
        partition:  output of partition.compute_partition.
        slot_index: which GPU's slab to build (0 .. len(partition.gpu_partitions)-1).

    Returns:
        A new Case whose particle_sources contain only the slot's particles
        and whose grid covers only the slot's own voxel x range.
    """
    if not (0 <= slot_index < len(partition.gpu_partitions)):
        raise ValueError(
            f"slot_index {slot_index} out of range "
            f"[0, {len(partition.gpu_partitions)})")
    gpu_partition = partition.gpu_partitions[slot_index]

    # ---- Filter particle sources by per-source mask ------------------------
    filtered_sources: list[ParticleSource] = []
    for source, mask in zip(case.particle_sources, gpu_partition.source_masks):
        kept_vertices = source.vertices[mask].copy()
        filtered_sources.append(ParticleSource(
            obj_path=source.obj_path,
            vertices=kept_vertices,
            material_name=source.material_name,
            material_group_id=source.material_group_id,
        ))

    # ---- Build the slab grid (own columns only; no ghost layer) ------------
    own_x_start, own_x_end = gpu_partition.own_voxel_x_range
    h = case.physics.h

    original_origin = case.grid["origin"]
    original_dimension = case.grid["dimension"]

    new_origin = (
        float(original_origin[0]) + own_x_start * h,
        float(original_origin[1]),
        float(original_origin[2]),
    )
    new_dimension = (
        int(own_x_end - own_x_start),
        int(original_dimension[1]),
        int(original_dimension[2]),
    )

    return Case(
        physics=case.physics,
        numerics=case.numerics,
        capacities=case.capacities,
        time=case.time,
        grid={"origin": new_origin, "dimension": new_dimension},
        materials=case.materials,
        particle_sources=filtered_sources,
        case_dir=case.case_dir,
    )
