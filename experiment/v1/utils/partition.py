"""
partition.py — V1.0 static 1D X-axis partition.

Pure CPU-side data computation: given a Case + a list of GPU names, decides
which voxel column is the split line K_split_voxel_x and which particles
each GPU owns. No Vulkan, no shaders.

Algorithm (matches docs/sph_v1_design.md):

  1. Resolve weights:
       - If `weights_override` is provided, use it.
       - Otherwise look each gpu_name up in KNOWN_GPU_SPH_WEIGHT.
  2. fractions[i] = weights[i] / sum(weights)
  3. Bin all FLUID particles by voxel x_index:
       x_index = floor((p.x - grid_origin_x) / h)
  4. Cumulative-sum the per-column fluid counts; binary-search for the
     column where cumulative reaches target_count_gpu0 = floor(N * fractions[0]).
     This is K_split_voxel_x.
  5. Each particle (any kind) joins its GPU based on its voxel x_index:
       x_index < K_split_voxel_x  → GPU 0
       x_index >= K_split_voxel_x → GPU 1
  6. Ghost columns are 1-thick on the interior side only:
       GPU 0 ghost = column K_split_voxel_x         (peer's leftmost own)
       GPU 1 ghost = column K_split_voxel_x - 1     (peer's rightmost own)

Particle-count-based (not domain-bbox-based) split handles non-uniform
initial distributions automatically. Voxel-aligned (not arbitrary X)
because neighbor search and atomic ops require single-owner voxels.

V1 supports exactly 2 GPUs. V3+ may generalize.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.sph.case import Case, KIND_FLUID
from experiment.v1.utils.gpu_capability import lookup_gpu_weight


@dataclass
class GpuPartition:
    """Per-GPU slice of the case after the 1D X partition."""
    slot_index: int                                  # 0..N-1
    gpu_name: str
    weight: float
    fraction: float                                  # weight / sum(weights)

    # Voxel x-index ranges on the GLOBAL grid (half-open [start, end)).
    own_voxel_x_range: tuple[int, int]               # interior-only
    ghost_voxel_x_range: Optional[tuple[int, int]]   # 1 column on peer side; None if no peer
    union_voxel_x_range: tuple[int, int]             # own ∪ ghost — local grid extent

    # Per-source boolean masks. source_masks[k] is a (N_k,) bool array; True
    # means that source's k-th particle belongs to this GPU. Length matches
    # case.particle_sources.
    source_masks: list[np.ndarray] = field(default_factory=list)

    def particle_count(self) -> int:
        return int(sum(int(mask.sum()) for mask in self.source_masks))

    def particle_count_by_kind(self, case: Case) -> dict[int, int]:
        """Aggregated count per material kind (FLUID/BOUNDARY/INLET/ROTOR)."""
        counts: dict[int, int] = {}
        for source, mask in zip(case.particle_sources, self.source_masks):
            kind = case.materials[source.material_group_id].kind
            counts[kind] = counts.get(kind, 0) + int(mask.sum())
        return counts


@dataclass
class Partition:
    """V1.0 1D X-axis static partition. Produced once at startup; immutable
    for the run's lifetime in V1.0 (dynamic re-partitioning is V3+)."""
    voxel_x_split: int                               # K_split_voxel_x
    grid_nx: int                                     # for context / sanity
    fluid_total: int                                 # total fluid particle count
    weights: list[float]
    fractions: list[float]
    gpu_partitions: list[GpuPartition]



# ============================================================================
# Public API
# ============================================================================


def compute_partition(
    case: Case,
    gpu_names: list[str],
    *,
    weights_override: Optional[list[float]] = None,
) -> Partition:
    """Compute the V1.0 1D X-axis static partition.

    Args:
        case:             fully-loaded Case object (CPU side; no Vulkan).
        gpu_names:        per-slot vkPhysicalDeviceProperties.deviceName.
                          Length determines the GPU count (V1: must be 2).
        weights_override: optional explicit weights (e.g. from case.yaml's
                          `partition.weights`); skips KNOWN_GPU_SPH_WEIGHT
                          lookup. Must match len(gpu_names) when given.

    Returns:
        Partition with voxel-x split + per-GPU GpuPartition masks.

    Raises:
        ValueError: weights mismatch, unknown GPU, fewer than 2 GPUs.
        NotImplementedError: more than 2 GPUs (V1 only supports 2).
    """
    n_gpus = len(gpu_names)
    if n_gpus < 2:
        raise ValueError(
            f"compute_partition requires at least 2 GPUs (V1.0 dual-GPU); "
            f"got {n_gpus}. For single-GPU runs use V0 directly.")
    if n_gpus > 2:
        raise NotImplementedError(
            f"V1 partition supports exactly 2 GPUs; got {n_gpus}. "
            f"3+ GPU partitioning is V3+ work.")

    # ---- 1. Resolve weights ------------------------------------------------
    if weights_override is not None:
        if len(weights_override) != n_gpus:
            raise ValueError(
                f"weights_override length {len(weights_override)} != "
                f"len(gpu_names) {n_gpus}")
        weights = [float(w) for w in weights_override]
    else:
        weights = []
        for name in gpu_names:
            w = lookup_gpu_weight(name)
            if w is None:
                raise ValueError(
                    f"GPU {name!r} is not in KNOWN_GPU_SPH_WEIGHT and no "
                    f"weights_override given. Either add it to "
                    f"experiment/v1/utils/gpu_capability.py (run "
                    f"tools/benchmark_calibration.py first) or pass "
                    f"weights_override.")
            weights.append(float(w))

    if any(w <= 0 for w in weights):
        raise ValueError(f"weights must be strictly positive, got {weights}")
    weight_sum = sum(weights)
    fractions = [w / weight_sum for w in weights]

    # ---- 2. Voxel-bin all FLUID particles ----------------------------------
    grid_nx = int(case.grid["dimension"][0])
    origin_x = float(case.grid["origin"][0])
    h = float(case.physics.h)

    fluid_counts_per_x = np.zeros(grid_nx, dtype=np.int64)
    source_voxel_x: list[np.ndarray] = []
    for source in case.particle_sources:
        x_indices = np.floor(
            (source.vertices[:, 0] - origin_x) / h).astype(np.int64)
        # Defensive clamp — particles on the boundary may compute to grid_nx
        # under float roundoff. Snap into [0, grid_nx).
        np.clip(x_indices, 0, grid_nx - 1, out=x_indices)
        source_voxel_x.append(x_indices)
        kind = case.materials[source.material_group_id].kind
        if kind == KIND_FLUID:
            np.add.at(fluid_counts_per_x, x_indices, 1)

    fluid_total = int(fluid_counts_per_x.sum())
    if fluid_total == 0:
        raise ValueError(
            "case has no fluid particles; partition algorithm requires "
            "at least one fluid particle to choose K_split.")

    # ---- 3. Choose K_split_voxel_x via cumulative searchsorted -------------
    target_count_gpu0 = max(1, int(fluid_total * fractions[0]))
    cumulative = np.cumsum(fluid_counts_per_x)
    voxel_x_split = int(
        np.searchsorted(cumulative, target_count_gpu0, side="left"))
    # Clamp to [1, grid_nx - 1] so both sides own at least one column.
    voxel_x_split = max(1, min(voxel_x_split, grid_nx - 1))

    # ---- 4. Build per-GPU partitions ---------------------------------------
    own_ranges = [(0, voxel_x_split), (voxel_x_split, grid_nx)]
    # Ghost columns: 1 thick on the interior side only.
    ghost_ranges = [
        (voxel_x_split, voxel_x_split + 1),       # GPU 0 ghost = peer's leftmost own
        (voxel_x_split - 1, voxel_x_split),       # GPU 1 ghost = peer's rightmost own
    ]
    union_ranges = [
        (own_ranges[0][0], ghost_ranges[0][1]),   # GPU 0: 0 .. K_split + 1
        (ghost_ranges[1][0], own_ranges[1][1]),   # GPU 1: K_split - 1 .. grid_nx
    ]

    gpu_partitions: list[GpuPartition] = []
    for slot_index in range(n_gpus):
        own_start, own_end = own_ranges[slot_index]
        masks = [
            (x_indices >= own_start) & (x_indices < own_end)
            for x_indices in source_voxel_x
        ]
        gpu_partitions.append(GpuPartition(
            slot_index=slot_index,
            gpu_name=gpu_names[slot_index],
            weight=weights[slot_index],
            fraction=fractions[slot_index],
            own_voxel_x_range=own_ranges[slot_index],
            ghost_voxel_x_range=ghost_ranges[slot_index],
            union_voxel_x_range=union_ranges[slot_index],
            source_masks=masks,
        ))

    return Partition(
        voxel_x_split=voxel_x_split,
        grid_nx=grid_nx,
        fluid_total=fluid_total,
        weights=weights,
        fractions=fractions,
        gpu_partitions=gpu_partitions,
    )


# ============================================================================
# Smoke test entry: load lid case, compute partition, print + assert
# ============================================================================


def _format_partition_with_kinds(partition: Partition, case: Case) -> str:
    from utils.sph.case import KIND_FLUID, KIND_BOUNDARY, KIND_INLET, KIND_ROTOR
    kind_names = {
        KIND_FLUID:    "fluid",
        KIND_BOUNDARY: "boundary",
        KIND_INLET:    "inlet",
        KIND_ROTOR:    "rotor",
    }
    lines = [
        f"Partition: K_split_voxel_x = {partition.voxel_x_split} / {partition.grid_nx}",
        f"  fluid total: {partition.fluid_total:,}",
        f"  weights:     {partition.weights}",
        f"  fractions:   [{', '.join(f'{f:.4f}' for f in partition.fractions)}]",
    ]
    for gp in partition.gpu_partitions:
        kind_counts = gp.particle_count_by_kind(case)
        kind_summary = ", ".join(
            f"{kind_names[k]}={c:,}" for k, c in sorted(kind_counts.items()))
        lines.append(
            f"  slot {gp.slot_index} {gp.gpu_name!r} "
            f"(weight={gp.weight:.3f}, fraction={gp.fraction:.4f}):")
        lines.append(
            f"    own   voxel-x = [{gp.own_voxel_x_range[0]}, "
            f"{gp.own_voxel_x_range[1]})  "
            f"({gp.own_voxel_x_range[1] - gp.own_voxel_x_range[0]} columns)")
        lines.append(
            f"    ghost voxel-x = [{gp.ghost_voxel_x_range[0]}, "
            f"{gp.ghost_voxel_x_range[1]})  (1 column)")
        lines.append(
            f"    union voxel-x = [{gp.union_voxel_x_range[0]}, "
            f"{gp.union_voxel_x_range[1]})")
        lines.append(
            f"    particles: total={gp.particle_count():,}  ({kind_summary})")
    return "\n".join(lines)


def main() -> None:
    import argparse
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.sph.case import load_case

    parser = argparse.ArgumentParser(
        description="Smoke test partition.py on a case.")
    parser.add_argument("--case",
                        default="cases/lid_driven_cavity_2d/case.yaml")
    parser.add_argument("--gpus",
                        default="NVIDIA GeForce RTX 4060 Ti,AMD Radeon RX 7900 XTX",
                        help="comma-separated GPU names to feed the partitioner")
    parser.add_argument("--weights", default=None,
                        help="comma-separated explicit weights (skips lookup); "
                             "e.g. '1,2.088'")
    args = parser.parse_args()

    case = load_case(args.case)
    gpu_names = [name.strip() for name in args.gpus.split(",")]
    weights_override = None
    if args.weights:
        weights_override = [float(w.strip()) for w in args.weights.split(",")]

    partition = compute_partition(
        case, gpu_names, weights_override=weights_override)

    print()
    print(_format_partition_with_kinds(partition, case))
    print()

    # Sanity asserts.
    n_total = sum(s.vertices.shape[0] for s in case.particle_sources)
    n_owned_total = sum(gp.particle_count() for gp in partition.gpu_partitions)
    assert n_owned_total == n_total, (
        f"particle conservation broken: total={n_total}, "
        f"sum-owned={n_owned_total}")

    # Per-source: every particle owned by exactly one GPU.
    for source_idx, source in enumerate(case.particle_sources):
        masks = [gp.source_masks[source_idx] for gp in partition.gpu_partitions]
        ownership_count = np.zeros(source.vertices.shape[0], dtype=np.int64)
        for m in masks:
            ownership_count += m.astype(np.int64)
        bad = (ownership_count != 1)
        if bad.any():
            raise AssertionError(
                f"source {source_idx} ({source.material_name!r}): "
                f"{int(bad.sum())} particles assigned to != 1 GPU")

    print(f"[partition-smoke] OK: total={n_total:,}  "
          f"sum-owned={n_owned_total:,}  "
          f"every particle owned by exactly one GPU")


if __name__ == "__main__":
    main()
