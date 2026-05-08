"""
transport_config.py — derive per-GPU SphSimulatorV1 kwargs (pool sizes,
ghost voxel counts, per-direction GhostTransportConfig) from a Partition.

V1.0a: 2-GPU only (1D x-axis split, 1-thick ghost on the interior side of
each GPU). Caller pipeline:

    partition = compute_partition(case, gpu_names)
    layouts   = build_per_gpu_layouts(partition, case)
    for slot, layout in enumerate(layouts):
        slab_case = build_slab_case(case, partition, slot)
        sim = SphSimulatorV1(
            ctx=multi_ctx[slot], case=slab_case,
            leading_ghost_pool_size=layout.leading_ghost_pool_size,
            trailing_ghost_pool_size=layout.trailing_ghost_pool_size,
            leading_ghost_voxel_count=layout.leading_ghost_voxel_count,
            trailing_ghost_voxel_count=layout.trailing_ghost_voxel_count,
            ghost_voxel_x_thickness_leading=layout.ghost_voxel_x_thickness_leading,
            ghost_voxel_x_thickness_trailing=layout.ghost_voxel_x_thickness_trailing,
            leading_transport_config=layout.leading_transport_config,
            trailing_transport_config=layout.trailing_transport_config,
        )

GhostTransportConfig field semantics (per ghost_send.comp spec const docs):

    BOUNDARY_VOXEL_X_LOCAL (id=91)
        My own's outermost column on send side, in extended-grid LOCAL x.
            leading send  → own_first_x_local       (= ghost_thickness_leading)
            trailing send → own_last_x_local        (= extended_nx - 1 - ghost_thickness_trailing)
    GHOST_VOXEL_X_LOCAL (id=92)
        My ghost column adjacent to my boundary on send side.
            leading send  → 0
            trailing send → extended_nx - 1
    GHOST_PID_OFFSET_TO_RECEIVER (id=93)
        peer.dest_first_pid - my.dest_first_pid  (signed int).
        For 2-GPU equal-pool-size, this collapses to ±OWN_POOL_SIZE.
    GHOST_VOXEL_ID_OFFSET_TO_RECEIVER (id=94)
        Option B per-pipeline offset. Signed int.
            offset = (peer.ghost_first_x_local - my.own_boundary_first_x_local) × NY × NZ
        Same offset works for both replicas and migrations on this pipeline
        (the boundary column's source-to-destination column shift cancels —
         see ghost_send.comp's "Option B convention" header).
"""

import pathlib
import sys

# Allow `python experiment/v1/utils/transport_config.py` (script entry).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataclasses import dataclass
from typing import Optional

from utils.sph.case import Case

from experiment.v1.utils.partition import Partition
from experiment.v1.utils.simulator_v1 import GhostTransportConfig


# Ghost-pool capacity per (y,z) face slot. Each face can carry at most:
#   replicas:    MAX_PARTICLES_PER_VOXEL  (one boundary voxel column's inside)
#   migrations:  MAX_INCOMING_PER_VOXEL   (one ghost voxel column's incoming)
# Sum bounds the per-step transport requirement. Multiplied by NY*NZ gives
# the per-direction ghost-pid pool size.
def _default_ghost_pool_size(case: Case) -> int:
    ny = int(case.grid["dimension"][1])
    nz = int(case.grid["dimension"][2])
    cap_inside   = int(case.capacities.max_per_voxel)
    cap_incoming = int(case.capacities.max_incoming)
    return ny * nz * (cap_inside + cap_incoming)


# V1.0a uses 1-voxel-thick ghost on the interior side of each GPU.
GHOST_THICKNESS_V1 = 1


@dataclass
class GpuLocalLayout:
    """Per-GPU SphSimulatorV1 construction kwargs derived from Partition."""
    slot_index: int
    leading_ghost_pool_size: int
    trailing_ghost_pool_size: int
    leading_ghost_voxel_count: int
    trailing_ghost_voxel_count: int
    ghost_voxel_x_thickness_leading: int
    ghost_voxel_x_thickness_trailing: int
    leading_transport_config: Optional[GhostTransportConfig]
    trailing_transport_config: Optional[GhostTransportConfig]


def build_per_gpu_layouts(
    partition: Partition,
    case: Case,
    *,
    ghost_pool_size: Optional[int] = None,
) -> list[GpuLocalLayout]:
    """Compute per-GPU layouts + transport configs for the partition.

    Args:
        partition:        compute_partition(case, gpu_names) output
        case:             full Case (capacities + grid dims used)
        ghost_pool_size:  per-direction ghost-pid pool size. Default =
                          NY*NZ * (max_per_voxel + max_incoming).

    Returns:
        list[GpuLocalLayout], indexed by partition.gpu_partitions slot.

    Raises:
        NotImplementedError: partition has != 2 GPUs.
    """
    n_gpus = len(partition.gpu_partitions)
    if n_gpus != 2:
        raise NotImplementedError(
            f"V1.0a transport_config supports exactly 2 GPUs; got {n_gpus}. "
            f"3+ GPU support is V3+ work.")

    if ghost_pool_size is None:
        ghost_pool_size = _default_ghost_pool_size(case)

    own_pool = int(case.capacities.pool_size)
    ny = int(case.grid["dimension"][1])
    nz = int(case.grid["dimension"][2])
    ny_nz = ny * nz
    ghost_voxel_count = ny_nz * GHOST_THICKNESS_V1

    # Compute each GPU's own_nx_local (= number of own voxel columns).
    own_nx_locals = []
    for gp in partition.gpu_partitions:
        x_start, x_end = gp.own_voxel_x_range
        own_nx_locals.append(x_end - x_start)
    nx0, nx1 = own_nx_locals

    # ------------------------------------------------------------------
    # GPU 0 (leftmost): no leading peer; trailing peer = GPU 1.
    # Extended grid: own [x_local=0..nx0-1] + 1 trailing ghost at x_local=nx0.
    # ------------------------------------------------------------------
    # Trailing send → receiver = GPU 1; data lands in GPU 1's leading-ghost slots.
    #   pid offset = peer.leading_first_pid - my.trailing_first_pid
    #              = 1 - (own_pool + 1) = -own_pool       (assumes shared pool size)
    #   vid offset = (peer.leading_ghost_x_local - my.own_boundary_x_local) * NY*NZ
    #              = (0 - (nx0 - 1)) * NY*NZ
    cfg_gpu0_trailing = GhostTransportConfig(
        boundary_voxel_x_local=nx0 - 1,
        ghost_voxel_x_local=nx0,                              # = extended_nx - 1
        ghost_pid_offset_to_receiver=-own_pool,
        ghost_voxel_id_offset_to_receiver=-(nx0 - 1) * ny_nz,
    )
    layout_gpu0 = GpuLocalLayout(
        slot_index=0,
        leading_ghost_pool_size=0,
        trailing_ghost_pool_size=ghost_pool_size,
        leading_ghost_voxel_count=0,
        trailing_ghost_voxel_count=ghost_voxel_count,
        ghost_voxel_x_thickness_leading=0,
        ghost_voxel_x_thickness_trailing=GHOST_THICKNESS_V1,
        leading_transport_config=None,
        trailing_transport_config=cfg_gpu0_trailing,
    )

    # ------------------------------------------------------------------
    # GPU 1 (rightmost): leading peer = GPU 0; no trailing peer.
    # Extended grid: 1 leading ghost at x_local=0 + own [x_local=1..nx1].
    # ------------------------------------------------------------------
    # Leading send → receiver = GPU 0; data lands in GPU 0's trailing-ghost slots.
    #   pid offset = peer.trailing_first_pid - my.leading_first_pid
    #              = (own_pool + 1) - 1 = +own_pool
    #   vid offset = (peer.trailing_ghost_x_local - my.own_boundary_x_local) * NY*NZ
    #              = ((nx0 + 1 - 1) - 1) * NY*NZ
    #              = (nx0 - 1) * NY*NZ              (peer.extended_nx = nx0 + 1)
    cfg_gpu1_leading = GhostTransportConfig(
        boundary_voxel_x_local=1,                             # own_first_x_local
        ghost_voxel_x_local=0,
        ghost_pid_offset_to_receiver=+own_pool,
        ghost_voxel_id_offset_to_receiver=+(nx0 - 1) * ny_nz,
    )
    layout_gpu1 = GpuLocalLayout(
        slot_index=1,
        leading_ghost_pool_size=ghost_pool_size,
        trailing_ghost_pool_size=0,
        leading_ghost_voxel_count=ghost_voxel_count,
        trailing_ghost_voxel_count=0,
        ghost_voxel_x_thickness_leading=GHOST_THICKNESS_V1,
        ghost_voxel_x_thickness_trailing=0,
        leading_transport_config=cfg_gpu1_leading,
        trailing_transport_config=None,
    )

    return [layout_gpu0, layout_gpu1]


# ============================================================================
# Smoke test
# ============================================================================

def _format_layout(layout: GpuLocalLayout) -> str:
    lines = [f"slot {layout.slot_index}:"]
    lines.append(f"  leading_ghost_pool_size  = {layout.leading_ghost_pool_size}")
    lines.append(f"  trailing_ghost_pool_size = {layout.trailing_ghost_pool_size}")
    lines.append(f"  leading_ghost_voxel_count  = {layout.leading_ghost_voxel_count}")
    lines.append(f"  trailing_ghost_voxel_count = {layout.trailing_ghost_voxel_count}")
    lines.append(f"  thickness leading={layout.ghost_voxel_x_thickness_leading} "
                 f"trailing={layout.ghost_voxel_x_thickness_trailing}")
    if layout.leading_transport_config:
        c = layout.leading_transport_config
        lines.append(f"  leading transport:")
        lines.append(f"    boundary_voxel_x_local            = {c.boundary_voxel_x_local}")
        lines.append(f"    ghost_voxel_x_local               = {c.ghost_voxel_x_local}")
        lines.append(f"    ghost_pid_offset_to_receiver      = {c.ghost_pid_offset_to_receiver:+d}")
        lines.append(f"    ghost_voxel_id_offset_to_receiver = {c.ghost_voxel_id_offset_to_receiver:+d}")
    if layout.trailing_transport_config:
        c = layout.trailing_transport_config
        lines.append(f"  trailing transport:")
        lines.append(f"    boundary_voxel_x_local            = {c.boundary_voxel_x_local}")
        lines.append(f"    ghost_voxel_x_local               = {c.ghost_voxel_x_local}")
        lines.append(f"    ghost_pid_offset_to_receiver      = {c.ghost_pid_offset_to_receiver:+d}")
        lines.append(f"    ghost_voxel_id_offset_to_receiver = {c.ghost_voxel_id_offset_to_receiver:+d}")
    return "\n".join(lines)


def _verify_round_trip(partition: Partition, case: Case,
                       layouts: list[GpuLocalLayout]) -> None:
    """End-to-end vid mapping check: pick a particle in GPU 0's last own column
    and a particle in GPU 0's trailing-ghost column, verify they map into
    GPU 1's leading-ghost range and own range respectively (and vice versa).
    """
    ny = int(case.grid["dimension"][1])
    nz = int(case.grid["dimension"][2])
    ny_nz = ny * nz
    own_pool = int(case.capacities.pool_size)

    nx0 = partition.gpu_partitions[0].own_voxel_x_range[1]
    nx1 = (partition.gpu_partitions[1].own_voxel_x_range[1]
           - partition.gpu_partitions[1].own_voxel_x_range[0])

    # ---- GPU 0 trailing send (data flow GPU 0 → GPU 1) ------------------
    cfg = layouts[0].trailing_transport_config
    assert cfg is not None

    # GPU 0 own_boundary vid (x_local=nx0-1, y=0, z=0) — replica source.
    own_boundary_vid = (nx0 - 1) * ny_nz + 1
    peer_replica_vid = own_boundary_vid + cfg.ghost_voxel_id_offset_to_receiver

    # GPU 1's leading-ghost-vid range = [1, ny_nz].
    assert 1 <= peer_replica_vid <= ny_nz, (
        f"replica vid {peer_replica_vid} not in GPU 1 leading-ghost range "
        f"[1, {ny_nz}]")

    # GPU 0 sender_ghost vid (x_local=nx0) — migration source.
    sender_ghost_vid = nx0 * ny_nz + 1
    peer_migration_vid = sender_ghost_vid + cfg.ghost_voxel_id_offset_to_receiver

    # GPU 1's own vid range = [ny_nz + 1, ny_nz + nx1*ny_nz]
    #                      = [ny_nz + 1, (1 + nx1)*ny_nz]
    gpu1_own_vid_lo = ny_nz + 1
    gpu1_own_vid_hi = (1 + nx1) * ny_nz
    assert gpu1_own_vid_lo <= peer_migration_vid <= gpu1_own_vid_hi, (
        f"migration vid {peer_migration_vid} not in GPU 1 own range "
        f"[{gpu1_own_vid_lo}, {gpu1_own_vid_hi}]")

    # GPU 0's trailing send pid offset: data dst on receiver's leading-pid range
    # First trailing slot on GPU 0: pid = own_pool + 1.
    my_trailing_first_pid = own_pool + 1
    peer_dest_pid = my_trailing_first_pid + cfg.ghost_pid_offset_to_receiver
    assert peer_dest_pid == 1, (
        f"peer dest pid {peer_dest_pid} != 1 (GPU 1 leading-ghost-first-pid)")

    # ---- GPU 1 leading send (data flow GPU 1 → GPU 0) -------------------
    cfg = layouts[1].leading_transport_config
    assert cfg is not None

    # GPU 1 own_boundary vid (x_local=1, y=0, z=0) = ny_nz + 1.
    own_boundary_vid_1 = 1 * ny_nz + 1
    peer_replica_vid_1 = own_boundary_vid_1 + cfg.ghost_voxel_id_offset_to_receiver

    # GPU 0's trailing-ghost-vid range = [extended_total - ny_nz + 1, extended_total]
    #                                  = [(nx0+1)*ny_nz - ny_nz + 1, (nx0+1)*ny_nz]
    #                                  = [nx0*ny_nz + 1, (nx0+1)*ny_nz]
    gpu0_trailing_lo = nx0 * ny_nz + 1
    gpu0_trailing_hi = (nx0 + 1) * ny_nz
    assert gpu0_trailing_lo <= peer_replica_vid_1 <= gpu0_trailing_hi, (
        f"GPU1→GPU0 replica vid {peer_replica_vid_1} not in GPU 0 trailing-ghost "
        f"range [{gpu0_trailing_lo}, {gpu0_trailing_hi}]")

    # GPU 1 sender_ghost vid (x_local=0) = 0*ny_nz + 1 = 1.
    sender_ghost_vid_1 = 0 * ny_nz + 1
    peer_migration_vid_1 = sender_ghost_vid_1 + cfg.ghost_voxel_id_offset_to_receiver

    # GPU 0's own vid range = [1, nx0*ny_nz].
    assert 1 <= peer_migration_vid_1 <= nx0 * ny_nz, (
        f"GPU1→GPU0 migration vid {peer_migration_vid_1} not in GPU 0 own "
        f"range [1, {nx0 * ny_nz}]")

    # GPU 1's leading send pid offset: dst on receiver's trailing-pid range
    my_leading_first_pid = 1
    peer_dest_pid = my_leading_first_pid + cfg.ghost_pid_offset_to_receiver
    assert peer_dest_pid == own_pool + 1, (
        f"peer dest pid {peer_dest_pid} != {own_pool + 1} "
        f"(GPU 0 trailing-ghost-first-pid)")

    print("[verify] all vid + pid mappings consistent")


def main() -> None:
    import argparse
    import pathlib
    import sys
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.sph.case import load_case
    from experiment.v1.utils.partition import compute_partition

    parser = argparse.ArgumentParser(description="V1.0a transport_config smoke test.")
    parser.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    parser.add_argument("--gpus",
                        default="NVIDIA GeForce RTX 4060 Ti,AMD Radeon RX 7900 XTX")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    case = load_case(args.case)
    gpu_names = [n.strip() for n in args.gpus.split(",")]
    weights_override = (
        [float(w.strip()) for w in args.weights.split(",")] if args.weights else None
    )
    partition = compute_partition(case, gpu_names, weights_override=weights_override)

    print(f"\nPartition: K_split={partition.voxel_x_split} / {partition.grid_nx}")
    print(f"Capacities: pool={case.capacities.pool_size:,} "
          f"max_per_voxel={case.capacities.max_per_voxel} "
          f"max_incoming={case.capacities.max_incoming}\n")

    layouts = build_per_gpu_layouts(partition, case)
    for layout in layouts:
        print(_format_layout(layout))
        print()

    _verify_round_trip(partition, case, layouts)


if __name__ == "__main__":
    main()
