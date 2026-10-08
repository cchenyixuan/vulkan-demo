"""
_test_seam_layout.py — CPU-only checks of the V6 seam layout (no Vulkan device).

1. V7_GHOST_LAYERS=1, V7_KEEP_DEPARTED=0: the v6 chain partition and the per-
   direction transport segments are field-for-field those of v5.
2. V7_KEEP_DEPARTED=1: only the departed pool size differs from (1).
3. V7_GHOST_LAYERS=2: for every link and direction, the column algebra of
   ghost_send.comp (outbox columns, migrant source columns, departed voxel ids)
   composed with the transport (ghost voxel range copied in x order) and the
   voxel-id / pid offsets puts every replica, migrant and departed particle at
   the right GLOBAL column on the right side: the receiver's inner ghost column
   holds the sender's seam column, its outer ghost column the column behind it,
   migrants land in the receiver's own seam column, departed copies sit in the
   sender's ghost column that mirrors the receiver's seam column.
4. V7_GHOST_LAYERS=2 transport segments: regions partition the ghost pool, the
   replica regions carry exactly the 4 sweep fields, every per-particle segment
   has a live-count word, staging is contiguous and ends with the frame stamp.
5. V7_LEAN_TRANSPORT=1 (both layouts, with / without V7_TRANSPORT_EXTENSION):
   the partition is unchanged, the particle segments are the full layout's
   segments restricted to the 4 read fields (+ extension_fields), same device
   ranges, count words and stamp; 44 (60) B per migrant slot.
6. V7_BAND_WIDTHS: parsing and rejection (c >= 2, d >= c, f >= d + 1; non-default
   widths with V7_BAND_COMPACT_DISPATCH), spec 82 of the split pipelines, no
   literal band width left in the simulator's spec entries and band dispatches.

Usage:
    .venv/Scripts/python.exe experiment/v7/_test_seam_layout.py
"""
from __future__ import annotations

import dataclasses
import importlib
import os
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _synthetic_global_case(case_module, column_count=48, row_count=14,
                           particles_per_voxel_side=4, depth_count=1):
    """A degenerate (no-peer) global case on a lattice, built in memory: 2-D, or
    3-D with depth_count voxels in z (depth_count > 1)."""
    smoothing_length = 0.01
    spacing = smoothing_length / particles_per_voxel_side
    x_values = (np.arange(column_count * particles_per_voxel_side) + 0.5) * spacing
    y_values = (np.arange(row_count * particles_per_voxel_side) + 0.5) * spacing
    z_values = ((np.arange(depth_count * particles_per_voxel_side) + 0.5) * spacing
                if depth_count > 1 else np.zeros(1))
    grid_x, grid_y, grid_z = np.meshgrid(x_values, y_values, z_values, indexing="ij")
    positions = np.zeros((grid_x.size, 3), dtype=np.float32)
    positions[:, 0] = grid_x.ravel()
    positions[:, 1] = grid_y.ravel()
    positions[:, 2] = grid_z.ravel()
    material_group = np.where(positions[:, 1] < 2 * spacing, 1, 0).astype(np.uint32)
    physics = case_module.PhysicsConstants(
        smoothing_length=smoothing_length, speed_of_sound=100.0, delta_coefficient=0.1,
        power_parameter=7.0, cfl_number=0.15, timestep=7.5e-6, gravity=(0.0, 0.0, 0.0),
        dimension=3 if depth_count > 1 else 2, neighbor_z_range=1 if depth_count > 1 else 0,
        kernel_coefficient=1.0,
        kernel_gradient_coefficient=1.0)
    numerics = case_module.NumericsConstants(
        regularization_xi=0.1, regularization_determinant_threshold=1e-4,
        regularization_max_frobenius_norm=10.0, eps_h_squared=1e-6,
        pst_main_shift_coefficient=0.1, pst_anti_shift_coefficient=0.0005)
    capacities = case_module.Capacities(
        max_particles_per_voxel=96, workgroup_size=128, max_incoming_per_voxel=16,
        own_pool_size=int(positions.shape[0] * 1.2), leading_ghost_pool_size=0,
        trailing_ghost_pool_size=0)
    grid = case_module.GridLayout(origin_x=0.0, origin_y=0.0, origin_z=0.0,
                                  grid_dimension_x=column_count,
                                  grid_dimension_y=row_count, grid_dimension_z=depth_count)
    materials = [
        case_module.MaterialParameter(kind=case_module.KIND_FLUID, rest_density=1000.0,
                                      viscosity=1e-3, eos_constant=1.0,
                                      smoothing_length=smoothing_length,
                                      radius=spacing / 2, volume=spacing ** 2),
        case_module.MaterialParameter(kind=case_module.KIND_BOUNDARY, rest_density=1000.0,
                                      viscosity=1e-3, eos_constant=1.0,
                                      smoothing_length=smoothing_length,
                                      radius=spacing / 2, volume=spacing ** 2),
    ]
    initial = case_module.InitialParticles(
        positions=positions, velocities=np.zeros_like(positions),
        material_group=material_group)
    return case_module.CaseV7(
        physics=physics, numerics=numerics, capacities=capacities, grid=grid,
        ghost_grid=case_module.GhostGridParams(0, 0),
        transport=case_module.TransportConfig(), materials=materials, initial=initial) \
        if hasattr(case_module, "CaseV7") else case_module.CaseV5(
        physics=physics, numerics=numerics, capacities=capacities, grid=grid,
        ghost_grid=case_module.GhostGridParams(0, 0),
        transport=case_module.TransportConfig(), materials=materials, initial=initial)


def _set_switches(ghost_layers: int, keep_departed: int, lean: int = 0,
                  extension: int = 0) -> None:
    """The pre-E6b defaults (partition_v7.LEGACY_DEFAULTS) for every switch, then
    the named ones: each check turns its switches on one at a time on top of them."""
    from experiment.v7.utils.partition_v7 import LEGACY_DEFAULTS
    os.environ.update(LEGACY_DEFAULTS)
    os.environ.pop("V7_MIGRANT_POOL_FACTOR", None)
    os.environ.pop("V7_BAND_COMPACT_DISPATCH", None)
    os.environ["V7_GHOST_LAYERS"] = str(ghost_layers)
    os.environ["V7_KEEP_DEPARTED"] = str(keep_departed)
    os.environ["V7_LEAN_TRANSPORT"] = str(lean)
    os.environ["V7_TRANSPORT_EXTENSION"] = str(extension)
    os.environ.pop("V7_DEPARTED_CAPACITY", None)


def _comparable(value):
    """Dataclass -> nested dict with numpy arrays reduced to bytes."""
    if dataclasses.is_dataclass(value):
        return {field.name: _comparable(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, list):
        return [_comparable(item) for item in value]
    return value


V7_ONLY_FIELDS = {"departed_pool_size", "replica_region_size", "ghost_layers",
                  "wall_boundary"}          # E37 numerics option (v6 only)


def _strip_v7_only(tree):
    if isinstance(tree, dict):
        return {key: _strip_v7_only(value) for key, value in tree.items()
                if key not in V7_ONLY_FIELDS}
    if isinstance(tree, list):
        return [_strip_v7_only(item) for item in tree]
    return tree


def _strip_unset_restart_densities(tree):
    """The v5 working tree carries an uncommitted checkpoint-restart field
    InitialParticles.densities (None for a fresh start); v6 is cut from HEAD."""
    if isinstance(tree, dict):
        return {key: _strip_unset_restart_densities(value) for key, value in tree.items()
                if not (key == "densities" and value is None)}
    if isinstance(tree, list):
        return [_strip_v7_only(item) for item in tree]
    return tree


def _fake_simulator(simulator_class, case):
    simulator = object.__new__(simulator_class)
    simulator.case = case
    return simulator


def check_v5_equivalence(failures: list) -> None:
    import experiment.v5.utils.case_v5 as case_v5
    import experiment.v5.utils.partition_v5 as partition_v5
    import experiment.v5.utils.simulator_v5 as simulator_v5
    import experiment.v7.utils.case_v7 as case_v7
    import experiment.v7.utils.partition_v7 as partition_v7
    import experiment.v7.utils.simulator_v7 as simulator_v7

    for keep_departed in (0, 1):
        _set_switches(1, keep_departed)
        for slab_count in (2, 3, 4):
            weights = [1.0] * slab_count
            chain_v5 = partition_v5.compute_chain_partition(
                _synthetic_global_case(case_v5), weights, pool_safety=1.2)
            # E31 moved v6's cuts to the column boundary nearest the target (partition_v7.nearest_cut);
            # v5 keeps searchsorted-left, at most one column left of it. v6's own cuts are checked
            # against the E31 rule, the layout comparison below runs at v5's cuts.
            global_v7 = _synthetic_global_case(case_v7)
            own_cuts = partition_v7.compute_chain_partition(global_v7, weights, pool_safety=1.2).cuts
            rule_cuts = partition_v7.chain_cuts_from_counts(
                partition_v7._bin_fluid_counts(global_v7), weights, partition_v7.MINIMUM_OWN_COLUMNS_HARD)
            if list(own_cuts) != rule_cuts or any(cut_v7 - cut_v5 not in (0, 1)
                                                   for cut_v5, cut_v7 in zip(chain_v5.cuts, own_cuts)):
                failures.append(f"K={slab_count}: v6 cuts {list(own_cuts)} are not the E31 rule's {rule_cuts} "
                                f"within one column right of v5's {list(chain_v5.cuts)}")
            compute_chain_cuts = partition_v7.compute_chain_cuts
            partition_v7.compute_chain_cuts = (lambda global_case, weights, minimum_own_columns,
                                               cuts=list(chain_v5.cuts): list(cuts))
            try:
                chain_v7 = partition_v7.compute_chain_partition(global_v7, weights, pool_safety=1.2)
            finally:
                partition_v7.compute_chain_cuts = compute_chain_cuts
            if chain_v5.cuts != chain_v7.cuts:
                failures.append(f"K={slab_count}: cuts differ {chain_v5.cuts} vs {chain_v7.cuts}")
            for index, (slab_v5, slab_v7) in enumerate(zip(chain_v5.slabs, chain_v7.slabs)):
                tree_v5 = _strip_unset_restart_densities(_comparable(slab_v5))
                tree_v7 = _strip_v7_only(_comparable(slab_v7))
                if tree_v5 != tree_v7:
                    differing = [key for key in tree_v5 if tree_v5[key] != tree_v7.get(key)]
                    failures.append(f"keep={keep_departed} K={slab_count} slab {index}: "
                                    f"partition differs from v5 in {differing}")
                peers = (slab_v7.transport.has_leading_peer + slab_v7.transport.has_trailing_peer)
                expected_departed = 0 if (keep_departed == 0 or peers == 0) else \
                    max(partition_v7.DEPARTED_CAPACITY_FLOOR,
                        int(np.ceil(0.25 * slab_v7.grid.grid_dimension_y * peers)))
                if slab_v7.capacities.departed_pool_size != expected_departed:
                    failures.append(f"keep={keep_departed} K={slab_count} slab {index}: departed "
                                    f"{slab_v7.capacities.departed_pool_size} != {expected_departed}")
                if slab_v7.capacities.replica_region_size != 0 or slab_v7.ghost_grid.ghost_layers != 1:
                    failures.append(f"K={slab_count} slab {index}: layers=1 slab has two-layer fields")
                # transport segments identical to v5 (ignoring the v6 count annotations)
                for direction in ("leading", "trailing"):
                    segments_v5, total_v5 = simulator_v5.SphSimulatorV5._compute_transport_segments(
                        _fake_simulator(simulator_v5.SphSimulatorV5, slab_v5), direction)
                    segments_v7, total_v7 = simulator_v7.SphSimulatorV7._compute_transport_segments(
                        _fake_simulator(simulator_v7.SphSimulatorV7, slab_v7), direction)
                    plain_v5 = [(s.buffer_name, s.device_offset, s.staging_offset, s.size) for s in segments_v5]
                    plain_v7 = [(s.buffer_name, s.device_offset, s.staging_offset, s.size) for s in segments_v7]
                    if plain_v5 != plain_v7 or total_v5 != total_v7:
                        failures.append(f"K={slab_count} slab {index} {direction}: transport "
                                        f"segments differ from v5")
                    if segments_v7:
                        count_offsets = {s.count_staging_offset for s in segments_v7[:9]}
                        if count_offsets != {segments_v7[-2].staging_offset}:
                            failures.append(f"K={slab_count} slab {index} {direction}: count-aware "
                                            f"plan does not point at the send count word")


def _global_column(geometry, local_x: int) -> int:
    return geometry.own_global_first_column + (local_x - geometry.leading_thickness)


def check_two_layer_algebra(failures: list) -> None:
    import experiment.v7.utils.case_v7 as case_v7
    import experiment.v7.utils.partition_v7 as partition_v7
    import experiment.v7.utils.simulator_v7 as simulator_v7

    _set_switches(2, 1)
    for slab_count in (2, 3, 4):
        chain = partition_v7.compute_chain_partition(
            _synthetic_global_case(case_v7), [1.0] * slab_count, pool_safety=1.2)
        face = chain.slabs[0].grid.grid_dimension_y * chain.slabs[0].grid.grid_dimension_z
        for sender_index, sender in enumerate(chain.slabs):
            sender_geometry = chain.geometry[sender_index]
            for direction in ("leading", "trailing"):
                spec = getattr(sender.transport, direction)
                if spec is None:
                    continue
                receiver_index = sender_index + (1 if direction == "trailing" else -1)
                receiver = chain.slabs[receiver_index]
                receiver_geometry = chain.geometry[receiver_index]
                trailing_send = direction == "trailing"
                boundary_x = spec.boundary_voxel_x_local
                inward_step = -1 if trailing_send else 1
                offset_columns = spec.ghost_voxel_id_offset_to_receiver // face
                if spec.ghost_voxel_id_offset_to_receiver % face:
                    failures.append(f"K={slab_count} s{sender_index} {direction}: vid offset not a column multiple")
                # transport: sender's ghost columns (this side) -> receiver's opposite ghost columns, x order
                if trailing_send:
                    sender_ghost_first_x = sender_geometry.leading_thickness + sender_geometry.own_column_count
                    receiver_ghost_first_x = 0
                else:
                    sender_ghost_first_x = 0
                    receiver_ghost_first_x = (receiver_geometry.leading_thickness
                                              + receiver_geometry.own_column_count)
                receiver_inner_x = (receiver_geometry.leading_thickness - 1 if trailing_send
                                    else receiver_geometry.leading_thickness + receiver_geometry.own_column_count)
                for layer_index in range(2):
                    source_x = boundary_x + inward_step * layer_index
                    outbox_x = (boundary_x + 2 - layer_index) if trailing_send else (boundary_x - 2 + layer_index)
                    landed_x = receiver_ghost_first_x + (outbox_x - sender_ghost_first_x)
                    if not (0 <= outbox_x - sender_ghost_first_x < 2):
                        failures.append(f"K={slab_count} s{sender_index} {direction} layer {layer_index}: "
                                        f"outbox x {outbox_x} outside the sender's ghost columns")
                    if _global_column(sender_geometry, source_x) != _global_column(receiver_geometry, landed_x):
                        failures.append(f"K={slab_count} s{sender_index} {direction} layer {layer_index}: replica "
                                        f"global column {_global_column(sender_geometry, source_x)} lands in "
                                        f"{_global_column(receiver_geometry, landed_x)}")
                    if source_x + offset_columns != landed_x:
                        failures.append(f"K={slab_count} s{sender_index} {direction} layer {layer_index}: replica "
                                        f"voxel id points at column {source_x + offset_columns}, listed in {landed_x}")
                    expected_inner = layer_index == 0
                    if (landed_x == receiver_inner_x) != expected_inner:
                        failures.append(f"K={slab_count} s{sender_index} {direction} layer {layer_index}: "
                                        f"inner/outer layer mismatch on the receiver")
                # migrants: sender inner ghost column -> receiver own seam column
                sender_inner_ghost_x = boundary_x - inward_step
                landed_x = sender_inner_ghost_x + offset_columns
                receiver_seam_x = (receiver_geometry.leading_thickness if trailing_send
                                   else receiver_geometry.leading_thickness + receiver_geometry.own_column_count - 1)
                if landed_x != receiver_seam_x:
                    failures.append(f"K={slab_count} s{sender_index} {direction}: migrant lands in local "
                                    f"{landed_x}, receiver seam column is {receiver_seam_x}")
                if _global_column(sender_geometry, sender_inner_ghost_x) != _global_column(receiver_geometry, receiver_seam_x):
                    failures.append(f"K={slab_count} s{sender_index} {direction}: sender inner ghost column "
                                    f"is not the receiver's seam column")
                # outer-column (two-column jump) migrant -> receiver own column 1
                landed_far_x = boundary_x - 2 * inward_step + offset_columns
                if _global_column(sender_geometry, boundary_x - 2 * inward_step) != _global_column(receiver_geometry, landed_far_x):
                    failures.append(f"K={slab_count} s{sender_index} {direction}: far migrant column mismatch")
                # departed copies keep the sender-frame ghost voxel: the sender's inbox for that
                # column (from the receiver's opposite send) must be the same global column
                reverse_spec = getattr(receiver.transport, "leading" if trailing_send else "trailing")
                reverse_boundary_x = reverse_spec.boundary_voxel_x_local
                reverse_inward = 1 if trailing_send else -1
                reverse_offset = reverse_spec.ghost_voxel_id_offset_to_receiver // face
                inner_replica_landing = reverse_boundary_x + reverse_offset    # receiver seam column -> sender
                if inner_replica_landing != sender_inner_ghost_x:
                    failures.append(f"K={slab_count} s{sender_index} {direction}: the receiver's seam-column "
                                    f"replicas land in sender column {inner_replica_landing}, departed copies "
                                    f"sit in {sender_inner_ghost_x}")
                outer_replica_landing = reverse_boundary_x + reverse_inward + reverse_offset
                if outer_replica_landing != sender_inner_ghost_x - inward_step:
                    failures.append(f"K={slab_count} s{sender_index} {direction}: outer replica column mismatch")
                # pid offsets map sender ghost range starts onto receiver ghost range starts
                capacities = sender.capacities
                sender_first = (1 if not trailing_send
                                else capacities.leading_ghost_pool_size + capacities.own_pool_size + 1)
                receiver_capacities = receiver.capacities
                receiver_first = (1 if trailing_send
                                  else receiver_capacities.leading_ghost_pool_size + receiver_capacities.own_pool_size + 1)
                if sender_first + spec.ghost_pid_offset_to_receiver != receiver_first:
                    failures.append(f"K={slab_count} s{sender_index} {direction}: pid offset mismatch")

            # transport segments of the two-layer layout
            replica_region = sender.capacities.replica_region_size
            if replica_region <= 0:
                failures.append(f"K={slab_count} s{sender_index}: no replica region in a two-layer partition")
                continue
            fake = _fake_simulator(simulator_v7.SphSimulatorV7, sender)
            for direction in ("leading", "trailing"):
                if getattr(sender.transport, direction) is None:
                    continue
                segments, total = simulator_v7.SphSimulatorV7._compute_transport_segments(fake, direction)
                pool = (sender.capacities.leading_ghost_pool_size if direction == "leading"
                        else sender.capacities.trailing_ghost_pool_size)
                pid_first = (1 if direction == "leading" else
                             sender.capacities.leading_ghost_pool_size + sender.capacities.own_pool_size + 1)
                coverage: dict[str, list] = {}
                expected_offset = 0
                for segment in segments:
                    if segment.staging_offset != expected_offset:
                        failures.append(f"K={slab_count} s{sender_index} {direction}: staging gap at {segment}")
                    expected_offset = segment.staging_offset + segment.size
                    if segment.stride:
                        first_slot = segment.device_offset // segment.stride - pid_first
                        coverage.setdefault(segment.buffer_name, []).append(
                            (first_slot, segment.size // segment.stride))
                        if segment.count_staging_offset is None:
                            failures.append(f"K={slab_count} s{sender_index} {direction}: per-particle "
                                            f"segment without a count word: {segment}")
                if expected_offset != total:
                    failures.append(f"K={slab_count} s{sender_index} {direction}: staging total mismatch")
                if segments[-1].device_offset != simulator_v7._OFFSET_FRAME_STAMP:
                    failures.append(f"K={slab_count} s{sender_index} {direction}: stamp is not the last segment")
                for name, ranges in coverage.items():
                    if name in simulator_v7._REPLICA_TRANSPORT_FIELDS:
                        expected = [(0, replica_region), (replica_region, replica_region),
                                    (2 * replica_region, pool - 2 * replica_region)]
                    else:
                        expected = [(2 * replica_region, pool - 2 * replica_region)]
                    if sorted(ranges) != expected:
                        failures.append(f"K={slab_count} s{sender_index} {direction}: {name} covers "
                                        f"{sorted(ranges)}, expected {expected}")
                overrides = fake._recv_status_overrides[direction]
                send_count = (simulator_v7._OFFSET_GHOST_SEND_LEADING if direction == "leading"
                              else simulator_v7._OFFSET_GHOST_SEND_TRAILING)
                receive_count = (simulator_v7._OFFSET_GHOST_RECV_LEADING if direction == "leading"
                                 else simulator_v7._OFFSET_GHOST_RECV_TRAILING)
                if overrides.get(send_count) != receive_count:
                    failures.append(f"K={slab_count} s{sender_index} {direction}: migrant count is not "
                                    f"uploaded into the install count slot")
            install_threads = simulator_v7.SphSimulatorV7._per_ghost_pid_dispatch_count(
                fake, "trailing" if sender.transport.has_trailing_peer else "leading")
            workgroup = sender.capacities.workgroup_size
            migrant_slots = (sender.capacities.trailing_ghost_pool_size or sender.capacities.leading_ghost_pool_size) \
                - 2 * replica_region
            if install_threads != (migrant_slots + workgroup - 1) // workgroup:
                failures.append(f"K={slab_count} s{sender_index}: install dispatch {install_threads} does not "
                                f"cover exactly the migrant region")


def check_lean_transport(failures: list, depth_count: int = 1) -> None:
    """V7_LEAN_TRANSPORT: the per-particle segments of the V5 mixed pool and
    of the two-layer migrant region are exactly the 4 read fields (+
    extension_fields with V7_TRANSPORT_EXTENSION), at the SAME device offsets
    and sizes as the full layout; voxel lists, count words and stamp are
    unchanged; the count-aware plan still points every particle segment at
    its live-count word; staging stays contiguous."""
    import experiment.v7.utils.case_v7 as case_v7
    import experiment.v7.utils.partition_v7 as partition_v7
    import experiment.v7.utils.simulator_v7 as simulator_v7

    lean_fields = {"position_voxel_id", "velocity_mass", "density_pressure", "material"}

    def key(segment):
        return (segment.buffer_name, segment.device_offset, segment.size, segment.stride)

    for ghost_layers, keep_departed in ((1, 0), (1, 1), (2, 1)):
        for extension in (0, 1):
            _set_switches(ghost_layers, keep_departed, lean=0)
            full_chain = partition_v7.compute_chain_partition(
                _synthetic_global_case(case_v7, depth_count=depth_count), [1.0, 1.0, 1.0], pool_safety=1.2)
            _set_switches(ghost_layers, keep_departed, lean=1, extension=extension)
            lean_chain = partition_v7.compute_chain_partition(
                _synthetic_global_case(case_v7, depth_count=depth_count), [1.0, 1.0, 1.0], pool_safety=1.2)
            expected_fields = lean_fields | ({"extension_fields"} if extension else set())
            tag = f"layers={ghost_layers} keep={keep_departed} ext={extension}"
            for index, (full_slab, lean_slab) in enumerate(zip(full_chain.slabs, lean_chain.slabs)):
                if _comparable(full_slab) != _comparable(lean_slab):
                    failures.append(f"{tag} slab {index}: the lean switch changed the partition")
                replica_region = lean_slab.capacities.replica_region_size

                def is_replica_segment(segment):
                    return (replica_region > 0
                            and segment.buffer_name in simulator_v7._REPLICA_TRANSPORT_FIELDS
                            and segment.size == segment.stride * replica_region)

                for direction in ("leading", "trailing"):
                    _set_switches(ghost_layers, keep_departed, lean=0)
                    full_segments, full_total = simulator_v7.SphSimulatorV7._compute_transport_segments(
                        _fake_simulator(simulator_v7.SphSimulatorV7, full_slab), direction)
                    _set_switches(ghost_layers, keep_departed, lean=1, extension=extension)
                    lean_segments, lean_total = simulator_v7.SphSimulatorV7._compute_transport_segments(
                        _fake_simulator(simulator_v7.SphSimulatorV7, lean_slab), direction)
                    if not full_segments:
                        if lean_segments:
                            failures.append(f"{tag} slab {index} {direction}: lean adds segments")
                        continue
                    # every non-particle segment, every replica-region segment
                    # and the lean fields of the migrant / mixed region survive
                    expected = [key(segment) for segment in full_segments
                                if not segment.stride or is_replica_segment(segment)
                                or segment.buffer_name in expected_fields]
                    observed = [key(segment) for segment in lean_segments]
                    if observed != expected:
                        failures.append(f"{tag} slab {index} {direction}: lean segments "
                                        f"{[item[0] for item in observed]} != "
                                        f"{[item[0] for item in expected]}")
                    offset = 0
                    for segment in lean_segments:
                        if segment.staging_offset != offset:
                            failures.append(f"{tag} slab {index} {direction}: staging gap")
                        offset = segment.staging_offset + segment.size
                        if segment.stride and segment.count_staging_offset is None:
                            failures.append(f"{tag} slab {index} {direction}: particle segment "
                                            f"{segment.buffer_name} without a count word")
                    if offset != lean_total:
                        failures.append(f"{tag} slab {index} {direction}: staging total mismatch")
                    if lean_segments[-1].device_offset != simulator_v7._OFFSET_FRAME_STAMP:
                        failures.append(f"{tag} slab {index} {direction}: stamp is not last")
                    full_words = [segment.device_offset for segment in full_segments
                                  if segment.buffer_name == "global_status"]
                    lean_words = [segment.device_offset for segment in lean_segments
                                  if segment.buffer_name == "global_status"]
                    if full_words != lean_words:
                        failures.append(f"{tag} slab {index} {direction}: count / stamp words differ")
                    migrant_slot_bytes = sum(segment.stride for segment in lean_segments
                                             if segment.stride and not is_replica_segment(segment))
                    expected_slot_bytes = 60 if extension else 44
                    if migrant_slot_bytes != expected_slot_bytes:
                        failures.append(f"{tag} slab {index} {direction}: {migrant_slot_bytes} B per "
                                        f"migrant slot, expected {expected_slot_bytes}")


def check_compact_ghost_lists(failures: list, depth_count: int = 1) -> None:
    """V7_COMPACT_GHOST_LISTS: the inside_particle_index segment (ghost voxels x
    MAX_PARTICLES_PER_VOXEL x 4 B) becomes one ghost_voxel_first_particle_id word
    per ghost voxel at device offset 4 x first ghost vid; every other segment
    keeps its device range; staging stays contiguous, count words / stamp last."""
    import experiment.v7.utils.case_v7 as case_v7
    import experiment.v7.utils.partition_v7 as partition_v7
    import experiment.v7.utils.simulator_v7 as simulator_v7

    def key(segment):
        return (segment.buffer_name, segment.device_offset, segment.size, segment.stride)

    for ghost_layers, keep_departed in ((1, 0), (1, 1), (2, 1)):
        _set_switches(ghost_layers, keep_departed, lean=1)
        chain = partition_v7.compute_chain_partition(
            _synthetic_global_case(case_v7, depth_count=depth_count), [1.0, 1.0, 1.0], pool_safety=1.2)
        tag = f"compact layers={ghost_layers} keep={keep_departed}"
        for index, slab in enumerate(chain.slabs):
            cap_inside = slab.capacities.max_particles_per_voxel
            for direction in ("leading", "trailing"):
                os.environ["V7_COMPACT_GHOST_LISTS"] = "0"
                full, _ = simulator_v7.SphSimulatorV7._compute_transport_segments(
                    _fake_simulator(simulator_v7.SphSimulatorV7, slab), direction)
                os.environ["V7_COMPACT_GHOST_LISTS"] = "1"
                compact, total = simulator_v7.SphSimulatorV7._compute_transport_segments(
                    _fake_simulator(simulator_v7.SphSimulatorV7, slab), direction)
                os.environ["V7_COMPACT_GHOST_LISTS"] = "0"
                if not full:
                    continue
                expected = []
                for segment in full:
                    if segment.buffer_name == "inside_particle_index":
                        voxels = segment.size // (4 * cap_inside)
                        first_vid = segment.device_offset // (4 * cap_inside)
                        expected.append(("ghost_voxel_first_particle_id", 4 * first_vid, 4 * voxels, 0))
                    else:
                        expected.append(key(segment))
                if [key(segment) for segment in compact] != expected:
                    failures.append(f"{tag} slab {index} {direction}: compact segments differ")
                offset = 0
                for segment in compact:
                    if segment.staging_offset != offset:
                        failures.append(f"{tag} slab {index} {direction}: staging gap")
                    offset = segment.staging_offset + segment.size
                if offset != total or compact[-1].device_offset != simulator_v7._OFFSET_FRAME_STAMP:
                    failures.append(f"{tag} slab {index} {direction}: staging total / stamp")


def check_packed_replicas(failures: list, depth_count: int = 1) -> None:
    """V7_PACKED_REPLICAS (two layers, compact lists): the 8 replica SoA segments
    become 4 ghost_packed_words blocks (G1 16/16 B, G2 16/16 B per replica: the
    same 32 B record, no pressure) at direction base d * 64 R bytes, with the
    inner / outer count words; the migrant region, voxel lists, count words and
    stamp are unchanged; the blocks of both directions tile the
    ghost_packed_words allocation exactly (2 x 64 R bytes)."""
    import experiment.v7.utils.case_v7 as case_v7
    import experiment.v7.utils.partition_v7 as partition_v7
    import experiment.v7.utils.simulator_v7 as simulator_v7

    def key(segment):
        return (segment.buffer_name, segment.device_offset, segment.size, segment.stride)

    _set_switches(2, 1, lean=1)
    os.environ["V7_COMPACT_GHOST_LISTS"] = "1"
    chain = partition_v7.compute_chain_partition(
        _synthetic_global_case(case_v7, depth_count=depth_count), [1.0, 1.0, 1.0], pool_safety=1.2)
    for index, slab in enumerate(chain.slabs):
        replica_region = slab.capacities.replica_region_size
        covered: list = []
        for direction in ("leading", "trailing"):
            os.environ["V7_PACKED_REPLICAS"] = "0"
            plain, _ = simulator_v7.SphSimulatorV7._compute_transport_segments(
                _fake_simulator(simulator_v7.SphSimulatorV7, slab), direction)
            os.environ["V7_PACKED_REPLICAS"] = "1"
            packed, total = simulator_v7.SphSimulatorV7._compute_transport_segments(
                _fake_simulator(simulator_v7.SphSimulatorV7, slab), direction)
            os.environ["V7_PACKED_REPLICAS"] = "0"
            if not plain:
                continue
            base = (0 if direction == "leading" else 1) * 64 * replica_region
            expected_blocks = [("ghost_packed_words", base + 4 * replica_region * words, stride * replica_region, stride)
                               for words, stride in ((0, 16), (4, 16), (8, 16), (12, 16))]
            replica_plain = [segment for segment in plain if segment.region in ("inner", "outer")]
            rest_plain = [key(segment) for segment in plain if segment.region not in ("inner", "outer")]
            blocks = [key(segment) for segment in packed if segment.buffer_name == "ghost_packed_words"]
            rest_packed = [key(segment) for segment in packed if segment.buffer_name != "ghost_packed_words"]
            tag = f"packed slab {index} {direction}"
            if blocks != expected_blocks:
                failures.append(f"{tag}: packed blocks {blocks} != {expected_blocks}")
            if rest_packed != rest_plain:
                failures.append(f"{tag}: non-replica segments changed")
            regions = [segment.region for segment in packed if segment.buffer_name == "ghost_packed_words"]
            if regions != ["inner", "inner", "outer", "outer"]:
                failures.append(f"{tag}: block regions {regions}")
            plain_bytes = sum(segment.stride for segment in replica_plain)
            packed_bytes = sum(segment.stride for segment in packed if segment.buffer_name == "ghost_packed_words")
            if (plain_bytes, packed_bytes) != (88, 64):
                failures.append(f"{tag}: bytes per G1+G2 replica pair {plain_bytes} -> {packed_bytes}, expected 88 -> 64")
            covered.extend((segment.device_offset, segment.device_offset + segment.size)
                           for segment in packed if segment.buffer_name == "ghost_packed_words")
            offset = 0
            for segment in packed:
                if segment.staging_offset != offset:
                    failures.append(f"{tag}: staging gap")
                offset = segment.staging_offset + segment.size
            if offset != total:
                failures.append(f"{tag}: staging total")
        # the blocks lie inside the allocation, never overlap, and a slab with both
        # peers covers it exactly (the shader's packed_layer_base addresses the same
        # words: see check_packed_shader_layout)
        os.environ["V7_PACKED_REPLICAS"] = "1"
        specs = {spec.name: spec for spec in simulator_v7.SphSimulatorV7._build_buffer_specs(
            _fake_simulator(simulator_v7.SphSimulatorV7, slab))}
        os.environ["V7_PACKED_REPLICAS"] = "0"
        allocation = specs["ghost_packed_words"].size
        if allocation != 2 * 64 * replica_region:
            failures.append(f"packed slab {index}: ghost_packed_words {allocation} B, expected {2 * 64 * replica_region}")
        covered.sort()
        if any(first[1] > second[0] for first, second in zip(covered, covered[1:])):
            failures.append(f"packed slab {index}: overlapping packed blocks {covered}")
        if covered and covered[-1][1] > allocation:
            failures.append(f"packed slab {index}: packed block beyond the allocation")
        both_peers = slab.transport.has_leading_peer and slab.transport.has_trailing_peer
        if both_peers and sum(end - start for start, end in covered) != allocation:
            failures.append(f"packed slab {index}: blocks do not tile the allocation")
    os.environ.pop("V7_COMPACT_GHOST_LISTS", None)
    os.environ.pop("V7_PACKED_REPLICAS", None)


def check_packed_shader_layout(failures: list) -> None:
    """The GLSL side of the packed format agrees with the segment table: the
    packed_layer_base word formula (16 words per direction, 8 per layer), both
    layers written / read as [x y z rho | vx vy vz material-bits] (no separate G1
    material word, no pressure), and every specialization constant id names one
    constant across the v6 shaders (V7_DIAG_POISON_G1 is id 99)."""
    import re
    shader_directory = pathlib.Path(__file__).resolve().parent / "shaders"
    helpers = (shader_directory / "helpers.glsl").read_text(encoding="utf-8")
    match = re.search(r"uint packed_layer_base\(uint direction, uint layer\) \{\s*return "
                      r"\(direction \* (\d+)u \+ layer \* (\d+)u\) \* REPLICA_REGION_SIZE;", helpers)
    if not match or (int(match.group(1)), int(match.group(2))) != (16, 8):
        failures.append(f"packed_layer_base is not (direction * 16u + layer * 8u) * R: "
                        f"{match.groups() if match else 'not found'}")
    sender = (shader_directory / "ghost_send.comp").read_text(encoding="utf-8")
    expander = (shader_directory / "expand_ghost_lists.comp").read_text(encoding="utf-8")
    for name, text in (("ghost_send.comp", sender), ("expand_ghost_lists.comp", expander)):
        if "8u * REPLICA_REGION_SIZE" in text:
            failures.append(f"{name}: still addresses the old separate G1 material block (8 R)")
    if "uintBitsToFloat(material[source_particle_id])" not in sender:
        failures.append("ghost_send.comp: the packed record does not carry the material bits")
    # the packed branch of ghost_send writes exactly the two vec4 blocks of the record (no extra G1 word) and reads
    # no pressure (.y of density_pressure or of any local holding it)
    start = sender.find("if (PACKED_REPLICAS) {")
    stop = sender.find("} else {", start)
    branch = sender[start:stop] if start >= 0 and stop > start else ""
    if not branch:
        failures.append("ghost_send.comp: PACKED_REPLICAS branch not found")
    else:
        if branch.count("store_packed_vec4(") != 2 or "ghost_packed_words[" in branch:
            failures.append(f"ghost_send.comp: the packed branch stores {branch.count('store_packed_vec4(')} vec4 "
                            "blocks and / or single words; expected the two blocks of the 32 B record")
        if re.search(r"pressure\w*(\[[^\]]*\])?\.y", branch):
            failures.append("ghost_send.comp: the packed branch still reads a pressure (.y)")
    if "floatBitsToUint(velocity_material.w)" not in expander:
        failures.append("expand_ghost_lists.comp: material is not read from the record's .w bits")
    # V7_DIAG_POISON_G1: the pressure branch writes the NaN into P, the density branch into rho, both for layer 0
    # (G1) only, and the unpacked SoA row is written from those two values
    poison_checks = (
        (r"layer == 0u && DIAGNOSTIC_POISON_INNER_REPLICA == DIAGNOSTIC_POISON_PRESSURE\)\s*\{\s*"
         r"pressure = uintBitsToFloat\(QUIET_NAN_BITS\);", "pressure poison"),
        (r"layer == 0u && DIAGNOSTIC_POISON_INNER_REPLICA == DIAGNOSTIC_POISON_DENSITY\)\s*\{\s*"
         r"density = uintBitsToFloat\(QUIET_NAN_BITS\);", "density poison"),
        (r"density_pressure\[particle_id\]\s*=\s*vec2\(density, pressure\);", "unpacked density / pressure store"),
        (r"QUIET_NAN_BITS\s*=\s*0x7FC00000u;", "quiet NaN bits"),
    )
    for pattern, label in poison_checks:
        if not re.search(pattern, expander):
            failures.append(f"expand_ghost_lists.comp: {label} not found")
    names_by_id: dict = {}
    for path in sorted(shader_directory.glob("*.comp")) + sorted(shader_directory.glob("*.glsl")):
        for constant_id, name in re.findall(r"constant_id\s*=\s*(\d+)\)\s*const\s+\w+\s+(\w+)",
                                            path.read_text(encoding="utf-8")):
            names_by_id.setdefault(int(constant_id), set()).add(name)
    for constant_id, names in sorted(names_by_id.items()):
        if len(names) > 1:
            failures.append(f"spec constant id {constant_id} names {sorted(names)}")
    if names_by_id.get(99) != {"DIAGNOSTIC_POISON_INNER_REPLICA"}:
        failures.append(f"spec constant 99 is {names_by_id.get(99)}, expected DIAGNOSTIC_POISON_INNER_REPLICA")


def check_spirv_current(failures: list) -> bool:
    """Every tracked SPIR-V file equals a fresh glslc build of its source (the
    compile_shaders_v7 flags; glslc output is deterministic): catches a commit
    whose shader edit was not recompiled. Skipped when glslc is absent."""
    import subprocess
    import tempfile
    from experiment.v7 import compile_shaders_v7
    if not os.path.isfile(compile_shaders_v7.GLSLC):
        print(f"[seam_layout] glslc not found ({compile_shaders_v7.GLSLC}): SPIR-V freshness check skipped")
        return False
    shader_directory = pathlib.Path(compile_shaders_v7.V7_SHADER_DIR)
    with tempfile.TemporaryDirectory() as temporary:
        for source in sorted(shader_directory.glob("*.comp")):
            if source.name.startswith("_"):
                continue
            output = pathlib.Path(temporary) / f"{source.name}.spv"
            result = subprocess.run([compile_shaders_v7.GLSLC, "--target-env=vulkan1.2", "-O", "-I",
                                     str(shader_directory), str(source), "-o", str(output)],
                                    capture_output=True, text=True)
            tracked = shader_directory / "spv" / f"{source.name}.spv"
            if result.returncode != 0:
                failures.append(f"{source.name}: glslc failed: {result.stderr.strip()[:200]}")
            elif not tracked.exists() or tracked.read_bytes() != output.read_bytes():
                failures.append(f"{source.name}: tracked SPIR-V is stale (run compile_shaders_v7.py)")
    return True


def check_packed_rejection(failures: list) -> None:
    """V7_PACKED_REPLICAS=1 without two ghost layers, without compact lists, or
    with V7_DIAG_GHOST_SELF lacking 'density' (no G1 pressure travels) is
    rejected; V7_DIAG_POISON_G1 parses off / pressure / density into spec 99 of
    the global entries, and rejects other values and a run without packing."""
    import experiment.v7.utils.case_v7 as case_v7
    import experiment.v7.utils.partition_v7 as partition_v7
    import experiment.v7.utils.simulator_v7 as simulator_v7
    for ghost_layers, keep_departed, compact in ((1, 1, 1), (2, 1, 0)):
        _set_switches(ghost_layers, keep_departed, lean=1)
        os.environ["V7_COMPACT_GHOST_LISTS"] = str(compact)
        os.environ["V7_PACKED_REPLICAS"] = "1"
        try:
            partition_v7.configured_packed_replicas()
            failures.append(f"packed accepted with layers={ghost_layers} compact={compact}")
        except ValueError:
            pass
    _set_switches(2, 1, lean=1)
    os.environ["V7_COMPACT_GHOST_LISTS"] = "1"
    os.environ["V7_PACKED_REPLICAS"] = "1"
    os.environ["V7_DIAG_GHOST_SELF"] = "correction"
    try:
        partition_v7.configured_packed_replicas()
        failures.append("packed accepted with V7_DIAG_GHOST_SELF=correction (force would read G1 P = 0)")
    except ValueError:
        pass
    os.environ.pop("V7_DIAG_GHOST_SELF", None)
    slab = partition_v7.compute_chain_partition(_synthetic_global_case(case_v7), [1.0, 1.0],
                                                pool_safety=1.2).slabs[0]
    fake = _fake_simulator(simulator_v7.SphSimulatorV7, slab)
    for text, expected in (("off", 0), ("", 0), ("0", 0), ("pressure", 1), ("Density", 2), (None, 0)):
        if text is None:
            os.environ.pop("V7_DIAG_POISON_G1", None)
        else:
            os.environ["V7_DIAG_POISON_G1"] = text
        try:
            entries = simulator_v7.SphSimulatorV7._global_entries(fake)
        except ValueError as error:
            failures.append(f"V7_DIAG_POISON_G1={text!r}: {error}")
            continue
        identifiers = [entry[0] for entry in entries]
        if len(identifiers) != len(set(identifiers)):
            failures.append(f"global spec entries repeat an id: {sorted(identifiers)}")
        if (99, "I", expected) not in entries:
            failures.append(f"V7_DIAG_POISON_G1={text!r}: spec 99 entry missing or != {expected}")
    parse_poison = getattr(partition_v7, "configured_diagnostic_poison_inner_replica", None)
    if parse_poison is None:
        failures.append("partition_v7.configured_diagnostic_poison_inner_replica is missing")
    for text, packed in ((("nan", "1"), ("pressure", "0")) if parse_poison else ()):
        os.environ["V7_DIAG_POISON_G1"] = text
        os.environ["V7_PACKED_REPLICAS"] = packed
        try:
            parse_poison()
            failures.append(f"V7_DIAG_POISON_G1={text} accepted with V7_PACKED_REPLICAS={packed}")
        except ValueError:
            pass
    os.environ.pop("V7_DIAG_POISON_G1", None)
    os.environ.pop("V7_COMPACT_GHOST_LISTS", None)
    os.environ.pop("V7_PACKED_REPLICAS", None)


def check_band_widths(failures: list) -> None:
    """V7_BAND_WIDTHS="c,d,f": parsing (default 2,3,4), rejection of c < 2, d < c,
    f < d + 1 and malformed values, rejection of non-default widths with
    V7_BAND_COMPACT_DISPATCH, spec 82 of the correction / density / force split
    pipelines, and no literal band width left at a spec 82 entry or a band
    dispatch site of the simulator source."""
    import re
    import experiment.v7.utils.case_v7 as case_v7
    import experiment.v7.utils.partition_v7 as partition_v7
    import experiment.v7.utils.simulator_v7 as simulator_v7
    parse = getattr(partition_v7, "configured_band_widths", None)
    if parse is None:
        failures.append("partition_v7.configured_band_widths is missing")
        return
    for text, expected in ((None, (2, 2, 3)), ("2,3,4", (2, 3, 4)), ("2,2,3", (2, 2, 3)), (" 3, 3, 5 ", (3, 3, 5))):
        if text is None:
            os.environ.pop("V7_BAND_WIDTHS", None)
        else:
            os.environ["V7_BAND_WIDTHS"] = text
        try:
            widths = parse()
            if widths != expected:
                failures.append(f"V7_BAND_WIDTHS={text!r} parsed as {widths}, expected {expected}")
        except ValueError as error:
            failures.append(f"V7_BAND_WIDTHS={text!r} rejected: {error}")
    for text in ("1,2,3", "3,2,4", "2,2,2", "2,3,3", "2,3", "2,3,4,5", "a,b,c", ""):
        os.environ["V7_BAND_WIDTHS"] = text
        try:
            parse()
            failures.append(f"V7_BAND_WIDTHS={text!r} accepted")
        except ValueError:
            pass
    # unset with V7_BAND_COMPACT_DISPATCH=1: the compact list's 2/3/4
    os.environ.pop("V7_BAND_WIDTHS", None)
    os.environ["V7_BAND_COMPACT_DISPATCH"] = "1"
    if parse() != (2, 3, 4):
        failures.append(f"V7_BAND_WIDTHS unset with V7_BAND_COMPACT_DISPATCH=1 parsed as {parse()}")
    os.environ.pop("V7_BAND_COMPACT_DISPATCH", None)
    _set_switches(2, 1)
    slab = partition_v7.compute_chain_partition(_synthetic_global_case(case_v7), [1.0, 1.0],
                                                pool_safety=1.2).slabs[0]
    fake = _fake_simulator(simulator_v7.SphSimulatorV7, slab)
    compact = simulator_v7._BAND_COMPACT
    try:
        for text, expected in (("2,3,4", (2, 3, 4)), ("2,2,3", (2, 2, 3))):
            os.environ["V7_BAND_WIDTHS"] = text
            simulator_v7._BAND_COMPACT = False
            fake.band_widths = simulator_v7.SphSimulatorV7._configured_band_widths(fake)
            for band_dispatch in (0, 1):
                entries = (simulator_v7.SphSimulatorV7._correction_mode_entries(fake, 2, band_dispatch)
                           + simulator_v7.SphSimulatorV7._density_mode_entries(fake, 2, band_dispatch)
                           + simulator_v7.SphSimulatorV7._force_mode_entries(fake, 2, 1, band_dispatch))
                ranges = tuple(entry[2] for entry in entries if entry[0] == 82)
                if ranges != expected:
                    failures.append(f"V7_BAND_WIDTHS={text}: spec 82 of correction / density / force = {ranges}")
            simulator_v7._BAND_COMPACT = True
            try:
                simulator_v7.SphSimulatorV7._configured_band_widths(fake)
                if expected != (2, 3, 4):
                    failures.append(f"V7_BAND_WIDTHS={text} accepted with V7_BAND_COMPACT_DISPATCH=1")
            except ValueError as error:
                if expected == (2, 3, 4):
                    failures.append(f"default band widths rejected with V7_BAND_COMPACT_DISPATCH=1: {error}")
    finally:
        simulator_v7._BAND_COMPACT = compact
        os.environ.pop("V7_BAND_WIDTHS", None)
    source = pathlib.Path(simulator_v7.__file__).read_text(encoding="utf-8")
    for pattern, label in ((r"\(82, 'I', \d", "spec 82 entry"),
                           (r"_per_band_dispatch_count\(\s*\d", "band dispatch size"),
                           (r"_band_thread_count\(\s*\d", "band thread count"),
                           (r"dispatch_boundary\(\"\w+\", \d", "single-GPU band dispatch")):
        if re.search(pattern, source):
            failures.append(f"simulator_v7.py: literal band width at a {label}")


def check_release_defaults(failures: list) -> None:
    """E6b: with no V7_* variable set the configuration is the recommended release
    set (v6_opt.md), per dimension for the pool factors; dependent switches follow
    an old value chosen alone; LEGACY_DEFAULTS restores every pre-E6b value."""
    import re
    import experiment.v7.utils.case_v7 as case_v7
    import experiment.v7.utils.partition_v7 as partition_v7
    saved = {key: value for key, value in os.environ.items() if key.startswith("V7_")}
    for key in saved:
        del os.environ[key]
    try:
        case_2d = _synthetic_global_case(case_v7)
        case_3d = _synthetic_global_case(case_v7, depth_count=3)
        expected = {"keep_departed": True, "ghost_layers": 2, "lean": True, "compact": True, "packed": True,
                    "clamp": True, "band_widths": (2, 2, 3),
                    "pools_2d": (0.29, 0.05, 0.8), "pools_3d": (0.5, 0.02, 0.64)}

        def current():
            return {"keep_departed": partition_v7.configured_keep_departed(),
                    "ghost_layers": partition_v7.configured_ghost_layers(),
                    "lean": partition_v7.configured_lean_transport(),
                    "compact": partition_v7.configured_compact_ghost_lists(),
                    "packed": partition_v7.configured_packed_replicas(),
                    "clamp": partition_v7.configured_init_seam_clamp(),
                    "band_widths": partition_v7.configured_band_widths(),
                    "pools_2d": tuple(function(case_2d) for function in (
                        partition_v7.configured_ghost_pool_factor, partition_v7.configured_migrant_pool_factor,
                        partition_v7.configured_departed_face_fraction)),
                    "pools_3d": tuple(function(case_3d) for function in (
                        partition_v7.configured_ghost_pool_factor, partition_v7.configured_migrant_pool_factor,
                        partition_v7.configured_departed_face_fraction))}
        if case_3d.physics.dimension != 3 or case_2d.physics.dimension != 2:
            failures.append("synthetic cases do not have dimension 2 / 3")
        got = current()
        for key, value in expected.items():
            if got[key] != value:
                failures.append(f"release default {key} = {got[key]}, expected {value}")
        # an old value chosen alone: the dependent switches follow, nothing raises
        for name, value, checks in (("V7_KEEP_DEPARTED", "0", {"ghost_layers": 1, "packed": False}),
                                    ("V7_BAND_VOXEL_DISPATCH", "0", {"ghost_layers": 1, "packed": False}),
                                    ("V7_GHOST_LAYERS", "1", {"packed": False}),
                                    ("V7_COMPACT_GHOST_LISTS", "0", {"packed": False})):
            os.environ[name] = value
            try:
                got = current()
                for key, wanted in checks.items():
                    if got[key] != wanted:
                        failures.append(f"{name}={value}: {key} = {got[key]}, expected {wanted}")
            except ValueError as error:
                failures.append(f"{name}={value} alone raised: {error}")
            del os.environ[name]
        os.environ["V7_GHOST_POOL_FACTOR"] = "0.25"
        if partition_v7.configured_migrant_pool_factor(case_2d) != 0.25:
            failures.append("V7_MIGRANT_POOL_FACTOR unset does not follow an explicit V7_GHOST_POOL_FACTOR")
        del os.environ["V7_GHOST_POOL_FACTOR"]
        os.environ.update(partition_v7.LEGACY_DEFAULTS)
        legacy = {"keep_departed": False, "ghost_layers": 1, "lean": False, "compact": False, "packed": False,
                  "clamp": False, "band_widths": (2, 3, 4), "pools_2d": (1.0, 1.0, 0.25), "pools_3d": (1.0, 1.0, 0.25)}
        got = current()
        for key, value in legacy.items():
            if got[key] != value:
                failures.append(f"LEGACY_DEFAULTS {key} = {got[key]}, expected {value}")
        for key in partition_v7.LEGACY_DEFAULTS:
            del os.environ[key]
        # E32: phase A's frame_done wait was the default until E32; the pinned baselines keep it
        if partition_v7.LEGACY_DEFAULTS.get("V7_PHASE_A_NO_WAIT") != "0":
            failures.append("LEGACY_DEFAULTS does not pin V7_PHASE_A_NO_WAIT=0")
    finally:
        for key in [key for key in os.environ if key.startswith("V7_")]:
            del os.environ[key]
        os.environ.update(saved)
    # defaults read outside partition_v7 (simulator at import, transport worker and
    # Vulkan context at construction): checked in the source
    sources = (("utils/simulator_v7.py", r'os\.environ\.get\("V7_BAND_SLOT_LANES", "64"\)'),
               ("utils/simulator_v7.py", r'os\.environ\.get\("V7_PHASE_A_NO_WAIT", "1"\) == "1"'),   # E32
               ("utils/transport_v7.py", r'os\.environ\.get\("V7_WORKER_COUNT_AWARE", "1"\)'),
               ("utils/vulkan_context_v7.py", r'os\.environ\.get\("V7_SPLIT_TRANSFER_QUEUES", "1"\)'))
    for relative, pattern in sources:
        text = (pathlib.Path(__file__).resolve().parent / relative).read_text(encoding="utf-8")
        if not re.search(pattern, text):
            failures.append(f"{relative}: release default {pattern} not found")


def main() -> int:
    failures: list = []
    check_release_defaults(failures)
    check_v5_equivalence(failures)
    check_two_layer_algebra(failures)
    for depth_count in (1, 3):            # 2-D and a 3-D case with NZ = 3
        check_lean_transport(failures, depth_count)
        check_compact_ghost_lists(failures, depth_count)
        check_packed_replicas(failures, depth_count)
    check_packed_rejection(failures)
    check_packed_shader_layout(failures)
    check_band_widths(failures)
    spirv_checked = check_spirv_current(failures)
    _set_switches(1, 0)
    if failures:
        print(f"[seam_layout] {len(failures)} FAILURE(S):")
        for failure in failures:
            print("  - " + failure)
        return 1
    print("[seam_layout] ALL PASS (layers=1 == v5 partition + transport; layers=2 column/pid algebra, "
          "segment layout, install range; lean / compact / packed segments in 2-D and 3-D, packed allocation "
          "tiling; packed rejection + V7_DIAG_POISON_G1 parsing; packed shader layout + poison branches + spec ids; "
          "V7_BAND_WIDTHS parsing / rejection / spec 82; release defaults (E6b, phase A no-wait E32) + "
          "LEGACY_DEFAULTS; "
          + ("SPIR-V current)" if spirv_checked else "SPIR-V check SKIPPED: no glslc)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
