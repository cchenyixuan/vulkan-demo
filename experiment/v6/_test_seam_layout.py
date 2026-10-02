"""
_test_seam_layout.py — CPU-only checks of the V6 seam layout (no Vulkan device).

1. V6_GHOST_LAYERS=1, V6_KEEP_DEPARTED=0: the v6 chain partition and the per-
   direction transport segments are field-for-field those of v5.
2. V6_KEEP_DEPARTED=1: only the departed pool size differs from (1).
3. V6_GHOST_LAYERS=2: for every link and direction, the column algebra of
   ghost_send.comp (outbox columns, migrant source columns, departed voxel ids)
   composed with the transport (ghost voxel range copied in x order) and the
   voxel-id / pid offsets puts every replica, migrant and departed particle at
   the right GLOBAL column on the right side: the receiver's inner ghost column
   holds the sender's seam column, its outer ghost column the column behind it,
   migrants land in the receiver's own seam column, departed copies sit in the
   sender's ghost column that mirrors the receiver's seam column.
4. V6_GHOST_LAYERS=2 transport segments: regions partition the ghost pool, the
   replica regions carry exactly the 4 sweep fields, every per-particle segment
   has a live-count word, staging is contiguous and ends with the frame stamp.
5. V6_LEAN_TRANSPORT=1 (both layouts, with / without V6_TRANSPORT_EXTENSION):
   the partition is unchanged, the particle segments are the full layout's
   segments restricted to the 4 read fields (+ extension_fields), same device
   ranges, count words and stamp; 44 (60) B per migrant slot.

Usage:
    .venv/Scripts/python.exe experiment/v6/_test_seam_layout.py
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
                           particles_per_voxel_side=4):
    """A degenerate (no-peer) 2-D global case on a lattice, built in memory."""
    smoothing_length = 0.01
    spacing = smoothing_length / particles_per_voxel_side
    x_values = (np.arange(column_count * particles_per_voxel_side) + 0.5) * spacing
    y_values = (np.arange(row_count * particles_per_voxel_side) + 0.5) * spacing
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="ij")
    positions = np.zeros((grid_x.size, 3), dtype=np.float32)
    positions[:, 0] = grid_x.ravel()
    positions[:, 1] = grid_y.ravel()
    material_group = np.where(positions[:, 1] < 2 * spacing, 1, 0).astype(np.uint32)
    physics = case_module.PhysicsConstants(
        smoothing_length=smoothing_length, speed_of_sound=100.0, delta_coefficient=0.1,
        power_parameter=7.0, cfl_number=0.15, timestep=7.5e-6, gravity=(0.0, 0.0, 0.0),
        dimension=2, neighbor_z_range=0, kernel_coefficient=1.0,
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
                                  grid_dimension_y=row_count, grid_dimension_z=1)
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
    return case_module.CaseV6(
        physics=physics, numerics=numerics, capacities=capacities, grid=grid,
        ghost_grid=case_module.GhostGridParams(0, 0),
        transport=case_module.TransportConfig(), materials=materials, initial=initial) \
        if hasattr(case_module, "CaseV6") else case_module.CaseV5(
        physics=physics, numerics=numerics, capacities=capacities, grid=grid,
        ghost_grid=case_module.GhostGridParams(0, 0),
        transport=case_module.TransportConfig(), materials=materials, initial=initial)


def _set_switches(ghost_layers: int, keep_departed: int, lean: int = 0,
                  extension: int = 0) -> None:
    os.environ["V6_GHOST_LAYERS"] = str(ghost_layers)
    os.environ["V6_KEEP_DEPARTED"] = str(keep_departed)
    os.environ["V6_LEAN_TRANSPORT"] = str(lean)
    os.environ["V6_TRANSPORT_EXTENSION"] = str(extension)
    os.environ.pop("V6_DEPARTED_CAPACITY", None)


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


V6_ONLY_FIELDS = {"departed_pool_size", "replica_region_size", "ghost_layers"}


def _strip_v6_only(tree):
    if isinstance(tree, dict):
        return {key: _strip_v6_only(value) for key, value in tree.items()
                if key not in V6_ONLY_FIELDS}
    if isinstance(tree, list):
        return [_strip_v6_only(item) for item in tree]
    return tree


def _strip_unset_restart_densities(tree):
    """The v5 working tree carries an uncommitted checkpoint-restart field
    InitialParticles.densities (None for a fresh start); v6 is cut from HEAD."""
    if isinstance(tree, dict):
        return {key: _strip_unset_restart_densities(value) for key, value in tree.items()
                if not (key == "densities" and value is None)}
    if isinstance(tree, list):
        return [_strip_v6_only(item) for item in tree]
    return tree


def _fake_simulator(simulator_class, case):
    simulator = object.__new__(simulator_class)
    simulator.case = case
    return simulator


def check_v5_equivalence(failures: list) -> None:
    import experiment.v5.utils.case_v5 as case_v5
    import experiment.v5.utils.partition_v5 as partition_v5
    import experiment.v5.utils.simulator_v5 as simulator_v5
    import experiment.v6.utils.case_v6 as case_v6
    import experiment.v6.utils.partition_v6 as partition_v6
    import experiment.v6.utils.simulator_v6 as simulator_v6

    for keep_departed in (0, 1):
        _set_switches(1, keep_departed)
        for slab_count in (2, 3, 4):
            weights = [1.0] * slab_count
            chain_v5 = partition_v5.compute_chain_partition(
                _synthetic_global_case(case_v5), weights, pool_safety=1.2)
            chain_v6 = partition_v6.compute_chain_partition(
                _synthetic_global_case(case_v6), weights, pool_safety=1.2)
            if chain_v5.cuts != chain_v6.cuts:
                failures.append(f"K={slab_count}: cuts differ {chain_v5.cuts} vs {chain_v6.cuts}")
            for index, (slab_v5, slab_v6) in enumerate(zip(chain_v5.slabs, chain_v6.slabs)):
                tree_v5 = _strip_unset_restart_densities(_comparable(slab_v5))
                tree_v6 = _strip_v6_only(_comparable(slab_v6))
                if tree_v5 != tree_v6:
                    differing = [key for key in tree_v5 if tree_v5[key] != tree_v6.get(key)]
                    failures.append(f"keep={keep_departed} K={slab_count} slab {index}: "
                                    f"partition differs from v5 in {differing}")
                peers = (slab_v6.transport.has_leading_peer + slab_v6.transport.has_trailing_peer)
                expected_departed = 0 if (keep_departed == 0 or peers == 0) else \
                    max(partition_v6.DEPARTED_CAPACITY_FLOOR,
                        int(np.ceil(0.25 * slab_v6.grid.grid_dimension_y * peers)))
                if slab_v6.capacities.departed_pool_size != expected_departed:
                    failures.append(f"keep={keep_departed} K={slab_count} slab {index}: departed "
                                    f"{slab_v6.capacities.departed_pool_size} != {expected_departed}")
                if slab_v6.capacities.replica_region_size != 0 or slab_v6.ghost_grid.ghost_layers != 1:
                    failures.append(f"K={slab_count} slab {index}: layers=1 slab has two-layer fields")
                # transport segments identical to v5 (ignoring the v6 count annotations)
                for direction in ("leading", "trailing"):
                    segments_v5, total_v5 = simulator_v5.SphSimulatorV5._compute_transport_segments(
                        _fake_simulator(simulator_v5.SphSimulatorV5, slab_v5), direction)
                    segments_v6, total_v6 = simulator_v6.SphSimulatorV6._compute_transport_segments(
                        _fake_simulator(simulator_v6.SphSimulatorV6, slab_v6), direction)
                    plain_v5 = [(s.buffer_name, s.device_offset, s.staging_offset, s.size) for s in segments_v5]
                    plain_v6 = [(s.buffer_name, s.device_offset, s.staging_offset, s.size) for s in segments_v6]
                    if plain_v5 != plain_v6 or total_v5 != total_v6:
                        failures.append(f"K={slab_count} slab {index} {direction}: transport "
                                        f"segments differ from v5")
                    if segments_v6:
                        count_offsets = {s.count_staging_offset for s in segments_v6[:9]}
                        if count_offsets != {segments_v6[-2].staging_offset}:
                            failures.append(f"K={slab_count} slab {index} {direction}: count-aware "
                                            f"plan does not point at the send count word")


def _global_column(geometry, local_x: int) -> int:
    return geometry.own_global_first_column + (local_x - geometry.leading_thickness)


def check_two_layer_algebra(failures: list) -> None:
    import experiment.v6.utils.case_v6 as case_v6
    import experiment.v6.utils.partition_v6 as partition_v6
    import experiment.v6.utils.simulator_v6 as simulator_v6

    _set_switches(2, 1)
    for slab_count in (2, 3, 4):
        chain = partition_v6.compute_chain_partition(
            _synthetic_global_case(case_v6), [1.0] * slab_count, pool_safety=1.2)
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
            fake = _fake_simulator(simulator_v6.SphSimulatorV6, sender)
            for direction in ("leading", "trailing"):
                if getattr(sender.transport, direction) is None:
                    continue
                segments, total = simulator_v6.SphSimulatorV6._compute_transport_segments(fake, direction)
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
                if segments[-1].device_offset != simulator_v6._OFFSET_FRAME_STAMP:
                    failures.append(f"K={slab_count} s{sender_index} {direction}: stamp is not the last segment")
                for name, ranges in coverage.items():
                    if name in simulator_v6._REPLICA_TRANSPORT_FIELDS:
                        expected = [(0, replica_region), (replica_region, replica_region),
                                    (2 * replica_region, pool - 2 * replica_region)]
                    else:
                        expected = [(2 * replica_region, pool - 2 * replica_region)]
                    if sorted(ranges) != expected:
                        failures.append(f"K={slab_count} s{sender_index} {direction}: {name} covers "
                                        f"{sorted(ranges)}, expected {expected}")
                overrides = fake._recv_status_overrides[direction]
                send_count = (simulator_v6._OFFSET_GHOST_SEND_LEADING if direction == "leading"
                              else simulator_v6._OFFSET_GHOST_SEND_TRAILING)
                receive_count = (simulator_v6._OFFSET_GHOST_RECV_LEADING if direction == "leading"
                                 else simulator_v6._OFFSET_GHOST_RECV_TRAILING)
                if overrides.get(send_count) != receive_count:
                    failures.append(f"K={slab_count} s{sender_index} {direction}: migrant count is not "
                                    f"uploaded into the install count slot")
            install_threads = simulator_v6.SphSimulatorV6._per_ghost_pid_dispatch_count(
                fake, "trailing" if sender.transport.has_trailing_peer else "leading")
            workgroup = sender.capacities.workgroup_size
            migrant_slots = (sender.capacities.trailing_ghost_pool_size or sender.capacities.leading_ghost_pool_size) \
                - 2 * replica_region
            if install_threads != (migrant_slots + workgroup - 1) // workgroup:
                failures.append(f"K={slab_count} s{sender_index}: install dispatch {install_threads} does not "
                                f"cover exactly the migrant region")


def check_lean_transport(failures: list) -> None:
    """V6_LEAN_TRANSPORT: the per-particle segments of the V5 mixed pool and
    of the two-layer migrant region are exactly the 4 read fields (+
    extension_fields with V6_TRANSPORT_EXTENSION), at the SAME device offsets
    and sizes as the full layout; voxel lists, count words and stamp are
    unchanged; the count-aware plan still points every particle segment at
    its live-count word; staging stays contiguous."""
    import experiment.v6.utils.case_v6 as case_v6
    import experiment.v6.utils.partition_v6 as partition_v6
    import experiment.v6.utils.simulator_v6 as simulator_v6

    lean_fields = {"position_voxel_id", "velocity_mass", "density_pressure", "material"}

    def key(segment):
        return (segment.buffer_name, segment.device_offset, segment.size, segment.stride)

    for ghost_layers, keep_departed in ((1, 0), (1, 1), (2, 1)):
        for extension in (0, 1):
            _set_switches(ghost_layers, keep_departed, lean=0)
            full_chain = partition_v6.compute_chain_partition(
                _synthetic_global_case(case_v6), [1.0, 1.0, 1.0], pool_safety=1.2)
            _set_switches(ghost_layers, keep_departed, lean=1, extension=extension)
            lean_chain = partition_v6.compute_chain_partition(
                _synthetic_global_case(case_v6), [1.0, 1.0, 1.0], pool_safety=1.2)
            expected_fields = lean_fields | ({"extension_fields"} if extension else set())
            tag = f"layers={ghost_layers} keep={keep_departed} ext={extension}"
            for index, (full_slab, lean_slab) in enumerate(zip(full_chain.slabs, lean_chain.slabs)):
                if _comparable(full_slab) != _comparable(lean_slab):
                    failures.append(f"{tag} slab {index}: the lean switch changed the partition")
                replica_region = lean_slab.capacities.replica_region_size

                def is_replica_segment(segment):
                    return (replica_region > 0
                            and segment.buffer_name in simulator_v6._REPLICA_TRANSPORT_FIELDS
                            and segment.size == segment.stride * replica_region)

                for direction in ("leading", "trailing"):
                    _set_switches(ghost_layers, keep_departed, lean=0)
                    full_segments, full_total = simulator_v6.SphSimulatorV6._compute_transport_segments(
                        _fake_simulator(simulator_v6.SphSimulatorV6, full_slab), direction)
                    _set_switches(ghost_layers, keep_departed, lean=1, extension=extension)
                    lean_segments, lean_total = simulator_v6.SphSimulatorV6._compute_transport_segments(
                        _fake_simulator(simulator_v6.SphSimulatorV6, lean_slab), direction)
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
                    if lean_segments[-1].device_offset != simulator_v6._OFFSET_FRAME_STAMP:
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


def check_compact_ghost_lists(failures: list) -> None:
    """V6_COMPACT_GHOST_LISTS: the inside_particle_index segment (ghost voxels x
    MAX_PARTICLES_PER_VOXEL x 4 B) becomes one ghost_voxel_first_particle_id word
    per ghost voxel at device offset 4 x first ghost vid; every other segment
    keeps its device range; staging stays contiguous, count words / stamp last."""
    import experiment.v6.utils.case_v6 as case_v6
    import experiment.v6.utils.partition_v6 as partition_v6
    import experiment.v6.utils.simulator_v6 as simulator_v6

    def key(segment):
        return (segment.buffer_name, segment.device_offset, segment.size, segment.stride)

    for ghost_layers, keep_departed in ((1, 0), (1, 1), (2, 1)):
        _set_switches(ghost_layers, keep_departed, lean=1)
        chain = partition_v6.compute_chain_partition(
            _synthetic_global_case(case_v6), [1.0, 1.0, 1.0], pool_safety=1.2)
        tag = f"compact layers={ghost_layers} keep={keep_departed}"
        for index, slab in enumerate(chain.slabs):
            cap_inside = slab.capacities.max_particles_per_voxel
            for direction in ("leading", "trailing"):
                os.environ["V6_COMPACT_GHOST_LISTS"] = "0"
                full, _ = simulator_v6.SphSimulatorV6._compute_transport_segments(
                    _fake_simulator(simulator_v6.SphSimulatorV6, slab), direction)
                os.environ["V6_COMPACT_GHOST_LISTS"] = "1"
                compact, total = simulator_v6.SphSimulatorV6._compute_transport_segments(
                    _fake_simulator(simulator_v6.SphSimulatorV6, slab), direction)
                os.environ["V6_COMPACT_GHOST_LISTS"] = "0"
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
                if offset != total or compact[-1].device_offset != simulator_v6._OFFSET_FRAME_STAMP:
                    failures.append(f"{tag} slab {index} {direction}: staging total / stamp")


def check_packed_replicas(failures: list) -> None:
    """V6_PACKED_REPLICAS (two layers, compact lists): the 8 replica SoA segments
    become 5 ghost_packed_words blocks (G1 16/16/4 B, G2 16/16 B per replica) at
    direction base d * 68 R bytes, with the inner / outer count words; the
    migrant region, voxel lists, count words and stamp are unchanged."""
    import experiment.v6.utils.case_v6 as case_v6
    import experiment.v6.utils.partition_v6 as partition_v6
    import experiment.v6.utils.simulator_v6 as simulator_v6

    def key(segment):
        return (segment.buffer_name, segment.device_offset, segment.size, segment.stride)

    _set_switches(2, 1, lean=1)
    os.environ["V6_COMPACT_GHOST_LISTS"] = "1"
    chain = partition_v6.compute_chain_partition(
        _synthetic_global_case(case_v6), [1.0, 1.0, 1.0], pool_safety=1.2)
    for index, slab in enumerate(chain.slabs):
        replica_region = slab.capacities.replica_region_size
        for direction in ("leading", "trailing"):
            os.environ["V6_PACKED_REPLICAS"] = "0"
            plain, _ = simulator_v6.SphSimulatorV6._compute_transport_segments(
                _fake_simulator(simulator_v6.SphSimulatorV6, slab), direction)
            os.environ["V6_PACKED_REPLICAS"] = "1"
            packed, total = simulator_v6.SphSimulatorV6._compute_transport_segments(
                _fake_simulator(simulator_v6.SphSimulatorV6, slab), direction)
            os.environ["V6_PACKED_REPLICAS"] = "0"
            if not plain:
                continue
            base = (0 if direction == "leading" else 1) * 68 * replica_region
            expected_blocks = [("ghost_packed_words", base + 4 * replica_region * words, stride * replica_region, stride)
                               for words, stride in ((0, 16), (4, 16), (8, 4), (9, 16), (13, 16))]
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
            if regions != ["inner", "inner", "inner", "outer", "outer"]:
                failures.append(f"{tag}: block regions {regions}")
            plain_bytes = sum(segment.stride for segment in replica_plain)
            packed_bytes = sum(segment.stride for segment in packed if segment.buffer_name == "ghost_packed_words")
            if (plain_bytes, packed_bytes) != (88, 68):
                failures.append(f"{tag}: bytes per G1+G2 replica pair {plain_bytes} -> {packed_bytes}, expected 88 -> 68")
            offset = 0
            for segment in packed:
                if segment.staging_offset != offset:
                    failures.append(f"{tag}: staging gap")
                offset = segment.staging_offset + segment.size
            if offset != total:
                failures.append(f"{tag}: staging total")
    os.environ.pop("V6_COMPACT_GHOST_LISTS", None)
    os.environ.pop("V6_PACKED_REPLICAS", None)


def main() -> int:
    failures: list = []
    check_v5_equivalence(failures)
    check_two_layer_algebra(failures)
    check_lean_transport(failures)
    check_compact_ghost_lists(failures)
    check_packed_replicas(failures)
    _set_switches(1, 0)
    if failures:
        print(f"[seam_layout] {len(failures)} FAILURE(S):")
        for failure in failures:
            print("  - " + failure)
        return 1
    print("[seam_layout] ALL PASS (layers=1 == v5 partition + transport; layers=2 column/pid algebra, "
          "segment layout, install range; lean transport segments; compact ghost lists; packed replicas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
