"""
particle_watch.py - a read-only monitor of a chain run, checked after every step: the debug switch --watch of
experiment/seam_audit/canonical_dump.py (off by default; report section 20, 2026-11-19).

After the bootstrap and after every step the monitor reads back from every slab the set-0 buffers it needs
(position_voxel_id, velocity_mass, density_pressure, material, extension_fields, shift, acceleration), the voxel
lists (inside_particle_count / inside_particle_index) and the global status, logs per slab the own particles that
sit in a ghost column or in no voxel list, and checks three events:

  wall_layer   an own alive fluid particle beyond a face of the fluid box on some axis (x_a < lower_a or
               x_a > upper_a). The box is the crossing box: numerics.wall_crossing's box as the loader resolved it,
               or, for simple / adami (no crossing settings), the same rule the loader applies when case.yaml gives
               no box: the fluid's initial bounding box widened by one particle radius.
  count        fewer own alive fluid particles on all slabs together (no ghost, no departed copy) than at the start.
  non_finite   a non-finite position or velocity component of an own alive particle (fluid or wall).

The run loop is the orchestrator's legacy loop at depth 1: frame n + 1 is submitted only after the monitor has read
frame n, so every read sees a finished step and an idle chain (the frame's transport included: phase C waits for
the upload, frame_done follows phase C). The monitor only reads (one-shot copy commands into host staging,
readback_buffers_batch / readback_global_status); it writes no device buffer and changes no dispatch, no spec
constant and no switch. Rows are identified by the global id canonical_dump puts in extension_fields, so
--transport-extension is required; replicas (packed, no extension_fields) are identified at the event by their
position bytes, which are copies of the sender's.

Memory, for the trajectories before an event (the solver does not repeat itself bit for bit, so an event cannot be
re-run from a checkpoint to be looked at): a ring of the last ring_length steps of every fluid particle, indexed by
global id (position, velocity, density, pressure, slab, row, whether a voxel list holds it), and the raw per-slab
buffers of the last raw_history steps. At the first event observe() raises WatchEvent; the caller then calls save_event(), which writes
the bundle read by particle_watch_report.py:

  watch_steps.jsonl      one line per checked step (counts, counter deltas, box margin, largest speed; the largest
                         fluid displacement since the previous step, shift and acceleration, each with the
                         particle's id, position and x - x_cut)
  event.json             what happened, the box, the cuts, the slabs' grids and pools, the event particles
  raw_step<n>.npz        for each of the last raw_history steps: every slab's buffers (all rows) and voxel lists
  ring.npz               the ring, for the fluid particles within ring_radius_factor * h of one of the first
                         RING_CENTRES event particles (at the event step) and for every event or vanished particle
"""
from __future__ import annotations

import collections
import json
import pathlib
import time

import numpy as np

GLOBAL_ID_LOW_BASE = 2 ** 20
STATUS_DELTA_KEYS = ("overflow_ghost_count", "overflow_inside_count", "overflow_incoming_count",
                     "overflow_install_tail", "overflow_install_inside", "overflow_departed_count",
                     "far_migration_count", "stamp_error_count", "migration_install_count",
                     "correction_fallback_count", "wall_crossing_gradient_fallback_count")
STATUS_LEVEL_KEYS = ("departed_count", "ghost_send_leading_count", "ghost_send_trailing_count",
                     "ghost_recv_leading_count", "ghost_recv_trailing_count",
                     "replica_inner_send_leading_count", "replica_inner_send_trailing_count",
                     "replica_outer_send_leading_count", "replica_outer_send_trailing_count",
                     "replica_inner_recv_leading_count", "replica_inner_recv_trailing_count",
                     "replica_outer_recv_leading_count", "replica_outer_recv_trailing_count", "frame_stamp")
READ_BUFFERS = ("position_voxel_id", "velocity_mass", "density_pressure", "material", "extension_fields",
                "shift", "acceleration", "inside_particle_count", "inside_particle_index")
EVENT_PARTICLES_SAVED = 64          # event particles named in event.json (the deepest first)
RING_CENTRES = 8                    # ring.npz keeps the neighbourhoods of the first (deepest) event particles


class WatchEvent(Exception):
    """The first step with an event; observe() has already recorded it, save_event() writes the bundle."""

    def __init__(self, step: int, events: dict) -> None:
        super().__init__(f"step {step}: " + "; ".join(f"{name}: {value['summary']}" for name, value in events.items()))
        self.step = step
        self.events = events


def fluid_box(global_case) -> tuple[np.ndarray, np.ndarray, str]:
    """(lower, upper, source) of the fluid box: the crossing box, or the loader's rule for a case without one."""
    from experiment.v7.utils.case_v7 import KIND_BOUNDARY
    from experiment.v7.utils.wall_crossing_v7 import WallCrossingSettings, resolve_wall_crossing_box
    settings = getattr(global_case.numerics, "wall_crossing", None)
    if settings is not None and settings.box_lower is not None:
        source = "numerics.wall_crossing box" + (" (case.yaml)" if settings.box_from_case else
                                                 " (loader: fluid bounding box + one particle radius)")
        return np.asarray(settings.box_lower, float), np.asarray(settings.box_upper, float), source
    kinds = np.asarray([material.kind for material in global_case.materials])
    is_wall = kinds[global_case.initial.material_group] == KIND_BOUNDARY
    radius = float(next(material for material in global_case.materials if material.kind == 0).radius)
    resolved = resolve_wall_crossing_box(WallCrossingSettings(), global_case.initial.positions, is_wall, radius,
                                         int(global_case.physics.dimension))
    return (np.asarray(resolved.box_lower, float), np.asarray(resolved.box_upper, float),
            "fluid bounding box + one particle radius (the crossing loader's rule; this case has no crossing box)")


class ParticleWatch:
    def __init__(self, simulators: list, global_case, cuts: list, out_directory: pathlib.Path,
                 ring_length: int = 128, raw_history: int = 3, ring_radius_factor: float = 4.0) -> None:
        from experiment.v7.utils.case_v7 import KIND_FLUID
        self.simulators = simulators
        self.global_case = global_case
        self.dimension = int(global_case.physics.dimension)
        self.support_radius = float(global_case.physics.smoothing_length)
        self.timestep = float(global_case.physics.timestep)
        self.out = pathlib.Path(out_directory)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ring_length = int(ring_length)
        self.raw_history = int(raw_history)
        self.ring_radius_factor = float(ring_radius_factor)
        self.lower, self.upper, self.box_source = fluid_box(global_case)
        self.spacing = 2.0 * min(float(material.radius) for material in global_case.materials
                                 if float(material.radius) > 0)
        self.fluid_groups = np.array([index for index, material in enumerate(global_case.materials)
                                      if material.kind == KIND_FLUID], dtype=np.uint32)
        groups = np.asarray(global_case.initial.material_group)
        self.global_count = int(groups.size)
        is_fluid = np.isin(groups, self.fluid_groups)
        self.fluid_global_ids = np.flatnonzero(is_fluid).astype(np.int64)
        self.fluid_ordinal = np.full(self.global_count, -1, dtype=np.int64)
        self.fluid_ordinal[self.fluid_global_ids] = np.arange(self.fluid_global_ids.size)
        self.initial_fluid_count = int(self.fluid_global_ids.size)
        origin_x = float(global_case.grid.origin_x)
        self.cut_columns = [int(cut) for cut in cuts]
        self.cut_world_x = [origin_x + cut * self.support_radius for cut in self.cut_columns]
        self.ring = collections.deque(maxlen=self.ring_length)
        self.raw = collections.deque(maxlen=self.raw_history)
        self.previous_status = None
        self.previous_present = None
        self.log_path = self.out / "watch_steps.jsonl"
        self.log_path.unlink(missing_ok=True)
        self.started = time.perf_counter()
        self.checked_steps = 0
        self.read_seconds = 0.0

    # ------------------------------------------------------------------ reading
    def _read_slab(self, slab_index: int) -> dict:
        simulator = self.simulators[slab_index]
        capacities = simulator.case.capacities
        capacity = capacities.total_pool_capacity()
        raw = simulator.readback_buffers_batch(list(READ_BUFFERS))
        status = simulator.readback_global_status()
        slots = capacities.max_particles_per_voxel
        counts = np.frombuffer(raw["inside_particle_count"], np.uint32).copy()
        voxel_slots = counts.size
        return {
            "position": np.frombuffer(raw["position_voxel_id"], np.float32)[:capacity * 4].reshape(capacity, 4).copy(),
            "velocity": np.frombuffer(raw["velocity_mass"], np.float32)[:capacity * 4].reshape(capacity, 4).copy(),
            "density_pressure": np.frombuffer(raw["density_pressure"], np.float32)[:capacity * 2]
            .reshape(capacity, 2).copy(),
            "material": np.frombuffer(raw["material"], np.uint32)[:capacity].copy(),
            "extension": np.frombuffer(raw["extension_fields"], np.float32)[:capacity * 4].reshape(capacity, 4).copy(),
            "shift": np.frombuffer(raw["shift"], np.float32)[:capacity * 4].reshape(capacity, 4).copy(),
            "acceleration": np.frombuffer(raw["acceleration"], np.float32)[:capacity * 4].reshape(capacity, 4).copy(),
            "list_count": counts,
            "list_index": np.frombuffer(raw["inside_particle_index"], np.uint32)[:voxel_slots * slots]
            .reshape(voxel_slots, slots).copy(),
            "status": status,
        }

    def _own_rows(self, slab_index: int, slab: dict) -> dict:
        simulator = self.simulators[slab_index]
        first = simulator.own_first_pid()
        stop = first + simulator.case.capacities.own_pool_size
        position = slab["position"][first:stop]
        velocity = slab["velocity"][first:stop]
        alive = (velocity[:, 3] > 0) & (position[:, 3] > 0.5)
        rows = np.flatnonzero(alive) + first
        extension = slab["extension"][rows].astype(np.float64)
        global_id = np.rint(extension[:, 2]).astype(np.int64) * GLOBAL_ID_LOW_BASE + np.rint(extension[:, 3]).astype(np.int64)
        material = slab["material"][rows]
        return {"rows": rows, "global_id": global_id, "fluid": np.isin(material, self.fluid_groups)}

    def _listed(self, slab_index: int, slab: dict) -> np.ndarray:
        """Boolean per row: referenced by some voxel list (within its count)."""
        capacity = slab["position"].shape[0]
        counts = slab["list_count"]
        slots = slab["list_index"].shape[1]
        within = np.arange(slots)[None, :] < np.minimum(counts, slots)[:, None]
        pids = slab["list_index"][within]
        listed = np.zeros(capacity, dtype=bool)
        listed[pids[pids < capacity]] = True
        return listed

    # ------------------------------------------------------------------ one step
    def observe(self, step: int) -> None:
        start = time.perf_counter()
        slabs = [self._read_slab(index) for index in range(len(self.simulators))]
        self.read_seconds += time.perf_counter() - start
        record = {"step": step, "t": step * self.timestep, "slabs": []}
        fluid_count = 0
        position_ring = np.full((self.initial_fluid_count, 3), np.nan, dtype=np.float32)
        velocity_ring = np.full((self.initial_fluid_count, 3), np.nan, dtype=np.float32)
        density_pressure_ring = np.full((self.initial_fluid_count, 2), np.nan, dtype=np.float32)
        slab_ring = np.full(self.initial_fluid_count, -1, dtype=np.int8)
        row_ring = np.full(self.initial_fluid_count, -1, dtype=np.int32)
        listed_ring = np.zeros(self.initial_fluid_count, dtype=np.uint8)
        outside_rows, non_finite_rows = [], []
        all_ids = []
        maximum_speed = 0.0
        smallest_margin = np.inf
        largest = {"shift": (0.0, -1, None), "acceleration": (0.0, -1, None)}
        for slab_index, slab in enumerate(slabs):
            own = self._own_rows(slab_index, slab)
            rows, global_id, fluid = own["rows"], own["global_id"], own["fluid"]
            all_ids.append(global_id[fluid])
            position = slab["position"][rows, :3]
            velocity = slab["velocity"][rows, :3]
            finite = np.isfinite(position).all(axis=1) & np.isfinite(velocity).all(axis=1)
            for row, gid in zip(rows[~finite], global_id[~finite]):
                non_finite_rows.append((slab_index, int(row), int(gid)))
            fluid_rows = rows[fluid]
            fluid_ids = global_id[fluid]
            fluid_position = position[fluid].astype(np.float64)
            margin = np.minimum(fluid_position - self.lower, self.upper - fluid_position)[:, :self.dimension]
            beyond = (margin < 0).any(axis=1)
            for row, gid, depth in zip(fluid_rows[beyond], fluid_ids[beyond], margin[beyond].min(axis=1)):
                outside_rows.append((slab_index, int(row), int(gid), float(depth)))
            if fluid_position.shape[0]:
                finite_margin = margin.min(axis=1)[np.isfinite(margin.min(axis=1))]
                if finite_margin.size:
                    smallest_margin = min(smallest_margin, float(finite_margin.min()))
                speed = np.linalg.norm(velocity[fluid].astype(np.float64), axis=1)
                speed = speed[np.isfinite(speed)]
                if speed.size:
                    maximum_speed = max(maximum_speed, float(speed.max()))
                for name in ("shift", "acceleration"):
                    magnitude = np.linalg.norm(slab[name][fluid_rows, :3].astype(np.float64), axis=1)
                    magnitude[~np.isfinite(magnitude)] = -1.0
                    best = int(np.argmax(magnitude))
                    if magnitude[best] > largest[name][0]:
                        largest[name] = (float(magnitude[best]), int(fluid_ids[best]),
                                         fluid_position[best].tolist())
            valid = (fluid_ids >= 0) & (fluid_ids < self.global_count)
            ordinal = np.full(fluid_ids.size, -1, dtype=np.int64)
            ordinal[valid] = self.fluid_ordinal[fluid_ids[valid]]
            keep = ordinal >= 0
            position_ring[ordinal[keep]] = slab["position"][fluid_rows[keep], :3]
            velocity_ring[ordinal[keep]] = slab["velocity"][fluid_rows[keep], :3]
            density_pressure_ring[ordinal[keep]] = slab["density_pressure"][fluid_rows[keep]]
            slab_ring[ordinal[keep]] = slab_index
            row_ring[ordinal[keep]] = fluid_rows[keep]
            fluid_count += int(fluid.sum())
            listed = self._listed(slab_index, slab)
            listed_ring[ordinal[keep]] = listed[fluid_rows[keep]]
            # own particles whose stored voxel id lies in a ghost column (helpers.glsl is_own_voxel)
            ghost_grid = self.simulators[slab_index].case.ghost_grid
            grid = self.simulators[slab_index].case.grid
            voxel = np.rint(slab["position"][rows, 3]).astype(np.int64)
            in_ghost = (voxel <= ghost_grid.leading_ghost_voxel_count) | (
                voxel > grid.total_voxel_count() - ghost_grid.trailing_ghost_voxel_count)
            status = slab["status"]
            previous = self.previous_status[slab_index] if self.previous_status else None
            record["slabs"].append({
                "own_alive": int(rows.size), "own_fluid": int(fluid.sum()), "own_wall": int((~fluid).sum()),
                "own_alive_unlisted": int((~listed[rows]).sum()),
                "own_in_ghost_column": int(in_ghost.sum()),
                "own_fluid_unlisted": int((~listed[fluid_rows]).sum()),
                "fluid_ids_not_fluid": int((~keep).sum()),
                "delta": {key: int(status.get(key, 0)) - (int(previous.get(key, 0)) if previous else 0)
                          for key in STATUS_DELTA_KEYS},
                "level": {key: int(status.get(key, 0)) for key in STATUS_LEVEL_KEYS},
            })
        self.previous_status = [slab["status"] for slab in slabs]
        ids = np.concatenate(all_ids) if all_ids else np.zeros(0, np.int64)
        unique_ids = np.unique(ids)
        present = slab_ring >= 0
        vanished = (np.flatnonzero(self.previous_present & ~present) if self.previous_present is not None
                    else np.zeros(0, np.int64))
        displacement_record = None
        if self.ring:
            moved = np.linalg.norm(position_ring.astype(np.float64) - self.ring[-1]["position"].astype(np.float64), axis=1)
            moved[~np.isfinite(moved)] = -1.0
            best = int(np.argmax(moved))
            displacement_record = {"value": float(moved[best]), "global_id": int(self.fluid_global_ids[best]),
                                   "position": position_ring[best].astype(float).tolist(),
                                   "median": float(np.median(moved[moved >= 0])) if (moved >= 0).any() else None}
        for name, (value, gid, where) in largest.items():
            record[f"largest_fluid_{name}"] = {"value": value, "global_id": gid, "position": where}
        record["largest_fluid_displacement_since_previous_check"] = displacement_record
        if self.cut_world_x:
            for key in ("largest_fluid_shift", "largest_fluid_acceleration",
                        "largest_fluid_displacement_since_previous_check"):
                item = record.get(key)
                if item and item.get("position"):
                    item["x_minus_cut"] = [item["position"][0] - x for x in self.cut_world_x]
        record.update({"fluid_total": fluid_count, "fluid_unique_ids": int(unique_ids.size),
                       "fluid_duplicate_rows": int(ids.size - unique_ids.size),
                       "fluid_missing_ids": int((~present).sum()),
                       "fluid_vanished_since_previous_step": int(vanished.size),
                       "fluid_outside_box": len(outside_rows), "non_finite": len(non_finite_rows),
                       "smallest_box_margin": smallest_margin if np.isfinite(smallest_margin) else None,
                       "largest_fluid_speed": maximum_speed,
                       "read_seconds_total": round(self.read_seconds, 3),
                       "wall_seconds": round(time.perf_counter() - self.started, 3)})
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.ring.append({"step": step, "position": position_ring, "velocity": velocity_ring,
                          "density_pressure": density_pressure_ring, "slab": slab_ring, "row": row_ring,
                          "listed": listed_ring})
        self.raw.append({"step": step, "slabs": slabs})
        self.previous_present = present
        self.checked_steps += 1

        events = {}
        if outside_rows:
            outside_rows.sort(key=lambda item: item[3])
            events["wall_layer"] = {"summary": f"{len(outside_rows)} fluid particle(s) beyond the box face, deepest "
                                               f"{-outside_rows[0][3]:.3e} beyond",
                                    "particles": [{"slab": slab, "row": row, "global_id": gid, "box_margin": depth}
                                                  for slab, row, gid, depth in outside_rows[:EVENT_PARTICLES_SAVED]],
                                    "count": len(outside_rows)}
        if fluid_count < self.initial_fluid_count:
            events["count"] = {"summary": f"{fluid_count} own fluid particles, {self.initial_fluid_count} at the "
                                          f"start ({fluid_count - self.initial_fluid_count:+d}); "
                                          f"{vanished.size} vanished since the previous check",
                               "fluid_total": fluid_count, "initial": self.initial_fluid_count,
                               "vanished_global_ids": [int(self.fluid_global_ids[ordinal])
                                                       for ordinal in vanished[:4096]],
                               "vanished_count": int(vanished.size)}
        if non_finite_rows:
            events["non_finite"] = {"summary": f"{len(non_finite_rows)} own particle(s) with a non-finite position "
                                               "or velocity",
                                    "particles": [{"slab": slab, "row": row, "global_id": gid}
                                                  for slab, row, gid in non_finite_rows[:EVENT_PARTICLES_SAVED]],
                                    "count": len(non_finite_rows)}
        if events:
            raise WatchEvent(step, events)

    # ------------------------------------------------------------------ the bundle
    def slab_geometry(self) -> list:
        geometry = []
        for simulator in self.simulators:
            case = simulator.case
            capacities = case.capacities
            leading = capacities.leading_ghost_pool_size
            own = capacities.own_pool_size
            trailing = capacities.trailing_ghost_pool_size
            geometry.append({
                "origin_float32": [float(np.float32(case.grid.origin_x)), float(np.float32(case.grid.origin_y)),
                                   float(np.float32(case.grid.origin_z))],
                "origin": [float(case.grid.origin_x), float(case.grid.origin_y), float(case.grid.origin_z)],
                "grid_dimension": [int(case.grid.grid_dimension_x), int(case.grid.grid_dimension_y),
                                   int(case.grid.grid_dimension_z)],
                "leading_ghost_voxel_count": int(case.ghost_grid.leading_ghost_voxel_count),
                "trailing_ghost_voxel_count": int(case.ghost_grid.trailing_ghost_voxel_count),
                "ghost_layers": int(case.ghost_grid.ghost_layers),
                "max_particles_per_voxel": int(capacities.max_particles_per_voxel),
                "max_incoming_per_voxel": int(capacities.max_incoming_per_voxel),
                "replica_region_size": int(capacities.replica_region_size),
                "pid_ranges": {"leading_ghost": [1, leading], "own": [leading + 1, leading + own],
                               "trailing_ghost": [leading + own + 1, leading + own + trailing],
                               "departed": [leading + own + trailing + 1,
                                            leading + own + trailing + capacities.departed_pool_size]},
                "stored_density_offset": float(simulator.stored_density_offset()),
            })
        return geometry

    def save_event(self, event: WatchEvent, meta: dict) -> dict:
        """Write the bundle of the first event (the chain is idle at the event step: nothing was submitted
        after the step observe() read)."""
        last = self.ring[-1]
        primary_ids = []
        for name in ("wall_layer", "non_finite"):
            if name in event.events:
                primary_ids += [item["global_id"] for item in event.events[name]["particles"]]
        vanished = event.events.get("count", {}).get("vanished_global_ids", [])
        primary_ids += vanished[:EVENT_PARTICLES_SAVED]
        # fluid particles near an event particle at the event step (or, for a vanished one, at its last step)
        centres = []
        for gid in primary_ids:
            ordinal = self.fluid_ordinal[gid] if 0 <= gid < self.global_count else -1
            if ordinal < 0:
                continue
            for entry in reversed(self.ring):
                if np.isfinite(entry["position"][ordinal]).all():
                    centres.append(entry["position"][ordinal].astype(np.float64))
                    break
        radius = self.ring_radius_factor * self.support_radius
        selected = set(int(self.fluid_ordinal[gid]) for gid in primary_ids
                       if 0 <= gid < self.global_count and self.fluid_ordinal[gid] >= 0)
        if centres:
            from scipy.spatial import cKDTree
            positions = last["position"].astype(np.float64)
            finite = np.isfinite(positions).all(axis=1)
            tree = cKDTree(positions[finite])
            finite_ordinals = np.flatnonzero(finite)
            for centre in centres[:RING_CENTRES]:
                for index in tree.query_ball_point(centre, r=radius):
                    selected.add(int(finite_ordinals[index]))
        ordinals = np.array(sorted(selected), dtype=np.int64)
        steps = np.array([entry["step"] for entry in self.ring], dtype=np.int64)
        np.savez_compressed(
            self.out / "ring.npz", steps=steps, global_id=self.fluid_global_ids[ordinals],
            position=np.stack([entry["position"][ordinals] for entry in self.ring]),
            velocity=np.stack([entry["velocity"][ordinals] for entry in self.ring]),
            density_pressure=np.stack([entry["density_pressure"][ordinals] for entry in self.ring]),
            slab=np.stack([entry["slab"][ordinals] for entry in self.ring]),
            row=np.stack([entry["row"][ordinals] for entry in self.ring]),
            listed=np.stack([entry["listed"][ordinals] for entry in self.ring]))
        raw_files = []
        for entry in self.raw:
            arrays = {}
            for slab_index, slab in enumerate(entry["slabs"]):
                for key, value in slab.items():
                    if key != "status":
                        arrays[f"slab{slab_index}_{key}"] = value
            path = self.out / f"raw_step{entry['step']:06d}.npz"
            np.savez_compressed(path, **arrays)
            raw_files.append({"step": entry["step"], "file": path.name,
                              "status": [slab["status"] for slab in entry["slabs"]]})
        bundle = dict(meta)
        bundle.update({
            "event_step": event.step, "event_time": event.step * self.timestep, "events": event.events,
            "box_lower": self.lower.tolist(), "box_upper": self.upper.tolist(), "box_source": self.box_source,
            "dimension": self.dimension, "support_radius": self.support_radius, "timestep": self.timestep,
            "spacing": self.spacing,
            "cut_columns": self.cut_columns, "cut_world_x": self.cut_world_x,
            "global_origin": [float(self.global_case.grid.origin_x), float(self.global_case.grid.origin_y),
                              float(self.global_case.grid.origin_z)],
            "fluid_groups": self.fluid_groups.tolist(),
            "material_kinds": [int(material.kind) for material in self.global_case.materials],
            "material_names": [str(getattr(material, "name", index))
                               for index, material in enumerate(self.global_case.materials)],
            "initial_fluid_count": self.initial_fluid_count, "global_count": self.global_count,
            "slabs": self.slab_geometry(), "raw_files": raw_files,
            "ring": {"file": "ring.npz", "steps": [int(steps[0]), int(steps[-1])], "length": int(steps.size),
                     "particles": int(ordinals.size), "radius": radius},
            "checked_steps": self.checked_steps, "read_seconds": round(self.read_seconds, 3),
            "wall_seconds": round(time.perf_counter() - self.started, 3)})
        with open(self.out / "event.json", "w", encoding="utf-8") as handle:
            json.dump(bundle, handle, indent=1)
        return bundle

    def summary(self) -> dict:
        return {"checked_steps": self.checked_steps, "read_seconds": round(self.read_seconds, 3),
                "wall_seconds": round(time.perf_counter() - self.started, 3), "log": str(self.log_path),
                "box_lower": self.lower.tolist(), "box_upper": self.upper.tolist(), "box_source": self.box_source}
