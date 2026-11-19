"""
particle_watch_report.py - what the bundle of a particle_watch event holds, laid out for a reader (report section 20).

    .venv/Scripts/python.exe -m experiment.wall_crossing.particle_watch_report <bundle directory> \
        [--figures claude/wall_crossing/figures --figure-prefix particle_watch_<name>]

Reads event.json, raw_step<n>.npz, ring.npz and watch_steps.jsonl of a bundle (particle_watch.py) and writes into
<bundle>/report/:

  event_state.npz        every slab's particles at the event step: own alive rows and every row a voxel list holds
                         (replicas, departed copies), with slab, row, kind (0 own, 1 leading replica, 2 trailing
                         replica, 3 departed copy), global id (own rows: extension_fields; the others: the owner's
                         row with the same position bytes, -1 when none), position, velocity, density, pressure,
                         material, stored voxel id, shift, acceleration, listed (own rows: held by a voxel list)
  event_particles.json   per event particle: the state, the face / edge / corner it is beyond, the step it went
                         beyond, the distance to the cut, its column, whether a voxel list holds it, and its history
                         over the ring (displacement per step against |v| dt); for a count event the vanished
                         particles' last appearance and the transport counters of that step
  neighbours.json / .csv the primary particle's neighbour lists at the event step (and the two steps before): the
                         entries of the 27 voxels around its stored voxel on its slab within the support radius, and
                         every own particle of any slab within the support radius that the lists do not hold, with
                         the reason (no replica received, beyond the ghost layers, own particle in no list, ...)
  trajectories.npz       the ring of the particles within 3 support radii of the primary particle
  vtp/*.vtp              ParaView point sets of the region around the primary particle at several steps (the three
                         raw steps with replicas, earlier ring steps with the fluid and the walls), and the primary
                         particle's path as a poly line
  figures                slices through the primary particle (planes perpendicular to the face it crossed and / or
                         to the cut) and its history, also copied to --figures when given
  summary.json           the numbers the report quotes
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

GLOBAL_ID_LOW_BASE = 2 ** 20
KIND_NAMES = {0: "own", 1: "leading replica", 2: "trailing replica", 3: "departed copy"}
BOUNDARY_KIND = 1
AXIS_NAMES = "xyz"


# ---------------------------------------------------------------------------------------------- loading
def load_bundle(directory: pathlib.Path) -> dict:
    event = json.loads((directory / "event.json").read_text(encoding="utf-8"))
    raws = {}
    for item in event["raw_files"]:
        with np.load(directory / item["file"]) as archive:
            raws[int(item["step"])] = {name: archive[name] for name in archive.files}
    with np.load(directory / "ring.npz") as archive:
        ring = {name: archive[name] for name in archive.files}
    steps_log = [json.loads(line) for line in (directory / "watch_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    return {"event": event, "raws": raws, "ring": ring, "steps_log": steps_log}


def slab_state(raw: dict, slab_index: int, geometry: dict, fluid_groups: list) -> dict:
    """Rows of one slab at one raw step: own alive and every non-own row a voxel list references."""
    prefix = f"slab{slab_index}_"
    position = raw[prefix + "position"]
    velocity = raw[prefix + "velocity"]
    capacity = position.shape[0]
    counts = raw[prefix + "list_count"]
    index = raw[prefix + "list_index"]
    slots = index.shape[1]
    within = np.arange(slots)[None, :] < np.minimum(counts, slots)[:, None]
    listed = np.zeros(capacity, dtype=bool)
    referenced = index[within]
    listed[referenced[referenced < capacity]] = True
    ranges = geometry["pid_ranges"]
    rows = np.arange(capacity)
    kind = np.full(capacity, -1, dtype=np.int8)
    for code, name in ((1, "leading_ghost"), (0, "own"), (2, "trailing_ghost"), (3, "departed")):
        low, high = ranges[name]
        if high >= low:
            kind[(rows >= low) & (rows <= high)] = code
    own_alive = (kind == 0) & (velocity[:, 3] > 0) & (position[:, 3] > 0.5)
    keep = own_alive | (listed & (kind > 0))
    selected = np.flatnonzero(keep)
    extension = raw[prefix + "extension"][selected].astype(np.float64)
    own_id = np.rint(extension[:, 2]).astype(np.int64) * GLOBAL_ID_LOW_BASE + np.rint(extension[:, 3]).astype(np.int64)
    density_pressure = raw[prefix + "density_pressure"][selected].astype(np.float64)
    density_pressure[:, 0] += geometry["stored_density_offset"]
    return {"slab": np.full(selected.size, slab_index, dtype=np.int8), "row": selected.astype(np.int64),
            "kind": kind[selected], "own_global_id": np.where(kind[selected] == 0, own_id, -1),
            "position": position[selected, :3], "voxel": np.rint(position[selected, 3]).astype(np.int64),
            "velocity": velocity[selected, :3], "mass": velocity[selected, 3],
            "density": density_pressure[:, 0], "pressure": density_pressure[:, 1],
            "material": raw[prefix + "material"][selected], "shift": raw[prefix + "shift"][selected, :3],
            "acceleration": raw[prefix + "acceleration"][selected, :3], "listed": listed[selected],
            "fluid": np.isin(raw[prefix + "material"][selected], fluid_groups)}


def merged_state(raw: dict, event: dict) -> dict:
    parts = [slab_state(raw, index, geometry, event["fluid_groups"]) for index, geometry in enumerate(event["slabs"])]
    state = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    # global ids of replicas and departed copies: the owner's row with the same position bytes
    own = state["kind"] == 0
    key_of = {bytes(row): gid for row, gid in zip(np.ascontiguousarray(state["position"][own]), state["own_global_id"][own])}
    global_id = state["own_global_id"].copy()
    for position_index in np.flatnonzero(~own):
        global_id[position_index] = key_of.get(bytes(np.ascontiguousarray(state["position"][position_index])), -1)
    state["global_id"] = global_id
    return state


# ---------------------------------------------------------------------------------------------- geometry helpers
def voxel_coordinate(voxel_id: int, geometry: dict) -> tuple:
    _, grid_y, grid_z = geometry["grid_dimension"]
    zero_based = int(voxel_id) - 1
    return zero_based // (grid_y * grid_z), zero_based % grid_y, (zero_based // grid_y) % grid_z


def voxel_id_of(column: int, y_index: int, z_index: int, geometry: dict) -> int:
    _, grid_y, grid_z = geometry["grid_dimension"]
    return 1 + y_index + z_index * grid_y + column * grid_y * grid_z


def column_of_position(position, geometry: dict, support: float) -> int:
    origin = np.float32(geometry["origin_float32"][0])
    return int(np.floor((np.float32(position[0]) - origin) / np.float32(support)))


def column_role(column: int, geometry: dict) -> str:
    grid_x, grid_y, grid_z = geometry["grid_dimension"]
    leading = geometry["leading_ghost_voxel_count"] // (grid_y * grid_z)
    trailing = geometry["trailing_ghost_voxel_count"] // (grid_y * grid_z)
    if column < 0 or column >= grid_x:
        return "outside the slab grid"
    if column < leading:
        return f"leading ghost column ({'inner' if column == leading - 1 else 'outer'})"
    if column >= grid_x - trailing:
        return f"trailing ghost column ({'inner' if column == grid_x - trailing else 'outer'})"
    return "own column"


def faces_beyond(position, lower, upper, dimension: int) -> list:
    names = []
    for axis in range(dimension):
        if position[axis] < lower[axis]:
            names.append(AXIS_NAMES[axis] + "-")
        elif position[axis] > upper[axis]:
            names.append(AXIS_NAMES[axis] + "+")
    return names


def box_margin(position, lower, upper, dimension: int) -> float:
    position = np.asarray(position, dtype=np.float64)
    return float(np.minimum(position - lower, upper - position)[:dimension].min())


# ---------------------------------------------------------------------------------------------- neighbours
def neighbour_analysis(state: dict, event: dict, primary_index: int) -> dict:
    """The primary particle's voxel-list neighbours on its slab against every own particle within h."""
    support = float(event["support_radius"])
    slab = int(state["slab"][primary_index])
    geometry = event["slabs"][slab]
    centre = state["position"][primary_index].astype(np.float64)
    primary_id = int(state["global_id"][primary_index])
    stored = int(state["voxel"][primary_index])
    column, y_index, z_index = voxel_coordinate(stored, geometry)
    grid_x, grid_y, grid_z = geometry["grid_dimension"]
    on_slab = np.flatnonzero(state["slab"] == slab)
    row_to_index = {int(row): int(index) for row, index in zip(state["row"][on_slab], on_slab)}
    return {"slab": slab, "stored_voxel": stored, "stored_voxel_coordinate": [column, y_index, z_index],
            "computed_column": column_of_position(centre, geometry, support),
            "primary_listed": bool(state["listed"][primary_index]),
            "primary_column_role": column_role(column, geometry), "grid_dimension": [grid_x, grid_y, grid_z],
            "row_to_index": row_to_index, "centre": centre, "primary_id": primary_id}


def list_neighbours(raw: dict, state: dict, event: dict, primary_global_id: int) -> dict:
    support = float(event["support_radius"])
    candidates = np.flatnonzero((state["global_id"] == primary_global_id) & (state["kind"] == 0))
    if candidates.size == 0:
        return {"present": False}
    primary_index = int(candidates[0])
    base = neighbour_analysis(state, event, primary_index)
    slab = base["slab"]
    geometry = event["slabs"][slab]
    prefix = f"slab{slab}_"
    counts = raw[prefix + "list_count"]
    index = raw[prefix + "list_index"]
    slots = index.shape[1]
    column, y_index, z_index = base["stored_voxel_coordinate"]
    grid_x, grid_y, grid_z = base["grid_dimension"]
    z_range = (-1, 0, 1) if int(event["dimension"]) == 3 else (0,)
    entries = []
    for x_offset in (-1, 0, 1):
        for z_offset in z_range:
            for y_offset in (-1, 0, 1):
                neighbour_column, neighbour_y, neighbour_z = column + x_offset, y_index + y_offset, z_index + z_offset
                if not (0 <= neighbour_column < grid_x and 0 <= neighbour_y < grid_y and 0 <= neighbour_z < grid_z):
                    continue
                voxel = voxel_id_of(neighbour_column, neighbour_y, neighbour_z, geometry)
                count = int(counts[voxel])
                for slot in range(min(count, slots)):
                    entries.append((voxel, int(index[voxel, slot])))
    centre = base["centre"]
    visible = {}
    listed_rows = []
    for voxel, row in entries:
        state_index = base["row_to_index"].get(row)
        if state_index is None:
            listed_rows.append({"voxel": voxel, "row": row, "note": "row not in the state (dead own row in a list)"})
            continue
        distance = float(np.linalg.norm(state["position"][state_index].astype(np.float64) - centre))
        gid = int(state["global_id"][state_index])
        if gid == base["primary_id"]:
            continue
        item = {"global_id": gid, "row": row, "voxel": voxel, "kind": KIND_NAMES[int(state["kind"][state_index])],
                "material": int(state["material"][state_index]), "fluid": bool(state["fluid"][state_index]),
                "distance": distance, "distance_over_h": distance / support,
                "position": state["position"][state_index].astype(float).tolist()}
        if distance < support:
            visible.setdefault(gid, item)
    # every own particle of any slab within h (the set the neighbour sum should see)
    own = np.flatnonzero(state["kind"] == 0)
    distances = np.linalg.norm(state["position"][own].astype(np.float64) - centre, axis=1)
    true = own[(distances < support) & (state["global_id"][own] != base["primary_id"])]
    missing = []
    for state_index in true:
        gid = int(state["global_id"][state_index])
        if gid in visible:
            continue
        owner = int(state["slab"][state_index])
        position = state["position"][state_index]
        column_here = column_of_position(position, geometry, support)
        role = column_role(column_here, geometry)
        if owner == slab:
            reason = ("own particle of this slab in no voxel list" if not state["listed"][state_index]
                      else "own particle listed in a voxel outside the 27 around the stored voxel")
        elif role.startswith("leading ghost") or role.startswith("trailing ghost"):
            replicas = np.flatnonzero((state["slab"] == slab) & (state["kind"] > 0) & (state["global_id"] == gid))
            reason = ("replica present on this slab but not in the 27 voxels' lists" if replicas.size
                      else "no replica of it on this slab (it sits in a ghost column)")
        elif role == "own column":
            reason = "owned by the other slab but located in an own column of this slab (not migrated)"
        else:
            reason = f"beyond this slab's ghost layers ({role})"
        missing.append({"global_id": gid, "owner_slab": owner, "material": int(state["material"][state_index]),
                        "fluid": bool(state["fluid"][state_index]), "owner_listed": bool(state["listed"][state_index]),
                        "distance": float(np.linalg.norm(position.astype(np.float64) - centre)),
                        "distance_over_h": float(np.linalg.norm(position.astype(np.float64) - centre)) / support,
                        "column_on_this_slab": column_here, "column_role_on_this_slab": role, "reason": reason,
                        "position": position.astype(float).tolist()})
    extra = [item for gid, item in visible.items() if gid < 0 or not np.any(state["global_id"][true] == gid)]
    return {"present": True, "slab": slab, "primary_global_id": base["primary_id"],
            "stored_voxel": base["stored_voxel"], "stored_voxel_coordinate": base["stored_voxel_coordinate"],
            "stored_column_role": base["primary_column_role"], "computed_column_from_position": base["computed_column"],
            "primary_listed": base["primary_listed"], "list_entries_in_27_voxels": len(entries),
            "visible_within_h": sorted(visible.values(), key=lambda item: item["distance"]),
            "true_within_h": int(true.size), "missing_within_h": sorted(missing, key=lambda item: item["distance"]),
            "extra_within_h": extra, "dead_rows_in_lists": listed_rows}


# ---------------------------------------------------------------------------------------------- VTP
def write_vtp(path: pathlib.Path, points: np.ndarray, arrays: dict, lines: list | None = None) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    count = points.shape[0]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
                     "<PolyData>\n")
        line_count = len(lines) if lines else 0
        handle.write(f'<Piece NumberOfPoints="{count}" NumberOfVerts="{0 if lines else count}" '
                     f'NumberOfLines="{line_count}" NumberOfStrips="0" NumberOfPolys="0">\n<PointData>\n')
        for name, values in arrays.items():
            values = np.asarray(values)
            components = 1 if values.ndim == 1 else values.shape[1]
            kind = "Float64" if values.dtype.kind == "f" else "Int64"
            text = " ".join(repr(float(item)) if kind == "Float64" else str(int(item)) for item in values.reshape(-1))
            handle.write(f'<DataArray type="{kind}" Name="{name}" NumberOfComponents="{components}" '
                         f'format="ascii">{text}</DataArray>\n')
        handle.write("</PointData>\n<Points>\n")
        handle.write('<DataArray type="Float64" NumberOfComponents="3" format="ascii">'
                     + " ".join(repr(float(item)) for item in points.reshape(-1)) + "</DataArray>\n</Points>\n")
        if lines:
            connectivity = " ".join(str(int(item)) for line in lines for item in line)
            offsets, total = [], 0
            for line in lines:
                total += len(line)
                offsets.append(total)
            handle.write('<Lines>\n<DataArray type="Int64" Name="connectivity" format="ascii">' + connectivity
                         + '</DataArray>\n<DataArray type="Int64" Name="offsets" format="ascii">'
                         + " ".join(map(str, offsets)) + "</DataArray>\n</Lines>\n")
        else:
            handle.write('<Verts>\n<DataArray type="Int64" Name="connectivity" format="ascii">'
                         + " ".join(map(str, range(count))) + '</DataArray>\n<DataArray type="Int64" Name="offsets" '
                         'format="ascii">' + " ".join(map(str, range(1, count + 1))) + "</DataArray>\n</Verts>\n")
        handle.write("</Piece>\n</PolyData>\n</VTKFile>\n")


# ---------------------------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle")
    parser.add_argument("--figures", default=None, help="directory the figures are copied to (the report's)")
    parser.add_argument("--figure-prefix", default=None)
    arguments = parser.parse_args()
    directory = pathlib.Path(arguments.bundle).resolve()
    out = directory / "report"
    (out / "vtp").mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(directory)
    event, raws, ring, steps_log = bundle["event"], bundle["raws"], bundle["ring"], bundle["steps_log"]
    support = float(event["support_radius"])
    dimension = int(event["dimension"])
    timestep = float(event["timestep"])
    lower, upper = np.asarray(event["box_lower"]), np.asarray(event["box_upper"])
    cut_x = event["cut_world_x"]
    event_step = int(event["event_step"])
    kinds = event["material_kinds"]
    state = merged_state(raws[event_step], event)
    np.savez_compressed(out / "event_state.npz", **{key: value for key, value in state.items()})

    # ---- event particles
    ring_steps = ring["steps"]
    ring_id_index = {int(gid): index for index, gid in enumerate(ring["global_id"])}
    events = event["events"]
    primary_ids = []
    for name in ("wall_layer", "non_finite"):
        if name in events:
            primary_ids += [int(item["global_id"]) for item in events[name]["particles"]]
    vanished = [int(gid) for gid in events.get("count", {}).get("vanished_global_ids", [])]

    def history(gid: int) -> dict | None:
        index = ring_id_index.get(gid)
        if index is None:
            return None
        position = ring["position"][:, index].astype(np.float64)
        velocity = ring["velocity"][:, index].astype(np.float64)
        displacement = np.linalg.norm(np.diff(position, axis=0), axis=1)
        speed_step = np.linalg.norm(velocity[1:], axis=1) * timestep
        margins = [box_margin(row, lower, upper, dimension) if np.isfinite(row).all() else None for row in position]
        beyond_steps = [int(step) for step, margin in zip(ring_steps, margins) if margin is not None and margin < 0]
        listed = ring["listed"][:, index]
        slabs = ring["slab"][:, index]
        unlisted_steps = [int(step) for step, flag, slab in zip(ring_steps, listed, slabs) if slab >= 0 and not flag]
        return {"ring_first_step": int(ring_steps[0]), "ring_last_step": int(ring_steps[-1]),
                "first_step_beyond_box": beyond_steps[0] if beyond_steps else None,
                "first_step_in_no_voxel_list": unlisted_steps[0] if unlisted_steps else None,
                "steps_in_no_voxel_list": len(unlisted_steps),
                "slab_changes": [[int(ring_steps[step + 1]), int(slabs[step]), int(slabs[step + 1])]
                                 for step in range(len(slabs) - 1) if slabs[step] != slabs[step + 1]],
                "mean_displacement_per_step": float(np.nanmean(displacement)) if displacement.size else None,
                "mean_speed_times_timestep": float(np.nanmean(speed_step)) if speed_step.size else None,
                "last_displacement_per_step": float(displacement[-1]) if displacement.size else None,
                "last_speed_times_timestep": float(speed_step[-1]) if speed_step.size else None,
                "box_margin_first": margins[0], "box_margin_last": margins[-1]}

    particles = []
    for gid in primary_ids:
        candidates = np.flatnonzero((state["global_id"] == gid) & (state["kind"] == 0))
        if candidates.size == 0:
            continue
        index = int(candidates[0])
        slab = int(state["slab"][index])
        geometry = event["slabs"][slab]
        position = state["position"][index].astype(np.float64)
        stored_column = voxel_coordinate(int(state["voxel"][index]), geometry)[0]
        faces = faces_beyond(position, lower, upper, dimension)
        particles.append({
            "global_id": gid, "slab": slab, "row": int(state["row"][index]),
            "position": position.tolist(), "velocity": state["velocity"][index].astype(float).tolist(),
            "speed": float(np.linalg.norm(state["velocity"][index].astype(np.float64))),
            "density": float(state["density"][index]), "pressure": float(state["pressure"][index]),
            "material": int(state["material"][index]), "shift": state["shift"][index].astype(float).tolist(),
            "acceleration": state["acceleration"][index].astype(float).tolist(),
            "box_margin": box_margin(position, lower, upper, dimension), "beyond": faces,
            "beyond_kind": {1: "face", 2: "edge", 3: "corner"}.get(len(faces), "inside"),
            "distance_to_cut": [float(position[0] - x) for x in cut_x],
            "stored_column": stored_column, "stored_column_role": column_role(stored_column, geometry),
            "computed_column_from_position": column_of_position(position, geometry, support),
            "listed": bool(state["listed"][index]), "history": history(gid)})
    vanished_records = []
    if vanished:
        last_log = steps_log[-1]
        for gid in vanished[:256]:
            index = ring_id_index.get(gid)
            record = {"global_id": gid}
            if index is not None:
                present = np.flatnonzero(ring["slab"][:, index] >= 0)
                if present.size:
                    last = present[-1]
                    record.update({"last_step_seen": int(ring_steps[last]), "last_slab": int(ring["slab"][last, index]),
                                   "last_row": int(ring["row"][last, index]),
                                   "last_position": ring["position"][last, index].astype(float).tolist(),
                                   "last_velocity": ring["velocity"][last, index].astype(float).tolist(),
                                   "last_listed": bool(ring["listed"][last, index])})
            vanished_records.append(record)
        transport = {"step": last_log["step"], "per_slab_counter_deltas": [slab["delta"] for slab in last_log["slabs"]],
                     "per_slab_levels": [slab["level"] for slab in last_log["slabs"]],
                     "defrag": "none (a canonical run ends before the first defrag)"}
    else:
        transport = None
    primary = particles[0]["global_id"] if particles else (vanished[0] if vanished else None)
    (out / "event_particles.json").write_text(json.dumps(
        {"event_step": event_step, "events": {name: value["summary"] for name, value in events.items()},
         "particles": particles, "vanished": vanished_records, "transport_at_event_step": transport,
         "primary_global_id": primary}, indent=1), encoding="utf-8")

    # ---- neighbours of the primary at the event step and the raw steps before it
    neighbours = {}
    if particles:
        for step in sorted(raws):
            step_state = state if step == event_step else merged_state(raws[step], event)
            neighbours[str(step)] = list_neighbours(raws[step], step_state, event, primary)
        (out / "neighbours.json").write_text(json.dumps(neighbours, indent=1), encoding="utf-8")
        with open(out / "neighbours.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "set", "global_id", "kind_or_owner", "material", "fluid", "distance_over_h",
                             "reason", "x", "y", "z"])
            for step, item in neighbours.items():
                if not item.get("present"):
                    continue
                for neighbour in item["visible_within_h"]:
                    writer.writerow([step, "in the lists", neighbour["global_id"], neighbour["kind"],
                                     neighbour["material"], neighbour["fluid"], f"{neighbour['distance_over_h']:.4f}",
                                     "", *neighbour["position"]])
                for neighbour in item["missing_within_h"]:
                    writer.writerow([step, "missing", neighbour["global_id"], f"own of slab {neighbour['owner_slab']}",
                                     neighbour["material"], neighbour["fluid"], f"{neighbour['distance_over_h']:.4f}",
                                     neighbour["reason"], *neighbour["position"]])

    # ---- trajectories of the particles within 3 h of the primary
    centre = None
    if primary is not None and primary in ring_id_index:
        primary_index = ring_id_index[primary]
        present = np.flatnonzero(ring["slab"][:, primary_index] >= 0)
        centre = ring["position"][present[-1], primary_index].astype(np.float64)
        last_positions = ring["position"][present[-1]].astype(np.float64)
        near = np.flatnonzero(np.linalg.norm(last_positions - centre, axis=1) < 3.0 * support)
        np.savez_compressed(out / "trajectories.npz", steps=ring_steps, global_id=ring["global_id"][near],
                            position=ring["position"][:, near], velocity=ring["velocity"][:, near],
                            density_pressure=ring["density_pressure"][:, near], slab=ring["slab"][:, near],
                            listed=ring["listed"][:, near], primary_global_id=primary)

    # ---- VTP files of the region around the primary
    vtp_files = []
    if centre is not None:
        half = 4.0 * support
        walls = (state["kind"] == 0) & ~state["fluid"]

        def region(positions):
            return (np.abs(positions.astype(np.float64) - centre)[:, :dimension] <= half).all(axis=1)

        for step in sorted(raws):
            step_state = state if step == event_step else merged_state(raws[step], event)
            mask = region(step_state["position"])
            classes = np.where(step_state["kind"] > 0, 2, np.where(step_state["fluid"], 0, 1))
            classes = np.where((step_state["kind"] == 0) & ~step_state["listed"] & step_state["fluid"], 3, classes)
            flag = np.where(step_state["global_id"] == primary, 1, 0)
            name = f"region_step{step:06d}.vtp"
            write_vtp(out / "vtp" / name, step_state["position"][mask], {
                "velocity": step_state["velocity"][mask].astype(np.float64), "density": step_state["density"][mask],
                "pressure": step_state["pressure"][mask], "material": step_state["material"][mask].astype(np.int64),
                "slab": step_state["slab"][mask].astype(np.int64),
                "class_0fluid_1wall_2replica_3unlisted": classes[mask].astype(np.int64),
                "row_kind_0own_1lead_2trail_3departed": step_state["kind"][mask].astype(np.int64),
                "global_id": step_state["global_id"][mask], "listed": step_state["listed"][mask].astype(np.int64),
                "primary": flag[mask].astype(np.int64)})
            vtp_files.append(name)
        offsets = sorted({event_step - value for value in (5, 10, 20, 50, 100)} | {int(ring_steps[0])})
        wall_mask = walls & region(state["position"])
        for step in offsets:
            hits = np.flatnonzero(ring_steps == step)
            if step in raws or hits.size == 0:
                continue
            position = ring["position"][hits[0]].astype(np.float64)
            present = (ring["slab"][hits[0]] >= 0) & np.isfinite(position).all(axis=1)
            fluid_mask = present & (np.abs(position - centre)[:, :dimension] <= half).all(axis=1)
            points = np.concatenate([position[fluid_mask], state["position"][wall_mask].astype(np.float64)])
            density_pressure = ring["density_pressure"][hits[0]][fluid_mask].astype(np.float64)
            wall_count = int(wall_mask.sum())
            name = f"region_step{step:06d}_fluid_and_walls.vtp"
            write_vtp(out / "vtp" / name, points, {
                "velocity": np.concatenate([ring["velocity"][hits[0]][fluid_mask].astype(np.float64),
                                            state["velocity"][wall_mask].astype(np.float64)]),
                "density": np.concatenate([density_pressure[:, 0], state["density"][wall_mask]]),
                "pressure": np.concatenate([density_pressure[:, 1], state["pressure"][wall_mask]]),
                "slab": np.concatenate([ring["slab"][hits[0]][fluid_mask].astype(np.int64),
                                        state["slab"][wall_mask].astype(np.int64)]),
                "class_0fluid_1wall_2replica_3unlisted": np.concatenate([
                    np.where(ring["listed"][hits[0]][fluid_mask] > 0, 0, 3), np.ones(wall_count, np.int64)]),
                "global_id": np.concatenate([ring["global_id"][fluid_mask], state["global_id"][wall_mask]]),
                "primary": np.concatenate([(ring["global_id"][fluid_mask] == primary).astype(np.int64),
                                           np.zeros(wall_count, np.int64)])})
            vtp_files.append(name)
        primary_index = ring_id_index[primary]
        path_points = ring["position"][:, primary_index].astype(np.float64)
        valid = np.isfinite(path_points).all(axis=1)
        write_vtp(out / "vtp" / "primary_path.vtp", path_points[valid],
                  {"step": ring_steps[valid].astype(np.int64),
                   "slab": ring["slab"][valid, primary_index].astype(np.int64),
                   "listed": ring["listed"][valid, primary_index].astype(np.int64)},
                  lines=[list(range(int(valid.sum())))])
        vtp_files.append("primary_path.vtp")

    # ---- figures
    figure_files = []
    if centre is not None:
        figure_files = draw_figures(out, event, state, ring, steps_log, primary, centre, particles, neighbours)
        if arguments.figures:
            target = pathlib.Path(arguments.figures).resolve()
            target.mkdir(parents=True, exist_ok=True)
            prefix = arguments.figure_prefix or "particle_watch"
            for name in figure_files:
                shutil.copyfile(out / name, target / f"{prefix}_{name}")
    neighbour_summary = {}
    for step, item in neighbours.items():
        if item.get("present"):
            reasons = {}
            for neighbour in item["missing_within_h"]:
                key = ("wall " if not neighbour["fluid"] else "fluid ") + neighbour["reason"]
                reasons[key] = reasons.get(key, 0) + 1
            neighbour_summary[step] = {"in_lists_within_h": len(item["visible_within_h"]),
                                       "true_within_h": item["true_within_h"],
                                       "missing_within_h": len(item["missing_within_h"]), "missing_by_reason": reasons,
                                       "primary_listed": item["primary_listed"],
                                       "stored_column_role": item["stored_column_role"]}
    summary = {"bundle": str(directory), "event_step": event_step, "event_time": event["event_time"],
               "events": {name: value["summary"] for name, value in events.items()},
               "primary_global_id": primary, "primary": particles[0] if particles else None,
               "event_particle_count": len(particles), "vanished_count": len(vanished),
               "neighbours": neighbour_summary, "vtp": vtp_files, "figures": figure_files,
               "steps_log_last": steps_log[-1]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("event_step", "events", "primary_global_id", "neighbours",
                                                    "figures")}, indent=1))
    return 0


def draw_figures(out, event, state, ring, steps_log, primary, centre, particles, neighbours) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, SURFACE, setup_style
    setup_style()
    support = float(event["support_radius"])
    dimension = int(event["dimension"])
    lower, upper = np.asarray(event["box_lower"]), np.asarray(event["box_upper"])
    cut_x = event["cut_world_x"]
    event_step = int(event["event_step"])
    primary_state = np.flatnonzero((state["global_id"] == primary) & (state["kind"] == 0))
    slab = int(state["slab"][primary_state[0]]) if primary_state.size else 0
    spacing = float(event.get("spacing", support / 4))
    faces = particles[0]["beyond"] if particles else []
    face_axes = sorted({"xyz".index(face[0]) for face in faces}) or [1]
    face_axis = face_axes[0]
    # (plane axes, the axis normal to the plane, what the plane is perpendicular to)
    planes = []
    other = [axis for axis in range(3) if axis not in (0, face_axis)]
    if face_axis != 0:
        normal = other[0] if other else 2
        planes.append(((0, face_axis), normal, "perpendicular to the face crossed and to the cut"))
        planes.append(((face_axis, normal), 0, "perpendicular to the face crossed, parallel to the cut"))
        planes.append(((0, normal), face_axis, "perpendicular to the cut, parallel to the face crossed"))
    else:
        planes.append(((0, 1), 2, "perpendicular to the face crossed (x) and to the cut"))
        planes.append(((0, 2), 1, "perpendicular to the face crossed (x) and to the cut"))
    if dimension == 2:
        planes = [((0, 1), 2, "the plane of the 2-D case")]
    colours = {"fluid": "#2a78d6", "replica": "#eb6834", "unlisted": "#eda100", "missing": "#1baf7a",
               "wall": "#b5b3ad", "peer": "#86b6ef"}
    on_slab = state["slab"] == slab
    own_slab = on_slab & (state["kind"] == 0)
    replica = on_slab & (state["kind"] > 0)
    peer_own = (~on_slab) & (state["kind"] == 0)
    replica_ids = np.array(sorted(set(int(item) for item in state["global_id"][replica] if item >= 0)), dtype=np.int64)
    # a particle of the other slab is expected as a replica here only inside this slab's ghost columns
    geometry = event["slabs"][slab]
    origin_x = np.float32(geometry["origin_float32"][0])
    columns = np.floor((state["position"][:, 0].astype(np.float32) - origin_x) / np.float32(support)).astype(np.int64)
    ghost_columns = [column for column in range(geometry["grid_dimension"][0])
                     if "ghost" in column_role(column, geometry)]
    peer_missing = peer_own & np.isin(columns, ghost_columns) & ~np.isin(state["global_id"], replica_ids)
    ring_ids = ring["global_id"]
    primary_ring = int(np.flatnonzero(ring_ids == primary)[0])
    path = ring["position"][:, primary_ring].astype(np.float64)
    near = np.flatnonzero(np.linalg.norm(ring["position"][-1].astype(np.float64) - centre, axis=1) < 3.0 * support)
    files = []
    for plane_index, ((first, second), normal, description) in enumerate(planes):
        figure, axis = plt.subplots(figsize=(7.4, 6.6))
        thickness = 0.6 * spacing
        layer = np.abs(state["position"][:, normal].astype(np.float64) - centre[normal]) <= thickness
        window = (np.abs(state["position"][:, first].astype(np.float64) - centre[first]) <= 4 * support) & \
                 (np.abs(state["position"][:, second].astype(np.float64) - centre[second]) <= 4 * support)
        selection = layer & window

        def scatter(mask, label, **style):
            chosen = selection & mask
            if chosen.any():
                axis.scatter(state["position"][chosen, first], state["position"][chosen, second], label=
                             f"{label} ({int(chosen.sum())})", **style)

        scatter(own_slab & ~state["fluid"], f"wall particle, own on slab {slab}", s=16, marker="s",
                color=colours["wall"], linewidths=0, zorder=2)
        scatter(peer_own & ~state["fluid"] & ~peer_missing, "wall particle of the other slab",
                s=16, marker="s", color=colours["wall"], alpha=0.45, linewidths=0, zorder=2)
        scatter(own_slab & state["fluid"] & state["listed"], f"fluid, own on slab {slab}", s=14,
                color=colours["fluid"], linewidths=0, zorder=3)
        scatter(peer_own & state["fluid"] & ~peer_missing, "fluid of the other slab", s=14,
                color=colours["peer"], linewidths=0, zorder=3)
        scatter(replica, f"replica on slab {slab}", s=40, facecolors="none",
                edgecolors=colours["replica"], linewidths=1.0, zorder=4)
        scatter(peer_missing, f"in a ghost column of slab {slab}, but no replica there", s=38, marker="x",
                color=colours["missing"], linewidths=1.3, zorder=5)
        scatter(own_slab & state["fluid"] & ~state["listed"], f"fluid, own on slab {slab}, in no voxel list", s=26,
                marker="D", color=colours["unlisted"], linewidths=0, zorder=5)
        for index in near:
            if ring_ids[index] == primary:
                continue
            track = ring["position"][:, index].astype(np.float64)
            if np.nanmax(np.abs(track[:, normal] - centre[normal])) <= 4 * thickness:
                axis.plot(track[:, first], track[:, second], color=INK_SECONDARY, linewidth=0.6, alpha=0.5,
                          zorder=6)
        axis.plot(path[:, first], path[:, second], color=INK_PRIMARY, linewidth=1.4, zorder=7,
                  label=f"path of particle {primary}, steps {int(ring['steps'][0])}-{int(ring['steps'][-1])}")
        axis.scatter([centre[first]], [centre[second]], marker="*", s=220, color=INK_PRIMARY, edgecolors=SURFACE,
                     linewidths=0.8, zorder=8, label=f"particle {primary} at step {event_step}")
        for plot_axis, value_axis in ((axis.axvline, first), (axis.axhline, second)):
            for bound in (lower[value_axis], upper[value_axis]):
                if abs(bound - centre[value_axis]) <= 4 * support:
                    plot_axis(bound, color=INK_PRIMARY, linewidth=1.0, zorder=1)
            if value_axis == 0:
                for x in cut_x:
                    if abs(x - centre[0]) <= 4 * support:
                        plot_axis(x, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (5, 3)), zorder=1)
        axis.set_xlim(centre[first] - 4 * support, centre[first] + 4 * support)
        axis.set_ylim(centre[second] - 4 * support, centre[second] + 4 * support)
        axis.set_aspect("equal")
        axis.set_xlabel(f"{AXIS_NAMES[first]} (m)")
        axis.set_ylabel(f"{AXIS_NAMES[second]} (m)")
        axis.set_title(f"step {event_step}: slice {AXIS_NAMES[normal]} = {centre[normal]:.4f} ± {thickness:.4f}, "
                       f"{description}", fontsize=9)
        axis.text(0.0, -0.13, "solid lines: faces of the fluid box; dashed: the cut between the slabs; thin grey "
                  f"lines: paths of fluid particles near it. Slab {slab}'s view: the other slab's particles beyond "
                  f"slab {slab}'s ghost columns are never sent", transform=axis.transAxes, fontsize=7.5,
                  color=INK_SECONDARY)

        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
        name = f"slice_{AXIS_NAMES[first]}{AXIS_NAMES[second]}.png"
        figure.savefig(out / name, bbox_inches="tight")
        plt.close(figure)
        files.append(name)

    # history of the primary particle and of the chain
    steps = ring["steps"]
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    figure.suptitle(f"particle {primary} (event at step {event_step}) and the chain", x=0.01, ha="left",
                    fontsize=10.5, fontweight="bold")
    margins = np.array([box_margin(row, lower, upper, dimension) if np.isfinite(row).all() else np.nan
                        for row in path])
    axes[0, 0].plot(steps, margins / support, color="#2a78d6")
    axes[0, 0].axhline(0.0, color=INK_PRIMARY, linewidth=0.8)
    axes[0, 0].set_title(f"particle {primary}: (distance to the nearest face) / h")
    axes[0, 0].set_xlabel("step")
    for x in cut_x:
        axes[0, 1].plot(steps, (path[:, 0] - x) / support, color="#2a78d6")
    axes[0, 1].axhline(0.0, color=INK_SECONDARY, linewidth=0.8, linestyle=(0, (5, 3)))
    axes[0, 1].set_title("(x - x_cut) / h")
    axes[0, 1].set_xlabel("step")
    displacement = np.r_[np.nan, np.linalg.norm(np.diff(path, axis=0), axis=1)]
    speed_step = np.linalg.norm(ring["velocity"][:, primary_ring].astype(np.float64), axis=1) * float(event["timestep"])
    axes[1, 0].plot(steps, displacement / spacing, color="#2a78d6", label="|Δx| per step")
    axes[1, 0].plot(steps, speed_step / spacing, color="#eb6834", label="|v| Δt")
    unlisted = (ring["listed"][:, primary_ring] == 0) & (ring["slab"][:, primary_ring] >= 0)
    if unlisted.any():
        axes[1, 0].scatter(steps[unlisted], np.zeros(int(unlisted.sum())), marker="|", color="#eda100", s=30,
                           label="in no voxel list")
    axes[1, 0].set_title("per step, in lattice spacings Δx")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].legend()
    log_steps = [record["step"] for record in steps_log]
    for slab_index, colour in zip(range(len(event["slabs"])), ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")):
        axes[1, 1].plot(log_steps, [record["slabs"][slab_index]["own_alive_unlisted"] for record in steps_log],
                        color=colour, label=f"slab {slab_index}: own particles in no voxel list")
        axes[1, 1].plot(log_steps, [record["slabs"][slab_index]["delta"]["overflow_ghost_count"] for record in steps_log],
                        color=colour, linestyle=(0, (4, 2)),
                        label=f"slab {slab_index}: overflow_ghost_count, this step (replicas not sent)")
    axes[1, 1].plot(log_steps, [event["initial_fluid_count"] - record["fluid_total"] for record in steps_log],
                    color=INK_PRIMARY, label="fluid particles fewer than at the start")
    axes[1, 1].set_title("chain, every step")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout()
    figure.savefig(out / "history.png", bbox_inches="tight")
    plt.close(figure)
    files.append("history.png")
    return files


if __name__ == "__main__":
    sys.exit(main())
