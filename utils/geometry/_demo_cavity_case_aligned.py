"""
_demo_cavity_case_aligned.py — lid-driven cavity cases whose fluid fills whole voxel columns (E32).

Same 4-part split (frame, domain = fluid, wall, wall_top = lid) and the same physics and numerics as
_demo_cavity_case.py (2-D) and _demo_cavity_case_3d.py (3-D): a 1 m cavity (height; width n_x / n_y m),
lid velocity 1 m/s, h/dx 5 (2-D) or 4 (3-D), c0 100, CFL 0.15, KCG xi 0.001, eps^2 0.0025 h^2, the same
wall layers. Two differences make equal-weight chain cuts exact:

  * Cell-centred lattice: the fluid is n_x x n_y (x n_z) particles at the centres of the dx cells of the
    cavity ([-n_x dx / 2, n_x dx / 2] x [-0.5, 0.5] (x [-0.5, 0.5]), dx = 1 / n_y); wall and lid are
    `border` more cell layers outside, as before. The node lattice of the other generators puts 2 half + 1
    particles on [-0.5, 0.5] inclusive: an odd count (effective width 1 + dx) that can neither fill whole
    voxel columns nor split evenly in two.
  * The frame is placed so that the voxel grid the loader derives from it (origin = frame minimum - h / 2,
    column width h = (h/dx) dx) has column boundaries exactly at both fluid edges. With n_x a multiple of
    h/dx the fluid fills n_x / (h/dx) whole columns, and with that a multiple of K every one of K
    equal-weight slabs gets the same number of fluid columns. Wall columns lie outside.

h, dx and the particle radius go into case.yaml with full precision and positions with 9 decimals (the
other generators round to 6 decimals: over 2,000+ columns that moves voxel edges by up to a column, and
the calibrated particle volume by ~0.5 % at the largest sizes). Before writing, the per-column fluid
histogram is rebuilt exactly as the loader and partition_v6 bin it (float32 positions, float32
arithmetic) and checked: whole columns, equal counts.

    .venv/Scripts/python.exe utils/geometry/_demo_cavity_case_aligned.py --dimension 2 --n 1000 \\
        --out cases/aligned/cavity2d_n1000
    .venv/Scripts/python.exe utils/geometry/_demo_cavity_case_aligned.py --dimension 3 --n 200 --n-x 1600 \\
        --out cases/aligned/cavity3d_n200_x1600          # stretched 64M, border 4
    add --objs-only to rebuild the particle files without touching case.yaml, --yaml-only to write case.yaml
    and generate.txt without the particle files (the repository keeps those two; the .obj are git-ignored)
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
POSITION_FORMAT = "%.9f"


def fluid_columns(n_x: int, hdx: float) -> int:
    columns = n_x / hdx
    if abs(columns - round(columns)) > 1e-9:
        raise ValueError(f"n_x = {n_x} is not a multiple of h/dx = {hdx:g}: the fluid cannot fill whole columns")
    return int(round(columns))


def wall_columns(border: int, hdx: float) -> int:
    """Voxel columns outside each fluid edge: the frame must reach the outermost wall cell centre,
    border - 1/2 cells out, and the frame minimum sits half a column inside the first column."""
    return max(1, math.ceil((border - 0.5) / hdx + 0.5))


def layout(dimension: int, n: int, n_x: int, border: int, hdx: float) -> dict:
    dx = 1.0 / n
    h = hdx * dx
    walls = wall_columns(border, hdx)
    half_x = 0.5 * n_x * dx
    frame_x = half_x + (walls - 0.5) * h
    frame_yz = 0.5 + (walls - 0.5) * h
    return {"dimension": dimension, "n": n, "n_x": n_x, "border": border, "hdx": hdx, "dx": dx, "h": h,
            "radius": 0.5 * dx, "wall_columns": walls, "fluid_columns": fluid_columns(n_x, hdx),
            "frame": (frame_x, frame_yz, frame_yz if dimension == 3 else 0.5 * dx)}


def cell_centres(count: int, border: int, dx: float, half: float) -> tuple[np.ndarray, np.ndarray]:
    """(centres, cell index) of cells -border .. count + border - 1 of a row of `count` cells starting at -half."""
    index = np.arange(-border, count + border, dtype=np.int64)
    return (index + 0.5) * dx - half, index


def loader_histogram(spec: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """(fluid, all particles) per voxel column and the column count, binned like case_loader_v6 +
    partition_v6._bin_fluid_counts: positions and frame parsed from their text to float32, origin =
    float64(frame minimum) - h / 2, nx = floor(span / h + 0.5) + 1, column = floor((x - origin) / h) in
    float32. Every x lattice column holds the same number of particles in total (the full cross-section)
    and of fluid (the fluid cross-section) when inside the fluid range."""
    dx, h = spec["dx"], spec["h"]
    frame_text = f"{spec['frame'][0]:.9f}"
    bbox_min = float(np.float32(-float(frame_text)))
    bbox_max = float(np.float32(float(frame_text)))
    origin = bbox_min - 0.5 * h
    nx = int(math.floor((bbox_max - bbox_min) / h + 0.5)) + 1
    x, index = cell_centres(spec["n_x"], spec["border"], dx, 0.5 * spec["n_x"] * dx)
    x32 = np.array([float(POSITION_FORMAT % value) for value in x], dtype=np.float32)
    columns = np.floor((x32 - origin) / h).astype(np.int64)
    if columns.min() < 0 or columns.max() > nx - 1:
        raise ValueError(f"particles outside the grid: columns {columns.min()}..{columns.max()} of {nx}")
    cross = spec["n"] + 2 * spec["border"]
    per_column_all = cross ** (spec["dimension"] - 1)
    per_column_fluid = spec["n"] ** (spec["dimension"] - 1)
    fluid = (index >= 0) & (index < spec["n_x"])
    everything = np.bincount(columns, minlength=nx).astype(np.int64) * per_column_all
    fluid_counts = np.bincount(columns[fluid], minlength=nx).astype(np.int64) * per_column_fluid
    return fluid_counts, everything, nx


def check(spec: dict) -> dict:
    fluid, everything, nx = loader_histogram(spec)
    filled = fluid[fluid > 0]
    expected = spec["fluid_columns"]
    full = round(spec["hdx"]) * spec["n"] ** (spec["dimension"] - 1)
    if len(filled) != expected or not np.all(filled == full):
        raise ValueError(f"fluid does not fill {expected} whole columns: {len(filled)} columns, counts "
                         f"{sorted(set(filled.tolist()))} (a full column holds {full})")
    first = int(np.flatnonzero(fluid)[0])
    if first != spec["wall_columns"] or nx != expected + 2 * spec["wall_columns"]:
        raise ValueError(f"unexpected grid: first fluid column {first}, nx {nx}")
    return {"nx": nx, "fluid_per_column": full, "fluid": fluid, "all": everything}


CASE_YAML_2D = """\
schema_version: 2

# 2D lid-driven cavity, aligned lattice: {n_fluid:,} fluid particles ({fluid_millions:.2f}M).
# GENERATED by utils/geometry/_demo_cavity_case_aligned.py --dimension 2 --n {n} --n-x {n_x} --border {border}
#   frame    = voxel-grid bounding box: column boundaries at the fluid edges x = -+{half_x:.6f} m,
#              {fluid_columns} whole fluid columns + {wall_columns} wall columns on each side ({nx} columns)
#   domain   = {n_x} x {n} = {n_fluid:,} fluid particles at the cell centres of the {size_x:g} m x 1 m cavity
#   wall     = {n_wall:,} static U-shaped wall (bottom + two sides + top corners), {border} layers
#   wall_top = {n_lid:,} lid particles (top, {border} layers); stick_lid has initial_velocity=[1,0,0]
# h = {h!r} m, dx = {dx!r} m -> h/dx = {hdx:g}; total particles = {total:,}

time:
  total: null
  max_steps: null
  output_cadence: null

physics:
  dimension: 2
  h: {h!r}
  particle_radius: {radius!r}
  lattice: grid
  calibrate_volume: true
  speed_of_sound: 100.0
  power: 7
  cfl: 0.15
  gravity: [0.0, 0.0, 0.0]

numerics:
  use_density_diffusion: true
  delta_coefficient: 0.1
  use_kcg_correction: true
  regularization:
    xi: 0.001
    det_threshold: 1.0e-4
    frobenius_max: 10.0
  # eps^2 = epsilon_squared_factor * h^2 (h = support radius) in the viscous / delta-diffusion
  # denominators; read by the V0 and v6 loaders (v2-v5 ignore it and use 0.01)
  epsilon_squared_factor: 0.0025
  use_pst: true
  pst_main: 0.1
  pst_anti: 0.0005
  defrag_enabled: true
  defrag_cadence: 1000
  use_prefix_sum_defrag: false

capacities:
  pool_size: {pool_size}
  max_per_voxel: {max_per_voxel}
  max_incoming: {max_incoming}
  workgroup: 128

material_library: {material_library}

geometry:
  frame: frame.obj
  particles:
    - {{file: domain.obj,   material: stick_water}}
    - {{file: wall.obj,     material: stick_wall}}
    - {{file: wall_top.obj, material: stick_lid}}
"""

CASE_YAML_3D = """\
schema_version: 2

# 3D lid-driven cavity, aligned lattice: {n_fluid:,} fluid particles ({fluid_millions:.2f}M).
# GENERATED by utils/geometry/_demo_cavity_case_aligned.py --dimension 3 --n {n} --n-x {n_x} --border {border}
#   frame    = voxel-grid bounding box: column boundaries at the fluid edges x = -+{half_x:.6f} m,
#              {fluid_columns} whole fluid columns + {wall_columns} wall columns on each side ({nx} columns)
#   domain   = {n_x} x {n} x {n} = {n_fluid:,} fluid particles at the cell centres of the {size_x:g} m x 1 m x 1 m cavity
#   wall     = {n_wall:,} static shell particles ({border} layers, 5 faces + edges)
#   wall_top = {n_lid:,} lid particles (band above the fluid footprint); stick_lid has initial_velocity=[1,0,0]
# h = {h!r} m, dx = {dx!r} m -> h/dx = {hdx:g}; total particles = {total:,}
# max_per_voxel {max_per_voxel} >= ceil(sqrt(2)*(h/dx)^3) = {packing_bound}

time:
  total: null
  max_steps: null
  output_cadence: null

physics:
  dimension: 3
  h: {h!r}
  particle_radius: {radius!r}
  lattice: grid
  calibrate_volume: true
  speed_of_sound: 100.0
  power: 7
  cfl: 0.15
  gravity: [0.0, 0.0, 0.0]

numerics:
  use_density_diffusion: true
  delta_coefficient: 0.1
  use_kcg_correction: true
  regularization:
    xi: 0.001
    det_threshold: 1.0e-4
    frobenius_max: 10.0
  # eps^2 = epsilon_squared_factor * h^2 (h = support radius) in the viscous / delta-diffusion
  # denominators; read by the V0 and v6 loaders (v2-v5 ignore it and use 0.01)
  epsilon_squared_factor: 0.0025
  use_pst: true
  pst_main: 0.1
  pst_anti: 0.0005
  defrag_enabled: true
  defrag_cadence: 1000
  use_prefix_sum_defrag: false

capacities:
  pool_size: {pool_size}
  max_per_voxel: {max_per_voxel}
  max_incoming: {max_incoming}
  workgroup: 128

material_library: {material_library}

geometry:
  frame: frame.obj
  particles:
    - {{file: domain.obj,   material: stick_water}}
    - {{file: wall.obj,     material: stick_wall}}
    - {{file: wall_top.obj, material: stick_lid}}
"""


def counts(spec: dict) -> tuple[int, int, int]:
    """(fluid, wall, lid) particle counts of the lattice."""
    n, n_x, b, dimension = spec["n"], spec["n_x"], spec["border"], spec["dimension"]
    if dimension == 2:
        total = (n_x + 2 * b) * (n + 2 * b)
        fluid, lid = n_x * n, n_x * b
    else:
        total = (n_x + 2 * b) * (n + 2 * b) ** 2
        fluid, lid = n_x * n * n, n_x * n * b
    return fluid, total - fluid - lid, lid


def capacities(dimension: int, hdx: float) -> tuple[int, int, int]:
    if dimension == 2:
        return 96, 16, math.ceil(2.0 / math.sqrt(3.0) * hdx ** 2)
    bound = math.ceil(math.sqrt(2.0) * hdx ** 3)
    per_voxel = max(32, int(math.ceil(bound * 1.3 / 32.0) * 32))
    return per_voxel, max(8, per_voxel // 4), bound


def write_frame_obj(path: pathlib.Path, frame: tuple) -> None:
    hx, hy, hz = frame
    vertices = np.array([[-hx, -hy, -hz], [-hx, hy, -hz], [hx, -hy, -hz], [hx, hy, -hz],
                         [-hx, -hy, hz], [-hx, hy, hz], [hx, -hy, hz], [hx, hy, hz]])
    faces = [(1, 2, 4), (1, 4, 3), (5, 7, 8), (5, 8, 6), (1, 5, 6), (1, 6, 2),
             (3, 4, 8), (3, 8, 7), (1, 3, 7), (1, 7, 5), (2, 6, 8), (2, 8, 4)]
    with open(path, "w") as handle:
        handle.write("# cavity frame bbox (aligned voxel grid)\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in faces:
            handle.write(f"f {face[0]} {face[1]} {face[2]}\n")


def write_particles(spec: dict, out_dir: pathlib.Path) -> tuple[int, int, int]:
    """Stream the lattice in x chunks (a 76M-site 3-D case does not fit in memory at once)."""
    n, n_x, b, dx, dimension = spec["n"], spec["n_x"], spec["border"], spec["dx"], spec["dimension"]
    x_all, ix_all = cell_centres(n_x, b, dx, 0.5 * n_x * dx)
    y_all, iy_all = cell_centres(n, b, dx, 0.5)
    if dimension == 3:
        z_all, iz_all = cell_centres(n, b, dx, 0.5)
    paths = {name: out_dir / f"{name}.obj" for name in ("domain", "wall", "wall_top")}
    handles = {name: open(path, "w") for name, path in paths.items()}
    written = {name: 0 for name in paths}
    try:
        chunk = 64 if dimension == 3 else 256    # ~3M sites per 2-D chunk at n = 11,320
        for start in range(0, x_all.size, chunk):
            stop = min(start + chunk, x_all.size)
            if dimension == 2:
                ix, iy = np.meshgrid(ix_all[start:stop], iy_all, indexing="ij")
                x, y = np.meshgrid(x_all[start:stop], y_all, indexing="ij")
                ix, iy, x, y = ix.ravel(), iy.ravel(), x.ravel(), y.ravel()
                inside_x = (ix >= 0) & (ix < n_x)
                fluid = inside_x & (iy >= 0) & (iy < n)
                lid = inside_x & (iy >= n)
                points = np.column_stack([x, y, np.zeros_like(x)])
            else:
                ix, iy, iz = np.meshgrid(ix_all[start:stop], iy_all, iz_all, indexing="ij")
                x, y, z = np.meshgrid(x_all[start:stop], y_all, z_all, indexing="ij")
                ix, iy, iz = ix.ravel(), iy.ravel(), iz.ravel()
                x, y, z = x.ravel(), y.ravel(), z.ravel()
                inside_xz = (ix >= 0) & (ix < n_x) & (iz >= 0) & (iz < n)
                fluid = inside_xz & (iy >= 0) & (iy < n)
                lid = inside_xz & (iy >= n)
                points = np.column_stack([x, y, z])
            wall = ~(fluid | lid)
            for name, mask in (("domain", fluid), ("wall", wall), ("wall_top", lid)):
                np.savetxt(handles[name], points[mask],
                           fmt=f"v {POSITION_FORMAT} {POSITION_FORMAT} {POSITION_FORMAT}")
                written[name] += int(mask.sum())
    finally:
        for handle in handles.values():
            handle.close()
    return written["domain"], written["wall"], written["wall_top"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--n", type=int, required=True, help="fluid cells over the 1 m cavity height: dx = 1 / n")
    parser.add_argument("--n-x", type=int, default=None, help="fluid cells along x (default n: square / cube; "
                        "K n for a stretched weak-scaling member)")
    parser.add_argument("--border", type=int, default=None, help="wall / lid layers (default 11 in 2-D, h/dx in 3-D)")
    parser.add_argument("--hdx", type=float, default=None, help="h/dx (default 5 in 2-D, 4 in 3-D)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--objs-only", action="store_true", help="write the .obj files only (case.yaml untouched)")
    parser.add_argument("--yaml-only", action="store_true",
                        help="write case.yaml and generate.txt only (no particle files)")
    parser.add_argument("--check-only", action="store_true", help="check the alignment, write nothing")
    arguments = parser.parse_args()
    dimension = arguments.dimension
    hdx = arguments.hdx if arguments.hdx is not None else (5.0 if dimension == 2 else 4.0)
    border = arguments.border if arguments.border is not None else (11 if dimension == 2 else math.ceil(hdx))
    n_x = arguments.n_x if arguments.n_x is not None else arguments.n
    spec = layout(dimension, arguments.n, n_x, border, hdx)
    result = check(spec)
    n_fluid, n_wall, n_lid = counts(spec)
    total = n_fluid + n_wall + n_lid
    print(f"n {arguments.n} n_x {n_x} border {border} h/dx {hdx:g}: dx {spec['dx']!r} h {spec['h']!r}; "
          f"{spec['fluid_columns']} whole fluid columns of {result['fluid_per_column']:,} + {spec['wall_columns']} "
          f"wall columns per side = {result['nx']} columns; fluid {n_fluid:,} wall {n_wall:,} lid {n_lid:,} "
          f"total {total:,}", flush=True)
    if arguments.check_only:
        return 0
    if arguments.objs_only and arguments.yaml_only:
        raise SystemExit("--objs-only and --yaml-only exclude each other")
    out_dir = _REPO_ROOT / arguments.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.yaml_only:
        written = write_particles(spec, out_dir)
        if written != (n_fluid, n_wall, n_lid):
            raise RuntimeError(f"wrote {written}, expected {(n_fluid, n_wall, n_lid)}")
        write_frame_obj(out_dir / "frame.obj", spec["frame"])
    if arguments.objs_only:
        print(f"wrote frame.obj domain.obj wall.obj wall_top.obj in {out_dir} (--objs-only: case.yaml untouched)")
        return 0
    per_voxel, incoming, bound = capacities(dimension, hdx)
    pool_size = int(math.ceil(total * 1.15 / 128) * 128)
    material_library = pathlib.Path(os.path.relpath(_REPO_ROOT / "materials" / "standard.yaml", out_dir)).as_posix()
    template = CASE_YAML_2D if dimension == 2 else CASE_YAML_3D
    (out_dir / "case.yaml").write_text(template.format(
        n=arguments.n, n_x=n_x, border=border, half_x=0.5 * n_x * spec["dx"], fluid_columns=spec["fluid_columns"],
        wall_columns=spec["wall_columns"], nx=result["nx"], n_fluid=n_fluid, fluid_millions=n_fluid / 1e6,
        size_x=n_x * spec["dx"], n_wall=n_wall, n_lid=n_lid, h=spec["h"], dx=spec["dx"], hdx=hdx, total=total,
        radius=spec["radius"], pool_size=pool_size, max_per_voxel=per_voxel, max_incoming=incoming,
        packing_bound=bound, material_library=material_library), encoding="utf-8")
    command = (f"utils/geometry/_demo_cavity_case_aligned.py --dimension {dimension} --n {arguments.n} --n-x {n_x} "
               f"--border {border}" + (f" --hdx {hdx:g}" if arguments.hdx is not None else "")
               + f" --out {pathlib.Path(arguments.out).as_posix()}")
    shape = f"{n_x} x {arguments.n}" + (f" x {arguments.n}" if dimension == 3 else "")
    (out_dir / "generate.txt").write_text(
        f"# {dimension}-D lid-driven cavity, aligned lattice (E32): fluid {shape} = {n_fluid:,}, wall {n_wall:,}, "
        f"lid {n_lid:,} = {total:,} particles,\n"
        f"# h/dx = {hdx:g}, border {border}; {spec['fluid_columns']} whole fluid voxel columns + {spec['wall_columns']} "
        f"wall columns per side = {result['nx']} columns.\n"
        f"# The particle files (.obj, incl. frame.obj) are git-ignored; rebuild them without touching case.yaml:\n"
        f".venv/Scripts/python.exe {command} --objs-only\n", encoding="utf-8")
    print(f"wrote {'case.yaml generate.txt' if arguments.yaml_only else 'frame.obj domain.obj wall.obj wall_top.obj case.yaml generate.txt'} in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
