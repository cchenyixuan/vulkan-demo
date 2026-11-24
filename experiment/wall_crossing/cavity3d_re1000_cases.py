"""
cavity3d_re1000_cases.py - claude/wall_store_report.md section 14 (round 8): the cases of the 3-D cubic cavity at
Re = 1000 exactly, for N = 50, 70, 100, 140 spacings per side, h/dx = 4 and 3.5, simple and crossing walls.

    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity3d_re1000_cases [--sizes 50,70,100,140]

Per N (one lattice for both h/dx and both wall modes):
  cases/cavity3d_re1000_n<N>/            the particles of utils/geometry/_demo_cavity_case_3d.py --half N/2 --border 9
                                         (domain / wall / wall_top / frame .obj, git-ignored; generated again when
                                         missing) and materials.yaml: the standard stick_water / stick_wall /
                                         stick_lid with the viscosity nu = U L / 1000, L = 1 + dx the wall-to-wall
                                         length of the crossing box (the fluid lattice's bounding box widened by one
                                         particle radius on every side), U = 1 the lid speed. The three materials get
                                         the same nu (force.comp uses the particle's own nu; the walls' value is
                                         never read), as in materials/standard.yaml where all three are 1e-3.
  cases/cavity3d_re1000_n<N>_hdx<4|3p5>_<simple|crossing>/case.yaml
                                         the generator's case.yaml for that h/dx (h = (h/dx) dx, dt = cfl h / c0,
                                         max_per_voxel / max_incoming by the generator's rule for that h/dx, pool
                                         size of the lattice), referencing the N directory's particles and
                                         materials.yaml; crossing adds what cases/cavity3d_n50_crossing adds
                                         (numerics.wall_boundary crossing, the volume constraint on, continuity
                                         extrapolate; everything else at the code defaults).
Nothing else differs from cases/cavity3d_n50 / _n70 / cavity3d_1m (report sections 24, 25): the earlier cases had
nu = 1e-3, i.e. Re = 1000 (1 + dx) on the box length.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import yaml

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
GENERATOR = REPOSITORY_ROOT / "utils" / "geometry" / "_demo_cavity_case_3d.py"
RATIOS = {"4": 4.0, "3p5": 3.5}
REYNOLDS = 1000.0
LID_SPEED = 1.0
BORDER = 9

CROSSING_LINES = """  wall_boundary: crossing   # crossing-fraction walls (first order, axis-aligned box)
  volume_constraint:        # the fluid's total volume is held (closed container)
    enabled: true
  wall_crossing:
    continuity: extrapolate   # extrapolate | wall_velocity
"""


def base_directory(size: int) -> pathlib.Path:
    return REPOSITORY_ROOT / "cases" / f"cavity3d_re1000_n{size}"


def variant_directory(size: int, ratio_label: str, mode: str) -> pathlib.Path:
    return REPOSITORY_ROOT / "cases" / f"cavity3d_re1000_n{size}_hdx{ratio_label}_{mode}"


def generate(size: int, ratio: float, out: pathlib.Path) -> str:
    """Run the generator into out; returns its case.yaml text."""
    if size % 2:
        raise ValueError(f"N = {size}: the generator needs an even number of spacings (--half N/2)")
    command = [sys.executable, str(GENERATOR), "--half", str(size // 2), "--border", str(BORDER),
               "--hdx", repr(ratio), "--out", str(out.relative_to(REPOSITORY_ROOT)), "--no-preview"]
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    return (out / "case.yaml").read_text(encoding="utf-8")


def materials_text(size: int) -> str:
    spacing = 1.0 / size
    length = 1.0 + spacing
    viscosity = LID_SPEED * length / REYNOLDS
    standard = yaml.safe_load((REPOSITORY_ROOT / "materials" / "standard.yaml").read_text(encoding="utf-8"))
    lines = [f"# cavity3d_re1000_n{size} (claude/wall_store_report.md section 14): Re = U L / nu = 1000 exactly on the",
             f"# wall-to-wall length L = 1 + dx = {length!r} (dx = 1/{size}, the crossing box), U = {LID_SPEED:g}.",
             f"# The standard materials with nu = {viscosity!r} instead of 1e-3; nothing else changed.", ""]
    for name in ("stick_water", "stick_wall", "stick_lid"):
        entry = dict(standard[name])
        entry["viscosity"] = viscosity
        lines.append(f"{name}:")
        lines.append(f"  kind: {entry['kind']}")
        lines.append(f"  rest_density: {float(entry['rest_density'])!r}")
        lines.append(f"  viscosity: {viscosity!r}")
        if "initial_velocity" in entry:
            lines.append(f"  initial_velocity: [{', '.join(repr(float(value)) for value in entry['initial_velocity'])}]")
        lines.append("")
    return "\n".join(lines)


def variant_text(generated: str, size: int, ratio_label: str, mode: str) -> str:
    base = f"../cavity3d_re1000_n{size}"
    text = generated
    text = text.replace("material_library: ../../materials/standard.yaml", f"material_library: {base}/materials.yaml")
    text = text.replace("frame: frame.obj", f"frame: {base}/frame.obj")
    for name in ("domain", "wall", "wall_top"):
        text = text.replace(f"{{file: {name}.obj,", f"{{file: {base}/{name}.obj,")
    if mode == "crossing":
        marker = "  defrag_order: prefix"
        index = text.index(marker)
        end = text.index("\n", index) + 1
        text = text[:end] + CROSSING_LINES + text[end:]
    header = (f"# cavity3d_re1000_n{size}_hdx{ratio_label}_{mode} (claude/wall_store_report.md section 14, "
              f"experiment/wall_crossing/cavity3d_re1000_cases.py): N = {size}, h/dx = {RATIOS[ratio_label]:g}, "
              f"{mode} walls, Re = 1000 on L = 1 + dx (materials.yaml of {base}); the particles are {base}'s.\n")
    return header + text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="50,70,100,140")
    arguments = parser.parse_args()
    scratch = REPOSITORY_ROOT / "cases" / "_cavity3d_re1000_scratch"
    for size in (int(item) for item in arguments.sizes.split(",")):
        base = base_directory(size)
        if not all((base / f"{name}.obj").exists() for name in ("domain", "wall", "wall_top", "frame")):
            generate(size, 4.0, base)
            (base / "case.yaml").unlink()          # the base holds particles and materials only
        (base / "materials.yaml").write_text(materials_text(size), encoding="utf-8", newline="\n")
        for ratio_label, ratio in RATIOS.items():
            generated = generate(size, ratio, scratch)
            for mode in ("simple", "crossing"):
                directory = variant_directory(size, ratio_label, mode)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "case.yaml").write_text(variant_text(generated, size, ratio_label, mode),
                                                     encoding="utf-8", newline="\n")
                print(f"wrote {directory.relative_to(REPOSITORY_ROOT)}/case.yaml")
        # the scratch lattice is the same as the base one (same --half / --border); compare, then drop it
        for name in ("domain", "wall", "wall_top"):
            if (scratch / f"{name}.obj").read_bytes() != (base / f"{name}.obj").read_bytes():
                raise SystemExit(f"N = {size}: {name}.obj of the scratch generation differs from the base's")
        for item in scratch.iterdir():
            item.unlink()
        scratch.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
