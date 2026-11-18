"""
quadratic_extension.py - the second-order continuation of the velocity of wall_boundary crossing
(numerics.wall_crossing.viscous_extension / continuity_extension, shaders/wall_boundary.glsl): its cases and the
tables and figures of section 10 of claude/wall_crossing_report.md.

    .venv/Scripts/python.exe -m experiment.wall_crossing.quadratic_extension make-cases
    .venv/Scripts/python.exe -m experiment.wall_crossing.quadratic_extension tables

Configurations (the names used in case directories, run names and tables):
    quadratic_viscous   viscous_extension quadratic, continuity_extension linear (the default continuity:
                        first order, normal component only)
    quadratic_full      viscous_extension quadratic, continuity_extension quadratic (the whole vector)
make-cases derives, from an existing crossing case, the same case with the two keys added (and, for the
screening's second attempt, distance_floor_factor 0.5): the cavity at 250^2 and 500^2 and the two 2-D
hydrostatic boxes. Existing case files are left alone (these directories hold only what this function writes).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

CONFIGURATIONS = {
    "quadratic_viscous": ("quadratic", "linear"),
    "quadratic_full": ("quadratic", "quadratic"),
}
CONFIGURATION_TEXT = {"quadratic_viscous": "二阶粘性", "quadratic_full": "全二阶"}
# source case (relative to cases/) -> the suffix-free name of its variants
SOURCE_CASES = (
    "lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing",
    "lid_driven_cavity_2d_n500_xi0p001_eps0p0025_crossing",
    "wall_crossing/hydrostatic_2d_n250_gy_crossing",
    "wall_crossing/hydrostatic_2d_n250_g30_crossing",
)
FLOOR_VARIANT_SOURCES = ("lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing",)
SECOND_FLOOR_FACTOR = 0.5


def variant_text(source_text: str, source_name: str, configuration: str, floor_factor: float | None) -> str:
    """The source case with the configuration's keys (and the floor factor) in numerics.wall_crossing."""
    viscous_extension, continuity_extension = CONFIGURATIONS[configuration]
    lines = source_text.split("\n")
    mapping_rows = [index for index, line in enumerate(lines) if line.startswith("  wall_crossing:")]
    if len(mapping_rows) != 1:
        raise ValueError(f"{source_name}: expected exactly one numerics.wall_crossing mapping")
    if any(line.strip().startswith(("viscous_extension", "continuity_extension", "distance_floor_factor"))
           for line in lines):
        raise ValueError(f"{source_name}: the source already sets an extension or the floor factor")
    added = [f"    viscous_extension: {viscous_extension}       # linear | quadratic (second-order continuation)",
             f"    continuity_extension: {continuity_extension}    # linear | quadratic"]
    if floor_factor is not None:
        added.append(f"    distance_floor_factor: {floor_factor!r}   # default 0.25")
    lines[mapping_rows[0] + 1:mapping_rows[0] + 1] = added
    if lines[0] != "schema_version: 2":
        raise ValueError(f"{source_name}: does not start with the schema line")
    header = [f"# VARIANT of cases/{source_name} (experiment/wall_crossing/quadratic_extension.py): the second-order",
              f"# continuation of the velocity, viscous_extension {viscous_extension}, continuity_extension "
              f"{continuity_extension}"
              + (f", distance_floor_factor {floor_factor!r}" if floor_factor is not None else "")
              + "; everything else identical."]
    lines[2:2] = header + [""]
    return "\n".join(lines)


def variant_name(source_name: str, configuration: str, floor_factor: float | None) -> str:
    return f"{source_name}_{configuration}" + ("" if floor_factor is None
                                               else "_floor" + repr(floor_factor).replace(".", "p"))


def make_cases() -> list:
    written = []
    for source_name in SOURCE_CASES:
        source = REPOSITORY_ROOT / "cases" / source_name / "case.yaml"
        source_text = source.read_text(encoding="utf-8")
        floors = (None, SECOND_FLOOR_FACTOR) if source_name in FLOOR_VARIANT_SOURCES else (None,)
        for configuration in CONFIGURATIONS:
            for floor_factor in floors:
                target = REPOSITORY_ROOT / "cases" / variant_name(source_name, configuration, floor_factor) \
                    / "case.yaml"
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(variant_text(source_text, source_name, configuration, floor_factor),
                                  encoding="utf-8", newline="\n")
                written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("make-cases")
    arguments = parser.parse_args()
    if arguments.command == "make-cases":
        for path in make_cases():
            print(f"[quadratic_extension] wrote {path.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
