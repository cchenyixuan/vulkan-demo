"""
switches.py — the three kernel switches of v7_single, read from the environment exactly as v7 reads them.

  V7_FUSED_CORRECTION_DENSITY  0 / 1, default 1   (E39 B1: correction + density in one traversal)
  V7_DENSITY_COPY_COMPUTE      0 / 1, default 1   (E39 B9: the scratch -> primary copy as a compute pass)
  V7_DEEP_WALL_SKIP            0 / 1 / auto, default auto   (E39 B4: deep walls skip correction / density)

The parsers are v7's (simulator_v7._parse_fused_correction_density / _parse_density_copy_compute /
_parse_deep_wall_skip): surrounding blanks ignored, V7_DEEP_WALL_SKIP case-insensitive, any other value
refused with the same message. Every other V7_* variable means nothing to v7_single (no ghost, no
transport, no band, no diagnostics), so a process that has one set is refused at import: a stale
release-set variable, V7_DEEP_WALL_CHECK, V7_DELTA_DENSITY, V7_SPV_DIR, V7_CONCURRENT_BUFFERS ... would
otherwise silently run a build other than the one the environment names. The values are read once at
import (module constant CONFIGURED), as in v7.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

FUSED_CORRECTION_DENSITY_ACCEPTED = ("0", "1")
DENSITY_COPY_COMPUTE_ACCEPTED = ("0", "1")
DEEP_WALL_SKIP_ACCEPTED = ("0", "1", "auto")
SWITCH_NAMES = ("V7_FUSED_CORRECTION_DENSITY", "V7_DENSITY_COPY_COMPUTE", "V7_DEEP_WALL_SKIP")


def parse_fused_correction_density(text: str) -> int:
    """V7_FUSED_CORRECTION_DENSITY: 0 or 1 (surrounding blanks ignored)."""
    value = text.strip()
    if value not in FUSED_CORRECTION_DENSITY_ACCEPTED:
        raise ValueError(f"V7_FUSED_CORRECTION_DENSITY={text!r}: accepted values are 0, 1 (1 = correction and "
                         "density in one neighbour traversal where the slab's bands allow it; 0 = the separate "
                         "kernels, the previous build)")
    return int(value)


def parse_density_copy_compute(text: str) -> int:
    """V7_DENSITY_COPY_COMPUTE: 0 or 1 (surrounding blanks ignored)."""
    value = text.strip()
    if value not in DENSITY_COPY_COMPUTE_ACCEPTED:
        raise ValueError(f"V7_DENSITY_COPY_COMPUTE={text!r}: accepted values are 0, 1 (1 = the density scratch -> "
                         "primary copy as a compute pass; 0 = the v6 vkCmdCopyBuffer recording, the previous build)")
    return int(value == "1")


def parse_deep_wall_skip(text: str) -> str:
    """V7_DEEP_WALL_SKIP: one of 0, 1, auto (case and surrounding blanks ignored)."""
    value = text.strip().lower()
    if value not in DEEP_WALL_SKIP_ACCEPTED:
        raise ValueError(f"V7_DEEP_WALL_SKIP={text!r}: accepted values are "
                         f"{', '.join(DEEP_WALL_SKIP_ACCEPTED)} (0 = off, the previous build; 1 = forced on; "
                         "auto = per-slab rule from the dimension and the initial deep-wall candidates)")
    return value


@dataclass(frozen=True)
class Switches:
    fused_correction_density: int = 1
    density_copy_compute: int = 1
    deep_wall_skip: str = "auto"

    def table(self) -> dict[str, int | str]:
        """The switch table of the run headers (v7's configured_v7_switches keys and spellings for these three)."""
        return {"V7_DENSITY_COPY_COMPUTE": int(self.density_copy_compute),
                "V7_DEEP_WALL_SKIP": self.deep_wall_skip if self.deep_wall_skip == "auto" else int(self.deep_wall_skip),
                "V7_FUSED_CORRECTION_DENSITY": int(self.fused_correction_density)}

    def header(self) -> str:
        return " ".join(f"{name}={value}" for name, value in self.table().items())


def unknown_switches(environment=None) -> list[str]:
    """The V7_* names of the environment v7_single does not know (sorted)."""
    source = os.environ if environment is None else environment
    return sorted(name for name in source if name.startswith("V7_") and name not in SWITCH_NAMES)


def read_switches(environment=None) -> Switches:
    """Parse the three switches of ``environment`` (os.environ by default); refuse any other V7_* name."""
    source = os.environ if environment is None else environment
    unknown = unknown_switches(source)
    if unknown:
        raise SystemExit(f"[v7_single] environment variables that mean nothing to v7_single: {', '.join(unknown)} "
                         f"(it reads only {', '.join(SWITCH_NAMES)}); unset them")
    return Switches(
        fused_correction_density=parse_fused_correction_density(source.get("V7_FUSED_CORRECTION_DENSITY", "1")),
        density_copy_compute=parse_density_copy_compute(source.get("V7_DENSITY_COPY_COMPUTE", "1")),
        deep_wall_skip=parse_deep_wall_skip(source.get("V7_DEEP_WALL_SKIP", "auto")))


CONFIGURED = read_switches()


def configured_switches() -> Switches:
    """The switches of this process (read once at import)."""
    return CONFIGURED
