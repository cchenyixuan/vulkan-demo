"""
switches.py — the three kernel switches of v7_single and their defaults.

v7_single runs the v7 kernels under v7's defaults: correction + density fused (E39 B1), the density
scratch -> primary copy as a compute pass (B9), the deep-wall skip by the auto rule (B4). This module
holds the values; the environment parsing (the same strict parsers as v7, and the refusal of every
other V7_* variable) is added by the switches commit.
"""

from __future__ import annotations

from dataclasses import dataclass

FUSED_CORRECTION_DENSITY_ACCEPTED = ("0", "1")
DENSITY_COPY_COMPUTE_ACCEPTED = ("0", "1")
DEEP_WALL_SKIP_ACCEPTED = ("0", "1", "auto")


@dataclass(frozen=True)
class Switches:
    fused_correction_density: int = 1       # V7_FUSED_CORRECTION_DENSITY: 0 / 1
    density_copy_compute: int = 1           # V7_DENSITY_COPY_COMPUTE: 0 / 1
    deep_wall_skip: str = "auto"            # V7_DEEP_WALL_SKIP: 0 / 1 / auto

    def table(self) -> dict[str, int | str]:
        """The switch table of the run headers (the same keys and spellings as v7's configured_v7_switches
        for these three switches)."""
        return {"V7_DENSITY_COPY_COMPUTE": int(self.density_copy_compute),
                "V7_DEEP_WALL_SKIP": self.deep_wall_skip if self.deep_wall_skip == "auto" else int(self.deep_wall_skip),
                "V7_FUSED_CORRECTION_DENSITY": int(self.fused_correction_density)}

    def header(self) -> str:
        return " ".join(f"{name}={value}" for name, value in self.table().items())


def configured_switches() -> Switches:
    """The switches of this process (the defaults until the environment parsing lands)."""
    return Switches()
