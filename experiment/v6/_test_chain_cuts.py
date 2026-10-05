"""
_test_chain_cuts.py — E31 unit tests of the chain cut rule (partition_v6.nearest_cut and
partition_v6.chain_cuts_from_counts). Zero-dependency runnable script (assert + non-zero exit on
failure), CPU only, no case files.

  1  symmetric: uniform fluid per column, wall columns (no fluid) at both ends, K = 2 / 4 / 8, even and
     odd fluid widths: the slabs' fluid column counts differ by <= 1. The pre-E31 rule
     (searchsorted side="left" as is) gives 2 on every even K = 2 split; checked too, so the test
     discriminates.
  2  non-uniform: random per-column counts (with empty columns), random weights, K = 2..8: every cut's
     prefix count is within half the straddled column of its target; every slab is within half a
     column at each of its cuts (one cut for an end slab, two for an interior slab).
  3  minimum width: extreme weights clamp the slabs to minimum_own_columns, cuts stay monotonic; a
     grid too small raises.
  4  nearest_cut on hand-made prefixes: exact boundary, nearer side, tie, ends.

Usage:
    .venv/Scripts/python.exe experiment/v6/_test_chain_cuts.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.v6.utils.partition_v6 import (  # noqa: E402
    MINIMUM_OWN_COLUMNS_HARD,
    chain_cuts_from_counts,
    nearest_cut,
)

_passed = 0
_failed: list[str] = []


def check(condition: bool, message: str) -> None:
    global _passed
    if condition:
        _passed += 1
    else:
        _failed.append(message)
        print(f"  FAIL: {message}")


def legacy_cuts(counts, weights, minimum_own_columns) -> list[int]:
    """The pre-E31 rule, for the discrimination check."""
    cumulative = np.cumsum(counts)
    total, weight_total = int(np.sum(counts)), sum(weights)
    cuts, running = [], 0.0
    for weight in weights[:-1]:
        running += weight
        cuts.append(int(np.searchsorted(cumulative, max(1, int(total * running / weight_total)), side="left")))
    for j in range(len(cuts)):
        low = (cuts[j - 1] if j > 0 else 0) + minimum_own_columns
        high = len(counts) - minimum_own_columns * (len(weights) - 1 - j)
        cuts[j] = max(low, min(cuts[j], high))
    return cuts


def fluid_columns_per_slab(counts, cuts) -> list[int]:
    boundaries = [0] + list(cuts) + [len(counts)]
    return [int(np.count_nonzero(counts[boundaries[i]:boundaries[i + 1]])) for i in range(len(boundaries) - 1)]


def test_symmetric() -> None:
    print("[1] symmetric uniform fluid with walls, K = 2 / 4 / 8")
    legacy_off_by_two = 0
    for slab_count in (2, 4, 8):
        for fluid_width in range(slab_count * MINIMUM_OWN_COLUMNS_HARD + 6, slab_count * MINIMUM_OWN_COLUMNS_HARD + 60):
            for wall in (0, 2, 3):
                for per_column in (101, 5005, 40804):
                    counts = np.array([0] * wall + [per_column] * fluid_width + [0] * wall, dtype=np.int64)
                    cuts = chain_cuts_from_counts(counts, [1.0] * slab_count, MINIMUM_OWN_COLUMNS_HARD)
                    widths = fluid_columns_per_slab(counts, cuts)
                    check(max(widths) - min(widths) <= 1,
                          f"K={slab_count} fluid {fluid_width} wall {wall} c {per_column}: fluid columns {widths}")
                    check(sum(widths) == fluid_width, f"K={slab_count} fluid {fluid_width}: columns lost {widths}")
                    if slab_count == 2 and fluid_width % 2 == 0:
                        old = fluid_columns_per_slab(counts, legacy_cuts(counts, [1.0, 1.0], MINIMUM_OWN_COLUMNS_HARD))
                        legacy_off_by_two += int(old[1] - old[0] == 2)
    # every even K = 2 split (3 walls x 3 column counts x the even widths) lands on a column boundary
    check(legacy_off_by_two > 0 and legacy_off_by_two % 9 == 0,
          f"the pre-E31 rule should give 2 extra columns to slab 1 on even splits ({legacy_off_by_two})")
    # the generator lattice: partial fluid columns at both edges (2-D 1M: 1 and 5 lattice columns of 1001)
    counts = np.array([0, 0, 1001] + [5005] * 200 + [0, 0, 0], dtype=np.int64)
    cuts = chain_cuts_from_counts(counts, [1.0, 1.0], MINIMUM_OWN_COLUMNS_HARD)
    check(cuts == [103], f"2-D 1M-like lattice: cut {cuts}, expected [103]")


def test_nonuniform() -> None:
    print("[2] non-uniform counts, random weights, K = 2..8")
    generator = np.random.default_rng(31)
    checked = 0
    for trial in range(400):
        nx = int(generator.integers(60, 400))
        counts = generator.integers(1, 2000, nx).astype(np.int64)
        counts[generator.random(nx) < 0.15] = 0                  # empty columns
        counts[:int(generator.integers(0, 6))] = 0               # walls
        slab_count = int(generator.integers(2, 9))
        weights = list(generator.uniform(0.5, 2.0, slab_count))
        cumulative = np.cumsum(counts)
        total, weight_total = int(counts.sum()), sum(weights)
        targets, running = [], 0.0
        for weight in weights[:-1]:
            running += weight
            targets.append(max(1, int(total * running / weight_total)))
        unclamped = [nearest_cut(cumulative, target) for target in targets]
        if any(b <= a for a, b in zip([0] + unclamped, unclamped + [nx])):
            continue                                             # the monotonic clamp moved a cut
        cuts = chain_cuts_from_counts(counts, weights, 1)
        check(cuts == unclamped, f"trial {trial}: cuts {cuts} != unclamped {unclamped}")
        halves = []
        for cut, target in zip(cuts, targets):
            straddled = int(np.searchsorted(cumulative, target, side="left"))
            half = counts[straddled] / 2.0
            prefix = int(cumulative[cut - 1]) if cut > 0 else 0
            check(abs(prefix - target) <= half, f"trial {trial}: cut {cut} prefix {prefix} target {target} half {half}")
            halves.append(half)
        boundaries = [0] + cuts + [nx]
        slab_targets = np.diff([0] + targets + [total])
        for index in range(slab_count):
            count = int(counts[boundaries[index]:boundaries[index + 1]].sum())
            allowed = (halves[index - 1] if index > 0 else 0.0) + (halves[index] if index < slab_count - 1 else 0.0)
            check(abs(count - slab_targets[index]) <= allowed,
                  f"trial {trial} slab {index}: {count} vs target {slab_targets[index]} (allowed {allowed})")
        checked += 1
    check(checked > 300, f"too few unclamped trials: {checked}")
    print(f"  {checked} trials")


def test_minimum_width() -> None:
    print("[3] minimum width")
    counts = np.full(200, 1000, dtype=np.int64)
    for weights in ([1000.0, 1.0, 1.0], [1.0, 1.0, 1000.0], [1.0, 1000.0, 1.0, 1.0], [1.0] * 8):
        cuts = chain_cuts_from_counts(counts, weights, MINIMUM_OWN_COLUMNS_HARD)
        widths = np.diff([0] + cuts + [len(counts)])
        check(all(width >= MINIMUM_OWN_COLUMNS_HARD for width in widths), f"w={weights}: widths {list(widths)}")
        check(all(b > a for a, b in zip(cuts, cuts[1:])), f"w={weights}: cuts not increasing {cuts}")
    clamped = chain_cuts_from_counts(counts, [1000.0, 1.0, 1.0], MINIMUM_OWN_COLUMNS_HARD)
    check(clamped == [200 - 2 * MINIMUM_OWN_COLUMNS_HARD, 200 - MINIMUM_OWN_COLUMNS_HARD],
          f"extreme weights should push the cuts to the right edge: {clamped}")
    try:
        chain_cuts_from_counts(np.ones(20, dtype=np.int64), [1.0, 1.0], MINIMUM_OWN_COLUMNS_HARD)
        check(False, "a 20-column grid split in 2 with minimum 12 should raise")
    except ValueError:
        check(True, "")


def test_nearest_cut() -> None:
    print("[4] nearest_cut")
    cumulative = np.array([10, 20, 30, 40])
    for target, expected, reason in ((20, 2, "exact boundary after column 1"), (17, 2, "nearer above"),
                                     (12, 1, "nearer below"), (15, 1, "tie keeps the pre-E31 cut"),
                                     (5, 0, "tie in the first column"), (40, 4, "the last boundary"),
                                     (45, 4, "beyond the total")):
        check(nearest_cut(cumulative, target) == expected,
              f"target {target}: {nearest_cut(cumulative, target)} != {expected} ({reason})")


def main() -> int:
    test_nearest_cut()
    test_symmetric()
    test_nonuniform()
    test_minimum_width()
    print(f"\n{_passed} checks passed, {len(_failed)} failed")
    if _failed:
        for message in _failed[:20]:
            print(f"  FAILED: {message}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
