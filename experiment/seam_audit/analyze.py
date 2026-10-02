"""
analyze.py — CPU analysis of seam-audit dumps (written by dump_state.py).

Question: how far does a K-slab chain (test B) depart from a reference run
(A, usually K=1) at the seams, measured against the reference's own
run-to-run noise floor?

Per-particle triplets: every selected particle i of B1 is matched to a
particle a(i) of A1, and a(i) to a particle b(a) of A2. On exactly those
particles

    test  = |B1_i    - A1_a(i)|        (decomposition + noise)
    noise = |A1_a(i) - A2_b(a)|        (reference run-to-run floor)

so every ratio test/noise compares identical particle sets. With a second
test run B2 the same is repeated for (B2, A2, A1) — a replicate of the ratio —
and |B1 - B2| is compared with |A1 - A2| on the B1 particles (is the
decomposed solver noisier than the reference?).

Matching: KD-tree nearest neighbour on positions (scipy cKDTree), tolerance
0.05 * smoothing_length; the global-id identity match is always computed too
and reported as an agreement rate (fraction of KD matches whose ids agree).
``match="id"`` makes the id match the primary method.

Binning (B's seams, from its JSON sidecar): global column
c = floor((x_B - origin_x) / h); for the nearest cut s, signed = c - s
(..., -2 = left slab column 1, -1 = left slab column 0, 0 = right slab
column 0, 1 = right slab column 1, ...) and unsigned distance
d = c - s if c >= s else s - 1 - c. Bins d = 0..15 and "interior" (d >= 16, or
no seam at all).

Column 0 split (the seam column on both sides), from B's crossed_last_step
(owner slab at N differs from N-1):
  flagged      another particle that crossed a seam in the last step lies
               within smoothing_length (strictly closer; Wendland W(r>=h)=0)
  unflagged    no such particle
  extras: crossed_self (the particle itself crossed), unflagged_not_crossed
  (cleanest "stale ghost density only" set), flagged_departed (a crossing
  neighbour LEFT this particle's slab — the departed migrant is missing from
  the sender's lists during phase C), flagged_arrived_only,
  flagged_including_self (the literal reading counting the particle itself).

Fields: acceleration, shift, velocity (norm of the vector difference),
density, pressure, kernel_sum (absolute difference); acceleration, shift
and density are the primary ones. Statistics are restricted to fluid
particles by default (walls have no acceleration/shift); ``particles="all"``
keeps every particle.

Outputs in out_dir: report.json (everything), report.md (compact tables),
difference_by_column.png, shift_profile.png, kernel_sum_column0.png.

Usage:
    .venv/Scripts/python.exe -m experiment.seam_audit.analyze \\
        --reference DUMPS/reference_v5_K1_dev1_t1_N300.npz DUMPS/reference_v5_K1_dev1_t2_N300.npz \\
        --test DUMPS/v5_K2_t1_N300.npz DUMPS/v5_K2_t2_N300.npz --out ANALYSIS/v5_K2_N300
    .venv/Scripts/python.exe -m experiment.seam_audit.analyze --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import tempfile
from dataclasses import dataclass

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

FLUID_KIND = 0
DISTANCE_BIN_COUNT = 16                 # bins d = 0 .. 15, then "interior"
INTERIOR_BIN = "interior"
NO_SEAM = -1
PROFILE_SIGNED_FIRST = -16
PROFILE_SIGNED_LAST = 15
FIELDS = ("acceleration", "shift", "density", "pressure", "kernel_sum", "velocity")
PRIMARY_FIELDS = ("acceleration", "shift", "density")
VECTOR_FIELDS = frozenset({"acceleration", "shift", "velocity"})
RATIO_METRICS = ("mean", "rms", "p99", "max")
COLUMN0_CATEGORIES = ("all", "flagged", "unflagged", "crossed_self",
                      "unflagged_not_crossed", "flagged_departed",
                      "flagged_arrived_only", "flagged_including_self")
# Every column-0 group, so the signed B1 - A1 kernel_sum exists for
# flagged_departed (the missing neighbour, directly) and its complements.
KERNEL_SUM_CATEGORIES = COLUMN0_CATEGORIES
KERNEL_SUM_FIGURE_CATEGORIES = ("flagged_departed", "flagged_arrived_only", "unflagged")
KERNEL_SUM_QUANTILES = (1, 5, 25, 50, 75, 95, 99)
DEFAULT_FAR_BIN_START = 8
DEFAULT_TOLERANCE_FACTOR = 0.05
REQUIRED_ARRAYS = ("id", "slab", "crossed_last_step", "position", "velocity",
                   "acceleration", "shift", "density", "pressure", "kernel_sum",
                   "material")

# Reference palette, first three categorical slots (validated all-pairs).
TEST_COLOR = "#eb6834"          # orange: test (B)
NOISE_COLOR = "#2a78d6"         # blue:   reference / noise floor (A1 vs A2)
REPLICATE_COLOR = "#1baf7a"     # aqua:   A2 / second test pair
NEUTRAL_COLOR = "#8a8984"


# =============================================================================
# Loading
# =============================================================================

@dataclass
class RunDump:
    label: str
    npz_path: pathlib.Path
    json_path: pathlib.Path
    metadata: dict
    arrays: dict

    @property
    def particle_count(self) -> int:
        return int(self.arrays["id"].shape[0])


def resolve_dump_paths(path) -> tuple[pathlib.Path, pathlib.Path]:
    """Accept the .npz, the .json sidecar or the common stem."""
    path = pathlib.Path(path)
    if path.suffix == ".npz":
        return path, path.with_suffix(".json")
    if path.suffix == ".json":
        return path.with_suffix(".npz"), path
    return path.with_name(path.name + ".npz"), path.with_name(path.name + ".json")


def load_dump(path, label: str) -> RunDump:
    npz_path, json_path = resolve_dump_paths(path)
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"{npz_path}: missing arrays {missing}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if "previous_slab" not in arrays:
        arrays["previous_slab"] = np.where(arrays["crossed_last_step"],
                                           255, arrays["slab"]).astype(np.uint8)
    return RunDump(label=label, npz_path=npz_path, json_path=json_path,
                   metadata=metadata, arrays=arrays)


def particle_selection(dump: RunDump, particle_filter: str) -> tuple[np.ndarray, str]:
    """Indices of the particles that enter the statistics."""
    every_particle = np.arange(dump.particle_count, dtype=np.int64)
    if particle_filter == "all":
        return every_particle, "all"
    kinds = dump.metadata.get("material_kinds")
    if not kinds:
        return every_particle, "all (no material_kinds in sidecar)"
    fluid_groups = [group for group, kind in enumerate(kinds) if int(kind) == FLUID_KIND]
    mask = np.isin(dump.arrays["material"], np.asarray(fluid_groups, dtype=np.int64))
    return np.flatnonzero(mask).astype(np.int64), "fluid"


# =============================================================================
# Matching
# =============================================================================

@dataclass
class MatchResult:
    target_index: np.ndarray     # int64, -1 where not matched
    valid: np.ndarray            # bool
    distance: np.ndarray         # float64 position distance (inf where unknown)


class Matcher:
    """Matches particles between dumps; caches one KD-tree per dump over that
    dump's selected particles."""

    def __init__(self, dumps: dict, selections: dict, tolerance: float) -> None:
        self.dumps = dumps
        self.selections = selections
        self.tolerance = tolerance
        self._trees: dict = {}
        self._selected_masks: dict = {}
        self._full_selection_matches: dict = {}

    def selected_mask(self, label: str) -> np.ndarray:
        if label not in self._selected_masks:
            mask = np.zeros(self.dumps[label].particle_count, dtype=bool)
            mask[self.selections[label]] = True
            self._selected_masks[label] = mask
        return self._selected_masks[label]

    def tree(self, label: str):
        if label not in self._trees:
            from scipy.spatial import cKDTree
            positions = self.dumps[label].arrays["position"][self.selections[label]]
            self._trees[label] = cKDTree(positions.astype(np.float64))
        return self._trees[label]

    def by_position(self, source_label: str, source_indices: np.ndarray,
                    target_label: str) -> MatchResult:
        points = self.dumps[source_label].arrays["position"][source_indices].astype(np.float64)
        target_selection = self.selections[target_label]
        if points.shape[0] == 0 or target_selection.size == 0:
            empty = np.zeros(points.shape[0], dtype=bool)
            return MatchResult(np.full(points.shape[0], -1, np.int64), empty,
                               np.full(points.shape[0], np.inf))
        distance, local_index = self.tree(target_label).query(
            points, k=1, distance_upper_bound=self.tolerance * (1.0 + 1e-9), workers=-1)
        valid = np.isfinite(distance) & (distance <= self.tolerance)
        target_index = np.full(points.shape[0], -1, dtype=np.int64)
        target_index[valid] = target_selection[local_index[valid]]
        return MatchResult(target_index, valid, distance)

    def by_id(self, source_label: str, source_indices: np.ndarray,
              target_label: str) -> MatchResult:
        source = self.dumps[source_label]
        target = self.dumps[target_label]
        source_ids = source.arrays["id"][source_indices].astype(np.int64)
        target_ids = target.arrays["id"].astype(np.int64)
        if target_ids.size == 0:
            empty = np.zeros(source_ids.size, dtype=bool)
            return MatchResult(np.full(source_ids.size, -1, np.int64), empty,
                               np.full(source_ids.size, np.inf))
        found = np.minimum(np.searchsorted(target_ids, source_ids), target_ids.size - 1)
        valid = (target_ids[found] == source_ids) & self.selected_mask(target_label)[found]
        target_index = np.where(valid, found, -1).astype(np.int64)
        distance = np.full(source_ids.size, np.inf)
        if valid.any():
            offset = (source.arrays["position"][source_indices[valid]].astype(np.float64)
                      - target.arrays["position"][found[valid]].astype(np.float64))
            distance[valid] = np.linalg.norm(offset, axis=1)
        return MatchResult(target_index, valid, distance)

    def match(self, source_label: str, source_indices: np.ndarray, target_label: str,
              method: str) -> MatchResult:
        """Match with the chosen method; results for a dump's FULL selection are
        cached (the matching summary and the triplets query the same pairs)."""
        is_full_selection = source_indices is self.selections[source_label]
        key = (source_label, target_label, method)
        if is_full_selection and key in self._full_selection_matches:
            return self._full_selection_matches[key]
        if method == "id":
            result = self.by_id(source_label, source_indices, target_label)
        else:
            result = self.by_position(source_label, source_indices, target_label)
        if is_full_selection:
            self._full_selection_matches[key] = result
        return result

    def unmatched_nearest_distance(self, source_label: str, source_indices: np.ndarray,
                                   target_label: str) -> np.ndarray:
        """Unbounded nearest distance (diagnostic for the unmatched few)."""
        if source_indices.size == 0 or self.selections[target_label].size == 0:
            return np.zeros(0)
        points = self.dumps[source_label].arrays["position"][source_indices].astype(np.float64)
        distance, _ = self.tree(target_label).query(points, k=1, workers=-1)
        return distance


def quantile_summary(values: np.ndarray, scale: float = 1.0) -> dict:
    if values.size == 0:
        return {"p50": None, "p99": None, "max": None}
    percentile_50, percentile_99 = np.percentile(values, [50, 99])
    return {"p50": float(percentile_50 / scale), "p99": float(percentile_99 / scale),
            "max": float(values.max() / scale)}


def matching_summary(matcher: Matcher, source_label: str, target_label: str,
                     smoothing_length: float) -> dict:
    """Both match methods for the selected particles of source -> target."""
    source_indices = matcher.selections[source_label]
    source_ids = matcher.dumps[source_label].arrays["id"][source_indices].astype(np.int64)
    target_ids = matcher.dumps[target_label].arrays["id"].astype(np.int64)

    by_position = matcher.match(source_label, source_indices, target_label, "kdtree")
    matched_targets = by_position.target_index[by_position.valid]
    agreement = source_ids[by_position.valid] == target_ids[matched_targets]
    unmatched_nearest = matcher.unmatched_nearest_distance(
        source_label, source_indices[~by_position.valid], target_label)

    by_id = matcher.match(source_label, source_indices, target_label, "id")
    id_offsets = by_id.distance[by_id.valid]
    return {
        "source": source_label,
        "target": target_label,
        "queried": int(source_indices.size),
        "kdtree": {
            "matched": int(by_position.valid.sum()),
            "unmatched": int((~by_position.valid).sum()),
            "id_agreement_rate": (float(agreement.mean()) if agreement.size else None),
            "id_disagreements": int((~agreement).sum()),
            "many_to_one": int(matched_targets.size - np.unique(matched_targets).size),
            "distance_over_h": quantile_summary(by_position.distance[by_position.valid],
                                                smoothing_length),
            "unmatched_nearest_over_h": quantile_summary(unmatched_nearest, smoothing_length),
        },
        "id": {
            "matched": int(by_id.valid.sum()),
            "unmatched": int((~by_id.valid).sum()),
            "position_offset_over_h": quantile_summary(id_offsets, smoothing_length),
            "offset_beyond_tolerance": int((id_offsets > matcher.tolerance).sum()),
        },
    }


@dataclass
class Triplets:
    test: np.ndarray        # indices into the test dump
    reference: np.ndarray   # indices into the reference dump
    repeat: np.ndarray      # indices into the reference repeat dump
    queried: int
    unmatched_first: int    # test -> reference
    unmatched_second: int   # reference -> repeat


def build_triplets(matcher: Matcher, test_label: str, reference_label: str,
                   repeat_label: str, method: str) -> Triplets:
    test_indices = matcher.selections[test_label]
    first = matcher.match(test_label, test_indices, reference_label, method)
    test_kept = test_indices[first.valid]
    reference_kept = first.target_index[first.valid]
    second = matcher.match(reference_label, reference_kept, repeat_label, method)
    return Triplets(test=test_kept[second.valid],
                    reference=reference_kept[second.valid],
                    repeat=second.target_index[second.valid],
                    queried=int(test_indices.size),
                    unmatched_first=int((~first.valid).sum()),
                    unmatched_second=int((~second.valid).sum()))


# =============================================================================
# Seam geometry
# =============================================================================

@dataclass
class SeamGeometry:
    cuts: list
    origin_x: float
    smoothing_length: float

    @classmethod
    def from_dump(cls, dump: RunDump) -> "SeamGeometry":
        metadata = dump.metadata
        return cls(cuts=[int(cut) for cut in metadata.get("cuts", [])],
                   origin_x=float(metadata["origin_x"]),
                   smoothing_length=float(metadata["smoothing_length"]))


def classify_columns(x_positions: np.ndarray, geometry: SeamGeometry) -> dict:
    """Global column, signed column and unsigned distance to the nearest seam."""
    column = np.floor((np.asarray(x_positions, dtype=np.float64) - geometry.origin_x)
                      / geometry.smoothing_length).astype(np.int64)
    count = column.size
    if not geometry.cuts:
        return {"column": column,
                "signed": np.full(count, NO_SEAM, dtype=np.int64),
                "distance": np.full(count, NO_SEAM, dtype=np.int64),
                "nearest_cut": np.full(count, -1, dtype=np.int64)}
    cuts = np.asarray(geometry.cuts, dtype=np.int64)
    offset = column[:, None] - cuts[None, :]
    unsigned = np.where(offset >= 0, offset, -offset - 1)
    nearest = np.argmin(unsigned, axis=1)
    rows = np.arange(count)
    return {"column": column, "signed": offset[rows, nearest],
            "distance": unsigned[rows, nearest], "nearest_cut": nearest}


def bin_masks(distance: np.ndarray) -> dict:
    masks = {str(bin_distance): distance == bin_distance
             for bin_distance in range(DISTANCE_BIN_COUNT)}
    masks[INTERIOR_BIN] = (distance >= DISTANCE_BIN_COUNT) | (distance < 0)
    return masks


def bin_labels() -> list:
    return [str(bin_distance) for bin_distance in range(DISTANCE_BIN_COUNT)] + [INTERIOR_BIN]


def is_far_bin(label: str, far_bin_start: int) -> bool:
    return label == INTERIOR_BIN or int(label) >= far_bin_start


def column0_categories(dump: RunDump, geometry: SeamGeometry) -> tuple[dict, dict]:
    """Boolean masks over ALL particles of a test dump (False outside column 0),
    plus the seam side / owner consistency count."""
    from scipy.spatial import cKDTree
    positions = dump.arrays["position"].astype(np.float64)
    columns = classify_columns(positions[:, 0], geometry)
    column0 = columns["distance"] == 0
    crossed = dump.arrays["crossed_last_step"].astype(bool)
    slab = dump.arrays["slab"].astype(np.int64)
    previous_slab = dump.arrays["previous_slab"].astype(np.int64)
    flagged = np.zeros(dump.particle_count, dtype=bool)
    departed = np.zeros(dump.particle_count, dtype=bool)
    crossing_indices = np.flatnonzero(crossed)
    column0_indices = np.flatnonzero(column0)
    if crossing_indices.size and column0_indices.size:
        tree = cKDTree(positions[crossing_indices])
        # strictly closer than h: a neighbour exactly at h has W = 0
        radius = float(np.nextafter(geometry.smoothing_length, 0.0))
        neighbour_lists = tree.query_ball_point(positions[column0_indices], r=radius,
                                                workers=-1)
        for particle_index, neighbour_list in zip(column0_indices, neighbour_lists):
            if not neighbour_list:
                continue
            neighbours = crossing_indices[np.asarray(neighbour_list, dtype=np.int64)]
            neighbours = neighbours[neighbours != particle_index]
            if neighbours.size == 0:
                continue
            flagged[particle_index] = True
            if np.any(previous_slab[neighbours] == slab[particle_index]):
                departed[particle_index] = True
    crossed_self = column0 & crossed
    categories = {
        "all": column0,
        "flagged": flagged,
        "unflagged": column0 & ~flagged,
        "crossed_self": crossed_self,
        "unflagged_not_crossed": column0 & ~flagged & ~crossed,
        "flagged_departed": departed,
        "flagged_arrived_only": flagged & ~departed,
        "flagged_including_self": flagged | crossed_self,
    }
    consistency = {"column0_particles": int(column0.sum()),
                   "crossing_particles": int(crossed.sum()),
                   "seam_side_owner_disagreements": 0}
    if geometry.cuts and column0.any():
        expected_owner = (columns["nearest_cut"][column0]
                          + (columns["signed"][column0] >= 0).astype(np.int64))
        consistency["seam_side_owner_disagreements"] = int(
            (expected_owner != slab[column0]).sum())
    return categories, consistency


# =============================================================================
# Statistics
# =============================================================================

def describe(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean": None, "rms": None, "p50": None, "p99": None, "max": None}
    values = values.astype(np.float64, copy=False)
    percentile_50, percentile_99 = np.percentile(values, [50, 99])
    return {"mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values * values))),
            "p50": float(percentile_50), "p99": float(percentile_99),
            "max": float(values.max())}


def safe_ratio(test_value, noise_value):
    """test / noise; None when both are 0 or unknown, inf when only noise is 0."""
    if test_value is None or noise_value is None:
        return None
    if noise_value > 0:
        return test_value / noise_value
    if test_value == 0:
        return None
    return math.inf


def compare_statistics(test_values: np.ndarray, noise_values: np.ndarray) -> dict:
    test = describe(test_values)
    noise = describe(noise_values)
    return {"count": int(test_values.size), "test": test, "noise": noise,
            "ratio": {metric: safe_ratio(test[metric], noise[metric])
                      for metric in RATIO_METRICS}}


def field_difference(first: RunDump, first_indices: np.ndarray, second: RunDump,
                     second_indices: np.ndarray, field: str) -> np.ndarray:
    difference = (first.arrays[field][first_indices].astype(np.float64)
                  - second.arrays[field][second_indices].astype(np.float64))
    if field in VECTOR_FIELDS:
        return np.linalg.norm(difference, axis=1)
    return np.abs(difference)


def field_magnitude(dump: RunDump, indices: np.ndarray, field: str) -> np.ndarray:
    values = dump.arrays[field][indices].astype(np.float64)
    return np.linalg.norm(values, axis=1) if field in VECTOR_FIELDS else np.abs(values)


def distribution(values: np.ndarray) -> dict:
    out = {"count": int(values.size)}
    if values.size == 0:
        out.update({"mean": None, "std": None})
        out.update({f"p{quantile}": None for quantile in KERNEL_SUM_QUANTILES})
        return out
    values = values.astype(np.float64, copy=False)
    out["mean"] = float(values.mean())
    out["std"] = float(values.std())
    for quantile, value in zip(KERNEL_SUM_QUANTILES,
                               np.percentile(values, KERNEL_SUM_QUANTILES)):
        out[f"p{quantile}"] = float(value)
    return out


def compare_fields(test_values_by_field: dict, noise_values_by_field: dict,
                   masks: dict, column0_masks: dict) -> dict:
    fields = {}
    for field in FIELDS:
        test_values = test_values_by_field[field]
        noise_values = noise_values_by_field[field]
        fields[field] = {
            "bins": {label: compare_statistics(test_values[mask], noise_values[mask])
                     for label, mask in masks.items()},
            "column0": {name: compare_statistics(test_values[mask], noise_values[mask])
                        for name, mask in column0_masks.items()},
        }
    return fields


def compare_triplets(dumps: dict, triplets: Triplets, test_label: str,
                     reference_label: str, repeat_label: str, geometry: SeamGeometry,
                     categories: dict) -> tuple[dict, dict]:
    """Per-bin and column-0 statistics of |test - reference| vs |reference - repeat|.
    Returns (report section, raw arrays for figures)."""
    test, reference, repeat = dumps[test_label], dumps[reference_label], dumps[repeat_label]
    columns = classify_columns(test.arrays["position"][triplets.test, 0], geometry)
    masks = bin_masks(columns["distance"])
    column0_masks = {name: categories[name][triplets.test] for name in COLUMN0_CATEGORIES}
    test_values = {field: field_difference(test, triplets.test, reference, triplets.reference,
                                           field) for field in FIELDS}
    noise_values = {field: field_difference(reference, triplets.reference, repeat,
                                            triplets.repeat, field) for field in FIELDS}
    section = {
        "label": f"|{test_label} - {reference_label}| vs noise "
                 f"|{reference_label} - {repeat_label}|",
        "particles": int(triplets.test.size),
        "queried": triplets.queried,
        "unmatched_test_to_reference": triplets.unmatched_first,
        "unmatched_reference_to_repeat": triplets.unmatched_second,
        "column0_counts": {name: int(mask.sum()) for name, mask in column0_masks.items()},
        "fields": compare_fields(test_values, noise_values, masks, column0_masks),
    }
    raw = {"columns": columns, "column0_masks": column0_masks,
           "test_values": test_values, "noise_values": noise_values}
    return section, raw


def compare_test_noise(matcher: Matcher, triplets: Triplets, method: str,
                       geometry: SeamGeometry, categories: dict) -> dict:
    """|B1 - B2| vs |A1 - A2| on the B1 triplet particles that also match into B2."""
    dumps = matcher.dumps
    partner = matcher.match("B1", triplets.test, "B2", method)
    keep = partner.valid
    first_test = triplets.test[keep]
    second_test = partner.target_index[keep]
    columns = classify_columns(dumps["B1"].arrays["position"][first_test, 0], geometry)
    masks = bin_masks(columns["distance"])
    column0_masks = {name: categories[name][first_test] for name in COLUMN0_CATEGORIES}
    test_values = {field: field_difference(dumps["B1"], first_test, dumps["B2"], second_test,
                                           field) for field in FIELDS}
    noise_values = {field: field_difference(dumps["A1"], triplets.reference[keep],
                                            dumps["A2"], triplets.repeat[keep], field)
                    for field in FIELDS}
    return {
        "label": "|B1 - B2| (test-config run-to-run) vs noise |A1 - A2|",
        "particles": int(keep.sum()),
        "unmatched_B1_to_B2": int((~keep).sum()),
        "column0_counts": {name: int(mask.sum()) for name, mask in column0_masks.items()},
        "fields": compare_fields(test_values, noise_values, masks, column0_masks),
    }


def kernel_sum_distributions(dumps: dict, triplets: Triplets, raw: dict) -> dict:
    test_kernel_sum = dumps["B1"].arrays["kernel_sum"][triplets.test].astype(np.float64)
    reference_kernel_sum = dumps["A1"].arrays["kernel_sum"][triplets.reference].astype(np.float64)
    out = {}
    for name in KERNEL_SUM_CATEGORIES:
        mask = raw["column0_masks"][name]
        out[name] = {
            "test_B1": distribution(test_kernel_sum[mask]),
            "reference_A1": distribution(reference_kernel_sum[mask]),
            "difference_B1_minus_A1": distribution(test_kernel_sum[mask]
                                                   - reference_kernel_sum[mask]),
        }
    return out


def shift_profile(dumps: dict, triplets: Triplets, raw: dict) -> dict:
    signed = np.where(raw["columns"]["distance"] >= 0, raw["columns"]["signed"],
                      np.iinfo(np.int64).min)          # no seam: in no profile column
    shift_test = field_magnitude(dumps["B1"], triplets.test, "shift")
    shift_reference = field_magnitude(dumps["A1"], triplets.reference, "shift")
    shift_repeat = field_magnitude(dumps["A2"], triplets.repeat, "shift")
    difference_test = raw["test_values"]["shift"]
    difference_noise = raw["noise_values"]["shift"]
    profile = {"signed_columns": [], "counts": [], "mean_shift_B1": [], "mean_shift_A1": [],
               "mean_shift_A2": [], "mean_difference_B1_A1": [],
               "mean_difference_A1_A2": []}
    for signed_column in range(PROFILE_SIGNED_FIRST, PROFILE_SIGNED_LAST + 1):
        mask = signed == signed_column
        count = int(mask.sum())
        profile["signed_columns"].append(signed_column)
        profile["counts"].append(count)
        for key, values in (("mean_shift_B1", shift_test), ("mean_shift_A1", shift_reference),
                            ("mean_shift_A2", shift_repeat),
                            ("mean_difference_B1_A1", difference_test),
                            ("mean_difference_A1_A2", difference_noise)):
            profile[key].append(float(values[mask].mean()) if count else None)
    return profile


def verdict(primary: dict, far_bin_start: int) -> dict:
    worst = {}
    far_worst = {}
    column0_ratio = {}
    for field in FIELDS:
        bins = primary["fields"][field]["bins"]
        worst[field] = {}
        for metric in RATIO_METRICS:
            candidates = [(statistics["ratio"][metric], label) for label, statistics in bins.items()
                          if statistics["ratio"][metric] is not None]
            if candidates:
                value, label = max(candidates, key=lambda item: item[0])
                worst[field][metric] = {"value": value, "bin": label}
            else:
                worst[field][metric] = {"value": None, "bin": None}
        far_candidates = [(statistics["ratio"]["rms"], label) for label, statistics in bins.items()
                          if is_far_bin(label, far_bin_start)
                          and statistics["ratio"]["rms"] is not None]
        if far_candidates:
            value, label = max(far_candidates, key=lambda item: item[0])
            far_worst[field] = {"value": value, "bin": label}
        else:
            far_worst[field] = {"value": None, "bin": None}
        column0_ratio[field] = {name: statistics["ratio"]["rms"] for name, statistics
                                in primary["fields"][field]["column0"].items()}
    return {
        "seam_excess": primary["fields"]["acceleration"]["bins"]["0"]["ratio"]["rms"],
        "seam_excess_definition": "rms(|B1-A1|) / rms(|A1-A2|) of acceleration at d=0",
        "worst_ratio": worst,
        "far_bin_start": far_bin_start,
        "far_bin_worst_rms_ratio": far_worst,
        "column0_rms_ratio": column0_ratio,
    }


# =============================================================================
# Driver
# =============================================================================

def run_metadata(dump: RunDump) -> dict:
    metadata = dump.metadata
    environment = metadata.get("environment", {})
    return {
        "path": str(dump.npz_path),
        "run_name": metadata.get("run_name"),
        "version": metadata.get("version"),
        "case_name": metadata.get("case_name"),
        "slabs": metadata.get("slabs"),
        "cuts": metadata.get("cuts"),
        "device_map": metadata.get("device_map"),
        "horizon": metadata.get("horizon"),
        "particles": dump.particle_count,
        "crossed_last_step": int(dump.arrays["crossed_last_step"].sum()),
        "solver_switches": {key: value for key, value in environment.items()
                            if key.startswith(("V5_", "V6_"))},
        "invariants": metadata.get("invariants", {}),
    }


def check_consistency(dumps: dict) -> dict:
    notes = []
    smoothing_lengths = {label: float(dump.metadata["smoothing_length"])
                         for label, dump in dumps.items()}
    dimensions = {label: int(dump.metadata.get("dimension", dump.arrays["position"].shape[1]))
                  for label, dump in dumps.items()}
    if len(set(smoothing_lengths.values())) != 1:
        raise ValueError(f"smoothing lengths differ between dumps: {smoothing_lengths}")
    if len(set(dimensions.values())) != 1:
        raise ValueError(f"dimensions differ between dumps: {dimensions}")
    horizons = {label: dump.metadata.get("horizon") for label, dump in dumps.items()}
    if len(set(horizons.values())) != 1:
        notes.append(f"horizons differ: {horizons}")
    cases = {label: dump.metadata.get("case_name") for label, dump in dumps.items()}
    if len(set(cases.values())) != 1:
        notes.append(f"case names differ: {cases}")
    if "B2" in dumps and dumps["B1"].metadata.get("cuts") != dumps["B2"].metadata.get("cuts"):
        notes.append("B1 and B2 have different cuts; each pair uses its own test run's seams")
    return {"horizons": horizons, "case_names": cases, "notes": notes}


def analyze(reference_paths, test_paths, match: str = "kdtree", out_dir=None, *,
            particle_filter: str = "fluid", far_bin_start: int = DEFAULT_FAR_BIN_START,
            tolerance_factor: float = DEFAULT_TOLERANCE_FACTOR,
            make_figures: bool = True) -> dict:
    """Compare test dumps (B1[, B2]) against reference dumps (A1, A2).
    Returns the report dict; writes report.json / report.md / figures into
    out_dir when given."""
    if match not in ("kdtree", "id"):
        raise ValueError(f"match must be 'kdtree' or 'id', got {match!r}")
    if len(reference_paths) != 2:
        raise ValueError("need exactly two reference dumps (A1, A2)")
    if not 1 <= len(test_paths) <= 2:
        raise ValueError("need one or two test dumps (B1[, B2])")
    dumps = {"A1": load_dump(reference_paths[0], "A1"),
             "A2": load_dump(reference_paths[1], "A2"),
             "B1": load_dump(test_paths[0], "B1")}
    if len(test_paths) == 2:
        dumps["B2"] = load_dump(test_paths[1], "B2")
    consistency = check_consistency(dumps)
    smoothing_length = float(dumps["B1"].metadata["smoothing_length"])
    tolerance = tolerance_factor * smoothing_length

    selections = {}
    filter_descriptions = {}
    for label, dump in dumps.items():
        selections[label], filter_descriptions[label] = particle_selection(dump, particle_filter)
    matcher = Matcher(dumps, selections, tolerance)

    matching = {
        "method": match,
        "tolerance": tolerance,
        "tolerance_over_h": tolerance_factor,
        "B1_to_A1": matching_summary(matcher, "B1", "A1", smoothing_length),
        "A1_to_A2": matching_summary(matcher, "A1", "A2", smoothing_length),
    }
    if "B2" in dumps:
        matching["B2_to_A2"] = matching_summary(matcher, "B2", "A2", smoothing_length)
        matching["A2_to_A1"] = matching_summary(matcher, "A2", "A1", smoothing_length)
        matching["B1_to_B2"] = matching_summary(matcher, "B1", "B2", smoothing_length)

    geometry_first = SeamGeometry.from_dump(dumps["B1"])
    categories_first, consistency["B1"] = column0_categories(dumps["B1"], geometry_first)
    triplets = build_triplets(matcher, "B1", "A1", "A2", match)
    primary, primary_raw = compare_triplets(dumps, triplets, "B1", "A1", "A2",
                                            geometry_first, categories_first)
    comparisons = {"primary": primary}
    secondary_raw = None
    if "B2" in dumps:
        geometry_second = SeamGeometry.from_dump(dumps["B2"])
        categories_second, consistency["B2"] = column0_categories(dumps["B2"], geometry_second)
        triplets_second = build_triplets(matcher, "B2", "A2", "A1", match)
        comparisons["secondary"], secondary_raw = compare_triplets(
            dumps, triplets_second, "B2", "A2", "A1", geometry_second, categories_second)
        comparisons["test_noise"] = compare_test_noise(matcher, triplets, match,
                                                       geometry_first, categories_first)

    report = {
        "tool": "experiment/seam_audit/analyze.py",
        "inputs": {"reference": [str(dumps["A1"].npz_path), str(dumps["A2"].npz_path)],
                   "test": [str(dumps[label].npz_path) for label in ("B1", "B2")
                            if label in dumps]},
        "settings": {"match": match, "tolerance": tolerance,
                     "tolerance_factor": tolerance_factor,
                     "smoothing_length": smoothing_length,
                     "particle_filter": particle_filter,
                     "particle_filter_applied": filter_descriptions,
                     "selected_particles": {label: int(selection.size)
                                            for label, selection in selections.items()},
                     "far_bin_start": far_bin_start,
                     "seams_from": "test run sidecar (cuts, origin_x)",
                     "cuts": geometry_first.cuts,
                     "origin_x": geometry_first.origin_x},
        "runs": {label: run_metadata(dump) for label, dump in dumps.items()},
        "consistency": consistency,
        "matching": matching,
        "comparisons": comparisons,
        "column0_kernel_sum": kernel_sum_distributions(dumps, triplets, primary_raw),
        "shift_profile": shift_profile(dumps, triplets, primary_raw),
    }
    report["verdict"] = verdict(primary, far_bin_start)
    report["summary"] = compact_summary(report)

    if out_dir is not None:
        out_directory = pathlib.Path(out_dir)
        out_directory.mkdir(parents=True, exist_ok=True)
        write_reports(report, out_directory)        # usable even if a figure fails
        if make_figures:
            figure_paths = write_figures(report, dumps, triplets, primary_raw,
                                         out_directory)
            report["figures"] = [str(path) for path in figure_paths]
            write_reports(report, out_directory)    # again, now with the Figures section
    return report


def write_reports(report: dict, out_directory: pathlib.Path) -> None:
    (out_directory / "report.json").write_text(
        json.dumps(json_safe(report), indent=1), encoding="utf-8")
    (out_directory / "report.md").write_text(render_markdown(report), encoding="utf-8")


def compact_summary(report: dict) -> dict:
    """The row run_matrix.py puts in summary.md / summary.json."""
    primary = report["comparisons"]["primary"]
    d0_rms_ratio = {}
    for field in PRIMARY_FIELDS:
        column0 = primary["fields"][field]["column0"]
        d0_rms_ratio[field] = {name: column0[name]["ratio"]["rms"]
                               for name in ("all", "flagged", "unflagged")}
    invariants = {}
    for label, run in report["runs"].items():
        run_invariants = run.get("invariants") or {}
        invariants[label] = {key: run_invariants.get(key) for key in
                             ("drift", "missing_ids", "duplicate_ids", "stamp_errors_gpu",
                              "stamp_errors_host", "overflow_total", "valid")}
    second_pair = report["comparisons"].get("secondary")
    test_noise = report["comparisons"].get("test_noise")
    b1_to_a1 = report["matching"]["B1_to_A1"]
    a1_to_a2 = report["matching"]["A1_to_A2"]
    return {
        "test_run": report["runs"]["B1"].get("run_name"),
        "reference_run": report["runs"]["A1"].get("run_name"),
        "case_name": report["runs"]["B1"].get("case_name"),
        "horizon": report["runs"]["B1"].get("horizon"),
        "slabs": report["runs"]["B1"].get("slabs"),
        "triplets": primary["particles"],
        "column0_counts": {name: primary["column0_counts"][name]
                           for name in ("all", "flagged", "unflagged", "crossed_self")},
        "d0_rms_ratio": d0_rms_ratio,
        "seam_excess": report["verdict"]["seam_excess"],
        "far_bin_worst_rms_ratio": {field: report["verdict"]["far_bin_worst_rms_ratio"][field]
                                    for field in PRIMARY_FIELDS},
        "second_pair_d0_rms_ratio": (
            {field: second_pair["fields"][field]["bins"]["0"]["ratio"]["rms"]
             for field in PRIMARY_FIELDS} if second_pair else None),
        "test_noise_d0_rms_ratio": (
            {field: test_noise["fields"][field]["bins"]["0"]["ratio"]["rms"]
             for field in PRIMARY_FIELDS} if test_noise else None),
        "unmatched": {
            "B1_to_A1_kdtree": b1_to_a1["kdtree"]["unmatched"],
            "A1_to_A2_kdtree": a1_to_a2["kdtree"]["unmatched"],
            "B1_to_A1_id": b1_to_a1["id"]["unmatched"],
            "A1_to_A2_id": a1_to_a2["id"]["unmatched"],
        },
        "id_agreement_rate": {"B1_to_A1": b1_to_a1["kdtree"]["id_agreement_rate"],
                              "A1_to_A2": a1_to_a2["kdtree"]["id_agreement_rate"]},
        "invariants": invariants,
        "all_runs_valid": all(bool(values.get("valid")) for values in invariants.values()),
    }


# =============================================================================
# JSON / Markdown output
# =============================================================================

def json_safe(value):
    """Recursively convert numpy types; inf -> "inf", NaN -> None."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    return value


def format_value(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):,}"
    return f"{float(value):.{digits}g}"


def markdown_table(header: list, rows: list) -> list:
    def cell(text) -> str:
        return str(text).replace("|", "\\|")

    lines = ["| " + " | ".join(cell(text) for text in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell(text) for text in row) + " |" for row in rows]
    return lines


def render_markdown(report: dict) -> str:
    runs = report["runs"]
    test_run = runs["B1"]
    reference_run = runs["A1"]
    primary = report["comparisons"]["primary"]
    settings = report["settings"]
    verdict_section = report["verdict"]
    lines = [
        f"# Seam audit: {test_run.get('run_name')} vs {reference_run.get('run_name')} "
        f"at N={test_run.get('horizon')}",
        "",
        f"- case: `{test_run.get('case_name')}`; test {test_run.get('version')} "
        f"K={test_run.get('slabs')} cuts={test_run.get('cuts')}; reference "
        f"{reference_run.get('version')} K={reference_run.get('slabs')}",
        f"- match: **{settings['match']}**, tolerance {format_value(settings['tolerance'])} m "
        f"({settings['tolerance_factor']} h); particles: {settings['particle_filter']} "
        f"({format_value(primary['particles'])} triplets of "
        f"{format_value(primary['queried'])} selected test particles)",
        f"- test = |B1 - A1|, noise = |A1 - A2| on the same particles; ratio = test / noise",
        "",
        "## Verdict",
        "",
        f"- **seam_excess** (acceleration, d=0, rms ratio): "
        f"**{format_value(verdict_section['seam_excess'])}**",
    ]
    for field in PRIMARY_FIELDS:
        worst = verdict_section["worst_ratio"][field]
        far = verdict_section["far_bin_worst_rms_ratio"][field]
        lines.append(
            f"- {field}: worst rms {format_value(worst['rms']['value'])} (d={worst['rms']['bin']}), "
            f"p99 {format_value(worst['p99']['value'])} (d={worst['p99']['bin']}), "
            f"max {format_value(worst['max']['value'])} (d={worst['max']['bin']}); far bins "
            f"(d>={verdict_section['far_bin_start']}) worst rms {format_value(far['value'])} "
            f"(d={far['bin']})")
    for note in report["consistency"].get("notes", []):
        lines.append(f"- NOTE: {note}")

    lines += ["", "## Ratio test/noise by distance to the seam (B1 vs A1, noise A1 vs A2)", ""]
    header = ["d", "n"]
    for field in PRIMARY_FIELDS:
        header += [f"{field} rms", f"{field} p99", f"{field} max"]
    rows = []
    for label in bin_labels():
        statistics_by_field = {field: primary["fields"][field]["bins"][label]
                               for field in PRIMARY_FIELDS}
        row = [label, format_value(statistics_by_field["acceleration"]["count"])]
        for field in PRIMARY_FIELDS:
            ratio = statistics_by_field[field]["ratio"]
            row += [format_value(ratio["rms"]), format_value(ratio["p99"]),
                    format_value(ratio["max"])]
        rows.append(row)
    lines += markdown_table(header, rows)

    lines += ["", "Absolute rms of the acceleration difference (test / noise):", ""]
    rows = []
    for label in bin_labels():
        statistics = primary["fields"]["acceleration"]["bins"][label]
        rows.append([label, format_value(statistics["test"]["rms"]),
                     format_value(statistics["noise"]["rms"])])
    lines += markdown_table(["d", "rms |B1-A1|", "rms |A1-A2|"], rows)

    lines += ["", "## Column 0 split (crossing particle within h in the last step)", "",
              "Ratios test/noise. flagged = another particle that crossed a seam in the last "
              "step lies closer than h; crossed_self = the particle itself crossed; "
              "flagged_departed = a crossing neighbour left this particle's slab.", ""]
    header = ["category", "n"]
    for field in PRIMARY_FIELDS:
        header += [f"{field} rms", f"{field} p99"]
    header += ["kernel_sum rms", "kernel_sum B1-A1 mean"]
    rows = []
    kernel_sum_section = report["column0_kernel_sum"]
    for name in COLUMN0_CATEGORIES:
        row = [name, format_value(primary["column0_counts"][name])]
        for field in PRIMARY_FIELDS:
            ratio = primary["fields"][field]["column0"][name]["ratio"]
            row += [format_value(ratio["rms"]), format_value(ratio["p99"])]
        row.append(format_value(primary["fields"]["kernel_sum"]["column0"][name]["ratio"]["rms"]))
        difference = kernel_sum_section.get(name, {}).get("difference_B1_minus_A1", {})
        row.append(format_value(difference.get("mean")) if difference else "-")
        rows.append(row)
    lines += markdown_table(header, rows)

    lines += ["", "## Column 0 kernel_sum (B1 vs matched A1)", ""]
    header = ["category", "run", "n", "mean", "std"] + [f"p{quantile}" for quantile
                                                        in KERNEL_SUM_QUANTILES]
    rows = []
    for name in KERNEL_SUM_CATEGORIES:
        for run_key in ("test_B1", "reference_A1", "difference_B1_minus_A1"):
            values = kernel_sum_section[name][run_key]
            rows.append([name, run_key, format_value(values["count"]),
                         format_value(values["mean"], 6), format_value(values["std"], 3)]
                        + [format_value(values[f"p{quantile}"], 6)
                           for quantile in KERNEL_SUM_QUANTILES])
    lines += markdown_table(header, rows)

    for key, title in (("secondary", "Second pair: B2 vs A2 (noise A2 vs A1)"),
                       ("test_noise", "Test-config noise: |B1 - B2| vs |A1 - A2|")):
        section = report["comparisons"].get(key)
        if not section:
            continue
        lines += ["", f"## {title}", "",
                  f"{format_value(section['particles'])} particles; rms ratios:", ""]
        header = ["d"] + [f"{field} rms" for field in PRIMARY_FIELDS]
        rows = []
        for label in bin_labels():
            rows.append([label] + [format_value(section["fields"][field]["bins"][label]
                                                ["ratio"]["rms"]) for field in PRIMARY_FIELDS])
        for name in ("flagged", "unflagged"):
            rows.append([f"col0 {name}"] + [format_value(
                section["fields"][field]["column0"][name]["ratio"]["rms"])
                for field in PRIMARY_FIELDS])
        lines += markdown_table(header, rows)

    lines += ["", "## Matching", ""]
    rows = []
    for key, summary in report["matching"].items():
        if not isinstance(summary, dict):
            continue
        kdtree = summary["kdtree"]
        identity = summary["id"]
        rows.append([key, format_value(summary["queried"]), format_value(kdtree["matched"]),
                     format_value(kdtree["unmatched"]),
                     format_value(kdtree["id_agreement_rate"], 6),
                     format_value(kdtree["many_to_one"]),
                     format_value(identity["unmatched"]),
                     format_value(identity["position_offset_over_h"]["max"])])
    lines += markdown_table(["pair", "queried", "kd matched", "kd unmatched",
                             "id agreement", "many-to-one", "id unmatched",
                             "id offset max (h)"], rows)
    for label in ("B1", "B2"):
        if label in report["consistency"]:
            check = report["consistency"][label]
            lines.append("")
            lines.append(f"- {label}: column-0 particles {format_value(check['column0_particles'])}, "
                         f"crossing particles {format_value(check['crossing_particles'])}, "
                         f"seam side / owner disagreements "
                         f"{format_value(check['seam_side_owner_disagreements'])}")

    lines += ["", "## Input runs", ""]
    rows = []
    for label, run in runs.items():
        invariants = run.get("invariants") or {}
        overflow_total = invariants.get("overflow_total")
        rows.append([label, f"`{pathlib.Path(run['path']).name}`", str(run.get("version")),
                     str(run.get("slabs")), str(run.get("horizon")),
                     format_value(run.get("particles")),
                     format_value(run.get("crossed_last_step")),
                     str(invariants.get("drift")), str(invariants.get("missing_ids")),
                     str(invariants.get("duplicate_ids")),
                     f"{invariants.get('stamp_errors_gpu')}/{invariants.get('stamp_errors_host')}",
                     str(overflow_total), str(invariants.get("valid")),
                     ", ".join(invariants.get("warnings") or []) or "-"])
    lines += markdown_table(["run", "file", "version", "K", "N", "particles", "crossed",
                             "drift", "missing", "duplicates", "stamps gpu/host",
                             "overflow", "valid", "warnings"], rows)
    if report.get("figures"):
        lines += ["", "## Figures", ""]
        lines += [f"![{pathlib.Path(path).stem}]({pathlib.Path(path).name})"
                  for path in report["figures"]]
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Figures
# =============================================================================

def _series(values) -> np.ndarray:
    """None / non-positive / inf -> NaN so log axes skip them."""
    array = np.array([np.nan if value is None else float(value) for value in values],
                     dtype=np.float64)
    array[~np.isfinite(array) | (array <= 0)] = np.nan
    return array


def _style_axes(axes) -> None:
    axes.grid(True, which="major", color="#d8d7d2", linewidth=0.6, alpha=0.8)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)


def _log_scale_if_positive(axes, *series_list) -> None:
    """Log y only when some plotted value is finite and positive."""
    if any(np.isfinite(series).any() for series in series_list):
        axes.set_yscale("log")


def write_figures(report: dict, dumps: dict, triplets: Triplets, raw: dict,
                  out_directory: pathlib.Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    primary = report["comparisons"]["primary"]
    secondary = report["comparisons"].get("secondary")
    test_run = report["runs"]["B1"]
    title_prefix = (f"{test_run.get('case_name')}: {test_run.get('run_name')} vs "
                    f"{report['runs']['A1'].get('run_name')}, N={test_run.get('horizon')}")
    labels = bin_labels()
    positions = np.array(list(range(DISTANCE_BIN_COUNT)) + [DISTANCE_BIN_COUNT + 1.5])
    tick_labels = [str(value) for value in range(DISTANCE_BIN_COUNT)] + ["int."]
    written = []

    # --- 1. difference and ratio by unsigned distance --------------------------
    figure, axes_grid = pyplot.subplots(len(PRIMARY_FIELDS), 2, figsize=(12, 10),
                                        squeeze=False)
    for row_index, field in enumerate(PRIMARY_FIELDS):
        bins = primary["fields"][field]["bins"]
        difference_axes, ratio_axes = axes_grid[row_index]
        plotted_differences = []
        for statistic_key, line_style in (("rms", "-"), ("p99", "--")):
            for side, color, legend in (("test", TEST_COLOR, "test |B1-A1|"),
                                        ("noise", NOISE_COLOR, "noise |A1-A2|")):
                series = _series([bins[label][side][statistic_key] for label in labels])
                plotted_differences.append(series)
                difference_axes.plot(positions, series, line_style, color=color, linewidth=2,
                                     marker="o", markersize=4,
                                     label=f"{legend} {statistic_key}")
        _log_scale_if_positive(difference_axes, *plotted_differences)
        difference_axes.set_ylabel(f"|difference| of {field}")
        difference_axes.set_title(f"{field}: test vs noise floor", fontsize=10)
        plotted_ratios = []
        for metric, line_style, marker in (("rms", "-", "o"), ("p99", "--", "s"),
                                           ("max", ":", "^")):
            series = _series([bins[label]["ratio"][metric] for label in labels])
            plotted_ratios.append(series)
            ratio_axes.plot(positions, series, line_style, color=TEST_COLOR, linewidth=2,
                            marker=marker, markersize=4, label=f"{metric} ratio (B1 vs A1)")
        if secondary:
            second_bins = secondary["fields"][field]["bins"]
            series = _series([second_bins[label]["ratio"]["rms"] for label in labels])
            plotted_ratios.append(series)
            ratio_axes.plot(positions, series, "-", color=REPLICATE_COLOR, linewidth=1.5,
                            marker="o", markersize=3, label="rms ratio (B2 vs A2)")
        ratio_axes.axhline(1.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
        _log_scale_if_positive(ratio_axes, *plotted_ratios)
        ratio_axes.set_ylabel("test / noise")
        ratio_axes.set_title(f"{field}: ratio", fontsize=10)
        for axes in (difference_axes, ratio_axes):
            _style_axes(axes)
            axes.set_xticks(positions)
            axes.set_xticklabels(tick_labels, fontsize=8)
        if row_index == 0:
            difference_axes.legend(fontsize=7, loc="best")
            ratio_axes.legend(fontsize=7, loc="best")
        if row_index == len(PRIMARY_FIELDS) - 1:
            for axes in (difference_axes, ratio_axes):
                axes.set_xlabel("voxel-column distance to the nearest seam "
                                "(0 = seam column, int. = d >= 16)")
    figure.suptitle(title_prefix, fontsize=11)
    figure.tight_layout()
    path = out_directory / "difference_by_column.png"
    figure.savefig(path, dpi=140)
    pyplot.close(figure)
    written.append(path)

    # --- 2. shift profile across the seam -------------------------------------
    profile = report["shift_profile"]
    signed_columns = np.array(profile["signed_columns"])
    figure, (magnitude_axes, difference_axes) = pyplot.subplots(2, 1, figsize=(11, 7),
                                                                sharex=True)
    magnitude_axes.plot(signed_columns, _series(profile["mean_shift_B1"]), "-o",
                        color=TEST_COLOR, linewidth=2, markersize=4, label="B1 (test)")
    magnitude_axes.plot(signed_columns, _series(profile["mean_shift_A1"]), "-o",
                        color=NOISE_COLOR, linewidth=2, markersize=4, label="A1 (reference)")
    magnitude_axes.plot(signed_columns, _series(profile["mean_shift_A2"]), "-o",
                        color=REPLICATE_COLOR, linewidth=1.5, markersize=3,
                        label="A2 (reference repeat)")
    magnitude_axes.set_ylabel("mean |shift| (m)")
    magnitude_axes.legend(fontsize=8)
    test_profile = _series(profile["mean_difference_B1_A1"])
    noise_profile = _series(profile["mean_difference_A1_A2"])
    difference_axes.plot(signed_columns, test_profile, "-o", color=TEST_COLOR, linewidth=2,
                         markersize=4, label="mean |B1 - A1|")
    difference_axes.plot(signed_columns, noise_profile, "-o", color=NOISE_COLOR, linewidth=2,
                         markersize=4, label="mean |A1 - A2|")
    _log_scale_if_positive(difference_axes, test_profile, noise_profile)
    difference_axes.set_ylabel("mean |shift difference| (m)")
    difference_axes.set_xlabel("signed column from the nearest seam (-1 = left slab "
                               "column 0, 0 = right slab column 0)")
    difference_axes.legend(fontsize=8)
    for axes in (magnitude_axes, difference_axes):
        _style_axes(axes)
        axes.axvline(-0.5, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
    magnitude_axes.set_title(f"{title_prefix}: particle shifting across the seam",
                             fontsize=10)
    figure.tight_layout()
    path = out_directory / "shift_profile.png"
    figure.savefig(path, dpi=140)
    pyplot.close(figure)
    written.append(path)

    # --- 3. column-0 kernel_sum distributions ---------------------------------
    # flagged is split into departed (a crossing neighbour LEFT this slab: it is
    # missing from the sender's lists, so B1's kernel_sum should sit lower) and
    # arrived-only; mixing them dilutes the missing-neighbour signal.
    test_kernel_sum = dumps["B1"].arrays["kernel_sum"][triplets.test].astype(np.float64)
    reference_kernel_sum = dumps["A1"].arrays["kernel_sum"][triplets.reference].astype(np.float64)
    masks = raw["column0_masks"]
    figure, axes_grid = pyplot.subplots(2, 2, figsize=(12, 8.4))
    for axes, name in zip(axes_grid.flat[:3], KERNEL_SUM_FIGURE_CATEGORIES):
        mask = masks[name]
        if mask.sum() < 2:
            axes.text(0.5, 0.5, f"no {name} column-0 particles", ha="center",
                      va="center", transform=axes.transAxes)
        else:
            values = np.concatenate([test_kernel_sum[mask], reference_kernel_sum[mask]])
            edges = np.histogram_bin_edges(values, bins=60)
            axes.hist(reference_kernel_sum[mask], bins=edges, histtype="step", linewidth=2,
                      color=NOISE_COLOR, label="A1 (reference)", density=True)
            axes.hist(test_kernel_sum[mask], bins=edges, histtype="step", linewidth=2,
                      color=TEST_COLOR, label="B1 (test)", density=True)
            axes.legend(fontsize=8)
        axes.set_title(f"column 0, {name} (n={int(mask.sum()):,})", fontsize=10)
        axes.set_xlabel("kernel_sum")
        _style_axes(axes)
    difference_axes = axes_grid.flat[3]
    plotted = False
    differences = {name: test_kernel_sum[masks[name]] - reference_kernel_sum[masks[name]]
                   for name in KERNEL_SUM_FIGURE_CATEGORIES}
    group_colors = {"flagged_departed": TEST_COLOR, "flagged_arrived_only": REPLICATE_COLOR,
                    "unflagged": NOISE_COLOR}
    combined = np.concatenate(list(differences.values()))
    if combined.size >= 2:
        edges = np.histogram_bin_edges(combined, bins=60)
        for name in KERNEL_SUM_FIGURE_CATEGORIES:
            if differences[name].size >= 2:
                difference_axes.hist(differences[name], bins=edges, histtype="step",
                                     linewidth=2, color=group_colors[name], density=True,
                                     label=f"{name} (n={differences[name].size:,})")
                plotted = True
    if plotted:
        difference_axes.legend(fontsize=8)
    else:
        difference_axes.text(0.5, 0.5, "no column-0 particles", ha="center", va="center",
                             transform=difference_axes.transAxes)
    difference_axes.axvline(0.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
    difference_axes.set_title("column 0: kernel_sum B1 - A1 by group", fontsize=10)
    difference_axes.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    difference_axes.set_xlabel("kernel_sum difference")
    _style_axes(difference_axes)
    figure.suptitle(f"{title_prefix}: kernel_sum in the seam column", fontsize=11)
    figure.tight_layout()
    path = out_directory / "kernel_sum_column0.png"
    figure.savefig(path, dpi=140)
    pyplot.close(figure)
    written.append(path)
    return written


# =============================================================================
# Self-test on synthetic dumps
# =============================================================================

SYNTHETIC_NOISE_SCALE = {"acceleration": 1e-3, "shift": 1e-7, "velocity": 1e-5,
                         "density": 1e-4, "pressure": 1e-2, "kernel_sum": 1e-6}
# Perturbation of the test runs on column-0 fluid particles, in multiples of the
# noise scale. A particle with a crossing neighbour that LEFT its slab gets the
# largest one and a kernel_sum that is always LOWER (the departed neighbour is
# missing from the sum), so a swapped or mis-keyed departed/arrived split shows
# in the ratios and in the signed kernel_sum mean, not only in the counts.
SYNTHETIC_DEPARTED_PERTURBATION = 100.0
SYNTHETIC_ARRIVED_ONLY_PERTURBATION = 30.0
SYNTHETIC_CROSSED_PERTURBATION = 8.0
SYNTHETIC_UNFLAGGED_PERTURBATION = 4.0
SYNTHETIC_RUN_NAMES = ("A1", "A2", "B1", "B2")
# Displaced particles (one run per matching leg: B1 -> A1 and A1 -> A2) move
# 0.1 h diagonally, twice the 0.05 h match tolerance. The lattice (spacing
# 0.2 h, jitter <= 0.02 h per axis) keeps every other particle >= 0.09 h away,
# so they are unmatched, never matched to a wrong particle; the generator
# asserts this for every pair of runs the analyzer matches.
SYNTHETIC_DISPLACED_RUNS = ("B1", "A2")
SYNTHETIC_DISPLACEMENT_OVER_H = 0.1
SYNTHETIC_MATCHED_PAIRS = (("B1", "A1"), ("A1", "A2"), ("B2", "A2"), ("B1", "B2"))
BRUTE_FORCE_RELATIVE_TOLERANCE = 1e-9


def _write_synthetic_dump(directory: pathlib.Path, name: str, arrays: dict,
                          metadata: dict) -> pathlib.Path:
    npz_path = directory / f"{name}.npz"
    np.savez(npz_path, **arrays)
    document = dict(metadata)
    document["run_name"] = name
    (directory / f"{name}.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    return npz_path


def _global_columns(positions: np.ndarray, origin_x: float,
                    smoothing_length: float) -> np.ndarray:
    return np.floor((np.asarray(positions[:, 0], dtype=np.float64) - origin_x)
                    / smoothing_length).astype(np.int64)


def _brute_force_seams(column: np.ndarray, cuts: list) -> tuple:
    """(owner slab, distance to the nearest seam, signed offset c - s to that
    seam) by explicit loops over the cuts; slab i owns [cut_(i-1), cut_i).
    Independent of classify_columns: used by the synthetic generator and by
    the brute-force check of the analyzer."""
    slab = np.zeros(column.size, dtype=np.int64)
    distance = np.full(column.size, np.iinfo(np.int64).max, dtype=np.int64)
    signed = np.zeros(column.size, dtype=np.int64)
    for cut in cuts:
        slab += column >= cut
        cut_distance = np.where(column >= cut, column - cut, cut - 1 - column)
        closer = cut_distance < distance
        distance[closer] = cut_distance[closer]
        signed[closer] = column[closer] - cut
    return slab, distance, signed


def _brute_force_flagged(positions: np.ndarray, column0: np.ndarray, crossed: np.ndarray,
                         slab: np.ndarray, previous_slab: np.ndarray,
                         smoothing_length: float) -> tuple:
    """(flagged, departed) over all particles by explicit pair distances:
    flagged = ANOTHER crossing particle lies strictly closer than h; departed =
    one of those came from this particle's slab (its previous_slab is it)."""
    flagged = np.zeros(column0.size, dtype=bool)
    departed = np.zeros(column0.size, dtype=bool)
    column0_indices = np.flatnonzero(column0)
    crossing_indices = np.flatnonzero(crossed)
    if column0_indices.size == 0 or crossing_indices.size == 0:
        return flagged, departed
    points = np.asarray(positions, dtype=np.float64)
    separation = np.linalg.norm(points[column0_indices][:, None, :]
                                - points[crossing_indices][None, :, :], axis=2)
    close = ((separation < smoothing_length)
             & (column0_indices[:, None] != crossing_indices[None, :]))
    from_this_slab = (np.asarray(previous_slab, dtype=np.int64)[crossing_indices][None, :]
                      == np.asarray(slab, dtype=np.int64)[column0_indices][:, None])
    flagged[column0_indices[close.any(axis=1)]] = True
    departed[column0_indices[(close & from_this_slab).any(axis=1)]] = True
    return flagged, departed


def _assert_displaced_isolated(run_positions: dict, displaced: dict,
                               tolerance: float) -> None:
    """For every pair of runs the analyzer matches, no particle of one run lies
    within the match tolerance of a displaced particle of the other, in either
    direction, so nearest-neighbour matching leaves a displaced particle
    unmatched instead of pairing it with a wrong one."""
    for first_run, second_run in SYNTHETIC_MATCHED_PAIRS:
        for moved_run, fixed_run in ((first_run, second_run), (second_run, first_run)):
            particle_indices = displaced.get(moved_run)
            if particle_indices is None:
                continue
            for source_run, target_run in ((moved_run, fixed_run), (fixed_run, moved_run)):
                points = run_positions[source_run][particle_indices].astype(np.float64)
                targets = run_positions[target_run].astype(np.float64)
                nearest = float(np.min(np.linalg.norm(targets[None, :, :]
                                                      - points[:, None, :], axis=2)))
                if nearest <= tolerance:
                    raise AssertionError(
                        f"synthetic: a particle displaced in {moved_run} lies "
                        f"{nearest:.3g} m from a {target_run} particle (tolerance "
                        f"{tolerance:.3g} m); change the random seed")


def build_synthetic_dumps(directory: pathlib.Path, *, cut_count: int = 1,
                          drop_fraction: float = 0.0, displaced_count: int = 0,
                          random_seed: int = 20261002) -> dict:
    """Synthetic dumps on a jittered 2-D lattice (h = 0.05, spacing h/5, 41
    voxel columns). References A1/A2 (K=1) differ by tiny noise; tests B1/B2
    (``cut_count`` evenly spaced cuts) carry independent noise of the same size
    PLUS a deterministic perturbation on column-0 particles near fake crossing
    particles (see SYNTHETIC_*_PERTURBATION).

    Harder variants for the bookkeeping checks: ``drop_fraction`` removes a
    different random subset of particles from every run, so array rows no
    longer line up between runs, and ``displaced_count`` particles of B1 and of
    A2 move 0.1 h, beyond the 0.05 h match tolerance (the kd-tree leaves them
    unmatched, the id match keeps them)."""
    directory.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(random_seed)
    smoothing_length = 0.05
    spacing = smoothing_length / 5.0
    domain_length = 2.0
    lattice = np.arange(0.0, domain_length + 0.5 * spacing, spacing)
    grid_x, grid_y = np.meshgrid(lattice, lattice, indexing="ij")
    base_positions = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    base_positions += generator.uniform(-0.1, 0.1, size=base_positions.shape) * spacing
    particle_count = base_positions.shape[0]
    origin_x = -0.5 * smoothing_length
    grid_column_count = int(np.floor(domain_length / smoothing_length + 0.5)) + 1
    cuts = [grid_column_count * (cut_index + 1) // (cut_count + 1)
            for cut_index in range(cut_count)]
    material = np.where(base_positions[:, 1] < 0.5 * spacing, 1, 0).astype(np.uint16)
    fluid = material == 0

    base_fields = {
        "acceleration": (10.0 * np.stack([np.sin(3.0 * base_positions[:, 1]),
                                          np.cos(2.0 * base_positions[:, 0])], axis=1)
                         + generator.normal(0.0, 1.0, (particle_count, 2))),
        "shift": generator.normal(0.0, 1e-4, (particle_count, 2)),
        "velocity": generator.normal(0.0, 0.1, (particle_count, 2)),
        "density": 1000.0 + generator.normal(0.0, 0.5, particle_count),
        "pressure": 1.0e4 * generator.normal(0.0, 1e-3, particle_count),
        "kernel_sum": 1.0 + generator.normal(0.0, 0.01, particle_count),
    }
    for field in ("acceleration", "shift"):
        base_fields[field][~fluid] = 0.0   # walls: no force, no shifting

    # Positions per run as stored (float32), a few of them displaced in B1 and A2
    run_positions = {name: base_positions + generator.normal(0.0, 1e-6 * spacing,
                                                             base_positions.shape)
                     for name in SYNTHETIC_RUN_NAMES}
    displaced = {}
    for name in (SYNTHETIC_DISPLACED_RUNS if displaced_count else ()):
        chosen = np.sort(generator.choice(np.flatnonzero(fluid), size=displaced_count,
                                          replace=False))
        diagonal = generator.choice([-1.0, 1.0], size=(displaced_count, 2)) / math.sqrt(2.0)
        run_positions[name][chosen] += (SYNTHETIC_DISPLACEMENT_OVER_H * smoothing_length
                                        * diagonal)
        displaced[name] = chosen
    run_positions = {name: positions.astype(np.float32)
                     for name, positions in run_positions.items()}
    _assert_displaced_isolated(run_positions, displaced,
                               DEFAULT_TOLERANCE_FACTOR * smoothing_length)

    # Seams, fake crossings and column-0 groups of B1, as the analyzer sees B1
    test_slab, test_distance, test_signed = _brute_force_seams(
        _global_columns(run_positions["B1"], origin_x, smoothing_length), cuts)
    column0 = test_distance == 0
    column0_fluid = np.flatnonzero(column0 & fluid)
    crossing = generator.choice(column0_fluid, size=max(4, column0_fluid.size // 100),
                                replace=False)
    crossed = np.zeros(particle_count, dtype=bool)
    crossed[crossing] = True

    def previous_slabs(slab: np.ndarray, signed: np.ndarray) -> np.ndarray:
        """A crossing particle came over its nearest seam: from the left slab
        when it sits right of that seam, from the right slab otherwise."""
        previous = slab.copy()
        previous[crossing] = np.where(signed[crossing] >= 0, slab[crossing] - 1,
                                      slab[crossing] + 1)
        return previous

    flagged, departed = _brute_force_flagged(
        run_positions["B1"], column0, crossed, test_slab,
        previous_slabs(test_slab, test_signed), smoothing_length)
    perturbation_scale = np.zeros(particle_count)
    perturbation_scale[column0] = SYNTHETIC_UNFLAGGED_PERTURBATION
    perturbation_scale[column0 & crossed & ~flagged] = SYNTHETIC_CROSSED_PERTURBATION
    perturbation_scale[flagged] = SYNTHETIC_ARRIVED_ONLY_PERTURBATION
    perturbation_scale[departed] = SYNTHETIC_DEPARTED_PERTURBATION
    perturbation_scale[~fluid] = 0.0
    perturbation = {}
    for field, scale in SYNTHETIC_NOISE_SCALE.items():
        if field in VECTOR_FIELDS:
            direction = generator.normal(0.0, 1.0, (particle_count, 2))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            perturbation[field] = direction * (perturbation_scale * scale)[:, None]
        else:
            sign = generator.choice([-1.0, 1.0], size=particle_count)
            if field == "kernel_sum":
                sign[departed] = -1.0       # the departed neighbour is missing from the sum
            perturbation[field] = sign * perturbation_scale * scale

    drop_count = int(round(drop_fraction * particle_count))
    kept_rows = {}
    for name in SYNTHETIC_RUN_NAMES:
        keep = np.ones(particle_count, dtype=bool)
        if drop_count:
            keep[generator.choice(particle_count, size=drop_count, replace=False)] = False
        kept_rows[name] = np.flatnonzero(keep)

    common_metadata = {"version": "synthetic", "case": "synthetic", "case_name": "synthetic",
                       "smoothing_length": smoothing_length, "origin_x": origin_x,
                       "grid_nx": grid_column_count, "dimension": 2, "horizon": 1,
                       "material_kinds": [0, 1],
                       "invariants": {"drift": 0, "missing_ids": 0, "duplicate_ids": 0,
                                      "stamp_errors_gpu": 0, "stamp_errors_host": 0,
                                      "overflow_total": 0, "valid": True}}
    paths = {}
    for name in SYNTHETIC_RUN_NAMES:
        is_test = name.startswith("B")
        fields = {}
        for field, scale in SYNTHETIC_NOISE_SCALE.items():
            values = base_fields[field] + generator.normal(0.0, scale, base_fields[field].shape)
            if field in ("acceleration", "shift"):
                values[~fluid] = 0.0
            if is_test:
                values = values + perturbation[field]
            fields[field] = values.astype(np.float32)
        if is_test:
            slab, _, signed = _brute_force_seams(
                _global_columns(run_positions[name], origin_x, smoothing_length), cuts)
            previous_slab = previous_slabs(slab, signed)
        else:
            slab = previous_slab = np.zeros(particle_count, dtype=np.int64)
        arrays = {
            "id": np.arange(particle_count, dtype=np.uint32),
            "slab": slab.astype(np.uint8),
            "previous_slab": previous_slab.astype(np.uint8),
            "crossed_last_step": crossed if is_test else np.zeros(particle_count, dtype=bool),
            "position": run_positions[name],
            "material": material,
            **fields,
        }
        arrays = {key: values[kept_rows[name]] for key, values in arrays.items()}
        metadata = dict(common_metadata)
        metadata.update({"slabs": cut_count + 1 if is_test else 1,
                         "cuts": cuts if is_test else []})
        paths[name] = _write_synthetic_dump(directory, f"synthetic_{name}", arrays, metadata)
    description = {
        "particles": particle_count,
        "fluid_particles": int(fluid.sum()),
        "cuts": cuts,
        "crossing": int(crossing.size),
        "flagged_fluid": int((flagged & fluid).sum()),
        "departed_fluid": int((departed & fluid).sum()),
        "dropped_per_run": drop_count,
        "displaced": {name: int(indices.size) for name, indices in displaced.items()},
    }
    return {"paths": paths, "description": description}


def _difference_magnitude(first_values: np.ndarray, second_values: np.ndarray) -> np.ndarray:
    difference = first_values.astype(np.float64) - second_values.astype(np.float64)
    if difference.ndim == 2:
        return np.sqrt(np.sum(difference * difference, axis=1))
    return np.abs(difference)


def _brute_force_rms_ratio(test_values: np.ndarray, noise_values: np.ndarray):
    """rms(test) / rms(noise) with safe_ratio's conventions (None when empty or
    both zero, inf when only the noise is zero)."""
    if test_values.size == 0:
        return None
    test_rms = math.sqrt(float(np.mean(test_values * test_values)))
    noise_rms = math.sqrt(float(np.mean(noise_values * noise_values)))
    if noise_rms > 0:
        return test_rms / noise_rms
    return None if test_rms == 0 else math.inf


def brute_force_primary(paths: dict, method: str,
                        tolerance_factor: float = DEFAULT_TOLERANCE_FACTOR) -> dict:
    """Recompute the analyzer's primary comparison (B1 vs A1, noise A1 vs A2,
    fluid particles) straight from the dump files with independent code:
    particles aligned through id -> row tables instead of the matcher's index
    plumbing, seams by explicit loops over the cuts, the column-0 groups by
    explicit pair distances. The 'kdtree' match is emulated as "same id and
    within the tolerance", which equals the nearest-neighbour match only
    because build_synthetic_dumps keeps every other particle farther away."""
    runs = {}
    for name in ("A1", "A2", "B1"):
        with np.load(paths[name], allow_pickle=False) as archive:
            runs[name] = {key: archive[key] for key in archive.files}
    metadata = json.loads(pathlib.Path(paths["B1"]).with_suffix(".json")
                          .read_text(encoding="utf-8"))
    smoothing_length = float(metadata["smoothing_length"])
    cuts = [int(cut) for cut in metadata["cuts"]]
    tolerance = tolerance_factor * smoothing_length
    fluid_groups = [group for group, kind in enumerate(metadata["material_kinds"])
                    if int(kind) == FLUID_KIND]
    id_limit = 1 + max(int(run["id"].max()) for run in runs.values())

    def fluid_rows(run) -> np.ndarray:
        return np.flatnonzero(np.isin(run["material"], fluid_groups))

    def match(source, source_rows, target) -> tuple:
        """Matched target row of every source row (-1 = none), and which
        source rows have their id in the target but beyond the tolerance."""
        row_by_id = np.full(id_limit, -1, dtype=np.int64)
        target_fluid_rows = fluid_rows(target)
        row_by_id[target["id"][target_fluid_rows].astype(np.int64)] = target_fluid_rows
        target_rows = row_by_id[source["id"][source_rows].astype(np.int64)]
        present = target_rows >= 0
        offset = np.full(source_rows.size, np.inf)
        offset[present] = _difference_magnitude(source["position"][source_rows[present]],
                                                target["position"][target_rows[present]])
        matched = present & (offset <= tolerance) if method == "kdtree" else present
        return np.where(matched, target_rows, -1), present & (offset > tolerance)

    test, reference, repeat = runs["B1"], runs["A1"], runs["A2"]
    test_rows = fluid_rows(test)
    reference_of_test, beyond_tolerance = match(test, test_rows, reference)
    first_matched = reference_of_test >= 0
    reference_rows = reference_of_test[first_matched]
    repeat_of_reference, _ = match(reference, reference_rows, repeat)
    second_matched = repeat_of_reference >= 0
    repeat_of_every_reference, _ = match(reference, fluid_rows(reference), repeat)
    triplet_test = test_rows[first_matched][second_matched]
    triplet_reference = reference_rows[second_matched]
    triplet_repeat = repeat_of_reference[second_matched]

    columns = _global_columns(test["position"], float(metadata["origin_x"]), smoothing_length)
    _, distance, _ = _brute_force_seams(columns, cuts)
    column0 = distance == 0
    crossed = test["crossed_last_step"].astype(bool)
    flagged, departed = _brute_force_flagged(test["position"], column0, crossed, test["slab"],
                                             test["previous_slab"], smoothing_length)
    groups = {
        "all": column0,
        "flagged": flagged,
        "unflagged": column0 & ~flagged,
        "crossed_self": column0 & crossed,
        "unflagged_not_crossed": column0 & ~flagged & ~crossed,
        "flagged_departed": departed,
        "flagged_arrived_only": flagged & ~departed,
        "flagged_including_self": flagged | (column0 & crossed),
    }
    triplet_distance = distance[triplet_test]
    bins = {str(value): triplet_distance == value for value in range(DISTANCE_BIN_COUNT)}
    bins[INTERIOR_BIN] = triplet_distance >= DISTANCE_BIN_COUNT
    group_masks = {name: mask[triplet_test] for name, mask in groups.items()}
    triplet_columns = columns[triplet_test]

    bin_ratio, group_ratio = {}, {}
    for field in PRIMARY_FIELDS:
        test_values = _difference_magnitude(test[field][triplet_test],
                                            reference[field][triplet_reference])
        noise_values = _difference_magnitude(reference[field][triplet_reference],
                                             repeat[field][triplet_repeat])
        bin_ratio[field] = {label: _brute_force_rms_ratio(test_values[mask], noise_values[mask])
                            for label, mask in bins.items()}
        group_ratio[field] = {name: _brute_force_rms_ratio(test_values[mask],
                                                           noise_values[mask])
                              for name, mask in group_masks.items()}
    return {
        "triplets": int(triplet_test.size),
        "unmatched_test_to_reference": int((~first_matched).sum()),
        "unmatched_reference_to_repeat": int((~second_matched).sum()),
        "unmatched_every_reference": int((repeat_of_every_reference < 0).sum()),
        "offset_beyond_tolerance": int(beyond_tolerance.sum()),
        "column0_per_cut": [int(((triplet_columns == cut) | (triplet_columns == cut - 1)).sum())
                            for cut in cuts],
        "bin_counts": {label: int(mask.sum()) for label, mask in bins.items()},
        "column0_counts": {name: int(mask.sum()) for name, mask in group_masks.items()},
        "bin_rms_ratio": bin_ratio,
        "column0_rms_ratio": group_ratio,
    }


def _same_ratio(value, expected) -> bool:
    if value is None or expected is None:
        return value is None and expected is None
    if math.isinf(value) or math.isinf(expected):
        return value == expected
    return abs(value - expected) <= BRUTE_FORCE_RELATIVE_TOLERANCE * abs(expected)


def brute_force_mismatches(report: dict, expected: dict) -> list:
    """Every disagreement between the report's primary comparison and
    brute_force_primary(), as short strings (empty: identical)."""
    primary = report["comparisons"]["primary"]
    problems = []
    for key, analyzer_key in (("triplets", "particles"),
                              ("unmatched_test_to_reference", "unmatched_test_to_reference"),
                              ("unmatched_reference_to_repeat", "unmatched_reference_to_repeat")):
        if primary[analyzer_key] != expected[key]:
            problems.append(f"{key} {primary[analyzer_key]} != {expected[key]}")
    for label, count in expected["bin_counts"].items():
        analyzer_count = primary["fields"]["acceleration"]["bins"][label]["count"]
        if analyzer_count != count:
            problems.append(f"bin d={label} count {analyzer_count} != {count}")
    for name, count in expected["column0_counts"].items():
        if primary["column0_counts"][name] != count:
            problems.append(f"column-0 {name} {primary['column0_counts'][name]} != {count}")
    for field in PRIMARY_FIELDS:
        for label, ratio in expected["bin_rms_ratio"][field].items():
            value = primary["fields"][field]["bins"][label]["ratio"]["rms"]
            if not _same_ratio(value, ratio):
                problems.append(f"{field} d={label} rms ratio {value} != {ratio}")
        for name, ratio in expected["column0_rms_ratio"][field].items():
            value = primary["fields"][field]["column0"][name]["ratio"]["rms"]
            if not _same_ratio(value, ratio):
                problems.append(f"{field} column-0 {name} rms ratio {value} != {ratio}")
    return problems


def _mismatch_detail(problems: list) -> str:
    if not problems:
        return "identical"
    return "; ".join(problems[:4]) + (f"; ... {len(problems)} in all" if len(problems) > 4 else "")


def _self_test_single_seam(directory: pathlib.Path) -> list:
    """K=2, one seam, every particle in every run: the expected contrasts."""
    synthetic = build_synthetic_dumps(directory)
    paths, description = synthetic["paths"], synthetic["description"]
    print(f"[seam_audit self-test] single seam: {description}")
    references, tests = [paths["A1"], paths["A2"]], [paths["B1"], paths["B2"]]
    report = analyze(references, tests, match="kdtree", out_dir=directory / "analysis_kdtree")
    report_by_id = analyze(references, tests, match="id", out_dir=directory / "analysis_id",
                           make_figures=False)
    expected = brute_force_primary(paths, "kdtree")
    primary = report["comparisons"]["primary"]
    matching = report["matching"]
    test_noise = report["comparisons"]["test_noise"]

    def column0_ratio(source, field, name):
        return source["comparisons"]["primary"]["fields"][field]["column0"][name]["ratio"]["rms"]

    far_ratios = {field: [statistics["ratio"]["rms"]
                          for label, statistics in primary["fields"][field]["bins"].items()
                          if is_far_bin(label, DEFAULT_FAR_BIN_START)
                          and statistics["count"] > 0]
                  for field in PRIMARY_FIELDS}
    group_problems = [name for name, count in expected["column0_counts"].items()
                      if primary["column0_counts"][name] != count]
    primary_problems = brute_force_mismatches(report, expected)
    kernel_sum_section = report["column0_kernel_sum"]
    departed_mean = (kernel_sum_section.get("flagged_departed", {})
                     .get("difference_B1_minus_A1", {}).get("mean"))
    arrived_only_mean = (kernel_sum_section.get("flagged_arrived_only", {})
                         .get("difference_B1_minus_A1", {}).get("mean"))
    departed_shift = SYNTHETIC_DEPARTED_PERTURBATION * SYNTHETIC_NOISE_SCALE["kernel_sum"]
    analysis_directory = directory / "analysis_kdtree"
    figure_names = ("difference_by_column.png", "shift_profile.png", "kernel_sum_column0.png")
    report_markdown = (analysis_directory / "report.md").read_text(encoding="utf-8")
    return [
        ("kd-tree matched every particle, ids agree",
         matching["B1_to_A1"]["kdtree"]["unmatched"] == 0
         and matching["A1_to_A2"]["kdtree"]["unmatched"] == 0
         and matching["B1_to_A1"]["kdtree"]["id_agreement_rate"] == 1.0
         and matching["A1_to_A2"]["kdtree"]["id_agreement_rate"] == 1.0,
         f"unmatched {matching['B1_to_A1']['kdtree']['unmatched']}/"
         f"{matching['A1_to_A2']['kdtree']['unmatched']}, agreement "
         f"{matching['B1_to_A1']['kdtree']['id_agreement_rate']}"),
        ("fluid filter + triplets cover every fluid particle",
         primary["particles"] == description["fluid_particles"],
         f"{primary['particles']} vs {description['fluid_particles']}"),
        ("column-0 groups match brute force, incl. the departed / arrived-only split",
         not group_problems and expected["column0_counts"]["flagged_departed"] > 0
         and expected["column0_counts"]["flagged_arrived_only"] > 0,
         ", ".join(f"{name} {primary['column0_counts'][name]}"
                   + (f" (brute force {expected['column0_counts'][name]})"
                      if name in group_problems else "")
                   for name in ("flagged", "unflagged", "flagged_departed",
                                "flagged_arrived_only", "crossed_self"))),
        ("primary comparison (triplets, bins, groups, rms ratios) equals brute force",
         not primary_problems, _mismatch_detail(primary_problems)),
        ("flagged column-0 rms ratio >> 1 (acceleration, shift, density)",
         all(column0_ratio(report, field, "flagged") > 20.0 for field in PRIMARY_FIELDS),
         ", ".join(f"{field} {column0_ratio(report, field, 'flagged'):.1f}"
                   for field in PRIMARY_FIELDS)),
        ("departed rms ratio > 2x arrived-only (acceleration, shift, density)",
         all(column0_ratio(report, field, "flagged_departed")
             > 2.0 * column0_ratio(report, field, "flagged_arrived_only")
             for field in PRIMARY_FIELDS),
         ", ".join(f"{field} {column0_ratio(report, field, 'flagged_departed'):.1f} vs "
                   f"{column0_ratio(report, field, 'flagged_arrived_only'):.1f}"
                   for field in PRIMARY_FIELDS)),
        ("unflagged column-0 rms ratio moderate (1.5 .. 4)",
         all(1.5 < column0_ratio(report, field, "unflagged") < 4.0
             for field in PRIMARY_FIELDS),
         ", ".join(f"{field} {column0_ratio(report, field, 'unflagged'):.2f}"
                   for field in PRIMARY_FIELDS)),
        ("far bins (d >= 8 and interior) rms ratio ~ 1 (0.8 .. 1.25)",
         all(far_ratios[field] and all(0.8 < value < 1.25 for value in far_ratios[field])
             for field in PRIMARY_FIELDS),
         ", ".join(f"{field} [{min(far_ratios[field]):.3f}, {max(far_ratios[field]):.3f}]"
                   for field in PRIMARY_FIELDS)),
        ("seam_excess > 1.5",
         report["verdict"]["seam_excess"] > 1.5,
         f"{report['verdict']['seam_excess']:.2f}"),
        ("kernel_sum B1 - A1 mean: lowered for departed, ~0 for arrived-only",
         departed_mean is not None and arrived_only_mean is not None
         and departed_mean < -0.5 * departed_shift
         and abs(arrived_only_mean) < 0.2 * abs(departed_mean),
         f"departed {format_value(departed_mean)}, arrived only "
         f"{format_value(arrived_only_mean)} (injected {-departed_shift:.3g})"),
        ("id matching reproduces the kd-tree ratios",
         all(abs(column0_ratio(report_by_id, field, "flagged")
                 / column0_ratio(report, field, "flagged") - 1.0) < 1e-9
             for field in PRIMARY_FIELDS),
         ", ".join(f"{field} {column0_ratio(report_by_id, field, 'flagged'):.3f}"
                   for field in PRIMARY_FIELDS)),
        ("second pair (B2 vs A2) flagged ratio >> 1",
         all(report["comparisons"]["secondary"]["fields"][field]["column0"]["flagged"]
             ["ratio"]["rms"] > 20.0 for field in PRIMARY_FIELDS), ""),
        ("test-config noise |B1-B2| ~ |A1-A2| everywhere (perturbation is deterministic)",
         all(0.8 < test_noise["fields"][field]["column0"]["flagged"]["ratio"]["rms"] < 1.25
             for field in PRIMARY_FIELDS),
         ", ".join(f"{field} {test_noise['fields'][field]['column0']['flagged']['ratio']['rms']:.3f}"
                   for field in PRIMARY_FIELDS)),
        ("seam side matches the owner slab",
         report["consistency"]["B1"]["seam_side_owner_disagreements"] == 0, ""),
        ("report.json, report.md with a Figures section linking the three figures",
         all((analysis_directory / name).exists()
             for name in ("report.json", "report.md") + figure_names)
         and "## Figures" in report_markdown
         and all(f"]({name})" in report_markdown for name in figure_names), ""),
    ]


def _self_test_three_seams(directory: pathlib.Path) -> list:
    """K=4, three seams; every run misses a different 1 % of the particles and
    25 particles of B1 and of A2 sit beyond the match tolerance: the analyzer's
    bookkeeping against brute_force_primary(), for both match methods."""
    synthetic = build_synthetic_dumps(directory, cut_count=3, drop_fraction=0.01,
                                      displaced_count=25, random_seed=20261003)
    paths, description = synthetic["paths"], synthetic["description"]
    print(f"[seam_audit self-test] three seams: {description}")
    references, tests = [paths["A1"], paths["A2"]], [paths["B1"], paths["B2"]]
    report = analyze(references, tests, match="kdtree", out_dir=directory / "analysis_kdtree",
                     make_figures=False)
    report_by_id = analyze(references, tests, match="id", out_dir=directory / "analysis_id",
                           make_figures=False)
    expected = brute_force_primary(paths, "kdtree")
    expected_by_id = brute_force_primary(paths, "id")
    matching = report["matching"]
    unmatched_pairs = [
        ("B1->A1 kd", matching["B1_to_A1"]["kdtree"]["unmatched"],
         expected["unmatched_test_to_reference"]),
        ("A1->A2 kd", matching["A1_to_A2"]["kdtree"]["unmatched"],
         expected["unmatched_every_reference"]),
        ("B1->A1 id", matching["B1_to_A1"]["id"]["unmatched"],
         expected_by_id["unmatched_test_to_reference"]),
        ("A1->A2 id", matching["A1_to_A2"]["id"]["unmatched"],
         expected_by_id["unmatched_every_reference"]),
    ]
    agreement = (matching["B1_to_A1"]["kdtree"]["id_agreement_rate"],
                 matching["A1_to_A2"]["kdtree"]["id_agreement_rate"])
    beyond_tolerance = matching["B1_to_A1"]["id"]["offset_beyond_tolerance"]
    exercised = {
        "drops (id-unmatched B1->A1)": expected_by_id["unmatched_test_to_reference"],
        "displaced B1 beyond tolerance": expected["offset_beyond_tolerance"],
        "displaced A2 (kd minus id unmatched A1->A2)":
            expected["unmatched_every_reference"] - expected_by_id["unmatched_every_reference"],
        "departed": expected["column0_counts"]["flagged_departed"],
        "arrived only": expected["column0_counts"]["flagged_arrived_only"],
        "fewest column-0 triplets at one seam": min(expected["column0_per_cut"]),
    }
    mismatches = brute_force_mismatches(report, expected)
    mismatches_by_id = brute_force_mismatches(report_by_id, expected_by_id)
    return [
        ("three seams: the synthetic runs exercise drops, displacements, departed and "
         "arrived crossings at every seam",
         all(value > 0 for value in exercised.values()),
         ", ".join(f"{name} {value}" for name, value in exercised.items())),
        ("three seams: unmatched counts (kd-tree and id) equal brute force, kd matches "
         "agree with ids",
         all(got == want for _, got, want in unmatched_pairs) and agreement == (1.0, 1.0),
         ", ".join(f"{name} {got}/{want}" for name, got, want in unmatched_pairs)
         + f", agreement {agreement}"),
        ("three seams: id-match offsets beyond 0.05 h are exactly the displaced B1 particles",
         beyond_tolerance == expected["offset_beyond_tolerance"],
         f"{beyond_tolerance} vs {expected['offset_beyond_tolerance']}"),
        ("three seams, kd-tree: triplets, bins, groups and rms ratios equal brute force",
         not mismatches,
         f"{expected['triplets']} triplets; " + _mismatch_detail(mismatches)),
        ("three seams, id match: triplets, bins, groups and rms ratios equal brute force",
         not mismatches_by_id,
         f"{expected_by_id['triplets']} triplets; " + _mismatch_detail(mismatches_by_id)),
        ("three seams: seam side matches the owner slab",
         report["consistency"]["B1"]["seam_side_owner_disagreements"] == 0, ""),
    ]


def run_self_test(keep_directory=None) -> int:
    """Two synthetic scenarios: one seam with every particle in every run (the
    expected contrasts), and three seams with dropped and displaced particles
    (the bookkeeping against an independent brute-force computation)."""
    with tempfile.TemporaryDirectory(prefix="seam_audit_self_test_") as temporary:
        directory = pathlib.Path(keep_directory) if keep_directory else pathlib.Path(temporary)
        directory.mkdir(parents=True, exist_ok=True)
        checks = (_self_test_single_seam(directory / "single_seam")
                  + _self_test_three_seams(directory / "three_seams"))
        all_passed = True
        for name, passed, detail in checks:
            all_passed &= bool(passed)
            print(f"[seam_audit self-test] {'PASS' if passed else 'FAIL'}  {name}"
                  + (f"  ({detail})" if detail else ""))
        print(f"[seam_audit self-test] {'ALL PASSED' if all_passed else 'FAILED'} "
              f"({len(checks)} checks)"
              + (f"; outputs kept in {directory}" if keep_directory else ""))
        return 0 if all_passed else 1


# =============================================================================
# CLI
# =============================================================================

def parse_arguments(argument_list=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seam audit analysis (CPU only).")
    parser.add_argument("--reference", nargs=2, metavar=("A1", "A2"),
                        help="two reference dumps (.npz, .json or stem)")
    parser.add_argument("--test", nargs="+", metavar="B",
                        help="one or two test dumps (B1 [B2])")
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--match", choices=("kdtree", "id"), default="kdtree")
    parser.add_argument("--particles", choices=("fluid", "all"), default="fluid")
    parser.add_argument("--far-bin-start", type=int, default=DEFAULT_FAR_BIN_START)
    parser.add_argument("--tolerance-factor", type=float, default=DEFAULT_TOLERANCE_FACTOR,
                        help="kd-tree match tolerance in units of smoothing_length")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="run on synthetic dumps and check the expected ratios")
    parser.add_argument("--keep", default=None,
                        help="(self-test) keep the synthetic dumps and outputs here")
    return parser.parse_args(argument_list)


def main(argument_list=None) -> int:
    arguments = parse_arguments(argument_list)
    if arguments.self_test:
        return run_self_test(arguments.keep)
    if not arguments.reference or not arguments.test or not arguments.out:
        print("need --reference A1 A2 --test B1 [B2] --out DIR (or --self-test)",
              file=sys.stderr)
        return 2
    if len(arguments.test) > 2:
        print("at most two test dumps", file=sys.stderr)
        return 2
    report = analyze(arguments.reference, arguments.test, match=arguments.match,
                     out_dir=arguments.out, particle_filter=arguments.particles,
                     far_bin_start=arguments.far_bin_start,
                     tolerance_factor=arguments.tolerance_factor,
                     make_figures=not arguments.no_figures)
    summary = report["summary"]
    print(f"[seam_audit analyze] {summary['case_name']} {summary['test_run']} N={summary['horizon']}: "
          f"seam_excess {format_value(summary['seam_excess'])}; d=0 rms ratios "
          + "; ".join(f"{field} all/flagged/unflagged "
                      f"{format_value(values['all'])}/{format_value(values['flagged'])}/"
                      f"{format_value(values['unflagged'])}"
                      for field, values in summary["d0_rms_ratio"].items())
          + f" -> {pathlib.Path(arguments.out) / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
