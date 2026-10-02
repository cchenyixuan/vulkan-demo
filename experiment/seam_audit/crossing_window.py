"""
crossing_window.py — capture the particles around seam crossings over a window
of frames, so the column-0 "a crossing particle is within h this step" group
has enough samples.

Why a window: at the audited horizons (300 / 2000 steps of the lid-driven
cavity) a seam sees only ~0.03-0.1 crossings per frame, so the single dumped
frame usually has NO crossing particle and the flagged column-0 group would be
empty. Over W consecutive frames the events add up.

What is stored: in every window frame, the particles that crossed any audit
cut line during that frame (x changed side of x_cut = origin_x + cut * h) and
every particle within capture_radius_factor * h of one of them, with their
state after that frame. In a K > 1 run a cut-line crossing IS a seam crossing
(slab ownership is the voxel column, floor((x - origin_x) / h) >= cut); a K = 1
reference run sees the same particles cross the same lines (its trajectory
differs only by floating-point divergence), so its captures overlap the test
run's and the analyzer can match them frame by frame.

The capture costs almost no disk (tens of particles per frame); the per-frame
cost is one readback of positions + ids (+ the remaining fields only in frames
that have a crossing).
"""
from __future__ import annotations

import numpy as np


def cut_line_positions(origin_x: float, smoothing_length: float, cut_columns) -> np.ndarray:
    """World x of every cut line (slab i owns global columns [cut_{i-1}, cut_i))."""
    return np.array([origin_x + int(cut) * smoothing_length for cut in sorted(set(cut_columns))],
                    dtype=np.float64)


def crossing_mask(previous_x: np.ndarray, current_x: np.ndarray, cut_lines: np.ndarray) -> np.ndarray:
    """True for particles whose x changed side of any cut line between the two
    frames (arrays aligned by global id; NaN = absent in either frame)."""
    present = np.isfinite(previous_x) & np.isfinite(current_x)
    crossed = np.zeros(previous_x.shape, dtype=bool)
    for line in cut_lines:
        crossed |= present & ((previous_x >= line) != (current_x >= line))
    return crossed


def select_capture(positions: np.ndarray, crossing_ids: np.ndarray, ids: np.ndarray,
                   radius: float) -> np.ndarray:
    """Indices (into ids/positions) of the crossing particles and every particle
    within `radius` of one of them."""
    if crossing_ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    from scipy.spatial import cKDTree
    index_of_id = {int(identifier): index for index, identifier in enumerate(ids)}
    crossing_rows = np.array([index_of_id[int(identifier)] for identifier in crossing_ids
                              if int(identifier) in index_of_id], dtype=np.int64)
    if crossing_rows.size == 0:
        return np.zeros(0, dtype=np.int64)
    tree = cKDTree(positions)
    neighbourhoods = tree.query_ball_point(positions[crossing_rows], r=radius)
    selected = set(crossing_rows.tolist())
    for neighbourhood in neighbourhoods:
        selected.update(neighbourhood)
    return np.array(sorted(selected), dtype=np.int64)


class CrossingWindowRecorder:
    """Accumulates one horizon's window. Call begin() with the positions after
    frame N-W, then add_frame() after every window frame, then arrays()."""

    def __init__(self, cut_lines: np.ndarray, smoothing_length: float,
                 capture_radius_factor: float = 1.5) -> None:
        self.cut_lines = cut_lines
        self.radius = capture_radius_factor * smoothing_length
        self.previous_x = None
        self.records: list[dict] = []
        self.frames_with_crossings = 0
        self.crossing_events = 0
        self.frame_count = 0

    def begin(self, total_ids: int, ids: np.ndarray, x: np.ndarray) -> None:
        self.previous_x = np.full(total_ids, np.nan, dtype=np.float64)
        self.previous_x[ids] = x

    def crossings(self, total_ids: int, ids: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Global ids that crossed a cut line since the previous frame (does not
        advance the window)."""
        current_x = np.full(total_ids, np.nan, dtype=np.float64)
        current_x[ids] = x
        return np.nonzero(crossing_mask(self.previous_x, current_x, self.cut_lines))[0].astype(np.uint32)

    def advance(self, total_ids: int, ids: np.ndarray, x: np.ndarray) -> None:
        """A window frame without crossings: only move the reference positions on."""
        self.frame_count += 1
        self.previous_x = np.full(total_ids, np.nan, dtype=np.float64)
        self.previous_x[ids] = x

    def add_frame(self, frame_number: int, total_ids: int, state: dict,
                  crossing_ids: np.ndarray) -> None:
        """state: per-particle arrays of this frame (sorted by id, keys id,
        slab, position, ... as produced by the dumper). crossing_ids from
        crossings() for the same frame."""
        self.frame_count += 1
        ids = state["id"]
        if crossing_ids.size:
            self.frames_with_crossings += 1
            self.crossing_events += int(crossing_ids.size)
            rows = select_capture(state["position"], crossing_ids, ids, self.radius)
            if rows.size:
                record = {key: value[rows] for key, value in state.items()
                          if isinstance(value, np.ndarray) and value.shape[:1] == ids.shape}
                record["crossed_this_frame"] = np.isin(record["id"], crossing_ids)
                record["frame"] = np.full(rows.size, frame_number, dtype=np.int32)
                self.records.append(record)
        self.previous_x = np.full(total_ids, np.nan, dtype=np.float64)
        self.previous_x[ids] = state["position"][:, 0]

    def arrays(self) -> dict:
        if not self.records:
            return {}
        keys = self.records[0].keys()
        return {key: np.concatenate([record[key] for record in self.records]) for key in keys}

    def summary(self) -> dict:
        return {"window_frames": self.frame_count,
                "frames_with_crossings": self.frames_with_crossings,
                "crossing_events": self.crossing_events,
                "captured_rows": int(sum(record["id"].size for record in self.records)),
                "capture_radius": self.radius,
                "cut_lines": [float(line) for line in self.cut_lines]}
