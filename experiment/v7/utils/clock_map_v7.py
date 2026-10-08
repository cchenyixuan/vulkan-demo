"""
clock_map_v7.py — device -> host clock map of the E29 step trace (numpy only, so the analysis can
refit traces without a Vulkan driver).

A map is fitted per device from VK_KHR_calibrated_timestamps pairs (device ns relative to an integer
origin tick, host ns, driver maxDeviation ns). Only pairs within ``deviation_margin_ns`` of the run's
smallest maxDeviation are used: a wider pair's host stamp sits late by up to its width (under load
25-44 % of the pairs on the 5090 / Windows driver are 15-43 us wide against a floor of ~12.3 us). Then a
least-squares line, the running median of its residual (2 half_window + 1 pairs, interpolated linearly in
device time; the frequency ratio drifts by a few ppm over minutes) and a 3-sigma clip on the residual to
line + drift, repeated until the kept set is stable.
"""
from __future__ import annotations

import numpy as np

CLOCK_FIT_VERSION = 2       # 1 = E29 first scan: all pairs, clip on the line residual, float ticks (256 ns grid)


def fit_clock(device_ns, host_ns, deviation_ns, deviation_margin_ns: float = 1000.0,
              half_window: int = 4) -> dict:
    device_ns = np.asarray(device_ns, dtype=np.float64)
    host_ns = np.asarray(host_ns, dtype=np.float64)
    deviation_ns = np.asarray(deviation_ns, dtype=np.float64)
    threshold = float(deviation_ns.min()) + deviation_margin_ns
    usable = deviation_ns <= threshold
    if usable.sum() < 5:                      # too few narrow pairs: take the 5 narrowest
        usable = np.zeros(len(device_ns), dtype=bool)
        usable[np.argsort(deviation_ns)[:5]] = True
    device_center, host_center = float(device_ns[usable].mean()), float(host_ns[usable].mean())
    x, y = device_ns - device_center, host_ns - host_center
    kept = usable.copy()
    for _ in range(5):
        slope, intercept = np.polyfit(x[kept], y[kept], 1)
        line_residual = y - (slope * x + intercept)
        order = np.argsort(x[kept])
        knot_x, knot_line = x[kept][order], line_residual[kept][order]
        drift = np.array([np.median(knot_line[max(0, index - half_window):index + half_window + 1])
                          for index in range(len(knot_line))])
        final = line_residual - np.interp(x, knot_x, drift)
        sigma = float(np.sqrt(np.mean(final[kept] ** 2)))
        new_kept = usable & (np.abs(final) <= max(3.0 * sigma, 1.0))
        if new_kept.sum() < 5 or np.array_equal(new_kept, kept):
            break
        kept = new_kept
    return {"version": CLOCK_FIT_VERSION, "device_center_ns": device_center, "host_center_ns": host_center,
            "slope": float(slope), "intercept_ns": float(intercept),
            "drift_knots_device_ns": [float(value) for value in knot_x],
            "drift_knots_ns": [float(value) for value in drift],
            "samples": int(len(device_ns)), "samples_low_deviation": int(usable.sum()),
            "samples_kept": int(kept.sum()), "deviation_threshold_ns": threshold,
            "line_residual_rms_ns": float(np.sqrt(np.mean(line_residual[kept] ** 2))),
            "residual_rms_ns": float(np.sqrt(np.mean(final[kept] ** 2))),
            "residual_max_ns": float(np.max(np.abs(final[kept]))),
            "driver_max_deviation_min_ns": float(deviation_ns.min()),
            "driver_max_deviation_median_ns": float(np.median(deviation_ns[kept]))}


def _drift(fit: dict, x):
    if "drift_knots_ns" not in fit:          # early version-1 maps: the line only
        return 0.0
    return np.interp(x, fit["drift_knots_device_ns"], fit["drift_knots_ns"])


def clock_to_host(fit: dict, device_ns):
    """Host ns of device ns (relative to the fit's origin), scalar or array."""
    x = np.asarray(device_ns, dtype=np.float64) - fit["device_center_ns"]
    return fit["host_center_ns"] + fit["intercept_ns"] + fit["slope"] * x + _drift(fit, x)


def host_to_clock(fit: dict, host_ns, iterations: int = 4, centered: bool = False):
    """Inverse of clock_to_host (the drift term varies by ppm, so a few fixed-point steps converge);
    ``centered``: relative to the fit's device_center_ns (version-1 fits are centred on absolute
    epoch ticks, where float64 has a 256 ns grid)."""
    target = np.asarray(host_ns, dtype=np.float64) - fit["host_center_ns"] - fit["intercept_ns"]
    x = target / fit["slope"]
    for _ in range(iterations):
        x = (target - _drift(fit, x)) / fit["slope"]
    return x if centered else x + fit["device_center_ns"]
