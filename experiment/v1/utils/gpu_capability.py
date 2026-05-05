"""
gpu_capability.py — GPU SPH compute-weight lookup for V1+ multi-GPU partitioning.

`KNOWN_GPU_SPH_WEIGHT` maps `vkPhysicalDeviceProperties.deviceName` to a
relative SPH-step throughput weight. Anchored at NVIDIA RTX 4060 Ti = 1.0.

Used by V1+ partitioner to compute fluid-particle fractions per GPU; see
`docs/sph_v1_design.md` for the K_split derivation. Override in case.yaml
via `partition.weights: [w0, w1, ...]` if your hardware isn't in this table
or you want a hand-tuned split.

Update procedure:
  1. .venv/Scripts/python.exe tools/benchmark_calibration.py --device-index N --wall-time 60
     (run for each GPU sequentially)
  2. .venv/Scripts/python.exe tools/analyze_calibration.py output/calibration/*
  3. Paste analyzer's suggested entries into KNOWN_GPU_SPH_WEIGHT below.
"""

from typing import Optional


# Calibrated 2026-05-06 on lid_driven_cavity_2d (1.046M particles, 60s wall-time
# headless run, post-warmup steady-state median). Methodology:
# tools/benchmark_calibration.py + tools/analyze_calibration.py.
KNOWN_GPU_SPH_WEIGHT: dict[str, float] = {
    "AMD Radeon RX 7900 XTX":      2.088,   # measured 2026-05-06: 293.6 M part-step/s, median 3.565 ms
    "NVIDIA GeForce RTX 4060 Ti":  1.000,   # baseline anchor: 140.6 M part-step/s, median 7.444 ms
    "NVIDIA GeForce RTX 5090":     6.0,     # provisional spec-sheet estimate, validate when 2x5090 rig exists
    "AMD Radeon(TM) Graphics":     0.3,     # iGPU lower bound, unmeasured
}


def lookup_gpu_weight(device_name: str) -> Optional[float]:
    """Return the SPH compute weight for a vkPhysicalDeviceProperties.deviceName,
    or None if the device is not in the table (caller should fall back to a
    case.yaml `partition.weights` override or surface a clear error).
    """
    return KNOWN_GPU_SPH_WEIGHT.get(device_name)
