"""CPU tests of cavity_sampling.py on synthetic particle fields (no GPU).

    .venv/Scripts/python.exe -m experiment.validation.test_cavity_sampling
"""
from __future__ import annotations

import numpy as np

from experiment.validation import cavity_sampling as sampling


def jittered_lattice(spacing: float, jitter: float, seed: int = 0) -> np.ndarray:
    count = int(round(1.0 / spacing)) + 1
    axis = np.linspace(-0.5, 0.5, count)
    xx, yy = np.meshgrid(axis, axis)
    positions = np.stack([xx.ravel(), yy.ravel()], axis=1)
    rng = np.random.default_rng(seed)
    return positions + rng.uniform(-jitter, jitter, positions.shape) * spacing


def check_linear_exactness() -> None:
    spacing = 1.0 / 250
    positions = jittered_lattice(spacing, 0.2)
    velocities = np.stack([0.3 + 2.0 * positions[:, 0] - 1.5 * positions[:, 1],
                           -0.1 + 0.7 * positions[:, 0] + 0.4 * positions[:, 1]], axis=1)
    particles = sampling.ParticleSet(positions, velocities, np.full(len(positions), spacing ** 2))
    points = np.random.default_rng(1).uniform(-0.4, 0.4, (500, 2))
    result = sampling.interpolate(particles, points, 5 * spacing)
    exact = np.stack([0.3 + 2.0 * points[:, 0] - 1.5 * points[:, 1], -0.1 + 0.7 * points[:, 0] + 0.4 * points[:, 1]],
                     axis=1)
    mls_error = np.abs(result["mls"] - exact).max()
    shepard_error = np.abs(result["shepard"] - exact).max()
    assert mls_error < 1e-10, mls_error
    assert shepard_error < 2e-3, shepard_error           # O(h) bias on a jittered lattice
    assert not result["mls_fallback"].any()
    print(f"linear field: MLS max error {mls_error:.1e}, Shepard {shepard_error:.1e}")


def check_smooth_field_convergence() -> None:
    errors = []
    for spacing in (1.0 / 125, 1.0 / 250, 1.0 / 500):
        positions = jittered_lattice(spacing, 0.1, seed=2)
        velocities = np.stack([np.sin(3 * positions[:, 0]) * np.cos(2 * positions[:, 1]),
                               np.cos(3 * positions[:, 0] + positions[:, 1])], axis=1)
        particles = sampling.ParticleSet(positions, velocities, np.full(len(positions), spacing ** 2))
        points = np.stack([np.zeros(50), np.linspace(-0.4, 0.4, 50)], axis=1)
        result = sampling.interpolate(particles, points, 5 * spacing)
        exact = np.stack([np.sin(3 * points[:, 0]) * np.cos(2 * points[:, 1]), np.cos(3 * points[:, 0] + points[:, 1])],
                         axis=1)
        errors.append(np.abs(result["mls"] - exact).max())
    orders = [np.log2(errors[index] / errors[index + 1]) for index in range(2)]
    assert all(order > 1.5 for order in orders), (errors, orders)
    print(f"smooth field: MLS max errors {[f'{e:.1e}' for e in errors]}, observed orders {[f'{o:.2f}' for o in orders]}")


def check_frames() -> None:
    frame = sampling.Frame.make("wall", 0.004)
    assert abs(frame.length - 1.008) < 1e-15
    assert np.allclose(frame.to_physical([0.0, 0.5, 1.0]), [-0.504, 0.0, 0.504])
    assert np.allclose(frame.to_reference(frame.to_physical([0.123, 0.987])), [0.123, 0.987])
    fluid = sampling.Frame.make("fluid", 0.004)
    assert np.allclose(fluid.to_physical([0.0, 0.5, 1.0]), [-0.5, 0.0, 0.5])
    points = sampling.centerline_points(frame, np.array([0.25, 0.75]), "u")
    assert np.allclose(points, [[0.0, -0.252], [0.0, 0.252]])
    print("frames: ok")


def check_extremum_and_stream_function() -> None:
    x = np.linspace(0, 1, 1001)
    values = 0.3 * (x - 0.4123) ** 2 - 0.2 + 0.05 * (x - 0.4123) ** 3
    position, value = sampling.refine_extremum(x, values, "min")
    assert abs(position - 0.4123) < 1e-5 and abs(value + 0.2) < 1e-9, (position, value)
    # psi of the solid-body-like field u = -dpsi/dy convention check: u = dpsi/dy with psi = sin(pi x) sin(pi y) * -0.1
    nx = ny = 257
    xr = np.linspace(0, 1, nx)
    yr = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(xr, yr)
    psi_exact = -0.1 * np.sin(np.pi * xx) * np.sin(np.pi * yy) ** 2
    u = -0.1 * np.sin(np.pi * xx) * 2 * np.sin(np.pi * yy) * np.cos(np.pi * yy) * np.pi
    psi = sampling.stream_function(u, yr)
    assert np.abs(psi - psi_exact).max() < 1e-4, np.abs(psi - psi_exact).max()
    xm, ym, vm = sampling.refine_grid_minimum(xr, yr, psi)
    assert abs(xm - 0.5) < 1e-4 and abs(ym - 0.5) < 1e-4 and abs(vm + 0.1) < 1e-5, (xm, ym, vm)
    print(f"extremum: ({position:.6f}, {value:.9f}); stream function minimum ({xm:.5f}, {ym:.5f}, {vm:.6f})")


if __name__ == "__main__":
    check_linear_exactness()
    check_smooth_field_convergence()
    check_frames()
    check_extremum_and_stream_function()
    print("ALL PASS")
