"""
taylor_green.py — the two-dimensional Taylor–Green vortex (decaying, periodic), the analytic solution the v8
TGV runs are measured against.

On the periodic box [x0, x0 + L) x [y0, y0 + L), wavenumber k = 2 pi n / L (n vortex pairs per side):

    u(x, y, t) =  U0 sin(k (x - x0)) cos(k (y - y0)) F(t)
    v(x, y, t) = -U0 cos(k (x - x0)) sin(k (y - y0)) F(t)
    p(x, y, t) =  rho0 U0^2 / 4 (cos(2 k (x - x0)) + cos(2 k (y - y0))) F(t)^2
    F(t)       =  exp(-2 nu k^2 t)

(u . grad) u = -grad p / rho0 holds for this (u, p) at every t, and the viscous decay is exact for the
incompressible Navier-Stokes equations. Kinetic energy over the box: E(t) = rho0 U0^2 L^2 F(t)^2 / 4
(1/2 rho0 integral |u|^2 = 1/2 rho0 U0^2 F^2 (L^2 / 4 + L^2 / 4)). Every function takes float64 arrays.
"""

from __future__ import annotations

import math

import numpy as np


def wavenumber(length: float, vortex_pairs: int = 1) -> float:
    return 2.0 * math.pi * vortex_pairs / length


def decay_factor(time_value: float, viscosity: float, k: float) -> float:
    """F(t) = exp(-2 nu k^2 t): the velocity amplitude relative to t = 0."""
    return math.exp(-2.0 * viscosity * k * k * time_value)


def velocity(x: np.ndarray, y: np.ndarray, time_value: float, *, amplitude: float, viscosity: float, k: float,
             x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """(N, 2) analytic velocity at the points (x, y) at time t."""
    factor = amplitude * decay_factor(time_value, viscosity, k)
    kx = k * (np.asarray(x, dtype=np.float64) - x0)
    ky = k * (np.asarray(y, dtype=np.float64) - y0)
    return np.column_stack([factor * np.sin(kx) * np.cos(ky), -factor * np.cos(kx) * np.sin(ky)])


def pressure(x: np.ndarray, y: np.ndarray, time_value: float, *, amplitude: float, viscosity: float, k: float,
             rest_density: float, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """Analytic pressure (zero mean over the box) at the points (x, y) at time t."""
    factor = decay_factor(time_value, viscosity, k) ** 2
    kx = k * (np.asarray(x, dtype=np.float64) - x0)
    ky = k * (np.asarray(y, dtype=np.float64) - y0)
    return 0.25 * rest_density * amplitude * amplitude * factor * (np.cos(2.0 * kx) + np.cos(2.0 * ky))


def kinetic_energy(time_value: float, *, amplitude: float, viscosity: float, k: float, rest_density: float,
                   length: float) -> float:
    """E(t) = rho0 U0^2 L^2 F(t)^2 / 4 over the whole box."""
    return 0.25 * rest_density * amplitude * amplitude * length * length * decay_factor(time_value, viscosity, k) ** 2


def vorticity(x: np.ndarray, y: np.ndarray, time_value: float, *, amplitude: float, viscosity: float, k: float,
              x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """omega = dv/dx - du/dy = 2 k U0 F sin(kx) sin(ky)."""
    factor = 2.0 * k * amplitude * decay_factor(time_value, viscosity, k)
    kx = k * (np.asarray(x, dtype=np.float64) - x0)
    ky = k * (np.asarray(y, dtype=np.float64) - y0)
    return factor * np.sin(kx) * np.sin(ky)


def momentum_balance_residual(*, amplitude: float, k: float, rest_density: float, samples: int = 64) -> float:
    """Self-check of the sign conventions above: max |(u . grad) u + grad p / rho0| at t = 0 on a lattice
    (finite differences on the analytic fields, spacing L / samples; O(h^2) small)."""
    length = 2.0 * math.pi / k
    spacing = length / samples
    grid = np.arange(samples) * spacing
    x, y = np.meshgrid(grid, grid, indexing="ij")
    keywords = dict(amplitude=amplitude, viscosity=0.0, k=k)
    field = velocity(x.ravel(), y.ravel(), 0.0, **keywords).reshape(samples, samples, 2)
    pressure_field = pressure(x.ravel(), y.ravel(), 0.0, rest_density=rest_density, **keywords).reshape(samples, samples)

    def derivative(values: np.ndarray, axis: int) -> np.ndarray:
        return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (2.0 * spacing)
    u, v = field[..., 0], field[..., 1]
    advection_x = u * derivative(u, 0) + v * derivative(u, 1)
    advection_y = u * derivative(v, 0) + v * derivative(v, 1)
    residual_x = advection_x + derivative(pressure_field, 0) / rest_density
    residual_y = advection_y + derivative(pressure_field, 1) / rest_density
    return float(max(np.abs(residual_x).max(), np.abs(residual_y).max()))
