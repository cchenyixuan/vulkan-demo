"""
viscous_operators.py - host-side (numpy, float64) discretizations of the viscous Laplacian on a particle set, with the
solver's own ingredients: the Wendland C4 kernel of support radius h, V_j = m_j / rho_j, the KGC matrix of
correction.comp with its Tikhonov regularisation, and EPS_H_SQUARED in the r^2 denominators.

Used by micropolar_tests.py operators-host (docs/viscous_operator_comparison.md). Nothing here is used by the solver.

Notation: r_ij = r_i - r_j, v_ij = v_i - v_j, gW_ij = grad_i W_ij (raw), L_i = (M_i + xi_reg I)^-1 with
M_i = sum_j V_j (r_j - r_i) (x) gW_ij, G_ij = L_i gW_ij, eps2 = EPS_H_SQUARED, d = dimension. Every operator returns
the discrete approximation of lap v (multiply by nu for the viscous acceleration).

  MG    2 (d + 2) sum_j V_j (v_ij . r_ij) / (r_ij^2 + eps2) G_ij                     (force.comp today)
  BM    2 sum_j V_j v_ij (r_ij . G_ij) / (r_ij^2 + eps2)                              (the form of xi lap w)
  BM-G  BM with v_ij -> v_ij - (grad v)_i r_ij, (grad v)_i = sum_j V_j (v_j - v_i) (x) G_ij
  DS    sum_j V_j ((grad v)_j - (grad v)_i) G_ij
  SW    (tr(Gamma_i^-1) / d) [ 2 sum_j V_j v_ij (r_ij . gW_ij) / (r_ij^2 + eps2) - 2 (grad v)_i sum_j V_j gW_ij ],
        Gamma_i = - sum_j V_j (r_ij . gW_ij) r_ij (x) r_ij / r_ij^2      (Schwaiger 2008, constant coefficient)
  FM    2 sum_j V_j (v_ij - (grad v)_i r_ij) (r_ij . Bhat_i gW_ij) / (r_ij^2 + eps2), Bhat_i (symmetric) from
        [ A_i^{bgt} L_i^{ta} C_i^{aem} + D_i^{bgem} ] Bhat_i^{em} = - delta^{bg},
        A^{bgt} = sum_j V_j r_ij^b r_ij^g gW_ij^t, C^{aem} = sum_j V_j e_ij^a e_ij^e gW_ij^m,
        D^{bgem} = sum_j V_j e_ij^b e_ij^g r_ij^e gW_ij^m, e_ij = r_ij / |r_ij|   (Fatehi & Manzari 2011)
  KK    sum_j V_j (v_j - v_i) lap W_ij, lap W = W'' + (d - 1) W' / r           (second derivative of the kernel)
"""

from __future__ import annotations

import numpy as np

OPERATORS = ("MG", "BM", "BM-G", "DS", "SW", "FM", "KK")
HAS_EPSILON = {"MG": True, "BM": True, "BM-G": True, "DS": False, "SW": True, "FM": True, "KK": False}


def wendland_c4(distance: np.ndarray, smoothing_length: float, dimension: int):
    """W, dW/dr, d2W/dr2 of the solver's kernel (helpers.glsl): W = C (1 - q)^6 (35/3 q^2 + 6 q + 1), q = r / h,
    C = 9 / (pi h^2) in 2-D, 495 / (32 pi h^3) in 3-D; zero for q >= 1."""
    coefficient = (9.0 / (np.pi * smoothing_length ** 2) if dimension == 2
                   else 495.0 / (32.0 * np.pi * smoothing_length ** 3))
    q = np.minimum(distance / smoothing_length, 1.0)
    one = 1.0 - q
    value = coefficient * one ** 6 * (35.0 / 3.0 * q * q + 6.0 * q + 1.0)
    polynomial = q * (-280.0 / 3.0 * q - 56.0 / 3.0)
    first = coefficient / smoothing_length * one ** 5 * polynomial
    second = coefficient / smoothing_length ** 2 * (-5.0 * one ** 4 * polynomial
                                                    + one ** 5 * (-560.0 / 3.0 * q - 56.0 / 3.0))
    return value, first, second


def _scatter(index: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    """sum of ``values`` rows per ``index`` -> (count, ...); real or complex."""
    flat = values.reshape(values.shape[0], -1)
    result = np.zeros((count, flat.shape[1]), dtype=values.dtype)
    for column in range(flat.shape[1]):
        if np.iscomplexobj(flat):
            result[:, column] = (np.bincount(index, flat[:, column].real, count)
                                 + 1j * np.bincount(index, flat[:, column].imag, count))
        else:
            result[:, column] = np.bincount(index, flat[:, column], count)
    return result.reshape((count,) + values.shape[1:])


class ParticleSet:
    """Pairs and the geometric sums of one arrangement; the fields are applied afterwards."""

    def __init__(self, positions: np.ndarray, volumes: np.ndarray, smoothing_length: float, dimension: int,
                 regularization_xi: float = 0.0) -> None:
        from scipy.spatial import cKDTree
        self.dimension = dimension
        self.count = positions.shape[0]
        self.positions = np.ascontiguousarray(positions[:, :dimension], dtype=np.float64)
        self.smoothing_length = float(smoothing_length)
        pairs = cKDTree(self.positions).query_pairs(self.smoothing_length, output_type="ndarray")
        first = np.concatenate([pairs[:, 0], pairs[:, 1]])
        second = np.concatenate([pairs[:, 1], pairs[:, 0]])
        offset = self.positions[first] - self.positions[second]                     # r_ij
        distance = np.sqrt((offset ** 2).sum(axis=1))
        keep = (distance < self.smoothing_length) & (distance >= 1e-12)             # the shader's pair test
        self.first, self.second = first[keep], second[keep]
        self.offset, self.distance = offset[keep], distance[keep]
        self.neighbour_volume = np.asarray(volumes, dtype=np.float64)[self.second]
        _value, derivative, second_derivative = wendland_c4(self.distance, self.smoothing_length, dimension)
        self.unit = self.offset / self.distance[:, None]
        self.kernel_gradient = derivative[:, None] * self.unit                      # gW_ij
        self.kernel_laplacian = second_derivative + (dimension - 1) * derivative / self.distance
        # KGC (correction.comp): M_i = sum_j V_j (r_j - r_i) (x) gW_ij, + xi_reg on the diagonal, inverse
        moment = _scatter(self.first, -self.neighbour_volume[:, None, None]
                          * self.offset[:, :, None] * self.kernel_gradient[:, None, :], self.count)
        moment = moment + regularization_xi * np.eye(dimension)[None]
        empty = np.abs(np.linalg.det(moment)) < 1e-12
        moment[empty] = np.eye(dimension)
        self.correction = np.linalg.inv(moment)                                     # L_i
        self.corrected_gradient = np.einsum("pab,pb->pa", self.correction[self.first], self.kernel_gradient)
        self._fatehi = None
        self._schwaiger = None

    # ---- helpers
    def sum_over_neighbours(self, pair_values: np.ndarray) -> np.ndarray:
        return _scatter(self.first, pair_values, self.count)

    def gradient(self, field: np.ndarray) -> np.ndarray:
        """(grad v)_i[a, b] = d v_a / d x_b = sum_j V_j (v_j - v_i)_a G_ij,b."""
        difference = field[self.second] - field[self.first]
        return self.sum_over_neighbours(self.neighbour_volume[:, None, None] * difference[:, :, None]
                                        * self.corrected_gradient[:, None, :])

    def smallest_pair_distance(self) -> float:
        return float(self.distance.min())

    # ---- operators (each returns the approximation of lap v, shape of ``field``)
    def laplacian(self, name: str, field: np.ndarray, eps_h_squared: float) -> np.ndarray:
        dimension = self.dimension
        volume = self.neighbour_volume
        difference = field[self.first] - field[self.second]                         # v_ij
        denominator = self.distance ** 2 + eps_h_squared
        if name == "MG":
            projection = np.einsum("pa,pa->p", difference, self.offset)
            return self.sum_over_neighbours((2.0 * (dimension + 2) * volume * projection / denominator)[:, None]
                                            * self.corrected_gradient)
        if name == "BM":
            weight = 2.0 * volume * np.einsum("pa,pa->p", self.offset, self.corrected_gradient) / denominator
            return self.sum_over_neighbours(weight[:, None] * difference)
        if name == "KK":
            return self.sum_over_neighbours((volume * self.kernel_laplacian)[:, None] * (-difference))
        gradient = self.gradient(field)
        if name == "DS":
            jump = gradient[self.second] - gradient[self.first]
            return self.sum_over_neighbours(volume[:, None] * np.einsum("pab,pb->pa", jump, self.corrected_gradient))
        reduced = difference - np.einsum("pab,pb->pa", gradient[self.first], self.offset)
        if name == "BM-G":
            weight = 2.0 * volume * np.einsum("pa,pa->p", self.offset, self.corrected_gradient) / denominator
            return self.sum_over_neighbours(weight[:, None] * reduced)
        if name == "SW":
            if self._schwaiger is None:
                radial = np.einsum("pa,pa->p", self.offset, self.kernel_gradient)
                gamma = self.sum_over_neighbours(-(volume * radial)[:, None, None]
                                                 * self.unit[:, :, None] * self.unit[:, None, :])
                singular = np.abs(np.linalg.det(gamma)) < 1e-12
                gamma[singular] = np.eye(dimension)
                self._schwaiger = (np.trace(np.linalg.inv(gamma), axis1=1, axis2=2) / dimension,
                                   self.sum_over_neighbours(volume[:, None] * self.kernel_gradient))
            scale, gradient_sum = self._schwaiger
            radial = np.einsum("pa,pa->p", self.offset, self.kernel_gradient)
            direct = self.sum_over_neighbours((2.0 * volume * radial / denominator)[:, None] * difference)
            return scale[:, None] * (direct - 2.0 * np.einsum("iab,ib->ia", gradient, gradient_sum))
        if name == "FM":
            if self._fatehi is None:
                self._fatehi = self._fatehi_tensor()
            corrected = np.einsum("pab,pb->pa", self._fatehi[self.first], self.kernel_gradient)
            weight = 2.0 * volume * np.einsum("pa,pa->p", self.offset, corrected) / denominator
            return self.sum_over_neighbours(weight[:, None] * reduced)
        raise ValueError(name)

    def _fatehi_tensor(self) -> np.ndarray:
        """Bhat_i of Fatehi & Manzari 2011 from the linear system quoted in the module docstring (Negi &
        Ramachandran, arXiv:2107.11859, Eq. 60), solved per particle in the minimum-norm sense (the system is
        symmetric in both index pairs, so the solution is the symmetric tensor)."""
        dimension, volume = self.dimension, self.neighbour_volume
        third = self.sum_over_neighbours(volume[:, None, None, None] * self.offset[:, :, None, None]
                                         * self.offset[:, None, :, None] * self.kernel_gradient[:, None, None, :])
        mixed = self.sum_over_neighbours(volume[:, None, None, None] * self.unit[:, :, None, None]
                                         * self.unit[:, None, :, None] * self.kernel_gradient[:, None, None, :])
        fourth = self.sum_over_neighbours(
            volume[:, None, None, None, None] * self.unit[:, :, None, None, None] * self.unit[:, None, :, None, None]
            * self.offset[:, None, None, :, None] * self.kernel_gradient[:, None, None, None, :])
        system = np.einsum("ibgt,ita,iaem->ibgem", third, self.correction, mixed) + fourth
        matrix = system.reshape(self.count, dimension * dimension, dimension * dimension)
        right = -np.eye(dimension).reshape(-1)
        solution = np.einsum("iab,b->ia", np.linalg.pinv(matrix), right)
        return solution.reshape(self.count, dimension, dimension)


def lattice_patch(spacing: float, smoothing_length: float, dimension: int, layers: int = 3) -> np.ndarray:
    """A cubic lattice patch around the origin, ``layers`` support radii wide (the centre particle's neighbours and
    their neighbours have full supports for layers >= 3). The centre is row 0."""
    half = int(np.ceil(layers * smoothing_length / spacing))
    axis = np.arange(-half, half + 1) * spacing
    grids = np.meshgrid(*([axis] * dimension), indexing="ij")
    positions = np.column_stack([grid.ravel() for grid in grids])
    centre = int(np.argmin((positions ** 2).sum(axis=1)))
    order = np.concatenate([[centre], np.delete(np.arange(positions.shape[0]), centre)])
    return positions[order]


def lattice_symbol(particles: ParticleSet, name: str, wave_vector: np.ndarray, eps_h_squared: float) -> np.ndarray:
    """The d x d symbol T(k) of an operator on a lattice patch at its centre particle (row 0, at the origin):
    lap_h [e exp(i k . r)] = T(k) e exp(i k . r). Columns = the response to e = unit vectors."""
    phase = np.exp(1j * particles.positions @ wave_vector[:particles.dimension])
    symbol = np.zeros((particles.dimension, particles.dimension), dtype=complex)
    for component in range(particles.dimension):
        field = np.zeros((particles.count, particles.dimension), dtype=complex)
        field[:, component] = phase
        symbol[:, component] = particles.laplacian(name, field, eps_h_squared)[0]
    return symbol


# ============================================================================
# The comparison (micropolar_tests.py operators-host)
# ============================================================================

def _plane_wave_ratio(particles, name, wave_vector, polarisation, eps_h_squared) -> complex:
    symbol = lattice_symbol(particles, name, wave_vector, eps_h_squared)
    return complex(polarisation @ symbol @ polarisation) / -float(wave_vector @ wave_vector)


def lattice_measurements(smoothing_length_2d: float, spacing_2d: float, smoothing_length_3d: float,
                         spacing_3d: float, epsilon_factors) -> dict:
    """Measurements 1 - 3 on the regular lattice (exact: the symbol at one particle of an infinite lattice)."""
    results = {"long_wave_2d": [], "long_wave_3d": [], "spectrum_2d": [], "longitudinal": []}
    positions = lattice_patch(spacing_2d, smoothing_length_2d, 2)
    lattice_2d = ParticleSet(positions, np.full(positions.shape[0], spacing_2d ** 2), smoothing_length_2d, 2)
    positions = lattice_patch(spacing_3d, smoothing_length_3d, 3)
    lattice_3d = ParticleSet(positions, np.full(positions.shape[0], spacing_3d ** 3), smoothing_length_3d, 3)
    small = 0.01
    for factor in epsilon_factors:
        eps_2d, eps_3d = factor * smoothing_length_2d ** 2, factor * smoothing_length_3d ** 2
        for angle in (0.0, 11.25, 22.5, 33.75, 45.0):
            radians = np.radians(angle)
            direction = np.array([np.cos(radians), np.sin(radians)])
            transverse = np.array([-np.sin(radians), np.cos(radians)])
            row = {"epsilon_factor": factor, "angle": angle}
            for name in OPERATORS:
                row[name] = _plane_wave_ratio(lattice_2d, name, small / smoothing_length_2d * direction, transverse,
                                              eps_2d).real
            results["long_wave_2d"].append(row)
        for label, direction, transverse in (
                ("[100], v along [010]", (1, 0, 0), (0, 1, 0)),
                ("[110], v along [1-10]", (1, 1, 0), (1, -1, 0)), ("[110], v along [001]", (1, 1, 0), (0, 0, 1)),
                ("[111], v along [1-10]", (1, 1, 1), (1, -1, 0))):
            direction = np.array(direction, float) / np.linalg.norm(direction)
            transverse = np.array(transverse, float) / np.linalg.norm(transverse)
            row = {"epsilon_factor": factor, "direction": label}
            for name in OPERATORS:
                row[name] = _plane_wave_ratio(lattice_3d, name, small / smoothing_length_3d * direction, transverse,
                                              eps_3d).real
            results["long_wave_3d"].append(row)
        # longitudinal wave (v parallel to k): MG approximates lap v + 2 grad div v, so 3; the others 1
        for label, particles, length, direction, eps in (
                ("2-D, axis", lattice_2d, smoothing_length_2d, (1, 0), eps_2d),
                ("2-D, diagonal", lattice_2d, smoothing_length_2d, (1, 1), eps_2d),
                ("3-D, [100]", lattice_3d, smoothing_length_3d, (1, 0, 0), eps_3d),
                ("3-D, [111]", lattice_3d, smoothing_length_3d, (1, 1, 1), eps_3d)):
            direction = np.array(direction, float) / np.linalg.norm(direction)
            row = {"epsilon_factor": factor, "case": label}
            for name in OPERATORS:
                row[name] = _plane_wave_ratio(particles, name, small / length * direction, direction, eps).real
            results["longitudinal"].append(row)
    # the whole spectrum, 2-D, default epsilon (the first factor): |k| dx / pi from long waves to the zone boundary
    eps_2d = epsilon_factors[0] * smoothing_length_2d ** 2
    for label, angle, fractions in (("axis", 0.0, (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
                                    ("diagonal", 45.0, (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2,
                                                        np.sqrt(2.0)))):
        radians = np.radians(angle)
        direction = np.array([np.cos(radians), np.sin(radians)])
        transverse = np.array([-np.sin(radians), np.cos(radians)])
        for fraction in fractions:
            row = {"direction": label, "k_dx_over_pi": float(fraction)}
            for name in OPERATORS:
                row[name] = _plane_wave_ratio(lattice_2d, name, fraction * np.pi / spacing_2d * direction, transverse,
                                              eps_2d).real
            results["spectrum_2d"].append(row)
    return results


def developed_measurements(positions, volumes, is_fluid, smoothing_length, spacing, epsilon_factors,
                           regularization_xi: float = 0.0, half_extent: float = 0.5, wave_counts=(2, 4, 8)) -> dict:
    """Measurements 4 - 7 on a developed 2-D arrangement in the cavity [-half_extent, half_extent]^2 (walls
    outside). The field is e exp(i k . r) on every particle (walls included: they are outside every evaluated
    support); factor(k) = the least-squares projection of the result onto the expected mode -k^2 e exp(i k . r)."""
    particles = ParticleSet(positions, volumes, smoothing_length, 2, regularization_xi)
    x, y = particles.positions[:, 0], particles.positions[:, 1]
    wall_distance = {"left": x + half_extent, "right": half_extent - x, "bottom": y + half_extent,
                     "top": half_extent - y}
    nearest = np.minimum.reduce(list(wall_distance.values()))
    bulk = is_fluid & (nearest > 3.0 * smoothing_length)
    quadrants = [bulk & (x < 0) & (y < 0), bulk & (x >= 0) & (y < 0), bulk & (x < 0) & (y >= 0),
                 bulk & (x >= 0) & (y >= 0)]
    bands = {}
    for wall, distance in wall_distance.items():
        others = np.minimum.reduce([value for name, value in wall_distance.items() if name != wall])
        bands[wall] = (is_fluid & (distance > smoothing_length) & (distance <= 3.0 * smoothing_length)
                       & (others > smoothing_length))
    normals = {"left": (1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, 1.0), "top": (0.0, 1.0)}
    result = {"particles": int(particles.count), "fluid": int(is_fluid.sum()), "bulk": int(bulk.sum()),
              "band": {wall: int(mask.sum()) for wall, mask in bands.items()},
              "smallest_pair_distance_over_dx": particles.smallest_pair_distance() / spacing,
              "regularization_xi": regularization_xi, "factors": [], "noise": [], "exactness": []}
    wave_numbers = [2.0 * np.pi * count / (2.0 * half_extent) for count in wave_counts]

    def factors(name, eps, direction, blocks):
        """Per block and wave number: (factor, relative residual rms)."""
        direction = np.asarray(direction, float)
        transverse = np.array([-direction[1], direction[0]])
        table = np.zeros((len(blocks), len(wave_numbers)))
        noise = np.zeros((len(blocks), len(wave_numbers)))
        for column, wave_number in enumerate(wave_numbers):
            mode = np.exp(1j * wave_number * (particles.positions @ direction))
            field = mode[:, None] * transverse[None, :]
            response = particles.laplacian(name, field, eps)
            expected = -wave_number ** 2 * field
            for row, block in enumerate(blocks):
                coefficient = (np.vdot(expected[block], response[block]) / np.vdot(expected[block], expected[block]))
                table[row, column] = coefficient.real
                residual = response[block] - coefficient * expected[block]
                noise[row, column] = np.sqrt((np.abs(residual) ** 2).sum() / (np.abs(expected[block]) ** 2).sum())
        return table, noise

    def intercept(table):
        """k -> 0 intercept of a least-squares line in k^2, per block; mean and standard error over the blocks."""
        design = np.column_stack([np.ones(len(wave_numbers)), np.square(wave_numbers)])
        values = np.array([np.linalg.lstsq(design, row, rcond=None)[0][0] for row in table])
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))

    for factor in epsilon_factors:
        eps = factor * smoothing_length ** 2
        for name in OPERATORS:
            if factor != epsilon_factors[0] and not HAS_EPSILON[name]:
                continue
            for angle in (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5):
                radians = np.radians(angle)
                table, noise = factors(name, eps, (np.cos(radians), np.sin(radians)), quadrants)
                mean, error = intercept(table)
                result["factors"].append({"operator": name, "epsilon_factor": factor, "region": "bulk",
                                          "direction": angle, "intercept": mean, "standard_error": error,
                                          "per_wave_count": table.mean(axis=0).tolist()})
                result["noise"].append({"operator": name, "epsilon_factor": factor, "region": "bulk",
                                        "direction": angle, "per_wave_count": noise.mean(axis=0).tolist()})
            for orientation in ("normal", "parallel"):
                tables, noises = [], []
                for wall, mask in bands.items():
                    normal = np.array(normals[wall])
                    direction = normal if orientation == "normal" else np.array([-normal[1], normal[0]])
                    table, noise = factors(name, eps, direction, [mask])
                    tables.append(table[0])
                    noises.append(noise[0])
                mean, error = intercept(np.array(tables))
                result["factors"].append({"operator": name, "epsilon_factor": factor, "region": "wall band",
                                          "direction": orientation, "intercept": mean, "standard_error": error,
                                          "per_wave_count": np.mean(tables, axis=0).tolist()})
                result["noise"].append({"operator": name, "epsilon_factor": factor, "region": "wall band",
                                        "direction": orientation, "per_wave_count": np.mean(noises, axis=0).tolist()})
    # exactness: a linear field and a rigid rotation; rms |lap_h v| in units of |grad v| / h
    eps = epsilon_factors[0] * smoothing_length ** 2
    all_bands = np.logical_or.reduce(list(bands.values()))
    for label, tensor in (("linear", np.array([[0.3, 0.7], [-0.4, 0.5]])),
                          ("rigid rotation", np.array([[0.0, -1.0], [1.0, 0.0]]))):
        field = particles.positions @ tensor.T
        scale = np.linalg.norm(tensor) / smoothing_length
        for name in OPERATORS:
            response = particles.laplacian(name, field, eps)
            magnitude = np.sqrt((response ** 2).sum(axis=1)) / scale
            result["exactness"].append({"operator": name, "field": label,
                                        "bulk_rms": float(np.sqrt((magnitude[bulk] ** 2).mean())),
                                        "band_rms": float(np.sqrt((magnitude[all_bands] ** 2).mean()))})
    return result
