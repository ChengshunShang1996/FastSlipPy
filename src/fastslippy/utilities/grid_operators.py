"""Polynomial-exact operators for the staggered finite-difference grid.

These routines are the Python counterpart of ``fdweights``, ``centred3``,
``onesided3``, and ``recovery_operators`` in the BP3 MATLAB reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy import sparse


def finite_difference_weights(
    target: float, points: np.ndarray, derivative: int
) -> np.ndarray:
    """Return polynomial-exact weights for a derivative at ``target``.

    The coordinates are scaled before solving the small moment system.  This
    keeps the three- and four-point systems used by BP3 well conditioned when
    coordinates are measured in kilometres.
    """

    points = np.asarray(points, dtype=float).reshape(-1)
    n = points.size
    if derivative < 0 or derivative >= n:
        raise ValueError(
            f"need at least {derivative + 1} points for derivative order "
            f"{derivative}; got {n}"
        )

    offsets = points - float(target)
    scale = float(np.max(np.abs(offsets)))
    if scale == 0.0:
        raise ValueError("finite-difference points are degenerate")

    scaled = offsets / scale
    system = np.empty((n, n), dtype=float)
    for power in range(n):
        system[power, :] = scaled**power / factorial(power)
    rhs = np.zeros(n, dtype=float)
    rhs[derivative] = 1.0
    return np.linalg.solve(system, rhs) / scale**derivative


def centred_first_weights(h_minus: float, h_plus: float) -> np.ndarray:
    """Three-point first derivative exact for quadratics."""

    return finite_difference_weights(
        0.0, np.array([-h_minus, 0.0, h_plus]), 1
    )


def one_sided_first_weights(h1: float, h2: float) -> np.ndarray:
    """Forward three-point first derivative exact for quadratics."""

    return finite_difference_weights(
        0.0, np.array([0.0, h1, h1 + h2]), 1
    )


def node_derivative_matrix(coords: np.ndarray) -> sparse.csr_matrix:
    """First derivative at nodes using three points at every row."""

    coords = np.asarray(coords, dtype=float).reshape(-1)
    n = coords.size
    if n < 3:
        raise ValueError(f"need at least 3 nodes; got {n}")

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for index in range(n):
        if index == 0:
            stencil = np.arange(3)
        elif index == n - 1:
            stencil = np.arange(n - 3, n)
        else:
            stencil = np.arange(index - 1, index + 2)
        weights = finite_difference_weights(
            coords[index], coords[stencil], 1
        )
        rows.extend([index] * 3)
        cols.extend(stencil.tolist())
        values.extend(weights.tolist())
    return sparse.csr_matrix((values, (rows, cols)), shape=(n, n))


def midpoint_to_node_matrix(coords: np.ndarray) -> sparse.csr_matrix:
    """Distance-weighted interpolation from n+1 midpoints to n nodes."""

    coords = np.asarray(coords, dtype=float).reshape(-1)
    n = coords.size
    if n < 2:
        raise ValueError(f"need at least 2 nodes; got {n}")
    spacing = np.diff(coords)
    padded = np.concatenate(([spacing[0]], spacing, [spacing[-1]]))
    left_weight = padded[1 : n + 1] / (padded[:n] + padded[1 : n + 1])
    rows = np.repeat(np.arange(n), 2)
    cols = np.column_stack((np.arange(n), np.arange(1, n + 1))).reshape(-1)
    values = np.column_stack((left_weight, 1.0 - left_weight)).reshape(-1)
    return sparse.csr_matrix((values, (rows, cols)), shape=(n, n + 1))


def centres_to_nodes_matrix(
    coords: np.ndarray, staggered: np.ndarray
) -> sparse.csr_matrix:
    """Map n-1 cell-centre values to n nodes.

    Interior rows use distance-weighted linear interpolation.  The two end
    rows use quadratic extrapolation from the nearest three cell centres.
    """

    coords = np.asarray(coords, dtype=float).reshape(-1)
    staggered = np.asarray(staggered, dtype=float).reshape(-1)
    n = coords.size
    n_centres = n - 1
    if staggered.size != n + 1:
        raise ValueError(
            f"expected {n + 1} staggered coordinates; got {staggered.size}"
        )
    if n_centres < 3:
        raise ValueError(f"need at least 3 cell centres; got {n_centres}")

    centre_coords = staggered[1:n]
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    end_weights = finite_difference_weights(coords[0], centre_coords[:3], 0)
    rows.extend([0] * 3)
    cols.extend([0, 1, 2])
    values.extend(end_weights.tolist())

    spacing = np.diff(coords)
    for index in range(1, n - 1):
        weight_above = spacing[index] / (spacing[index - 1] + spacing[index])
        rows.extend([index, index])
        cols.extend([index - 1, index])
        values.extend([weight_above, 1.0 - weight_above])

    end_weights = finite_difference_weights(
        coords[-1], centre_coords[-3:], 0
    )
    rows.extend([n - 1] * 3)
    cols.extend([n_centres - 3, n_centres - 2, n_centres - 1])
    values.extend(end_weights.tolist())

    return sparse.csr_matrix(
        (values, (rows, cols)), shape=(n, n_centres)
    )


@dataclass(frozen=True)
class RecoveryOperators:
    """Sparse staggered-grid operators used by MATLAB BP3 stress recovery."""

    derivative_x: sparse.csr_matrix
    derivative_y: sparse.csr_matrix
    midpoint_x_to_node: sparse.csr_matrix
    midpoint_y_to_node: sparse.csr_matrix
    sigma_centres_to_nodes: sparse.csr_matrix


def build_recovery_operators(
    x: np.ndarray, y: np.ndarray, xp: np.ndarray, yp: np.ndarray
) -> RecoveryOperators:
    """Build the complete MATLAB-equivalent stress-recovery operator set."""

    return RecoveryOperators(
        derivative_x=node_derivative_matrix(x),
        derivative_y=node_derivative_matrix(y),
        midpoint_x_to_node=midpoint_to_node_matrix(x),
        midpoint_y_to_node=midpoint_to_node_matrix(y),
        sigma_centres_to_nodes=centres_to_nodes_matrix(y, yp),
    )
