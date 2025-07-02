"""Utilities for thin-plate spline smoothing of implied volatility surfaces.

This module provides helper functions to fit a thin-plate spline surface to
sparse implied volatility data points. The implementation avoids third party
dependencies to work in constrained environments.
"""

from typing import Callable, Iterable, Tuple, List
import math


Point = Tuple[float, float, float]  # (K, T, sigma)


def _gauss_solve(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve a linear system using Gaussian elimination."""
    n = len(b)
    # Forward elimination
    for i in range(n):
        # Pivot
        pivot = a[i][i]
        if abs(pivot) < 1e-12:
            # find non-zero pivot below
            for j in range(i + 1, n):
                if abs(a[j][i]) > 1e-12:
                    a[i], a[j] = a[j], a[i]
                    b[i], b[j] = b[j], b[i]
                    pivot = a[i][i]
                    break
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix")
        inv_pivot = 1.0 / pivot
        for j in range(i, n):
            a[i][j] *= inv_pivot
        b[i] *= inv_pivot
        for k in range(i + 1, n):
            factor = a[k][i]
            for j in range(i, n):
                a[k][j] -= factor * a[i][j]
            b[k] -= factor * b[i]
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i] - sum(a[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s
    return x


def fit_thin_plate_spline(points: Iterable[Point]) -> Callable[[float, float], float]:
    """Fit a thin-plate spline surface to the given points.

    Parameters
    ----------
    points:
        Iterable of (K, T, sigma) tuples representing strikes, maturities and
        observed volatilities.

    Returns
    -------
    Callable[[float, float], float]
        A function f(K, T) that evaluates the smooth surface.
    """
    pts = list(points)
    n = len(pts)
    if n < 3:
        raise ValueError("Need at least 3 points to fit a surface")

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]

    m = n + 3
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m

    # Fill radial basis component
    for i in range(n):
        for j in range(n):
            if i == j:
                r = 0.0
            else:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                r = math.sqrt(dx * dx + dy * dy)
            A[i][j] = 0.0 if r == 0 else r * r * math.log(r)
        A[i][n] = 1.0
        A[i][n + 1] = xs[i]
        A[i][n + 2] = ys[i]
        b[i] = zs[i]

    # Polynomial constraints
    for j in range(n):
        A[n][j] = 1.0
        A[n + 1][j] = xs[j]
        A[n + 2][j] = ys[j]

    # Solve system
    weights = _gauss_solve(A, b)

    def surface(x: float, y: float) -> float:
        result = weights[n] + weights[n + 1] * x + weights[n + 2] * y
        for i in range(n):
            dx = x - xs[i]
            dy = y - ys[i]
            r = math.sqrt(dx * dx + dy * dy)
            term = 0.0 if r == 0 else r * r * math.log(r)
            result += weights[i] * term
        return result

    return surface


def fit_surfaces(
    low_points: Iterable[Point],
    med_points: Iterable[Point],
    high_points: Iterable[Point],
) -> Tuple[Callable[[float, float], float], Callable[[float, float], float], Callable[[float, float], float]]:
    """Fit thin-plate spline surfaces for low, medium, and high vol data."""
    f_low = fit_thin_plate_spline(low_points)
    f_med = fit_thin_plate_spline(med_points)
    f_high = fit_thin_plate_spline(high_points)
    return f_low, f_med, f_high

