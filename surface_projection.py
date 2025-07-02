import numpy as np
from scipy.interpolate import griddata


def project_surfaces(surfaces, strikes, maturities, n):
    """Project implied volatility surfaces onto a global (K, T) grid.

    Parameters
    ----------
    surfaces : list of numpy.ndarray
        List containing IV surfaces for each asset. Each surface should be a
        2-D array shaped ``(len(strikes[i]), len(maturities[i]))``.
    strikes : list of numpy.ndarray
        Strike coordinates for each surface. ``strikes[i]`` should have shape
        ``(len(strikes[i]),)``.
    maturities : list of numpy.ndarray
        Maturity coordinates for each surface in **days**. ``maturities[i]``
        should have shape ``(len(maturities[i]),)``.
    n : int
        Desired grid size for both strike and maturity dimensions.

    Returns
    -------
    tuple
        ``(global_strike_grid, global_maturity_grid, projected_surfaces)``
        where ``projected_surfaces`` is a list of ``n x n`` arrays containing
        the surfaces sampled on the common grid.
    """
    if not surfaces:
        raise ValueError("surfaces list cannot be empty")
    if not (len(surfaces) == len(strikes) == len(maturities)):
        raise ValueError("surfaces, strikes and maturities must have same length")

    # Determine global ranges across all assets
    k_min = min(float(np.min(s)) for s in strikes)
    k_max = max(float(np.max(s)) for s in strikes)
    t_min = min(float(np.min(t)) for t in maturities)
    t_max = max(float(np.max(t)) for t in maturities)

    global_strikes = np.linspace(k_min, k_max, n)
    global_maturities = np.linspace(t_min, t_max, n)
    kk, tt = np.meshgrid(global_strikes, global_maturities, indexing="ij")
    target_points = np.column_stack([kk.ravel(), tt.ravel()])

    projected = []
    for surf, k, t in zip(surfaces, strikes, maturities):
        sk, mt = np.meshgrid(k, t, indexing="ij")
        points = np.column_stack([sk.ravel(), mt.ravel()])
        values = surf.ravel()

        # Primary interpolation using linear method
        interp = griddata(points, values, target_points, method="linear")
        interp = interp.reshape(n, n)

        # Fill missing values using nearest neighbor
        mask = np.isnan(interp)
        if np.any(mask):
            nearest = griddata(points, values, target_points[mask.ravel()], method="nearest")
            interp[mask] = nearest

        # Conservative fallback if there are still NaNs (e.g., constant 0)
        interp = np.nan_to_num(interp, nan=0.0)
        projected.append(interp)

    return global_strikes, global_maturities, projected
