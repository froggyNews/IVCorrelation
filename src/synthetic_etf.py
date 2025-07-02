import pandas as pd
from typing import Dict, Callable, Optional


def combine_surfaces(
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]],
    rhos: Dict[str, float],
    weight_grids: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Combine normalized surfaces across tickers to form a synthetic ETF surface.

    Parameters
    ----------
    surfaces : dict
        Mapping of ticker -> date -> DataFrame where the index represents strike
        and the columns represent maturity (in days). Each value is the
        implied volatility for that (strike, maturity) pair.
    rhos : dict
        Mapping of ticker -> weighting coefficient (rho).
    weight_grids : dict, optional
        Mapping of ticker -> DataFrame of the same shape as each surface giving
        the weight function :math:`\omega_i(K, T)`. If omitted, weights of 1 are
        used for that ticker.

    Returns
    -------
    dict
        Mapping of date -> DataFrame representing the synthetic ETF volatility
        surface on that date.
    """

    all_dates = set()
    for surf_by_date in surfaces.values():
        all_dates.update(surf_by_date.keys())

    result = {}
    for date in sorted(all_dates):
        numerator = None
        denominator = None
        for ticker, surf_by_date in surfaces.items():
            if date not in surf_by_date:
                continue
            sigma = surf_by_date[date]
            rho = rhos.get(ticker, 1.0)
            weight_grid = None
            if weight_grids is not None and ticker in weight_grids:
                weight_grid = weight_grids[ticker]
            if weight_grid is None:
                weight_grid = pd.DataFrame(
                    1.0, index=sigma.index, columns=sigma.columns
                )
            contrib_num = rho * weight_grid * sigma
            contrib_den = rho * weight_grid

            if numerator is None:
                numerator = contrib_num
                denominator = contrib_den
            else:
                numerator = numerator.add(contrib_num, fill_value=0)
                denominator = denominator.add(contrib_den, fill_value=0)

        if numerator is not None:
            result[date] = numerator.divide(denominator)
    return result
