from typing import Optional, Dict, Tuple, List
from collections import defaultdict
from statistics import median

Row = Dict[str, float | str]
Grid = Dict[Tuple[str, str], Dict[float, Dict[float, float]]]


def _closest_strike(strikes: List[float], target: float) -> float:
    """Return strike value closest to target."""
    return min(strikes, key=lambda k: abs(k - target))


def normalize_surface(
    rows: List[Row],
    *,
    strike_col: str = "strike",
    vol_col: str = "implied_vol",
    maturity_col: str = "maturity",
    ticker_col: str = "ticker",
    date_col: str = "date",
    atm_strike: Optional[float] = None,
) -> Grid:
    """Normalize IV surface by ATM implied volatility.

    Parameters
    ----------
    rows: list of dict
        Each row should contain strike, implied vol, maturity, ticker and date.
    atm_strike: float, optional
        Use this strike as ATM when available. If not found for a given maturity
        fallback to the median strike for that maturity.

    Returns
    -------
    dict
        Mapping ``(ticker, date)`` -> ``{maturity: {strike: sigma_norm}}``.
    """

    grouped: Dict[Tuple[str, str, float], List[Row]] = defaultdict(list)
    for row in rows:
        key = (str(row[ticker_col]), str(row[date_col]), float(row[maturity_col]))
        grouped[key].append(row)

    result: Grid = defaultdict(lambda: defaultdict(dict))

    for (ticker, day, maturity), group_rows in grouped.items():
        strikes = [float(r[strike_col]) for r in group_rows]
        vols = [float(r[vol_col]) for r in group_rows]
        if atm_strike is not None and atm_strike in strikes:
            atm = atm_strike
        else:
            atm = _closest_strike(strikes, median(strikes))
        sigma_atm = vols[strikes.index(atm)] if atm in strikes else vols[0]
        for strike, vol in zip(strikes, vols):
            result[(ticker, day)][maturity][strike] = vol / sigma_atm
    return result
