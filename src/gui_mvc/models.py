import pickle
from pathlib import Path
from typing import Tuple
import numpy as np


def load_data(theme: str, ticker: str, month: str):
    """Load stock and ETF volatility surfaces from pickle files."""
    base = Path("data/vol_surfaces") / theme
    stock_path = base / f"{ticker}_{month}.pkl"
    med_path = base / f"ETF_MEDIAN_{month}.pkl"
    low_path = base / f"ETF_LOW_{month}.pkl"
    high_path = base / f"ETF_HIGH_{month}.pkl"

    def _load(path: Path):
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    stock = _load(stock_path)
    median = _load(med_path)
    low = _load(low_path)
    high = _load(high_path)
    return stock, median, low, high


def atm_normalize(surface: dict) -> dict:
    """Normalize a surface by its ATM slice."""
    K = np.asarray(surface["K"], dtype=float)
    IV = np.asarray(surface["IV"], dtype=float)
    idx_atm = np.argmin(np.abs(K - np.median(K)))
    atm = IV[idx_atm, :]
    IV = IV / atm
    return {"K": K, "T": surface["T"], "IV": IV}


def to_moneyness(surface: dict) -> Tuple[dict, float]:
    """Convert a surface to moneyness coordinates."""
    K = np.asarray(surface["K"], dtype=float)
    s0 = np.median(K)
    return {"K": K / s0, "T": surface["T"], "IV": surface["IV"]}, s0

