from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "model_params.parquet"

def append_params(
    asof_date: str,
    ticker: str,
    expiry: Optional[str],
    model: str,
    params: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Log fitted parameters to disk."""
    meta = meta or {}
    asof_ts = pd.to_datetime(asof_date).normalize()
    expiry_dt = pd.to_datetime(expiry) if expiry else None
    tenor_d = (expiry_dt - asof_ts).days if expiry_dt is not None else None
    rows = []
    for key, val in params.items():
        try:
            fval = float(val)
        except Exception:
            continue
        rows.append({
            "asof_date": asof_ts,
            "ticker": ticker,
            "expiry": expiry_dt,
            "tenor_d": tenor_d,
            "model": model.lower(),
            "param": key,
            "value": fval,
            "fit_meta": meta,
        })
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if STORE_PATH.exists():
        df_old = pd.read_parquet(STORE_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = (
            df.sort_values(["asof_date","ticker","expiry","model","param"])
              .drop_duplicates(["asof_date","ticker","expiry","model","param"], keep="last")
        )
    else:
        df = df_new
    df.to_parquet(STORE_PATH, index=False)

def load_model_params() -> pd.DataFrame:
    """Load the logged parameters as a DataFrame."""
    if not STORE_PATH.exists():
        return pd.DataFrame(columns=["asof_date","ticker","expiry","tenor_d","model","param","value","fit_meta"])
    df = pd.read_parquet(STORE_PATH)
    df["asof_date"] = pd.to_datetime(df["asof_date"])
    if "expiry" in df:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    return df
