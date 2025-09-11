#!/usr/bin/env python3
"""
Correlation walkthrough: ATM term-structure correlation between IONQ and RGTI.

What this does (simple, reproducible steps):
- Resolves the requested as-of date to the nearest available trading day in the DB
- Extracts per-expiry ATM vol for each ticker for that date
- Interpolates those ATM vols onto common DTE pillars
- Computes Pearson/Spearman correlations across pillars (with and without demeaning)
- Saves small CSVs and a quick plot under `outputs/`

Usage:
  python scripts/correlation_example.py 2025-08-20
  python scripts/correlation_example.py            # uses most recent date in DB

Notes:
- Uses DB at data/iv_data.db via data.db_utils
- Falls back to the nearest available date if the requested one is missing
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analysis_pipeline import available_dates, get_smile_slice
from analysis.pillars import compute_atm_by_expiry


# ----------------------------
# Config
# ----------------------------
TICKERS = ["IONQ", "QBTS"]
PILLARS_DAYS = [7, 14, 21, 28, 35, 49, 84, 112]
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def _resolve_asof_nearest(requested: Optional[str], tickers: Iterable[str]) -> Optional[str]:
    """Pick the calendar date in DB closest to `requested` that exists for ALL tickers.

    If `requested` is None, use the most recent date common to all tickers.
    """
    tix = [t.upper() for t in tickers]
    per = [available_dates(t) for t in tix]
    # intersect lists as sets then sort
    common = sorted(set(per[0]).intersection(*per[1:])) if per and all(per) else []
    if not common:
        return None
    if not requested:
        return common[-1]
    try:
        req = pd.to_datetime(requested).normalize()
    except Exception:
        return common[-1]
    cands = pd.to_datetime(pd.Series(common)).dt.normalize()
    j = int(np.argmin(np.abs((cands - req).dt.days)))
    return str(cands.iloc[j].date())


def _atm_series_for_ticker_date(ticker: str, asof: str) -> pd.DataFrame:
    """Return DataFrame with columns [T_days, atm_vol] for one ticker on one date.

    - Pulls day slice via get_smile_slice
    - Computes ATM vol per expiry using a robust median/fit fallback
    - Outputs sorted by T
    """
    df = get_smile_slice(ticker, asof, T_target_years=None, max_expiries=None)
    if df is None or df.empty:
        return pd.DataFrame(columns=["T_days", "atm_vol"])  # empty

    # Use a fit with automatic fallback to median/poly if needed
    atm = compute_atm_by_expiry(df, method="fit", model="auto", vega_weighted=True, n_boot=0)
    if atm is None or atm.empty:
        return pd.DataFrame(columns=["T_days", "atm_vol"])  # empty

    out = pd.DataFrame({
        "T_days": (pd.to_numeric(atm["T"], errors="coerce") * 365.25).astype(float),
        "atm_vol": pd.to_numeric(atm["atm_vol"], errors="coerce").astype(float),
    }).dropna()
    out = out[np.isfinite(out["T_days"]) & np.isfinite(out["atm_vol"])].sort_values("T_days").reset_index(drop=True)
    return out


def _interp_to_pillars(curve: pd.DataFrame, pillars_days: List[int]) -> pd.Series:
    """Linear interpolation in DTE-days with edge clipping."""
    if curve is None or curve.empty:
        return pd.Series({int(d): np.nan for d in pillars_days})
    x = curve["T_days"].to_numpy(float)
    y = curve["atm_vol"].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return pd.Series({int(d): np.nan for d in pillars_days})
    o = np.argsort(x); x, y = x[o], y[o]
    z = np.interp(pillars_days, x, y, left=y[0], right=y[-1])
    return pd.Series({int(d): float(v) for d, v in zip(pillars_days, z)})


def _corr_across_pillars(a: pd.Series, b: pd.Series) -> Tuple[float, float, float, float]:
    """Return (pearson, spearman, pearson_demean, spearman_demean) across common pillars."""
    df = pd.concat([a.rename("A"), b.rename("B")], axis=1).dropna()
    if df.empty or len(df) < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    pear = float(df["A"].corr(df["B"], method="pearson"))
    spear = float(df["A"].corr(df["B"], method="spearman"))
    A0 = df["A"] - df["A"].mean()
    B0 = df["B"] - df["B"].mean()
    pear_dm = float(A0.corr(B0, method="pearson"))
    spear_dm = float(A0.corr(B0, method="spearman"))
    return pear, spear, pear_dm, spear_dm


# ----------------------------
# Main
# ----------------------------
def main(argv: List[str]) -> int:
    req = argv[1] if len(argv) > 1 else None
    asof = _resolve_asof_nearest(req, TICKERS)
    if not asof:
        print("No common as-of dates available in DB for:", TICKERS)
        return 2
    if req and req != asof:
        print(f"Requested as-of {req} not found; using nearest available {asof}.")
    else:
        print(f"Using as-of {asof}.")

    # 1) Build ATM curves per ticker
    curves = {}
    for t in TICKERS:
        c = _atm_series_for_ticker_date(t, asof)
        curves[t] = c
        csv_path = OUT_DIR / f"atm_curve_{t}_{asof}.csv"
        c.to_csv(csv_path, index=False)
        print(f"Saved ATM per-expiry curve for {t} -> {csv_path}")

    # 2) Interpolate to common pillars
    at_pillars = {t: _interp_to_pillars(curves[t], PILLARS_DAYS) for t in TICKERS}
    pillars_df = pd.DataFrame({t: at_pillars[t] for t in TICKERS})
    pillars_df.index.name = "DTE"
    pillars_csv = OUT_DIR / f"atm_pillars_{'-'.join(TICKERS)}_{asof}.csv"
    pillars_df.to_csv(pillars_csv)
    print(pillars_df)
    print(f"Saved ATM-by-pillar table -> {pillars_csv}")

    # 3) Correlations across pillars
    pear, spear, pear_dm, spear_dm = _corr_across_pillars(at_pillars[TICKERS[0]], at_pillars[TICKERS[1]])
    print("\nCorrelation across pillars (common DTEs):")
    print(f"- Pearson:          {pear:.4f}" if np.isfinite(pear) else "- Pearson:          NaN")
    print(f"- Spearman:         {spear:.4f}" if np.isfinite(spear) else "- Spearman:         NaN")
    print(f"- Pearson (demean): {pear_dm:.4f}" if np.isfinite(pear_dm) else "- Pearson (demean): NaN")
    print(f"- Spearman (demean):{spear_dm:.4f}" if np.isfinite(spear_dm) else "- Spearman (demean):NaN")

    # 4) Quick visualization
    plt.figure(figsize=(6.4, 4.0))
    for t, style in zip(TICKERS, ["o-", "s-"]):
        plt.plot(pillars_df.index, pillars_df[t], style, label=f"{t} ATM")
    plt.title(f"ATM IV vs DTE — {TICKERS[0]} vs {TICKERS[1]} @ {asof}")
    plt.xlabel("DTE (days)")
    plt.ylabel("Implied Vol")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png = OUT_DIR / f"atm_pillars_{'-'.join(TICKERS)}_{asof}.png"
    plt.savefig(png, dpi=150)
    plt.close()
    print(f"Saved plot -> {png}")

    # Also write a small text summary
    summary = OUT_DIR / f"atm_corr_summary_{'-'.join(TICKERS)}_{asof}.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"As-of: {asof}\n")
        f.write("Tickers: " + ",".join(TICKERS) + "\n\n")
        f.write("Correlations across pillars (common DTEs)\n")
        f.write(f"Pearson: {pear}\n")
        f.write(f"Spearman: {spear}\n")
        f.write(f"Pearson_demean: {pear_dm}\n")
        f.write(f"Spearman_demean: {spear_dm}\n")
    print(f"Saved summary -> {summary}")

    # Guidance for interpretation
    print("\nInterpretation tips:")
    print("- High raw correlations imply both level and shape co-move.")
    print("- High de-meaned correlations isolate term-structure SHAPE similarity (ignoring level).")
    print("- If raw >> de-meaned, levels dominate; if close, shapes co-move strongly.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
