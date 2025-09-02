#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
from typing import Iterable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root on sys.path BEFORE importing local packages
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# === Use your db_utils (no direct sqlite calls) ===
from data.db_utils import (
    get_conn,
    ensure_initialized,
    check_db_health,
    get_most_recent_date,
)
from analysis.beta_builder.unified_weights import compute_unified_weights
from display.plotting.weights_plot import plot_weights

# ----------------------------
# Config
# ----------------------------
TARGET = "IONQ"
PEER_POOL = ["RGTI", "QUBT", "QBTS", "IONQ"]     # pool includes target; we'll exclude it for the composite
PILLARS_D = [7, 14, 21, 28, 30, 56, 94, 112]     # includes 30 DTE focus
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def _resolve_asof(conn, requested: Optional[str], ticker_hint: Optional[str]) -> Optional[str]:
    if requested:
        return str(pd.to_datetime(requested).date())
    hint = ticker_hint or (PEER_POOL[0] if PEER_POOL else None)
    d = get_most_recent_date(conn, hint) if hint else get_most_recent_date(conn, None)
    return d

def _load_quotes_df(conn, tickers: Iterable[str], asof: str) -> pd.DataFrame:
    """Query the minimal columns we need from options_quotes using the provided connection."""
    tix = [t.upper() for t in tickers]
    placeholders = ",".join(["?"] * len(tix))
    sql = f"""
        SELECT
            asof_date,
            UPPER(ticker) AS ticker,
            expiry,
            strike           AS K,
            call_put,
            iv               AS sigma,
            spot             AS S,
            ttm_years        AS T,
            moneyness,
            log_moneyness,
            delta,
            is_atm
        FROM options_quotes
        WHERE asof_date = ?
          AND UPPER(ticker) IN ({placeholders})
    """
    df = pd.read_sql_query(sql, conn, params=[asof, *tix])
    if df.empty:
        return df
    # sanitize types
    df["T"] = pd.to_numeric(df["T"], errors="coerce")
    df["sigma"] = pd.to_numeric(df["sigma"], errors="coerce")
    for c in ("moneyness", "log_moneyness", "delta", "S", "K"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_atm"] = df.get("is_atm", 0).astype(bool)
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return df.dropna(subset=["T", "sigma"])

def _pick_atm_per_expiry(dft: pd.DataFrame) -> pd.DataFrame:
    """One ATM row per expiry: prefer is_atm; else min |moneyness-1|; else delta~0.5; else median strike."""
    rows = []
    for exp, g in dft.groupby("expiry"):
        g = g.copy()
        if "is_atm" in g and g["is_atm"].any():
            best = g[g["is_atm"]].iloc[0]
        elif "moneyness" in g and g["moneyness"].notna().any():
            best = g.loc[(g["moneyness"] - 1.0).abs().idxmin()]
        elif "delta" in g and g["delta"].notna().any():
            best = g.loc[(g["delta"] - 0.5).abs().idxmin()]
        else:
            medK = g["K"].median()
            best = g.loc[(g["K"] - medK).abs().idxmin()]
        rows.append(best)
    out = pd.DataFrame(rows)
    return out.sort_values("T").reset_index(drop=True)[["ticker", "expiry", "K", "S", "T", "sigma"]]

def _interp_at_pillars(atm_rows: pd.DataFrame, pillars_d: list[int]) -> pd.Series:
    """Linear interpolation of sigma along maturity; clip beyond ends."""
    if atm_rows.empty:
        return pd.Series({int(d): np.nan for d in pillars_d})
    x = (atm_rows["T"].to_numpy(float) * 365.25)
    y = atm_rows["sigma"].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return pd.Series({int(d): np.nan for d in pillars_d})
    o = np.argsort(x); x, y = x[o], y[o]
    z = np.interp(pillars_d, x, y, left=y[0], right=y[-1])
    return pd.Series({int(d): float(v) for d, v in zip(pillars_d, z)})

def _diag_plots(df: pd.DataFrame, asof: str) -> None:
    """Small plots/tables to show data source and quickly flag gaps."""
    # Spot (median)
    sp = df.groupby("ticker")["S"].median().reindex(PEER_POOL)
    plt.figure(figsize=(5.2, 3.0)); sp.plot(kind="bar")
    plt.title(f"Spot (median) — {asof}"); plt.ylabel("Spot"); plt.tight_layout()
    plt.savefig(OUT_DIR / f"diag_spot_{asof}.png", dpi=150); plt.close()

    # ATM expiries recovered
    atm_counts = []
    for t in PEER_POOL:
        dft = df[df["ticker"] == t]
        atm = _pick_atm_per_expiry(dft) if not dft.empty else pd.DataFrame()
        atm_counts.append((t, len(atm)))
    atm_df = pd.DataFrame(atm_counts, columns=["ticker", "atm_expiries"])
    atm_df.to_csv(OUT_DIR / f"diag_atm_counts_{asof}.csv", index=False)
    plt.figure(figsize=(5.2, 3.0))
    plt.bar(atm_df["ticker"], atm_df["atm_expiries"])
    plt.title(f"ATM expiries recovered — {asof}"); plt.ylabel("# expiries"); plt.tight_layout()
    plt.savefig(OUT_DIR / f"diag_atm_counts_{asof}.png", dpi=150); plt.close()

    # Null rate
    core = df[["T", "sigma", "K", "S"]].copy()
    null_rate = core.isna().mean() * 100
    null_rate.to_csv(OUT_DIR / f"diag_null_rate_{asof}.csv")
    plt.figure(figsize=(5.2, 3.0)); null_rate.plot(kind="bar")
    plt.title(f"Null % (core cols) — {asof}"); plt.ylabel("% null"); plt.tight_layout()
    plt.savefig(OUT_DIR / f"diag_null_rate_{asof}.png", dpi=150); plt.close()

# ----------------------------
# Main: IONQ vs Peer Composite (equal-weight)
# ----------------------------
def main(asof: Optional[str] = None, pool: Iterable[str] = PEER_POOL) -> None:
    target = TARGET.upper()
    pool = [p.upper() for p in pool]
    # peers for composite EXCLUDE the target
    peers = [p for p in pool if p != target]

    conn = get_conn()
    ensure_initialized(conn)
    check_db_health(conn)

    asof_res = _resolve_asof(conn, asof, ticker_hint=target if target in pool else (pool[0] if pool else None))
    if not asof_res:
        raise SystemExit("No asof date available in DB.")

    # Load quotes for both target and peers
    df = _load_quotes_df(conn, [target] + peers, asof_res)
    if df.empty:
        raise SystemExit(f"No quotes for {asof_res} and tickers={[target]+peers}.")

    # Diagnostics
    _diag_plots(df, asof_res)

    # Compute per-ticker ATM series and 30D numbers
    per_rows: List[Dict] = []
    iv_at_pillar: Dict[str, pd.Series] = {}
    for t in [target] + peers:
        dft = df[df["ticker"] == t]
        if dft.empty:
            iv_at_pillar[t] = pd.Series({d: np.nan for d in PILLARS_D})
            per_rows.append({
                "Ticker": t, "Spot": np.nan,
                "Original_TTE_List": "(none)",
                "Comparison_Strike_30D": np.nan,
                "IV_30D": np.nan
            })
            continue

        atm = _pick_atm_per_expiry(dft)
        ttes = [int(round(T * 365.25)) for T in atm["T"].tolist()]
        iv_series = _interp_at_pillars(atm, PILLARS_D)
        iv_at_pillar[t] = iv_series

        # comparison strike: ATM row nearest to 30 DTE
        idx30 = int(np.abs((atm["T"].to_numpy(float) * 365.25) - 30.0).argmin()) if not atm.empty else 0
        compK = float(atm.iloc[idx30]["K"]) if not atm.empty else np.nan
        spot = float(dft["S"].median()) if len(dft) else np.nan

        per_rows.append({
            "Ticker": t,
            "Spot": spot,
            "Original_TTE_List": ",".join(map(str, ttes)) if ttes else "(none)",
            "Comparison_Strike_30D": compK,
            "IV_30D": float(iv_series.get(30, np.nan)),
        })

    # Equal weights across PEERS ONLY (exclude target)
    w = 1.0 / max(len(peers), 1)
    for r in per_rows:
        r["Weight"] = (w if r["Ticker"] in peers else 0.0)
        r["Weighted_IV_30D"] = (r["IV_30D"] * r["Weight"]) if np.isfinite(r["IV_30D"]) else np.nan

    per_df = pd.DataFrame(per_rows, columns=[
        "Ticker", "Spot", "Original_TTE_List", "Comparison_Strike_30D",
        "IV_30D", "Weight", "Weighted_IV_30D"
    ])

    # --- Composite from peers only (by pillar) ---
    # If no peers (edge case), composite is NaN series
    if peers:
        comp_series = pd.concat([iv_at_pillar[p].rename(p) for p in peers], axis=1).mean(axis=1, skipna=True)
    else:
        comp_series = pd.Series({d: np.nan for d in PILLARS_D})

    # --- Build comparison (IONQ vs Composite) ---
    ionq_series = iv_at_pillar.get(target, pd.Series({d: np.nan for d in PILLARS_D}))
    compare_df = pd.DataFrame({
        "DTE": PILLARS_D,
        "IONQ_IV": [float(ionq_series.get(d, np.nan)) for d in PILLARS_D],
        "Composite_IV": [float(comp_series.get(d, np.nan)) for d in PILLARS_D],
    })
    compare_df["Diff"] = compare_df["Composite_IV"] - compare_df["IONQ_IV"]

    # Add Composite_IV_30D to the per-ticker table (same scalar for all rows)
    composite_30 = float(comp_series.get(30, np.nan))
    per_df["Composite_IV_30D"] = composite_30

    # Save + print
    per_df.to_csv(OUT_DIR / f"ionq_vs_peers_per_ticker_30d_{asof_res}.csv", index=False)
    compare_df.to_csv(OUT_DIR / f"ionq_vs_composite_smile_{asof_res}.csv", index=False)

    # --- Plot: IONQ vs Composite ATM smiles ---
    plt.figure(figsize=(6.8, 4.0))
    plt.plot(compare_df["DTE"], compare_df["IONQ_IV"], marker="o", label="IONQ ATM")
    plt.plot(compare_df["DTE"], compare_df["Composite_IV"], marker="o", label="Composite ATM (EW peers)")
    plt.title(f"IONQ vs Equal-Weight Peer Composite — ATM Smile @ {asof_res}")
    plt.xlabel("DTE")
    plt.ylabel("Implied Vol")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ionq_vs_composite_atm_{asof_res}.png", dpi=160)
    plt.close()

    # ---------------------------------------------------------------
    # (Optional) Slide 10 weights grid demo kept intact (unchanged)
    # ---------------------------------------------------------------
    try:
        unrelated_peers = ["CVS", "T"]
        asof_w = asof_res
        import matplotlib as mpl
        plt.style.use("seaborn-v0_8-whitegrid")
        mpl.rcParams.update({
            "figure.facecolor": "#f6f9ff",
            "axes.facecolor": "#f6f9ff",
            "axes.labelcolor": "#0d2b52",
            "axes.titlecolor": "#0d2b52",
        })
        features = [("iv_atm", "ATM"), ("surface_grid", "Surface"), ("ul", "UL")]

        # PCA grid
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        for ax, (feat, label) in zip(axes, features):
            try:
                mode = f"pca_{feat}"
                wts = compute_unified_weights(target=TARGET, peers=unrelated_peers, mode=mode, asof=asof_w)
                plot_weights(ax, wts); ax.set_title(f"PCA • {label}")
            except Exception as e:
                ax.clear(); ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
        fig.suptitle(f"Slide 10 • PCA Weights • {TARGET} @ {asof_w}")
        fig.savefig(OUT_DIR / f"slide10_weights_grid_pca_{asof_w}.png", dpi=160); plt.close(fig)

        # Correlation grid
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        for ax, (feat, label) in zip(axes, features):
            try:
                mode = f"corr_{feat}"
                wts = compute_unified_weights(target=TARGET, peers=unrelated_peers, mode=mode, asof=asof_w)
                plot_weights(ax, wts); ax.set_title(f"Correlation • {label}")
            except Exception as e:
                ax.clear(); ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
        fig.suptitle(f"Slide 10 • Correlation Weights • {TARGET} @ {asof_w}")
        fig.savefig(OUT_DIR / f"slide10_weights_grid_corr_{asof_w}.png", dpi=160); plt.close(fig)
    except Exception as e:
        print(f"Slide 10 plotting error: {e}")

    # Console output
    print(f"\nAs-of: {asof_res}")
    print("\n=== Per-Ticker ATM @ ~30D (IONQ target; peers equal-weight) ===")
    print(per_df.to_string(index=False))
    print("\n=== IONQ vs Equal-Weight Peer Composite (ATM by DTE) ===")
    print(compare_df.to_string(index=False))
    print(f"\nSaved tables/plots to {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
