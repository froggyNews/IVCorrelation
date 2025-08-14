# src/spillover/spillover_engine.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# --------------------------
# 0) Utilities
# --------------------------
def _zscore_changes(df: pd.DataFrame, iv_col: str, lookback: int = 60) -> pd.DataFrame:
    df = df.sort_values(["ticker","date"]).copy()
    df["dIV"] = df.groupby("ticker")[iv_col].diff()
    roll_std = (
        df.groupby("ticker")["dIV"]
          .rolling(lookback, min_periods=20)
          .std()
          .reset_index(level=0, drop=True)
    )
    df["zIV"] = df["dIV"] / roll_std
    return df

def _build_peers(df: pd.DataFrame, iv_col: str, lookback: int = 60, top_k: int = 5,
                 restrict_same_sector: bool = False) -> Dict[str, List[str]]:
    """
    Build related set R(i) as top-K correlations of ΔIV over a rolling lookback
    using the most recent window per ticker.
    """
    # compute ΔIV on last L days per ticker
    last_dates = df["date"].max()
    window_start = df["date"].sort_values().drop_duplicates().iloc[-lookback]
    w = df[(df["date"] >= window_start) & (df["date"] <= last_dates)].copy()
    w = w.sort_values(["ticker","date"])
    w["dIV"] = w.groupby("ticker")[iv_col].diff()
    piv = w.pivot(index="date", columns="ticker", values="dIV")
    corr = piv.corr(min_periods=int(0.5*lookback))  # pairwise corr of ΔIV

    peers: Dict[str, List[str]] = {}
    for tkr in corr.columns:
        s = corr[tkr].drop(index=tkr).dropna()
        if restrict_same_sector and "sector" in df.columns:
            sector = df.loc[df["ticker"] == tkr, "sector"].mode().iloc[0] if \
                     (df["ticker"] == tkr).any() else None
            same = df[df["sector"] == sector]["ticker"].unique().tolist()
            s = s[s.index.isin(same)]
        peers[tkr] = s.sort_values(ascending=False).head(top_k).index.tolist()
    return peers

def _residualize_common_shocks(df: pd.DataFrame, iv_col: str,
                               common_cols: List[str]) -> pd.DataFrame:
    """
    Residualize ΔIV against common shocks (e.g., ΔVIX, ΔSectorIV).
    Works per-ticker with rolling OLS betas to avoid lookahead.
    """
    out = df.sort_values(["ticker","date"]).copy()
    out["dIV"] = out.groupby("ticker")[iv_col].diff()

    for c in common_cols:
        out[f"d{c}"] = out.groupby("ticker")[c].diff() if c not in ("dVIX","dSECTOR") else out[c]

    Xcols = [f"d{c}" if f"d{c}" in out.columns else c for c in common_cols]
    # Rolling betas via expanding window after a warmup
    def _fit_resid(g: pd.DataFrame, warmup: int = 60):
        g = g.copy()
        g["resid_dIV"] = np.nan
        for idx in range(len(g)):
            if idx < warmup:
                continue
            y = g["dIV"].iloc[:idx].values
            X = g[Xcols].iloc[:idx].values
            X = np.c_[np.ones(len(X)), X]  # intercept
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                # residual at idx
                x_now = np.r_[1.0, g[Xcols].iloc[idx].values]
                g.loc[g.index[idx], "resid_dIV"] = g["dIV"].iloc[idx] - x_now @ beta
            except Exception:
                pass
        return g
    out = out.groupby("ticker", group_keys=False).apply(_fit_resid)
    return out

# --------------------------
# 1) Event detection
# --------------------------
def detect_iv_events(df: pd.DataFrame,
                     iv_col: str = "atm_iv",
                     z_thresh: float = 2.0,
                     lookback: int = 60,
                     use_residuals: bool = True,
                     common_cols: Optional[List[str]] = None,
                     min_gap_days: int = 3) -> pd.DataFrame:
    """
    Returns a DataFrame of trigger events with columns:
    ['date','ticker','dIV_trigger','z_trigger','sign']
    """
    data = df.copy()
    if use_residuals and common_cols:
        r = _residualize_common_shocks(data, iv_col, common_cols)
        data["dIV_eff"] = r["resid_dIV"]
    else:
        z = _zscore_changes(data, iv_col, lookback)
        data["dIV_eff"] = z["dIV"]

    # z-score of effective changes (if we residualized, recompute z on resid)
    data = data.sort_values(["ticker","date"])
    roll_std = (
        data.groupby("ticker")["dIV_eff"]
            .rolling(lookback, min_periods=20).std()
            .reset_index(level=0, drop=True)
    )
    data["z_eff"] = data["dIV_eff"] / roll_std

    events = (
        data.loc[np.abs(data["z_eff"]) >= z_thresh,
                 ["date","ticker","dIV_eff","z_eff"]]
            .rename(columns={"dIV_eff":"dIV_trigger","z_eff":"z_trigger"})
            .copy()
    )
    # Debounce multiple hits: keep the first hit, then block next min_gap_days for that ticker
    events = events.sort_values(["ticker","date"])
    keep_idx = []
    last_date_by_ticker = {}
    for i, row in events.iterrows():
        tkr, d = row["ticker"], row["date"]
        if (tkr not in last_date_by_ticker) or ((d - last_date_by_ticker[tkr]).days > min_gap_days):
            keep_idx.append(i)
            last_date_by_ticker[tkr] = d
    events = events.loc[keep_idx]
    events["sign"] = np.sign(events["dIV_trigger"]).astype(int)
    return events.reset_index(drop=True)

# --------------------------
# 2) Compute peer responses
# --------------------------
def compute_spillovers(df: pd.DataFrame,
                       events: pd.DataFrame,
                       peers: Dict[str, List[str]],
                       iv_col: str = "atm_iv",
                       horizons: List[int] = [0,1,3,5,10],
                       response_iv_col: Optional[str] = None) -> pd.DataFrame:
    """
    For each (event i at t0) and peer j in R(i), compute responses over horizons.
    response_iv_col lets you choose 'atm_iv' (synthetic) vs a raw/bumpy column.
    """
    iv_resp = response_iv_col or iv_col
    panel = df.set_index(["date","ticker"]).sort_index()

    rows = []
    for _, e in events.iterrows():
        t0 = e["date"]; i = e["ticker"]
        j_peers = peers.get(i, [])
        # base levels at t0-1 (or earliest available)
        t_minus1 = panel.loc[: (t0,), :].index.get_level_values(0).max()
        for j in j_peers:
            try:
                base = panel.loc[(t_minus1, j), iv_resp]
            except KeyError:
                continue
            for h in horizons:
                t_h = panel.index.get_level_values(0).unique().searchsorted(t0) + h
                dates = panel.index.get_level_values(0).unique()
                if t_h >= len(dates):
                    continue
                d_h = dates[t_h]
                if (d_h, j) not in panel.index:
                    continue
                resp = panel.loc[(d_h, j), iv_resp] - base
                rows.append({
                    "trigger_ticker": i,
                    "peer": j,
                    "t0": t0,
                    "h": h,
                    "trigger_dIV": e["dIV_trigger"],
                    "trigger_z": e["z_trigger"],
                    "resp_dIV": resp
                })
    return pd.DataFrame(rows)

# --------------------------
# 3) Summaries & metrics
# --------------------------
def summarize_spillovers(sp: pd.DataFrame,
                         df: pd.DataFrame,
                         iv_col: str = "atm_iv",
                         threshold_mode: str = "sigma",  # or "bps"
                         threshold_value: float = 1.0
                         ) -> pd.DataFrame:
    """
    Compute hit-rate, sign-concordance, mean/median response, elasticity by horizon.
    If threshold_mode == 'sigma', threshold_value means 1 * rolling σ of peer ΔIV.
    """
    # Build per-peer rolling σ to set thresholds
    tmp = df.sort_values(["ticker","date"]).copy()
    tmp["dIV"] = tmp.groupby("ticker")[iv_col].diff()
    tmp["sigma"] = (
        tmp.groupby("ticker")["dIV"]
           .rolling(60, min_periods=20).std()
           .reset_index(level=0, drop=True)
    )
    sigma_map = tmp.groupby(["date","ticker"])["sigma"].last()

    sp = sp.copy()
    # Map each (t0+h, peer) sigma
    sp["date_h"] = sp["t0"]  # base date; horizons vary, but we used levels array in compute step
    key_index = sigma_map.index
    # Conservative: use the last available sigma for peer (ignoring exact h-date mismatch)
    last_sigma_by_peer = tmp.groupby("ticker")["sigma"].last().to_dict()
    sp["peer_sigma"] = sp["peer"].map(last_sigma_by_peer)

    if threshold_mode == "sigma":
        sp["thr"] = threshold_value * sp["peer_sigma"].replace({0: np.nan})
    else:  # bps: treat IV in decimals; convert threshold_value bps -> decimal
        sp["thr"] = np.abs(threshold_value) / 10000.0

    def _agg(g):
        hr = np.mean(np.abs(g["resp_dIV"]) >= g["thr"]) if len(g) else np.nan
        sc = np.mean(np.sign(g["resp_dIV"]) == np.sign(g["trigger_dIV"])) if len(g) else np.nan
        mean_resp = np.nanmean(g["resp_dIV"])
        med_resp  = np.nanmedian(g["resp_dIV"])
        # elasticity conditioned on nonzero trigger
        valid = g[np.abs(g["trigger_dIV"]) > 1e-12]
        elast = np.nanmedian(valid["resp_dIV"] / valid["trigger_dIV"]) if len(valid) else np.nan
        return pd.Series({
            "hit_rate": hr,
            "sign_concord": sc,
            "mean_resp": mean_resp,
            "median_resp": med_resp,
            "elasticity_med": elast,
            "n": len(g)
        })

    summary = sp.groupby("h").apply(_agg).reset_index()
    return summary
