#!/usr/bin/env python3
"""
Pipeline Debugger (real-data, GUI-lite).

Goals
-----
- Exercise the *same* code-paths the GUI uses, but from CLI.
- Minimal interactive loop: list dates, pick asof, build smiles/term,
  compute peer weights, build composite, and dump intermediate data.
- Focused debug logging (opt-in) + warning-to-error to reveal NaN stacks.

Usage
-----
# Basic (download/ingest, build, then interactive loop)
python scripts/pipeline_debugger.py --tickers SPY QQQ --max-expiries 6 --interactive

# Specify a target + peers and weight mode up front
python scripts/pipeline_debugger.py --tickers SPY QQQ XLK \
    --target SPY --peers QQQ XLK --weight-mode corr_iv_atm --interactive

# Run with verbose debug + raise on warnings (good for tracing NaNs)
ANALYSIS_DEBUG=1 python scripts/pipeline_debugger.py --tickers CI UNH --interactive

Commands (inside REPL)
----------------------
help               - show commands
dates              - list available dates (global and per ticker)
asof YYYY-MM-DD    - set active asof date (default: most recent)
smile [T_days]     - build smile data (default uses nearest expiry to 30d)
term               - build ATM term curve (+ optional composite overlay)
weights            - compute peer weights with current config
composite              - build composite ATM pillar series (weights required)
dump [what]        - write current artifacts to /tmp parquet/csv (see options)
quit / exit        - leave
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---- Make project importable when run from repo root or scripts/ ---------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- Project imports ------------------------------------------------------
from analysis.analysis_pipeline import (  # noqa: E402
    PipelineConfig,
    ingest_and_process,
    build_surfaces,
    list_surface_dates,
    surface_to_frame_for_date,
    available_tickers,
    available_dates,
    get_most_recent_date_global,
    compute_peer_weights,
    build_composite_iv_series_weighted,
    build_composite_iv_series_corrweighted,
    prepare_smile_data,
    prepare_term_data,
)

# -----------------------------------------------------------------------------
# Logging & warnings
# -----------------------------------------------------------------------------
logger = logging.getLogger("pipeline_debugger")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)

if os.environ.get("ANALYSIS_DEBUG", "").strip().lower() in ("1", "true", "yes"):
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Raise warnings as errors optionally (surfacing NaN stacktraces)
if os.environ.get("RAISE_WARNINGS", "").strip().lower() in ("1", "true", "yes"):
    warnings.filterwarnings("error")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _df_info(df: pd.DataFrame, cols: Iterable[str] = ()):
    if df is None:
        return "None"
    if df.empty:
        return "empty"
    cols = list(cols) or list(df.columns[:6])
    na_bits = []
    for c in cols:
        if c in df.columns:
            na_bits.append(f"{c}:na={df[c].isna().mean()*100:.1f}%")
    return f"rows={len(df):,}, cols={df.shape[1]} | " + ", ".join(na_bits)


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class Session:
    tickers: List[str]
    target: Optional[str] = None
    peers: List[str] = field(default_factory=list)
    weight_mode: str = "corr_iv_atm"
    max_expiries: int = 6
    cfg: PipelineConfig = field(default_factory=PipelineConfig)

    # runtime state
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = field(default_factory=dict)
    asof: Optional[str] = None
    smile_last: Dict[str, object] | None = None
    term_last: Dict[str, object] | None = None
    weights_last: pd.Series | None = None
    composite_iv_last: pd.DataFrame | None = None

    def ensure_asof(self):
        if self.asof:
            return self.asof
        date = get_most_recent_date_global()
        self.asof = date
        return self.asof


# -----------------------------------------------------------------------------
# Core run steps
# -----------------------------------------------------------------------------
def run_ingest(sess: Session):
    logger.info("Ingesting tickers: %s (max_expiries=%s)", ",".join(sess.tickers), sess.max_expiries)
    rows = ingest_and_process(sess.tickers, max_expiries=sess.max_expiries)
    logger.info("Inserted rows: %s", rows)


def run_build_surfaces(sess: Session, most_recent_only: bool = True):
    logger.info("Building surfaces (most_recent_only=%s)...", most_recent_only)
    sess.surfaces = build_surfaces(sess.tickers, cfg=sess.cfg, most_recent_only=most_recent_only)
    counts = {t: len(d) for t, d in sess.surfaces.items()}
    logger.info("Surfaces dates per ticker: %s", counts)
    if most_recent_only:
        sess.asof = get_most_recent_date_global()
        logger.info("Active asof set to most recent: %s", sess.asof)


def run_smile(sess: Session, T_days: float | None = 30.0):
    if not sess.target:
        logger.warning("No target set. Use --target or 'asof YYYY-MM-DD' then set .target in CLI flags.")
        return
    asof = sess.ensure_asof()
    logger.info("Preparing smile: target=%s asof=%s T_days=%s", sess.target, asof, T_days)
    data = prepare_smile_data(
        target=sess.target,
        asof=asof,
        T_days=T_days if T_days is not None else 30.0,
        overlay_composite=False,
        peers=sess.peers,
        weights=(None if sess.weights_last is None else sess.weights_last.to_dict()),
        overlay_peers=True if sess.peers else False,
        max_expiries=sess.max_expiries,
    )
    sess.smile_last = data or {}
    if not data:
        logger.warning("Smile data is empty.")
        return

    T_arr = data["T_arr"]; K_arr = data["K_arr"]; sigma_arr = data["sigma_arr"]
    logger.info("Smile arrays: T=%s K=%s sigma=%s", T_arr.shape, K_arr.shape, sigma_arr.shape)
    fit = data.get("fit_info", {})
    logger.info("Fit info (SVI keys=%s SABR keys=%s): %s / %s",
                list(fit.get("svi", {}).keys())[:5], list(fit.get("sabr", {}).keys())[:5],
                "expiry="+str(fit.get("expiry")), "asof="+str(fit.get("asof")))


def run_term(sess: Session, ci: float = 68.0):
    if not sess.target:
        logger.warning("No target set.")
        return
    asof = sess.ensure_asof()
    logger.info("Preparing term structure: target=%s asof=%s", sess.target, asof)
    out = prepare_term_data(
        target=sess.target, asof=asof, ci=ci,
        peers=sess.peers,
        weights=(None if sess.weights_last is None else sess.weights_last.to_dict()),
        max_expiries=sess.max_expiries,
    )
    sess.term_last = out or {}
    atm = out.get("atm_curve", pd.DataFrame())
    composite = out.get("composite_curve", pd.DataFrame())
    logger.info("ATM curve: %s", _df_info(atm, cols=("T", "atm_vol")))
    if not composite is None:
        logger.info("composite ATM curve: %s", _df_info(composite, cols=("T", "atm_vol")))


def run_weights(sess: Session, asof: Optional[str] = None):
    if not sess.target or not sess.peers:
        logger.warning("Need target and peers to compute weights.")
        return
    asof = asof or sess.ensure_asof()
    logger.info("Computing weights: target=%s peers=%s mode=%s asof=%s",
                sess.target, ",".join(sess.peers), sess.weight_mode, asof)
    w = compute_peer_weights(
        target=sess.target,
        peers=sess.peers,
        weight_mode=sess.weight_mode,
        asof=asof,
    )
    sess.weights_last = w
    logger.info("Weights:\n%s", w.to_string())


def run_composite_iv(sess: Session, pillar_days: Tuple[int, ...] | int = (7, 30, 60, 90), tolerance_days: float = 7.0):
    if sess.weights_last is None and (not sess.target or not sess.peers):
        logger.warning("Need weights or (target+peers) to build composite.")
        return

    if sess.weights_last is None:
        logger.info("No precomputed weights; computing on-the-fly for composite...")
        df, w = build_composite_iv_series_corrweighted(
            target=sess.target, peers=sess.peers, weight_mode=sess.weight_mode,
            pillar_days=pillar_days, tolerance_days=tolerance_days, asof=sess.ensure_asof(),
        )
        sess.weights_last = w
        sess.composite_iv_last = df
    else:
        df = build_composite_iv_series_weighted(
            weights=sess.weights_last.to_dict(),
            pillar_days=pillar_days, tolerance_days=tolerance_days,
        )
        sess.composite_iv_last = df

    logger.info("composite ATM pillars: %s", _df_info(sess.composite_iv_last, cols=("pillar_days", "iv")))


def list_dates(sess: Session):
    if not sess.surfaces:
        logger.info("No surfaces built yet.")
        return
    dates = list_surface_dates(sess.surfaces)
    logger.info("All unique dates across tickers: %s", [pd.to_datetime(d).date().isoformat() for d in dates])
    for t, dct in sess.surfaces.items():
        logger.info("Ticker %s has %d dates; example latest=%s", t, len(dct), max(dct.keys()) if dct else None)


def dump(sess: Session, what: str = "smile"):
    outdir = Path("/tmp")  # adjust if needed on Windows
    outdir.mkdir(parents=True, exist_ok=True)

    if what == "smile" and sess.smile_last:
        df = pd.DataFrame({
            "T": pd.Series(sess.smile_last["T_arr"]).astype(float),
            "K": pd.Series(sess.smile_last["K_arr"]).astype(float),
            "sigma": pd.Series(sess.smile_last["sigma_arr"]).astype(float),
            "S": pd.Series(sess.smile_last["S_arr"]).astype(float),
        })
        p = outdir / "debug_smile_slice.csv"
        df.to_csv(p, index=False)
        logger.info("Wrote %s (%s)", p, _df_info(df, cols=("T", "K", "sigma")))
        return

    if what == "term" and sess.term_last:
        atm = sess.term_last.get("atm_curve", pd.DataFrame())
        p = outdir / "debug_atm_curve.parquet"
        if not atm.empty:
            atm.to_parquet(p, index=False)
            logger.info("Wrote %s (%s)", p, _df_info(atm, cols=("T", "atm_vol")))
        return

    if what == "composite" and sess.composite_iv_last is not None:
        p = outdir / "debug_composite_pillars.parquet"
        sess.composite_iv_last.to_parquet(p, index=False)
        logger.info("Wrote %s (%s)", p, _df_info(sess.composite_iv_last))
        return

    if what == "surface":
        if not sess.surfaces:
            logger.info("No surfaces loaded.")
            return
        asof = sess.ensure_asof()
        frames = []
        for t, dct in sess.surfaces.items():
            if asof in dct:
                g = dct[asof].copy()
                g.insert(0, "mny_bin", g.index.astype(str))
                tidy = g.melt(id_vars="mny_bin", var_name="tenor_days", value_name="iv")
                tidy["ticker"] = t
                tidy["asof_date"] = pd.to_datetime(asof).strftime("%Y-%m-%d")
                frames.append(tidy)
        if frames:
            df = pd.concat(frames, ignore_index=True)
            p = outdir / "debug_surface_tidy.parquet"
            df.to_parquet(p, index=False)
            logger.info("Wrote %s (%s)", p, _df_info(df))
        return

    logger.info("Nothing to dump for '%s'. Options: smile | term | composite | surface", what)


# -----------------------------------------------------------------------------
# REPL
# -----------------------------------------------------------------------------
def repl(sess: Session):
    print("\nPipeline Debugger REPL. Type 'help' for commands.")
    while True:
        try:
            raw = input("dbg> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            break

        if cmd == "help":
            print("""
help               - show this
dates              - list known dates (global and per ticker)
asof YYYY-MM-DD    - set active asof date (or 'asof latest')
smile [T_days]     - build smile slice & fit (default 30)
term               - build ATM term structure (+ composite if peers/weights set)
weights            - compute peer weights with current config
composite              - build composite ATM pillar series
dump [what]        - write artifacts to /tmp (smile|term|composite|surface)
""")
            continue

        if cmd == "dates":
            list_dates(sess); continue

        if cmd == "asof":
            if len(parts) == 1 or parts[1].lower() == "latest":
                sess.asof = get_most_recent_date_global()
            else:
                sess.asof = parts[1]
            logger.info("Active asof: %s", sess.asof)
            continue

        if cmd == "smile":
            T = _safe_float(parts[1], 30.0) if len(parts) > 1 else 30.0
            run_smile(sess, T_days=T)
            continue

        if cmd == "term":
            run_term(sess); continue

        if cmd == "weights":
            run_weights(sess); continue

        if cmd == "composite":
            run_composite_iv(sess); continue

        if cmd == "dump":
            what = parts[1].lower() if len(parts) > 1 else "smile"
            dump(sess, what=what); continue

        print("Unknown command. Type 'help'.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a real-data pipeline debugger.")
    p.add_argument("--tickers", nargs="+", required=True, help="Tickers to ingest/build.")
    p.add_argument("--target", type=str, default=None, help="Target ticker for smiles/weights.")
    p.add_argument("--peers", nargs="*", default=[], help="Peer tickers.")
    p.add_argument("--weight-mode", type=str, default="corr_iv_atm", help="Weight mode (unified).")
    p.add_argument("--max-expiries", type=int, default=6, help="Max expiries to include per day.")
    p.add_argument("--use-atm-only", action="store_true", help="Surface grids: ATM-only mode.")
    p.add_argument("--interactive", action="store_true", help="Enter REPL after initial run.")
    p.add_argument("--no-ingest", action="store_true", help="Skip ingestion (use existing DB).")
    p.add_argument("--asof", type=str, default=None,
               help="Force an asof date (YYYY-MM-DD). Default=most recent.")
    p.add_argument("--t-days", type=float, default=30.0,
                help="Target maturity in days for smile fitting.")
    p.add_argument("--pillar-days", nargs="*", type=int, default=[7,30,60,90],
                help="Pillar days for composite IV series.")
    p.add_argument("--tolerance-days", type=float, default=7.0,
                help="Tolerance in days when aligning expiries.")
    p.add_argument("--ci", type=float, default=68.0,
                help="Confidence interval level for term/smile bands.")
    p.add_argument("--dump", type=str, choices=["smile","term","composite","surface"], default=None,
                help="Immediately dump an artifact to /tmp after run.")
    p.add_argument("--list-dates", action="store_true",
                help="Just list available dates for tickers and exit.")
    p.add_argument("--plot", action="store_true",
                help="Render matplotlib plots (smile/term) after run.")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = PipelineConfig(use_atm_only=args.use_atm_only, max_expiries=args.max_expiries)

    sess = Session(
        tickers=[t.upper() for t in args.tickers],
        target=(args.target.upper() if args.target else None),
        peers=[p.upper() for p in args.peers],
        weight_mode=args.weight_mode,
        max_expiries=args.max_expiries,
        cfg=cfg,
    )

    if not args.no_ingest:
        run_ingest(sess)
    run_build_surfaces(sess, most_recent_only=True)

    # If target provided, do one pass of smile/term/weights to warm things up
    if sess.target:
        run_smile(sess, T_days=30.0)
        run_term(sess)
        if sess.peers:
            run_weights(sess)

    if args.interactive:
        repl(sess)
    else:
        logger.info("Done. Use --interactive for REPL.")

if __name__ == "__main__":
    main()
