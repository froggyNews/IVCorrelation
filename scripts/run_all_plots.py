#!/usr/bin/env python
"""
Batch runner to generate all key plot variants with one command.

It invokes scripts/generate_plots.py multiple times with sensible presets:
- ATM Corr/PCA
- Surface Grid Corr/PCA
- Underlying Corr/PCA
- Weights grids (PCA and Corr) for iv_atm, surface_grid, ul
- Slide 10 smile overlay (target SVI + CI, thin peer SVI, composite)

Example:
  python scripts/run_all_plots.py \
      --target IONQ --peers PLTR,MSFT,GOOGL \
      --asof 2025-08-28 --out-root outputs

Outputs are written under: {out-root}/{scenario}/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


SCRIPT = Path(__file__).resolve().parents[0] / "generate_plots.py"


def _run(cmd: list[str]) -> int:
    print("\n>>>", " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Error running: {' '.join(cmd)}\n  -> {e}")
        return 1


def main() -> int:
    p = argparse.ArgumentParser(description="Batch generate plots for multiple modes")
    p.add_argument("--target", required=True)
    p.add_argument("--peers", default="", help="Comma-separated peers")
    p.add_argument("--asof", default=None, help="Date YYYY-MM-DD (omit to auto-resolve)")
    p.add_argument("--latest", action="store_true", help="Use most recent date and ignore --asof")
    p.add_argument("--out-root", default="outputs", help="Root output folder for all scenarios")
    p.add_argument("--max-expiries", type=int, default=6)
    p.add_argument("--t-days", type=float, default=30.0, help="Term smile target days")
    p.add_argument("--slide10-days", type=float, default=28.0)
    p.add_argument("--slide10-ci", type=float, default=68.0)
    p.add_argument("--features", default="iv_atm,surface_grid,ul", help="Grid features for weights grids")
    p.add_argument("--skip-slide10", action="store_true")
    p.add_argument("--skip-grids", action="store_true")
    args = p.parse_args()

    target = args.target.upper()
    peers = ",".join([t.strip().upper() for t in args.peers.split(',') if t.strip()]) if args.peers else ""

    ts = datetime.now().strftime("%Y-%m-%d")
    root = Path(args.out_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    common = []  # flags common to all runs
    if args.latest:
        common += ["--latest"]
    elif args.asof:
        common += ["--asof", args.asof]
    common += ["--max-expiries", str(int(args.max_expiries))]

    scenarios = [
        ("atm_corr", "corr_iv_atm"),
        ("atm_pca",  "pca_iv_atm"),
        ("surface_corr", "corr_surface_grid"),
        ("surface_pca",  "pca_surface_grid"),
        ("ul_corr", "corr_ul"),
        ("ul_pca",  "pca_ul"),
    ]

    # Run per-mode scenarios
    for label, mode in scenarios:
        out_dir = root / f"{target}_{label}_{ts}"
        cmd = [
            sys.executable, str(SCRIPT),
            "--target", target,
            "--peers", peers,
            "--weight-mode", mode,
            "--t-days", str(float(args.t_days)),
            "--out-dir", str(out_dir),
        ] + common
        rc = _run(cmd)
        if rc != 0:
            print(f"Warning: scenario {label} exited with code {rc}")

    # Weights grids (PCA and Corr) in a single run for convenience
    if not args.skip_grids and peers:
        out_dir = root / f"{target}_grids_{ts}"
        cmd = [
            sys.executable, str(SCRIPT),
            "--target", target,
            "--peers", peers,
            "--weight-mode", "corr_iv_atm",
            "--weights-grids",
            "--features", args.features,
            "--out-dir", str(out_dir),
        ] + common
        _run(cmd)

    # Slide 10 smile overlay
    if not args.skip_slide10:
        out_dir = root / f"{target}_slide10_{ts}"
        cmd = [
            sys.executable, str(SCRIPT),
            "--target", target,
            "--peers", peers,
            "--weight-mode", "corr_iv_atm",
            "--slide10-smile",
            "--slide10-days", str(float(args.slide10_days)),
            "--slide10-ci", str(float(args.slide10_ci)),
            "--out-dir", str(out_dir),
        ] + common
        _run(cmd)

    print(f"\nAll scenarios done. Root: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

