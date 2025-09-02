from analysis.compositeETFBuilder import combine_surfaces
import pandas as pd
from analysis.compositeETFBuilder import build_surface_grids
from analysis.pillars import get_feasible_expiry_pillars, get_target_expiry_pillars
from analysis.model_params_logger import compute_or_load, WarmupWorker


def get_target_and_composite_grid(self, target: str, peers: list[str], asof: str, weights: dict[str, float] | None, max_expiries: int):
    """
    Returns (tgt_grid, syn_grid, date_used) where date_used is a pd.Timestamp actually present in both.
    If exact `asof` not present on one side, falls back to each side's latest and uses their intersection if possible.
    """
    tickers = list({target, *peers})
    surfaces = self._get_surface_grids(tickers, max_expiries)  # dict[ticker][date]->DF

    if not surfaces or target not in surfaces or not surfaces[target]:
        return None, None, None

    # build composite by date from peer surfaces if weights provided
    composite_by_date = {}
    if peers and weights:
        composite_surfaces = {t: surfaces[t] for t in peers if t in surfaces}
        try:
            composite_by_date = combine_surfaces(composite_surfaces, weights)
        except Exception:
            composite_by_date = {}

    # choose a date
    asof_ts = pd.to_datetime(asof).floor("min")
    tgt_dates = sorted(surfaces[target].keys())
    syn_dates = sorted(composite_by_date.keys())

    # prefer common date
    common = sorted(set(tgt_dates).intersection(syn_dates))
    date_used = None
    if common:
        date_used = common[-1]
    else:
        # fallback: prefer target date at/near asof, else latest target; and pick closest composite date
        date_used = asof_ts if asof_ts in surfaces[target] else (tgt_dates[-1] if tgt_dates else None)
        if date_used and date_used not in composite_by_date and syn_dates:
            # pick latest composite date as fallback
            date_used = syn_dates[-1] if date_used is None else date_used

    if date_used is None:
        return None, None, None

    tgt_grid = surfaces[target].get(date_used)
    syn_grid = composite_by_date.get(date_used)
    return tgt_grid, syn_grid, date_used

def get_surface_grids(self, tickers, max_expiries):
    """Return surface grids for ``tickers`` using cache if available."""
    key = (tuple(sorted(set(tickers))), int(max_expiries))
    if key not in self._surface_cache:
        payload = {"tickers": list(key[0]), "max_expiries": key[1]}

        def _builder():
            return build_surface_grids(
                tickers=list(key[0]),
                tenors=None,
                mny_bins=None,
                use_atm_only=False,
                max_expiries=key[1],
            )

        try:
            grids = compute_or_load("surface_grids", payload, _builder)
        except Exception:
            grids = _builder()
        self._surface_cache[key] = grids if grids is not None else {}
    return self._surface_cache.get(key, {})

def get_target_pillars(self, target: str, asof: str, max_expiries: int = 6, peers: list[str] = None) -> list[int]:
    """Get feasible expiry pillars that work across target and peers."""
    if peers:
        # Use flexible approach that considers peer coverage
        return get_feasible_expiry_pillars(
            get_smile_slice=self.get_smile_slice,  # Use bounded slicer
            target_ticker=target,
            peer_tickers=peers,
            asof=asof,
            max_expiries=max_expiries,
            tol_days=14,  # 7 day tolerance for expiry matching
            min_peer_coverage=0.3  # At least 50% of peers must be able to match each pillar
        )
    else:
        # No peers, just use target expiries
        return get_target_expiry_pillars(
            get_smile_slice=self.get_smile_slice,  # Use bounded slicer
            target_ticker=target,
            asof=asof,
            max_expiries=max_expiries
        )
    return []
# -------------------- main entry --------------------
