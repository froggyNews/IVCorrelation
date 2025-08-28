from analysis.beta_builder.unified_weights import compute_unified_weights, normalize
from analysis.beta_builder.correlation import corr_weights


def weights_from_ui_or_matrix(self, target: str, peers: list[str], weight_mode: str, asof=None, pillars=None):
    """
    Single source of truth for peer weights.

    Tries multiple `compute_unified_weights` signatures for backward compat:
    (target, peers, mode=..., asof=..., pillars_days=...)
    (target, peers, mode=..., asof=..., pillar_days=...)
    (target, peers, weight_mode=..., asof=..., pillar_days=...)
    positional: (target, peers, mode, asof, pillar_days)

    Fallbacks to relative-weight matrix-derived weights (if cached meta matches) then equal weights.
    """
    import numpy as np
    import pandas as pd
    from analysis.beta_builder.correlation import corr_weights

    target = (target or "").upper()
    peers = [p.upper() for p in (peers or [])]

    # Use target ticker's actual expiries instead of fixed day pillars
    pillars = self.target_pillars

    settings = getattr(self, "last_settings", {})

    # 1) Unified weights with signature shims
    try:
        
        attempts = (
            lambda: compute_unified_weights(target=target, peers=peers, mode=weight_mode, asof=asof, pillars_days=pillars),
            lambda: compute_unified_weights(target=target, peers=peers, mode=weight_mode, asof=asof, pillar_days=pillars),
            lambda: compute_unified_weights(target=target, peers=peers, weight_mode=weight_mode, asof=asof, pillar_days=pillars),
            lambda: compute_unified_weights(target, peers, weight_mode, asof, pillars),
        )
        for fn in attempts:
            uw = fn()
            nw = normalize(uw, peers)
            if nw is not None:
                return nw

    except Exception as e:
        print(f"Unified weight computation failed: {e}")

    # 2) Relative-weight matrix derived (only if cached meta matches exactly)
    try:
        if (
            isinstance(self.last_relative_weight_df, pd.DataFrame)
            and not self.last_relative_weight_df.empty
            and self.last_relative_weight_meta.get("weight_mode") == weight_mode
            and self.last_relative_weight_meta.get("clip_negative") == settings.get("clip_negative", True)
            and self.last_relative_weight_meta.get("weight_power") == settings.get("weight_power", 1.0)
            and self.last_relative_weight_meta.get("pillars", []) == list(pillars)
            and self.last_relative_weight_meta.get("asof") == asof
            and set(self.last_relative_weight_meta.get("tickers", [])) >= set([target] + peers)
        ):
            w = corr_weights(
                self.last_relative_weight_df,
                self.target,
                self.peers,
                clip_negative=settings.get("clip_negative", True),
                power=settings.get("weight_power", 1.0),
            )
            if w is not None and not w.empty and np.isfinite(w.to_numpy(dtype=float)).any():
                w = w.dropna().astype(float)
                w = w[w.index.isin(self.peers)]
                s = float(w.sum())
                if s > 0 and np.isfinite(s):
                    return (w / s).reindex(self.peers).fillna(0.0).astype(float)
    except Exception:
        pass

    # 3) Equal weights fallback
    eq = 1.0 / max(len(self.peers), 1)
    return pd.Series(eq, index=self.peers, dtype=float)