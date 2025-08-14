"""
Animation utilities for IV correlation analysis with robust error handling.

This module provides functions for creating animated plots of implied volatility data,
including smile evolution over time, surface animations, and spillover visualizations.

Key improvements for robustness:
- Graceful handling of matplotlib blitting failures
- Automatic fallback from blitted to non-blitted animations
- Backend-aware animation creation to prevent crashes
- Exception handling in update functions to prevent animation crashes

The functions are designed to work reliably across different matplotlib backends,
especially TkAgg which can be problematic for blitted animations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.legend import Legend
from typing import Dict, List, Tuple, Iterable


def animate_smile_over_time(
    k: np.ndarray,
    iv_tk: np.ndarray,
    dates: Iterable[str],
    *,
    interval_ms: int = 120,
    iv_raw_tk: np.ndarray | None = None,
    ci_lo_tk: np.ndarray | None = None,
    ci_hi_tk: np.ndarray | None = None,
) -> Tuple[plt.Figure, FuncAnimation, Dict[str, List[plt.Artist]]]:
    """Animate smile slices over time.

    Parameters
    ----------
    k : array, shape (N_k,)
        Moneyness grid.
    iv_tk : array, shape (T, N_k)
        Synthetic IV values for each time.
    dates : list of str
        Labels for each frame.
    iv_raw_tk : optional array, shape (T, N_k)
        Raw IV series to overlay.
    ci_lo_tk, ci_hi_tk : optional arrays, shape (T, N_k)
        Confidence interval bounds.

    Returns
    -------
    (fig, ani, artists_dict)
    """

    k = np.asarray(k)
    iv_tk = np.asarray(iv_tk)
    T = iv_tk.shape[0]
    dates = list(dates)

    fig, ax = plt.subplots()

    line_synth, = ax.plot(k, iv_tk[0], label="Synthetic", lw=2)
    artists: Dict[str, List[plt.Artist]] = {"Synthetic": [line_synth]}

    line_raw = None
    if iv_raw_tk is not None:
        iv_raw_tk = np.asarray(iv_raw_tk)
        line_raw, = ax.plot(k, iv_raw_tk[0], label="Raw", lw=1.5, alpha=0.7)
        artists["Raw"] = [line_raw]

    band: PolyCollection | None = None
    if ci_lo_tk is not None and ci_hi_tk is not None:
        ci_lo_tk = np.asarray(ci_lo_tk)
        ci_hi_tk = np.asarray(ci_hi_tk)
        verts = np.column_stack(
            [np.r_[k, k[::-1]], np.r_[ci_hi_tk[0], ci_lo_tk[0][::-1]]]
        )
        band = PolyCollection([verts], facecolor="grey", alpha=0.3, label="CI", closed=False)
        ax.add_collection(band)
        artists["CI"] = [band]

    # Fix limits using robust percentiles across all provided data
    vals = [iv_tk]
    if iv_raw_tk is not None:
        vals.append(iv_raw_tk)
    if ci_lo_tk is not None and ci_hi_tk is not None:
        vals.extend([ci_lo_tk, ci_hi_tk])
    stack = np.vstack([v.reshape(-1) for v in vals])
    lo, hi = np.nanpercentile(stack, [1, 99])
    ax.set_xlim(k.min(), k.max())
    ax.set_ylim(lo, hi)

    def update(i: int):
        try:
            line_synth.set_ydata(iv_tk[i])
            updated = [line_synth]
            if line_raw is not None and iv_raw_tk is not None:
                line_raw.set_ydata(iv_raw_tk[i])
                updated.append(line_raw)
            if band is not None:
                verts = np.column_stack(
                    [np.r_[k, k[::-1]], np.r_[ci_hi_tk[i], ci_lo_tk[i][::-1]]]
                )
                # Safely update band vertices
                if len(band.get_paths()) > 0:
                    band.get_paths()[0].vertices[:] = verts
                    updated.append(band)
            ax.set_title(str(dates[i]))
            return updated
        except Exception:
            # If blitting fails, return empty list to prevent crash
            return []

    # Use safer animation settings to prevent blitting issues
    try:
        ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=True)
    except Exception:
        # Fallback to non-blitted animation if blitting fails
        ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
    
    return fig, ani, artists


def animate_surface_timesweep(
    k: np.ndarray,
    tau: np.ndarray,
    iv_tktau: np.ndarray,
    dates: Iterable[str],
    *,
    interval_ms: int = 120,
) -> Tuple[plt.Figure, FuncAnimation, Dict[str, List[plt.Artist]]]:
    """Animate a surface through time using a single AxesImage."""

    k = np.asarray(k)
    tau = np.asarray(tau)
    iv_tktau = np.asarray(iv_tktau)
    T = iv_tktau.shape[0]
    dates = list(dates)

    fig, ax = plt.subplots()

    vmin, vmax = np.nanpercentile(iv_tktau, [1, 99])
    im = ax.imshow(
        iv_tktau[0],
        origin="lower",
        aspect="auto",
        extent=[tau.min(), tau.max(), k.min(), k.max()],
        vmin=vmin,
        vmax=vmax,
        animated=True,
    )
    ax.set_xlabel("Tau")
    ax.set_ylabel("k")
    artists = {"Surface": [im]}

    def update(i: int):
        try:
            im.set_array(iv_tktau[i])
            ax.set_title(str(dates[i]))
            return [im]
        except Exception:
            # If blitting fails, return empty list to prevent crash
            return []

    # Use safer animation settings to prevent blitting issues
    try:
        ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=True)
    except Exception:
        # Fallback to non-blitted animation if blitting fails
        ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
    
    return fig, ani, artists


def animate_spillover(
    times: Iterable[str],
    peers_xy: Dict[str, Tuple[float, float]],
    responses: Dict[str, np.ndarray],
    *,
    interval_ms: int = 250,
) -> Tuple[plt.Figure, FuncAnimation, Dict[str, List[plt.Artist]], Dict[str, np.ndarray]]:
    """Animate peer spillover responses using a single scatter."""

    labels = list(peers_xy.keys())
    xs = np.array([peers_xy[l][0] for l in labels])
    ys = np.array([peers_xy[l][1] for l in labels])
    times = list(times)
    T = len(times)

    mags = np.vstack([responses[l] for l in labels])  # shape (n, T)
    max_mag = np.nanmax(np.abs(mags)) or 1.0

    fig, ax = plt.subplots()
    base_sizes = np.full(len(labels), 40.0)
    colors = np.repeat(np.array([[0.2, 0.4, 0.8, 0.6]]), len(labels), axis=0)
    scat = ax.scatter(xs, ys, s=base_sizes, facecolors=colors)
    scat._base_sizes = base_sizes.copy()  # type: ignore[attr-defined]
    scat._base_facecolors = colors.copy()  # type: ignore[attr-defined]

    artists = {"Peers": [scat]}

    def update(i: int):
        try:
            resp = mags[:, i]
            sizes = base_sizes * (1.0 + np.abs(resp) / max_mag)
            alpha = np.clip(np.abs(resp) / max_mag, 0.0, 1.0)
            fc = scat.get_facecolors()
            fc[:, 3] = alpha
            scat.set_sizes(sizes)
            scat.set_facecolors(fc)
            ax.set_title(str(times[i]))
            return [scat]
        except Exception:
            # If blitting fails, return empty list to prevent crash
            return []

    # Use safer animation settings to prevent blitting issues
    try:
        ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=True)
    except Exception:
        # Fallback to non-blitted animation if blitting fails
        ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
    
    state = {"labels": labels, "xs": xs, "ys": ys}
    return fig, ani, artists, state


def add_checkboxes(
    fig: plt.Figure,
    series_map: Dict[str, List[plt.Artist]],
    *,
    rect: Tuple[float, float, float, float] = (0.82, 0.35, 0.16, 0.25),
) -> CheckButtons:
    """Attach CheckButtons to toggle visibility of series."""

    axcb = fig.add_axes(rect)
    labels = list(series_map.keys())
    visibility = [series_map[l][0].get_visible() for l in labels]
    check = CheckButtons(axcb, labels, visibility)

    def func(label: str):
        for art in series_map[label]:
            art.set_visible(not art.get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(func)
    return check


def add_keyboard_toggles(
    fig: plt.Figure,
    series_map: Dict[str, List[plt.Artist]],
    keymap: Dict[str, str],
) -> None:
    """Bind key presses to toggle visibility."""

    def on_key(event):
        if event.key in keymap:
            label = keymap[event.key]
            arts = series_map.get(label, [])
            if not arts:
                return
            visible = not arts[0].get_visible()
            for art in arts:
                art.set_visible(visible)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)


def add_legend_toggles(ax: plt.Axes, series_map: Dict[str, List[plt.Artist]]) -> Legend:
    """Make legend entries clickable to toggle series with improved visual feedback."""

    leg = ax.legend()
    fig = ax.figure
    handles = leg.legend_handles if hasattr(leg, "legend_handles") else leg.legendHandles
    texts = leg.get_texts()
    
    # Make handles clickable
    for handle in handles:
        handle.set_picker(True)
        # Only set pickradius if the method exists
        if hasattr(handle, 'set_pickradius'):
            handle.set_pickradius(15)  # Increase click area
    
    # Also make text labels clickable  
    for text in texts:
        text.set_picker(True)
        if hasattr(text, 'set_pickradius'):
            text.set_pickradius(15)

    def on_pick(event):
        artist = event.artist
        
        # Determine which legend entry was clicked
        if hasattr(artist, 'get_label'):
            # Clicked on a handle
            label_text = artist.get_label()
        else:
            # Clicked on text - find corresponding handle
            try:
                text_index = texts.index(artist)
                label_text = texts[text_index].get_text()
            except (ValueError, IndexError):
                return
        
        # Find matching series in series_map
        # Try exact match first, then partial match
        matched_key = None
        for key in series_map.keys():
            if key == label_text or label_text in key or key in label_text:
                matched_key = key
                break
        
        if not matched_key:
            # Fallback: try to match based on common words or special cases
            label_words = label_text.lower().split()
            for key in series_map.keys():
                key_words = key.lower().split()
                # Check for word overlap
                if any(word in key_words for word in label_words):
                    matched_key = key
                    break
                # Special case: CI/Confidence Interval matching
                if ('ci' in label_text.lower() or '%' in label_text) and 'confidence' in key.lower():
                    matched_key = key
                    break
                # Special case: fit matching
                if 'fit' in label_text.lower() and 'fit' in key.lower():
                    matched_key = key
                    break
        
        if not matched_key:
            print(f"Warning: Could not find series for legend label '{label_text}'")
            print(f"Available series: {list(series_map.keys())}")
            return
            
        arts = series_map[matched_key]
        if not arts:
            return
            
        # Toggle visibility
        current_visible = arts[0].get_visible()
        new_visible = not current_visible
        
        for art in arts:
            art.set_visible(new_visible)
        
        # Update legend visual feedback
        # Find the corresponding handle and text
        for i, handle in enumerate(handles):
            if (hasattr(handle, 'get_label') and handle.get_label() == label_text) or \
               (i < len(texts) and texts[i].get_text() == label_text):
                # Update handle appearance
                handle.set_alpha(1.0 if new_visible else 0.3)
                # Update text appearance  
                if i < len(texts):
                    texts[i].set_alpha(1.0 if new_visible else 0.5)
                    texts[i].set_weight('normal' if new_visible else 'normal')
                    texts[i].set_style('normal' if new_visible else 'italic')
                break
        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    
    # Add instruction text
    ax.text(0.02, 0.98, "Click legend entries to toggle visibility", 
            transform=ax.transAxes, fontsize=8, alpha=0.7,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", alpha=0.1))
    
    return leg


def apply_profile_visibility(
    series_map: Dict[str, List[plt.Artist]],
    profile: Dict[str, bool],
) -> None:
    """Apply a visibility profile to mapped series."""

    for label, visible in profile.items():
        for art in series_map.get(label, []):
            art.set_visible(bool(visible))


def set_scatter_group_visibility(
    scat: PathCollection,
    groups: Dict[str, np.ndarray],
    name: str,
    visible: bool,
) -> None:
    """Mask/unmask scatter points belonging to a group."""

    if name not in groups:
        return
    mask = np.asarray(groups[name], bool)
    sizes = scat.get_sizes()
    fc = scat.get_facecolors()
    if not hasattr(scat, "_base_sizes"):
        scat._base_sizes = sizes.copy()  # type: ignore[attr-defined]
    if not hasattr(scat, "_base_facecolors"):
        scat._base_facecolors = fc.copy()  # type: ignore[attr-defined]
    base_sizes = scat._base_sizes  # type: ignore[attr-defined]
    base_fc = scat._base_facecolors  # type: ignore[attr-defined]
    if visible:
        sizes[mask] = base_sizes[mask]
        fc[mask, 3] = base_fc[mask, 3]
    else:
        sizes[mask] = 1e-6
        fc[mask, 3] = 0.0
    scat.set_sizes(sizes)
    scat.set_facecolors(fc)

import pandas as pd
import numpy as np
from typing import List, Dict, Iterable

"""Tools to detect implied-volatility events and measure spillovers.

This module loads a daily IV dataset, flags events where a ticker's ATM IV
moves by a configurable percentage threshold and then measures how those shocks
propagate to its peers.
"""


def load_iv_data(path: str, use_raw: bool = False) -> pd.DataFrame:
    """Load IV data from a Parquet file.

    Parameters
    ----------
    path: str
        Location of the ``iv_daily`` Parquet file.
    use_raw: bool
        If ``True`` use the ``atm_iv_raw`` column, otherwise use
        ``atm_iv_synth``.
    """
    df = pd.read_parquet(path)
    col = "atm_iv_raw" if use_raw else "atm_iv_synth"
    df = df[["date", "ticker", col]].rename(columns={col: "atm_iv"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


def detect_events(df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Flag dates where a ticker's IV changes by ``threshold`` or more.

    Returns a DataFrame with columns ``ticker``, ``date``, ``rel_change`` and
    ``sign`` (1 or -1).
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["rel_change"] = df.groupby("ticker")["atm_iv"].pct_change()
    events = df.loc[df["rel_change"].abs() >= threshold,
                    ["ticker", "date", "rel_change"]].copy()
    events["sign"] = np.sign(events["rel_change"]).astype(int)
    return events.reset_index(drop=True)


def select_peers(df: pd.DataFrame, lookback: int = 60, top_k: int = 3) -> Dict[str, List[str]]:
    """Identify top-K peers for each ticker using rolling correlation of ΔIV."""
    df = df.sort_values(["ticker", "date"]).copy()
    df["dIV"] = df.groupby("ticker")["atm_iv"].pct_change()
    piv = df.pivot(index="date", columns="ticker", values="dIV")
    peers: Dict[str, List[str]] = {}
    for t in piv.columns:
        corr = piv.rolling(lookback).corr(piv[t]).iloc[-1]
        corr = corr.drop(index=t).dropna().sort_values(ascending=False).head(top_k)
        peers[t] = list(corr.index)
    return peers


def compute_responses(df: pd.DataFrame,
                      events: pd.DataFrame,
                      peers: Dict[str, List[str]],
                      horizons: Iterable[int] = (1, 3, 5)) -> pd.DataFrame:
    """Compute peer responses for each event over given horizons.

    Response for peer j at horizon h is the percentage change in j's IV from
    t0-1 to t0+h.
    """
    panel = df.set_index(["date", "ticker"]).sort_index()
    dates = panel.index.get_level_values(0).unique()
    rows = []
    for _, e in events.iterrows():
        t0 = e["date"]
        i = e["ticker"]
        idx0 = dates.searchsorted(t0)
        if idx0 == 0:
            continue  # need t-1
        t_minus1 = dates[idx0 - 1]
        for j in peers.get(i, []):
            if (t_minus1, j) not in panel.index:
                continue
            base = panel.loc[(t_minus1, j), "atm_iv"]
            for h in horizons:
                idx_h = idx0 + h
                if idx_h >= len(dates):
                    continue
                d_h = dates[idx_h]
                if (d_h, j) not in panel.index:
                    continue
                resp = panel.loc[(d_h, j), "atm_iv"]
                pct = (resp - base) / base
                rows.append({
                    "ticker": i,
                    "peer": j,
                    "t0": t0,
                    "h": int(h),
                    "trigger_pct": e["rel_change"],
                    "peer_pct": pct,
                    "sign": e["sign"],
                })
    return pd.DataFrame(rows)


def summarise(responses: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Summarise peer responses across events."""
    def _agg(g: pd.DataFrame) -> pd.Series:
        hr = (g["peer_pct"].abs() >= threshold).mean()
        sc = (np.sign(g["peer_pct"]) == g["sign"]).mean()
        med_resp = g["peer_pct"].median()
        med_elast = (g["peer_pct"] / g["trigger_pct"]).median()
        return pd.Series({
            "hit_rate": hr,
            "sign_concord": sc,
            "median_resp": med_resp,
            "median_elasticity": med_elast,
            "n": len(g),
        })
    return responses.groupby(["ticker", "peer", "h"]).apply(_agg).reset_index()


def persist_events(events: pd.DataFrame, path: str) -> None:
    """Write event table to Parquet."""
    events.to_parquet(path)


def persist_summary(summary: pd.DataFrame, path: str) -> None:
    """Write summary metrics to Parquet."""
    summary.to_parquet(path)


def run_spillover(
    iv_path: str,
    *,
    tickers: Iterable[str] | None = None,
    threshold: float = 0.10,
    lookback: int = 60,
    top_k: int = 3,
    horizons: Iterable[int] = (1, 3, 5),
    use_raw: bool = False,
    events_path: str = "spill_events.parquet",
    summary_path: str = "spill_summary.parquet",
) -> Dict[str, pd.DataFrame]:
    """High level helper that runs the full spillover analysis.

    Returns a dictionary with keys ``events`` and ``summary``.
    """
    df = load_iv_data(iv_path, use_raw=use_raw)
    if tickers is not None:
        tickers = [t.upper() for t in tickers]
        df = df[df["ticker"].str.upper().isin(tickers)]
    events = detect_events(df, threshold=threshold)
    peers = select_peers(df, lookback=lookback, top_k=top_k)
    responses = compute_responses(df, events, peers, horizons=horizons)
    summary = summarise(responses, threshold=threshold)
    persist_events(events, events_path)
    persist_summary(summary, summary_path)
    return {"events": events, "responses": responses, "summary": summary}

def create_safe_animation(fig, update_func, frames, interval_ms=120, repeat=True):
    """
    Create a matplotlib animation with graceful fallback for blitting issues.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to animate
    update_func : callable
        Animation update function
    frames : int or iterable
        Number of frames or iterable of frame data
    interval_ms : int
        Interval between frames in milliseconds
    repeat : bool
        Whether animation should repeat
        
    Returns
    -------
    matplotlib.animation.FuncAnimation
        The created animation
    """
    import matplotlib
    backend = matplotlib.get_backend()
    
    # For problematic backends (especially TkAgg), disable blitting by default
    use_blit = backend not in ['TkAgg', 'Qt4Agg', 'Qt5Agg'] 
    
    try:
        if use_blit:
            ani = FuncAnimation(
                fig, update_func, frames=frames, 
                interval=interval_ms, blit=True, repeat=repeat
            )
        else:
            ani = FuncAnimation(
                fig, update_func, frames=frames, 
                interval=interval_ms, blit=False, repeat=repeat
            )
        return ani
    except Exception:
        # Final fallback - always use non-blitted animation
        return FuncAnimation(
            fig, update_func, frames=frames, 
            interval=interval_ms, blit=False, repeat=repeat
        )
