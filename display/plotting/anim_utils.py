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
        line_synth.set_ydata(iv_tk[i])
        updated = [line_synth]
        if line_raw is not None:
            line_raw.set_ydata(iv_raw_tk[i])
            updated.append(line_raw)
        if band is not None:
            verts = np.column_stack(
                [np.r_[k, k[::-1]], np.r_[ci_hi_tk[i], ci_lo_tk[i][::-1]]]
            )
            band.get_paths()[0].vertices[:] = verts
            updated.append(band)
        ax.set_title(str(dates[i]))
        return updated

    ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=True)
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
        im.set_array(iv_tktau[i])
        ax.set_title(str(dates[i]))
        return [im]

    ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=True)
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
        resp = mags[:, i]
        sizes = base_sizes * (1.0 + np.abs(resp) / max_mag)
        alpha = np.clip(np.abs(resp) / max_mag, 0.0, 1.0)
        fc = scat.get_facecolors()
        fc[:, 3] = alpha
        scat.set_sizes(sizes)
        scat.set_facecolors(fc)
        ax.set_title(str(times[i]))
        return [scat]

    ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=True)
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
    """Make legend entries clickable to toggle series."""

    leg = ax.legend()
    fig = ax.figure
    handles = leg.legend_handles if hasattr(leg, "legend_handles") else leg.legendHandles
    for handle in handles:
        handle.set_picker(True)

    def on_pick(event):
        artist = event.artist
        label = artist.get_label()
        arts = series_map.get(label, [])
        if not arts:
            return
        visible = not arts[0].get_visible()
        for art in arts:
            art.set_visible(visible)
        artist.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
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
