############################################################
##### Imports
############################################################

import os

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch


############################################################
##### Plotting Style
############################################################


CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red",
    ["#104281", "#256abf", "#86b6ef", "#f0efec", "#ef9a99", "#e34948", "#8f2020"],
)

CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

ORDINAL_BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]

# Colour scheme used for plotting
C_INK = "#2b2b28"  # primary text
C_INK_MUTED = "#6b6b66"  # secondary text, reference lines
C_RAW = "0.72"  # individual waves
C_ROLL = "#1f4e9c"  # rolling mean
C_ROLL2 = "#e07b39"  # rolling mean
C_ROLL3 = "#1baf7a"  # rolling mean
C_ROGUE = "#c1121f"  # rogue waves / threshold

C_FOLD = [
    "#c7e9c0",  # Fold 1
    "#74c476",  # Fold 2
    "#41ab5d",  # Fold 3
    "#238b45",  # Fold 4
    "#005a32",  # Fold 5
]
C_PURGE = "#cccccc"  # neutral, dark enough to be visible at hairline width
C_TEST = "#064d69"  # warm tint: a different kind of block, not a later fold


def set_plotting_style():
    # Set the plotting style
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 500,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )


def save_figure(fig, filename, dir_output, ext="png"):
    # Save fig as <dir_output>/<filename>.<ext> at the savefig.dpi set in
    # set_plotting_style(). Pass dir_output=None to skip saving.
    if dir_output is None or filename is None:
        return
    os.makedirs(dir_output, exist_ok=True)
    fig.savefig(os.path.join(dir_output, f"{filename}.{ext}"), bbox_inches="tight")


############################################################
##### Utility Functions
############################################################


def decimate_minmax(y, n_out=6000):
    # Reduce a long series to ~n_out samples, preserving the extremes of each bucket.
    # Returns the indices of the retained samples, so that any parallel array
    # (e.g. the time axis) can be sliced consistently.
    y = np.asarray(y)
    n = len(y)
    if n <= n_out:
        return np.arange(n)
    bucket = int(np.ceil(n / (n_out // 2)))
    n_buckets = n // bucket
    blocks = y[: n_buckets * bucket].reshape(n_buckets, bucket)
    offset = np.arange(n_buckets) * bucket
    idx = np.union1d(blocks.argmin(axis=1) + offset, blocks.argmax(axis=1) + offset)
    return np.concatenate([idx, np.arange(n_buckets * bucket, n)])


def roll(series, window, fn="mean"):
    # Centred rolling statistic over `window` waves.
    r = pd.Series(np.asarray(series)).rolling(window, center=True, min_periods=max(1, window // 10))
    return getattr(r, fn)().to_numpy()


def ordinal_cmap(n):
    # n ordered steps of the single blue hue, sampled from ORDINAL_BLUE.
    idx = np.linspace(0, len(ORDINAL_BLUE) - 1, n).round().astype(int)
    return ListedColormap([ORDINAL_BLUE[i] for i in idx])


############################################################
##### Plotting Functions
############################################################


def plot_abni_series(
    wave_index,
    abni,
    dec,
    is_rogue,
    ROGUE_WAVE_THRESHOLD,
    rolling_window,
    title="Abnormality Index over the recording period",
    dir_output=None,
    filename="abni_series",
):
    set_plotting_style()

    rolling_mean = roll(abni, rolling_window)

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(
        wave_index[dec],
        abni[dec],
        marker=".",
        ms=1.2,
        lw=0.35,
        color=C_RAW,
        rasterized=True,
        label="AbnI, individual waves (min/max-decimated)",
        zorder=1,
    )
    ax.scatter(
        wave_index[is_rogue],
        abni[is_rogue],
        s=2,
        color=C_ROGUE,
        alpha=0.5,
        rasterized=True,
        label=f"rogue waves (AbnI $\\geq$ {ROGUE_WAVE_THRESHOLD:g})",
        zorder=2,
    )

    step = max(1, len(wave_index) // 8000)
    ax.plot(
        wave_index[::step],
        rolling_mean[::step],
        lw=1.0,
        color=C_ROLL,
        alpha=0.85,
        label="rolling mean",
        zorder=3,
    )

    ax.axhline(ROGUE_WAVE_THRESHOLD, ls="--", lw=1.2, color=C_ROGUE, zorder=5)
    ax.text(
        wave_index[dec][-1],
        ROGUE_WAVE_THRESHOLD,
        "  rogue",
        va="center",
        ha="left",
        color=C_ROGUE,
        fontsize=9,
    )

    ax.set_ylabel("AbnI  =  $H_{max}/H_s$  (next 10 min)")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=2, framealpha=0.9, markerscale=4)
    ax.set_xlim(wave_index[dec][0], wave_index[dec][-1])
    ax.set_xlabel("wave index")
    plt.tight_layout()
    if dir_output is not None:
        save_figure(fig, filename, dir_output)
    plt.show()

    return fig, ax


def plot_rogue_wave_gaps(
    gaps,
    title="Waiting-time distribution",
    dir_output=None,
    filename="rogue_wave_gaps",
):
    set_plotting_style()

    fig, ax = plt.subplots(figsize=(7, 4))

    bins = np.logspace(np.log10(max(gaps.min(), 1e-3)), np.log10(gaps.max()), 35)
    ax.hist(gaps, bins=bins, color=C_ROLL, alpha=0.8, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("waiting time between rogue events [waves]")
    ax.set_ylabel("count")

    plt.tight_layout()
    if dir_output is not None:
        save_figure(fig, filename, dir_output)
    plt.show()

    return fig, ax


def plot_abni_feature_relationship(
    panels,
    wave_index,
    abni,
    data,
    ev_start,
    rolling_window,
    ROGUE_WAVE_THRESHOLD,
    title_panel="Metocean Features",
    event_anchor="mixed",
    show_events_on_features=True,
    add_rogue_event_scatter=True,
    dir_output=None,
    filename="abni_feature_relationship",
):
    set_plotting_style()

    step = max(1, len(wave_index) // 8000)

    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 2.1 * len(panels)), sharex=True)

    for pos, (ax, (col, title, ylabel)) in enumerate(zip(axes, panels)):
        y = data[col].to_numpy()
        y_roll = roll(y, rolling_window)

        d = decimate_minmax(y, n_out=4000)

        ax.plot(wave_index[d], y[d], lw=0.3, color=C_RAW, rasterized=True)
        ax.plot(wave_index[::step], y_roll[::step], lw=1.6, color=C_ROLL)
        if len(ev_start) and (pos == 0 or show_events_on_features):
            on_rolling = event_anchor == "rolling" or (event_anchor == "mixed" and pos > 0)
            y_events = y_roll[ev_start] if on_rolling else y[ev_start]
            ax.scatter(
                wave_index[ev_start],
                y_events,
                s=15,
                color=C_ROGUE,
                edgecolors="white",
                linewidths=0.3,
                zorder=5,
                label="rogue events" if pos == 0 else None,
            )

        ax.set_ylabel(ylabel)
        ax.text(
            0.006,
            0.90,
            title,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color=C_ROLL,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
        )
        ax.set_xlim(wave_index[0], wave_index[-1])

    if add_rogue_event_scatter:
        axes[0].scatter(
            wave_index[ev_start], abni[ev_start], s=12, color=C_ROGUE, zorder=5, label="rogue events"
        )
        axes[0].axhline(ROGUE_WAVE_THRESHOLD, ls="--", lw=1.0, color=C_ROGUE)
    axes[0].set_title(title_panel)

    axes[-1].set_xlabel("wave index")
    plt.tight_layout()
    if dir_output is not None:
        save_figure(fig, filename, dir_output)
    plt.show()
    return fig, axes


def plot_acf(acf, n_waves, dir_output, filename="acf"):
    set_plotting_style()

    short_max_lag = min(500, n_waves - 1)
    long_max_lag = min(10_000, n_waves - 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))

    axes[0].plot(
        np.arange(short_max_lag + 1),
        acf[: short_max_lag + 1],
        color=C_ROLL,
        lw=1.5,
    )
    axes[0].axhline(0, color="0.4", lw=0.8)
    axes[0].set_xlabel("Lag [waves]")
    axes[0].set_ylabel("Pearson autocorrelation of AbnI")
    axes[0].set_title("Short lags")

    axes[1].plot(
        np.arange(long_max_lag + 1),
        acf[: long_max_lag + 1],
        color=C_ROLL,
        lw=1.5,
    )
    axes[1].axhline(0, color="0.4", lw=0.8)
    axes[1].set_xlabel("Lag [waves]")
    axes[1].set_title("Long lags")

    plt.tight_layout()
    if dir_output is not None:
        save_figure(fig, filename, dir_output)
    plt.show()

    return fig, axes


def plot_correlation_matrix(
    frame,
    method="spearman",
    title="Spearman correlation",
    annot=True,
    dir_output=None,
    filename="correlation_matrix",
):
    set_plotting_style()

    corr = frame.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(
        corr,
        mask=mask,
        cmap=CMAP_DIVERGING,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot=annot,
        fmt=".2f",
        annot_kws={"size": 5, "color": C_INK},
        cbar_kws={"shrink": 0.6, "label": f"{method.capitalize()} $\\rho$"},
    )
    ax.set_title(title or f"{method.capitalize()} correlation matrix")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)

    plt.tight_layout()
    save_figure(fig, filename, dir_output)
    plt.show()

    return fig, ax


def plot_feature_vs_target(
    frame,
    target,
    dec,
    columns,
    highlight=None,
    kind="scatter",
    n_bins=20,
    ncols=5,
    panel_size=3.0,
    target_label="AbnI",
    threshold=None,
    title="Features vs. AbnI",
    dir_output=None,
    filename="feature_vs_target",
):
    set_plotting_style()

    highlight = None if highlight is None else np.asarray(highlight, dtype=bool)

    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_size, nrows * panel_size))
    axes = np.atleast_1d(axes).ravel()

    for k, (ax, col) in enumerate(zip(axes, columns)):
        v = frame[col].to_numpy()

        if kind == "hexbin":
            ax.hexbin(v, target, gridsize=45, bins="log", cmap=CMAP_SEQUENTIAL, linewidths=0)
        else:
            ax.scatter(
                v[dec],
                target[dec],
                s=2,
                color=C_RAW,
                alpha=0.30,
                rasterized=True,
                label=f"{len(dec):,} waves (min/max-decimated)",
            )
            if highlight is not None:
                ax.scatter(
                    v[highlight],
                    target[highlight],
                    s=2,
                    color=C_ROGUE,
                    alpha=0.35,
                    rasterized=True,
                    label="rogue rows",
                )

        if threshold is not None:
            ax.axhline(threshold, ls="--", lw=1.0, color=C_ROGUE)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel(target_label if k % ncols == 0 else "", fontsize=9)
        if k == 0:
            ax.legend(loc="upper right", fontsize=6.5, markerscale=3, framealpha=0.9)

    for ax in axes[len(columns) :]:
        ax.axis("off")

    fig.suptitle(title or f"Features vs. {target_label}", y=1.0, fontsize=12)
    plt.tight_layout()
    if dir_output is not None:
        save_figure(fig, filename, dir_output)
    plt.show()

    return fig, axes


def plot_group_boxplots(
    frame,
    mask,
    columns,
    group_names=("Background", "Rogue Waves"),
    whis=(1, 99),
    ncols=5,
    panel_size=3.0,
    title="Feature Distributions",
    dir_output=None,
    filename="group_boxplots",
):
    set_plotting_style()

    mask = np.asarray(mask, dtype=bool)
    group = np.where(mask, group_names[1], group_names[0])
    nrows = int(np.ceil(len(columns) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_size, nrows * panel_size))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, columns):
        sns.boxplot(
            x=group,
            y=frame[col].to_numpy(),
            hue=group,
            order=list(group_names),
            hue_order=list(group_names),
            palette=["lightgrey", "red"],
            whis=whis,
            showfliers=False,
            linewidth=0.9,
            width=0.6,
            ax=ax,
            legend=False,
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8)

    for ax in axes[len(columns) :]:
        ax.axis("off")

    fig.suptitle(
        title
        or f"Feature distributions: {group_names[1]} vs. {group_names[0]} "
        f"(box = quartiles, whiskers = {whis[0]}th/{whis[1]}th percentile)",
        y=1.0,
        fontsize=12,
    )
    plt.tight_layout()
    if dir_output is not None:
        save_figure(fig, filename, dir_output)
    plt.show()
    return fig, axes


def plot_scree(
    explained_variance_ratio,
    n_show=None,
    title=None,
    figsize=(6.5, 4),
    dir_output=None,
    filename="scree",
):
    set_plotting_style()

    evr = np.asarray(explained_variance_ratio)[:n_show]
    x = np.arange(1, len(evr) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, 100 * evr, color=C_ROLL, alpha=0.9, label="per component")
    ax.plot(x, 100 * np.cumsum(evr), marker="o", ms=4, lw=1.6, color=C_ROLL2, label="cumulative")
    ax.axhline(95, ls="--", lw=1.0, color=C_INK_MUTED)
    ax.set_xticks(x)
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance [%]")
    ax.set_title(title or "PCA scree plot")
    ax.legend()
    plt.tight_layout()
    if dir_output is not None:
        save_figure(fig, filename, dir_output)
    plt.show()
    return fig, ax


def plot_embedding(
    emb,
    mask,
    blocks,
    block_labels=None,
    method="PCA",
    axis_labels=("component 1", "component 2"),
    n_sample=50_000,
    seed=42,
    figsize=(15, 6),
    dir_output=None,
    filename=None,
):
    set_plotting_style()

    rng = np.random.default_rng(seed)
    n = len(emb)
    n_blocks = int(blocks.max()) + 1

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    bg = np.flatnonzero(~mask)
    if n_sample and len(bg) > n_sample:
        bg = rng.choice(bg, size=n_sample, replace=False)
    fg = np.flatnonzero(mask)

    ax = axes[0]
    ax.scatter(
        emb[bg, 0],
        emb[bg, 1],
        s=2,
        color=C_RAW,
        alpha=0.25,
        rasterized=True,
        label=f"background ({len(bg):,} shown)",
    )
    ax.scatter(
        emb[fg, 0],
        emb[fg, 1],
        s=3,
        color=C_ROGUE,
        alpha=0.45,
        rasterized=True,
        label=f"rogue ({len(fg):,})",
    )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(f"{method}: rogue vs. background")
    ax.legend(loc="best", markerscale=4)

    ax = axes[1]
    show = np.arange(n)
    if n_sample and n > n_sample:
        show = rng.choice(n, size=n_sample, replace=False)
    show = rng.permutation(show)
    sc = ax.scatter(
        emb[show, 0],
        emb[show, 1],
        c=blocks[show],
        s=2,
        alpha=0.45,
        cmap=ordinal_cmap(n_blocks),
        vmin=-0.5,
        vmax=n_blocks - 0.5,
        rasterized=True,
    )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(f"{method}: position in the record")
    cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(n_blocks))
    cbar.set_label("block of the record (early $\\rightarrow$ late)")
    if block_labels is not None:
        cbar.ax.set_yticklabels(block_labels, fontsize=7)

    plt.tight_layout()
    filename = filename or f"embedding_{method.lower()}"
    if dir_output is not None:
        save_figure(fig, filename, dir_output)
    plt.show()
    return fig, axes


def plot_cv_folds(
    data,
    y_col,
    dec,
    num_cv,
    fold_col="fold",
    rolling_window=None,
    ev_start=None,
    threshold=None,
    event_on_rolling=False,
    raw_color="0.45",
    figsize=(15, 4),
    title=None,
    dir_output=None,
    filename="cv_folds",
):
    """Time series with the chronological split shaded: purge gaps, folds 1..num_cv, test.

    ev_start: positional indices of the rogue-wave events, drawn as dots.
    event_on_rolling places them on the rolling mean instead of the raw value.
    """
    set_plotting_style()

    idx, y, fold = data.index.to_numpy(), data[y_col].to_numpy(), data[fold_col].to_numpy()
    i = np.linspace(0, len(C_FOLD) - 1, num_cv).round().astype(int)
    tints = [C_FOLD[k] for k in i]
    color = lambda f: C_PURGE if f == 0 else (C_TEST if f > num_cv else tints[f - 1])

    fig, ax = plt.subplots(figsize=figsize)
    edges = np.flatnonzero(np.r_[True, fold[1:] != fold[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        ax.axvspan(idx[a], idx[b - 1], color=color(fold[a]), lw=0, zorder=0)

    ax.plot(idx[dec], y[dec], lw=0.3, color=raw_color, rasterized=True, zorder=1)

    y_roll = roll(y, rolling_window) if rolling_window else None
    if y_roll is not None:
        step = max(1, len(idx) // 8000)
        ax.plot(idx[::step], y_roll[::step], lw=1.4, color=C_ROLL, zorder=2)

    handles = [Patch(facecolor=t, label=f"fold {g}") for g, t in enumerate(tints, 1)] + [
        Patch(facecolor=C_PURGE, label="purge gap"),
        Patch(facecolor=C_TEST, label="test"),
    ]

    if ev_start is not None and len(ev_start):
        ev_start = np.asarray(ev_start)
        y_events = y_roll[ev_start] if (event_on_rolling and y_roll is not None) else y[ev_start]
        handles.append(
            ax.scatter(
                idx[ev_start],
                y_events,
                s=12,
                color=C_ROGUE,
                edgecolors="white",
                linewidths=0.3,
                zorder=5,
                label="rogue events",
            )
        )
    if threshold is not None:
        ax.axhline(threshold, ls="--", lw=1.0, color=C_ROGUE, zorder=4)

    ax.grid(False)
    ax.set_xlim(idx[0], idx[-1])
    ax.set_xlabel("wave index")
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{y_col} with the chronological split")
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(handles),
        fontsize=8,
        frameon=False,
    )
    plt.tight_layout()
    if dir_output is not None:
        save_figure(fig, filename, dir_output)
    plt.show()
    return fig, ax


def plot_predictions(
    y_true,
    y_pred,
    title="True vs. Predicted Values",
    xlabel="True Values",
    ylabel="Predicted Values",
    textstr="",
    kind="hexbin",
    gridsize=60,
    figsize=(4.5, 4.5),
    dir_output=None,
    filename="predictions",
):
    """True vs. predicted values as a 2-d density (or scatter), with the y = x reference.

    kind="hexbin" shows the full joint density on a single-hue ramp; kind="scatter" thins
    nothing and is only sensible for small samples. Both axes share identical limits and an
    equal aspect, so the diagonal is a true 45 degree line.
    """
    set_plotting_style()

    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    fig, ax = plt.subplots(figsize=figsize)

    if kind == "hexbin":
        hb = ax.hexbin(
            y_true,
            y_pred,
            gridsize=gridsize,
            bins="log",
            cmap=CMAP_SEQUENTIAL,
            linewidths=0,
            extent=(lo, hi, lo, hi),
        )
        fig.colorbar(hb, ax=ax, shrink=0.8, label="count (log)")
    else:
        ax.scatter(y_true, y_pred, s=2, color=C_RAW, alpha=0.3, rasterized=True)

    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.4, color=C_ROGUE)

    if textstr:
        ax.text(
            0.04,
            0.96,
            textstr,
            transform=ax.transAxes,
            fontsize=8,
            color=C_INK,
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.9),
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    if dir_output is not None:
        save_figure(fig, filename, dir_output)
    plt.show()
    return fig, ax


def plot_coefficients(
    coefficients,
    column="Coefficient",
    drop=("intercept",),
    title="Elastic Net Model Coefficients",
    ylabel="coefficient (standardised features)",
    figsize=(8, 5),
    dir_output=None,
    filename="coefficients",
):
    """Signed model coefficients as a bar chart, coloured on the diverging map.

    coefficients: a Series, or a DataFrame holding `column`. Labels listed in `drop` are
    excluded: the intercept is on the scale of the target and is orders of magnitude
    larger than the coefficients, so including it flattens all of them.
    """
    set_plotting_style()

    values = coefficients[column] if isinstance(coefficients, pd.DataFrame) else coefficients
    values = values.drop(labels=[d for d in (drop or ()) if d in values.index])

    lim = float(np.abs(values).max())
    norm = plt.Normalize(-lim, lim)
    colors = [CMAP_DIVERGING(norm(v)) for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(values.index, values.to_numpy(), color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(0, lw=1.0, color=C_INK_MUTED)

    ax.set_xlabel("feature")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Model coefficients")
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    if dir_output is not None:
        save_figure(fig, filename, dir_output)
    plt.show()
    return fig, ax


def plot_shap_dependence(
    explanation,
    num_cols=5,
    panel_size=3.2,
    interaction_index="auto",
    share_y=True,
    dot_size=6,
    alpha=0.4,
    dir_output=None,
    filename="shap_dependence",
):
    """SHAP dependence panels, ordered by descending mean |SHAP|.

    interaction_index="auto" keeps SHAP's interaction colouring (and its per-panel colour
    bar); None drops it for a single-colour, lower-ink version. share_y puts every panel on
    the same SHAP scale, so the importance ordering is visible in the spread of each panel
    rather than being normalised away.
    """
    set_plotting_style()

    mean_shap = np.abs(explanation.values).mean(axis=0)
    order = np.argsort(-mean_shap)
    features = [explanation.feature_names[i] for i in order]

    rows = int(np.ceil(len(features) / num_cols))
    fig, axes = plt.subplots(rows, num_cols, figsize=(num_cols * panel_size, rows * panel_size))
    axes = np.atleast_1d(axes).ravel()

    # symmetric limits from a high quantile: the extreme tail would squash every panel
    lim = float(np.quantile(np.abs(explanation.values), 0.999))
    ymin, ymax = (-lim, lim) if share_y else (None, None)

    for k, (feature, i) in enumerate(zip(features, order)):
        shap.dependence_plot(
            ind=feature,
            shap_values=explanation.values,
            features=explanation.data,
            feature_names=explanation.feature_names,
            interaction_index=interaction_index,
            cmap=CMAP_DIVERGING,
            color=C_ROLL,
            dot_size=dot_size,
            alpha=alpha,
            ymin=ymin,
            ymax=ymax,
            ax=axes[k],
            show=False,
        )
        axes[k].set_title(f"{feature}   mean |SHAP| = {mean_shap[i]:.3f}", fontsize=9.5)
        axes[k].set_xlabel(feature, fontsize=9)
        axes[k].set_ylabel("SHAP value" if k % num_cols == 0 else "", fontsize=9)

    for ax in axes[len(features) :]:
        fig.delaxes(ax)

    plt.tight_layout()
    if dir_output is not None:
        save_figure(fig, filename, dir_output)
    plt.show()
    return fig, axes[: len(features)]
