"""Generate publication figures and manuscript/supplementary tables from saved results."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from string import ascii_uppercase

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_io import ROOT


# ---------------------------------------------------------------------------
# Publication design system
# ---------------------------------------------------------------------------
# Hues remain the Okabe-Ito colour-blind-safe set so that every categorical
# contrast survives deuteranopia, protanopia, and greyscale printing. The
# supporting neutrals, tints, and rules below are derived from those hues and
# give the figures a single, consistent visual language.

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
BLUE, VERMILION, GREEN, PURPLE, AMBER, SKY, BLACK = OKABE_ITO

NAVY = "#13293D"          # primary ink: axes, labels, emphasis text
SLATE = "#4F6272"         # secondary ink: annotations, captions in-figure
MUTED = "#8B9AA6"         # de-emphasised categorical fill
RULE = "#D5DFE6"          # hairlines and reference rules
ZEBRA = "#F3F7F9"         # alternating row wash
LIGHT_BLUE = "#DCECF4"
LIGHT_GRAY = "#EEF2F4"
DEEP_VERMILION = "#A33E00"


def tint(color: str, amount: float) -> str:
    """Mix ``color`` toward white; ``amount`` of 0 keeps it, 1 returns white."""

    red, green, blue = mpl.colors.to_rgb(color)
    amount = float(np.clip(amount, 0.0, 1.0))
    return mpl.colors.to_hex(
        (
            red + (1.0 - red) * amount,
            green + (1.0 - green) * amount,
            blue + (1.0 - blue) * amount,
        )
    )


def shade(color: str, amount: float) -> str:
    """Mix ``color`` toward black; ``amount`` of 0 keeps it, 1 returns black."""

    red, green, blue = mpl.colors.to_rgb(color)
    amount = float(np.clip(amount, 0.0, 1.0))
    return mpl.colors.to_hex((red * (1 - amount), green * (1 - amount), blue * (1 - amount)))

MODEL_LABELS = {
    "catboost": "CatBoost",
    "extra_trees": "Extra Trees",
    "random_forest": "Random Forest",
    "rbf_svr": "RBF SVR",
    "xgboost": "XGBoost",
    "direct_unconstrained_row": "Direct target, row-weighted",
    "bounded_unconstrained_row": "Bounded target, unconstrained",
    "bounded_monotonic_row": "Bounded monotonic, row-weighted",
    "bounded_monotonic_study_weighted": "Proposed study-weighted framework",
    "bounded_monotonic_study_weighted_cleaned_target": "Proposed, truncated target",
    "bounded_monotonic_study_weighted_cleaned_time": "Proposed, truncated time",
    "bounded_monotonic_study_weighted_no_temperature": "Proposed, no temperature",
}


def configure_style() -> None:
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.6,
            "axes.labelsize": 8,
            "axes.labelpad": 3.5,
            "axes.titlesize": 8.4,
            "axes.titlepad": 4.0,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.55,
            "legend.columnspacing": 1.4,
            "legend.labelcolor": NAVY,
            "text.color": NAVY,
            "axes.edgecolor": NAVY,
            "axes.labelcolor": NAVY,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": False,
            "grid.color": RULE,
            "grid.linewidth": 0.55,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.major.pad": 2.4,
            "ytick.major.pad": 2.4,
            "lines.linewidth": 1.4,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    sns.set_palette(OKABE_ITO)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def panel_label(ax, letter: str, dx: float = -0.155, dy: float = 1.035, fontsize: float = 9.5) -> None:
    """Place one bold panel letter in axes coordinates."""

    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        color=NAVY,
        fontweight="bold",
        fontsize=fontsize,
        ha="left",
        va="baseline",
        clip_on=False,
    )


def panel_labels(axes, dx: float = -0.155, dy: float = 1.035, fontsize: float = 9.5) -> None:
    for letter, ax in zip(ascii_uppercase, np.ravel(axes)):
        panel_label(ax, letter, dx=dx, dy=dy, fontsize=fontsize)


def soft_grid(ax, axis: str = "y") -> None:
    """Hairline reference grid drawn beneath the data."""

    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=RULE, linewidth=0.55, zorder=0)
    ax.grid(False, axis="x" if axis == "y" else "y")


def zebra_rows(ax, n_rows: int, color: str = ZEBRA, start: int = 0) -> None:
    """Alternating horizontal wash that ties wide categorical panels together."""

    for row in range(start, n_rows, 2):
        ax.axhspan(row - 0.5, row + 0.5, color=color, linewidth=0, zorder=0)
    ax.set_axisbelow(True)


def pill(ax, x: float, y: float, text: str, color: str, *, fontsize: float = 6.4, ha: str = "left", va: str = "center", transform=None):
    """Compact rounded tag used for diagnostics and reference callouts."""

    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes if transform is None else transform,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight="bold",
        color=shade(color, 0.18),
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": tint(color, 0.86),
            "edgecolor": tint(color, 0.55),
            "linewidth": 0.6,
        },
    )


def _text_data_extent(text_artist, ax, renderer, axis: str):
    box = text_artist.get_window_extent(renderer=renderer)
    inverse = ax.transData.inverted()
    lower = inverse.transform((box.x0, box.y0))
    upper = inverse.transform((box.x1, box.y1))
    index = 0 if axis == "x" else 1
    return min(lower[index], upper[index]), max(lower[index], upper[index])


def fit_axis_to_texts(ax, texts, axis: str = "x", margin: float = 0.02) -> None:
    """Grow one axis until every supplied label fits inside the axes.

    Value labels are the most common source of overlap and clipping in
    multi-panel figures. Measuring the rendered text boxes and expanding the
    limits guarantees that no annotation is cut off or pushed onto a
    neighbouring artist, at any figure size or font.
    """

    texts = [artist for artist in texts if artist is not None and artist.get_text()]
    if not texts:
        return
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    low, high = ax.get_xlim() if axis == "x" else ax.get_ylim()
    inverted = low > high
    current_low, current_high = (high, low) if inverted else (low, high)
    for artist in texts:
        text_low, text_high = _text_data_extent(artist, ax, renderer, axis)
        current_low = min(current_low, text_low)
        current_high = max(current_high, text_high)
    span = current_high - current_low
    if not np.isfinite(span) or span <= 0:
        return
    current_low -= margin * span
    current_high += margin * span
    setter = ax.set_xlim if axis == "x" else ax.set_ylim
    setter((current_high, current_low) if inverted else (current_low, current_high))


def label_barh(ax, values, positions, labels=None, *, color=NAVY, fontsize: float = 6.6, gap: float = 0.014, fontweight: str = "normal"):
    """Right-hand value labels for horizontal bars, with guaranteed headroom."""

    span = abs(np.diff(ax.get_xlim())[0])
    offset = gap * span
    texts = []
    labels = list(values if labels is None else labels)
    for value, position, label in zip(values, positions, labels):
        text = label if isinstance(label, str) else f"{label:.1f}"
        texts.append(
            ax.text(
                float(value) + offset,
                position,
                text,
                va="center",
                ha="left",
                fontsize=fontsize,
                color=color,
                fontweight=fontweight,
            )
        )
    fit_axis_to_texts(ax, texts, axis="x")
    return texts


def label_bar(ax, values, positions, labels=None, *, color=NAVY, fontsize: float = 7.0, gap: float = 0.02, fontweight: str = "bold"):
    """Value labels above vertical bars, with guaranteed headroom."""

    span = abs(np.diff(ax.get_ylim())[0])
    offset = gap * span
    texts = []
    labels = list(values if labels is None else labels)
    for value, position, label in zip(values, positions, labels):
        text = label if isinstance(label, str) else f"{label:.1f}"
        texts.append(
            ax.text(
                position,
                float(value) + offset,
                text,
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=color,
                fontweight=fontweight,
            )
        )
    fit_axis_to_texts(ax, texts, axis="y")
    return texts


def save_figure(fig, stem: str) -> None:
    output = ROOT / "figures" / stem
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = output.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    # Matplotlib emits trailing spaces in SVG path data; normalize them so
    # generated publication assets pass Git's whitespace checks.
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_table(frame: pd.DataFrame, stem: str, caption: str) -> None:
    table_dir = ROOT / "tables"
    frame.to_csv(table_dir / f"{stem}.csv", index=False)
    try:
        markdown = frame.to_markdown(index=False)
    except ImportError:
        markdown = frame.to_csv(index=False)
    (table_dir / f"{stem}.md").write_text(
        f"# {caption}\n\n{markdown}\n", encoding="utf-8"
    )


def _flow_card(ax, y0: float, height: float, accent: str, headline: str, detail: str, *, x0: float = 0.0, width: float = 0.655) -> None:
    """One rounded provenance card with a coloured left rule."""

    ax.add_patch(
        mpl.patches.FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=0.028",
            facecolor=tint(accent, 0.93),
            edgecolor=tint(accent, 0.45),
            linewidth=0.7,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.add_patch(
        mpl.patches.Rectangle(
            (x0 + 0.012, y0 + 0.022),
            0.016,
            height - 0.044,
            facecolor=accent,
            edgecolor="none",
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    centre = x0 + 0.028 + (width - 0.028) / 2
    ax.text(centre, y0 + height * 0.66, headline, transform=ax.transAxes, ha="center", va="center", fontsize=7.0, fontweight="bold", color=shade(accent, 0.28))
    ax.text(centre, y0 + height * 0.30, detail, transform=ax.transAxes, ha="center", va="center", fontsize=6.3, color=SLATE, linespacing=1.35)


def figure_1(table: pd.DataFrame) -> None:
    comparison = pd.read_csv(ROOT / "results" / "raw_vs_cleaned_target_comparison.csv")
    split = pd.read_csv(ROOT / "splits" / "task_a_random_row.csv")
    joined = table[["row_id", "curve_id"]].merge(split, on="row_id")
    curve_partitions = joined.groupby("curve_id").partition.nunique()

    fig = plt.figure(figsize=(7.2, 5.3))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.94], wspace=0.34, hspace=0.46)

    # (A) Provenance flow -----------------------------------------------------
    ax = fig.add_subplot(grid[0, 0])
    ax.axis("off")
    _flow_card(ax, 0.775, 0.205, GREEN, "Primary experiments", "17 source publications")
    _flow_card(ax, 0.450, 0.205, BLUE, "Curated trajectory corpus", "133 trajectories · 1,546 observations")
    _flow_card(ax, 0.050, 0.265, NAVY, "Primary analysis population", "16 studies · 129 trajectories\n1,467 observations")
    for tail, head in [((0.328, 0.775), (0.328, 0.672)), ((0.328, 0.450), (0.328, 0.330))]:
        ax.annotate("", xy=head, xytext=tail, xycoords="axes fraction", arrowprops={"arrowstyle": "-|>,head_width=0.17,head_length=0.34", "color": SLATE, "lw": 0.9})
    ax.annotate(
        "",
        xy=(0.700, 0.390),
        xytext=(0.336, 0.390),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>,head_width=0.15,head_length=0.30", "color": VERMILION, "lw": 0.8, "linestyle": (0, (3, 2))},
    )
    ax.text(
        0.845,
        0.390,
        "79 observations\nremoved: incomplete\ndescriptors",
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=shade(VERMILION, 0.18),
        fontsize=6.2,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.38", "facecolor": tint(VERMILION, 0.90), "edgecolor": tint(VERMILION, 0.55), "linewidth": 0.6},
    )
    panel_label(ax, "A", dx=-0.04, dy=1.02)

    # (B) Truncation identity plot with a magnified staircase ------------------
    ax = fig.add_subplot(grid[0, 1])
    soft_grid(ax, axis="both")
    ax.plot([-2, 102], [-2, 102], color=SLATE, lw=0.9, ls=(0, (4, 2)), zorder=2, label="Identity")
    ax.scatter(
        comparison.weight_loss_decimal,
        comparison.weight_loss_cleaned,
        s=9,
        alpha=0.30,
        color=BLUE,
        edgecolor="none",
        rasterized=True,
        zorder=3,
    )
    ax.set(xlabel="Original decimal mass loss (%)", ylabel="Released truncated mass loss (%)", xlim=(-2, 102), ylim=(-2, 102))
    ax.set_xticks(np.arange(0, 101, 25))
    ax.set_yticks(np.arange(0, 101, 25))
    ax.text(
        0.035,
        0.965,
        "96.7% of observations\nlost decimal precision",
        transform=ax.transAxes,
        color=NAVY,
        va="top",
        ha="left",
        fontsize=7.0,
        linespacing=1.35,
    )
    window = (39.0, 43.0)
    inset = ax.inset_axes([0.550, 0.060, 0.380, 0.380])
    zoom = comparison[(comparison.weight_loss_decimal >= window[0]) & (comparison.weight_loss_decimal <= window[1])]
    inset.plot(window, window, color=SLATE, lw=0.8, ls=(0, (4, 2)))
    inset.scatter(zoom.weight_loss_decimal, zoom.weight_loss_cleaned, s=11, color=VERMILION, edgecolor="none", alpha=0.9, zorder=3)
    inset.set(xlim=window, ylim=window, xticks=[39, 41, 43], yticks=[39, 41, 43])
    inset.yaxis.tick_right()
    inset.tick_params(labelsize=5.4, length=1.8, pad=1.2)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(tint(NAVY, 0.55))
        spine.set_linewidth(0.6)
    inset.set_facecolor("white")
    inset.set_title("Truncation staircase", fontsize=5.8, color=SLATE, pad=2.0)
    ax.indicate_inset_zoom(inset, edgecolor=tint(NAVY, 0.5), linewidth=0.6, alpha=0.9)
    panel_label(ax, "B")

    # (C) Magnitude of the recovered precision --------------------------------
    ax = fig.add_subplot(grid[1, 0])
    soft_grid(ax, axis="y")
    errors = comparison.target_absolute_difference_pct
    ax.hist(errors, bins=25, color=VERMILION, alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set(xlabel="Absolute truncation error (percentage points)", ylabel="Observations", xlim=(0, 1.0))
    cumulative = ax.twinx()
    ordered = np.sort(errors.to_numpy())
    cumulative.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), color=NAVY, lw=1.1, zorder=4)
    cumulative.set(ylim=(0, 1.02), ylabel="Cumulative fraction")
    cumulative.spines["right"].set_visible(True)
    cumulative.spines["right"].set_color(NAVY)
    cumulative.spines["top"].set_visible(False)
    cumulative.tick_params(colors=NAVY)
    ax.axvline(errors.mean(), color=NAVY, ls=(0, (4, 2)), lw=1.0, zorder=5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.24)
    ax.text(
        errors.mean() + 0.025,
        ax.get_ylim()[1] * 0.965,
        f"Mean = {errors.mean():.3f} pp",
        color=NAVY,
        ha="left",
        va="top",
        fontsize=6.6,
        fontweight="bold",
    )
    panel_label(ax, "C")

    # (D) Trajectory-level leakage under random row division -------------------
    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    mixed = int((curve_partitions == 2).sum())
    confined = int((curve_partitions == 1).sum())
    total = mixed + confined
    columns = 13
    ax.set(xlim=(-0.9, columns - 0.1), ylim=(-2.25, np.ceil(total / columns) + 0.55))
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    for index in range(total):
        row, column = divmod(index, columns)
        is_mixed = index < mixed
        ax.scatter(
            column,
            row,
            s=46,
            marker="o",
            color=VERMILION if is_mixed else GREEN,
            alpha=0.92 if is_mixed else 0.92,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    for offset, count, colour, wording in [
        (-1.90, mixed, VERMILION, "span both portions"),
        (-1.10, confined, GREEN, "stay within one portion"),
    ]:
        ax.scatter(-0.55, offset, s=46, color=colour, edgecolor="white", linewidth=0.55, zorder=3)
        ax.text(
            0.05,
            offset,
            f"{count} trajectories ({count / total:.0%}) {wording}",
            color=shade(colour, 0.20),
            fontsize=6.7,
            fontweight="bold",
            va="center",
            ha="left",
        )
    ax.text(
        -0.85,
        np.ceil(total / columns) + 0.35,
        "One marker per trajectory (n = 129) after random division of observations",
        fontsize=6.2,
        color=SLATE,
        va="center",
        ha="left",
    )
    panel_label(ax, "D", dx=-0.04, dy=1.02)

    save_figure(fig, "Figure_1_data_reconstruction_and_leakage")


def figure_2(table: pd.DataFrame) -> None:
    study = table.groupby("study_id_normalized").agg(observations=("row_id", "size"), curves=("curve_id", "nunique")).reset_index()
    study["order"] = pd.to_numeric(study.study_id_normalized)
    study = study.sort_values("order")
    curve_lengths = table.groupby("curve_id").size()
    environment = table.groupby("degradation_environment").agg(observations=("row_id", "size"), curves=("curve_id", "nunique")).reset_index()

    environment_names = {
        "soil": "Soil",
        "compost_organic_fertilizer": "Compost / organic fertilizer",
        "marine_aquatic": "Marine / aquatic",
        "laboratory_mineral_media": "Laboratory mineral medium",
    }
    environment["display_environment"] = environment.degradation_environment.map(environment_names)
    environment = environment.sort_values("observations", ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=(7.2, 5.35))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.18, 1.0], wspace=0.30, hspace=0.42)

    # (A) Observations and trajectories per source study -----------------------
    ax = fig.add_subplot(grid[0, 0])
    study_plot = study.sort_values("observations", ascending=False).reset_index(drop=True)
    y = np.arange(len(study_plot))
    soft_grid(ax, axis="x")
    zebra_rows(ax, len(study_plot))
    ax.barh(y, study_plot.observations, color=BLUE, alpha=0.92, height=0.62, zorder=3)
    upper = float(study_plot.observations.max())
    column_x = upper * 1.10
    ax.set(
        yticks=y,
        yticklabels=[f"S{value}" for value in study_plot.study_id_normalized],
        xlabel="Repeated observations",
        ylabel="Source study",
        xlim=(0, upper * 1.34),
        ylim=(len(study_plot) - 0.5, -0.5),
    )
    ax.set_xticks(np.arange(0, upper + 1, 100))
    for index, row in study_plot.iterrows():
        ax.text(column_x, index, f"{int(row.curves)}", va="center", ha="left", color=SLATE, fontsize=6.3, zorder=4)
    ax.text(column_x, -1.15, "Trajectories", va="center", ha="left", color=SLATE, fontsize=6.0, fontweight="bold")
    panel_label(ax, "A", dx=-0.20)

    # (B) Trajectory-length distribution --------------------------------------
    ax = fig.add_subplot(grid[0, 1])
    soft_grid(ax, axis="y")
    bins = np.arange(1, curve_lengths.max() + 3) - 0.5
    ax.hist(curve_lengths, bins=bins, color=GREEN, alpha=0.92, edgecolor="white", linewidth=0.25, zorder=3)
    median_length = float(curve_lengths.median())
    ax.set(xlabel="Observations per trajectory", ylabel="Trajectories", xlim=(0, 56))
    ax.set_ylim(0, float(np.histogram(curve_lengths, bins=bins)[0].max()) * 1.30)
    ax.annotate(
        f"Median = {median_length:.0f} observations",
        xy=(median_length, ax.get_ylim()[1] * 0.80),
        xytext=(14.0, ax.get_ylim()[1] * 0.95),
        arrowprops={"arrowstyle": "-|>,head_width=0.14,head_length=0.30", "color": NAVY, "lw": 0.8, "shrinkA": 1.0, "shrinkB": 2.0},
        color=NAVY,
        fontsize=6.6,
        fontweight="bold",
        va="center",
        ha="left",
    )
    ax.text(
        0.985,
        0.62,
        "Longest trajectory:\n" + f"{int(curve_lengths.max())} observations",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=6.2,
        color=SLATE,
        linespacing=1.35,
    )
    panel_label(ax, "B")

    # (C) Environment coverage -------------------------------------------------
    ax_counts = fig.add_subplot(grid[1, 0])
    positions = np.arange(len(environment))
    soft_grid(ax_counts, axis="x")
    zebra_rows(ax_counts, len(environment))
    ax_counts.barh(positions, environment.observations, color=BLUE, alpha=0.92, height=0.58, zorder=3)
    ax_counts.set(
        yticks=positions,
        yticklabels=environment.display_environment,
        xlabel="Repeated observations",
        ylim=(len(environment) - 0.5, -0.5),
        xlim=(0, float(environment.observations.max()) * 1.30),
    )
    labels = label_barh(
        ax_counts,
        environment.observations,
        positions,
        [f"{int(count)} trajectories" for count in environment.curves],
        color=SLATE,
        fontsize=6.3,
    )
    del labels
    panel_label(ax_counts, "C", dx=-0.62)

    # (D) Observed mass-loss distribution by environment -----------------------
    ax_target = fig.add_subplot(grid[1, 1], sharey=ax_counts)
    soft_grid(ax_target, axis="x")
    zebra_rows(ax_target, len(environment))
    rng = np.random.default_rng(11)
    for index, key in enumerate(environment.degradation_environment):
        values = table.loc[table.degradation_environment.eq(key), "weight_loss"].to_numpy()
        ax_target.scatter(
            values,
            index + rng.uniform(-0.17, 0.17, len(values)),
            s=5,
            color=tint(BLUE, 0.55),
            alpha=0.45,
            edgecolor="none",
            zorder=3,
            rasterized=True,
        )
        box = ax_target.boxplot(
            values,
            positions=[index],
            vert=False,
            widths=0.44,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            medianprops={"color": shade(VERMILION, 0.15), "linewidth": 1.3},
            boxprops={"facecolor": "white", "edgecolor": NAVY, "linewidth": 0.7, "alpha": 0.92},
            whiskerprops={"color": NAVY, "linewidth": 0.7},
            capprops={"color": NAVY, "linewidth": 0.7},
            zorder=4,
        )
        for element in box["boxes"]:
            element.set_zorder(4)
    ax_target.set(xlabel="Observed mass loss (%)", xlim=(-3, 104))
    ax_target.set_xticks(np.arange(0, 101, 25))
    ax_target.tick_params(axis="y", left=False, labelleft=False)
    ax_target.text(
        0.985,
        1.045,
        "Box: median and IQR · whiskers: 1.5 × IQR · points: observations",
        transform=ax_target.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
        color=SLATE,
    )
    panel_label(ax_target, "D", dx=-0.10)

    save_figure(fig, "Figure_2_dataset_composition")


def figure_3() -> None:
    metrics = pd.read_csv(ROOT / "results" / "metrics" / "screening_metrics.csv")
    order = ["random_row", "curve_grouped", "leave_one_study_out"]
    labels = ["Observations\nmixed", "Trajectories\nwithheld", "Studies\nwithheld"]
    fixed = (
        metrics.loc[metrics.model.eq("xgboost") & metrics.task.isin(order)]
        .set_index("task")
        .loc[order]
    )
    repeated = pd.read_csv(ROOT / "results" / "metrics" / "repeated_split_metrics.csv")
    loso = pd.read_csv(
        ROOT / "results" / "metrics" / "screening_study_metrics.csv",
        dtype={"study_id": str},
    )
    loso = loso.loc[
        loso.task.eq("leave_one_study_out") & loso.model.eq("xgboost"), "mae"
    ]
    distributions = [
        repeated.loc[repeated.task.eq("repeated_random_row"), "mae"].to_numpy(),
        repeated.loc[repeated.task.eq("repeated_curve_grouped"), "mae"].to_numpy(),
        loso.to_numpy(),
    ]

    fig = plt.figure(figsize=(7.2, 3.30))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.42], wspace=0.28)
    colors = [BLUE, AMBER, VERMILION]

    # (A) Fixed-assignment escalation -----------------------------------------
    ax = fig.add_subplot(grid[0, 0])
    soft_grid(ax, axis="y")
    values = fixed.point_mae.to_numpy(float)
    ax.bar(np.arange(3), values, color=colors, width=0.62, zorder=3)
    ax.set(
        xticks=np.arange(3),
        xticklabels=labels,
        xlabel="Same gradient-boosting relation",
        ylabel="Point MAE (percentage points)",
        xlim=(-0.62, 2.62),
        ylim=(0, values.max() * 1.42),
    )
    ax.tick_params(axis="x", labelsize=6.4)
    label_bar(ax, values, np.arange(3), [f"{value:.1f}" for value in values], fontsize=7.4)
    for index in range(2):
        step = values[index + 1] / values[index]
        ax.annotate(
            "",
            xy=(index + 0.72, values[index + 1]),
            xytext=(index + 0.28, values[index]),
            arrowprops={"arrowstyle": "-|>,head_width=0.13,head_length=0.28", "color": SLATE, "lw": 0.8, "connectionstyle": "arc3,rad=-0.22"},
        )
        ax.text(
            index + 0.5,
            (values[index] + values[index + 1]) / 2 - values.max() * 0.055,
            f"×{step:.1f}",
            ha="center",
            va="top",
            fontsize=6.6,
            color=SLATE,
            fontweight="bold",
        )
    inflation = values[-1] / values[0]
    ax.text(
        0.02,
        0.975,
        f"Withholding a whole study\ninflates error {inflation:.1f}-fold",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=shade(VERMILION, 0.2),
        fontweight="bold",
        linespacing=1.4,
    )
    panel_label(ax, "A", dx=-0.235)

    # (B) Repeated-assignment distributions ------------------------------------
    ax = fig.add_subplot(grid[0, 1])
    soft_grid(ax, axis="y")
    box = ax.boxplot(
        distributions,
        positions=np.arange(3),
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": NAVY, "linewidth": 1.3},
        whiskerprops={"linewidth": 0.8, "color": NAVY},
        capprops={"linewidth": 0.8, "color": NAVY},
        zorder=3,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(tint(color, 0.80))
        patch.set_edgecolor(color)
        patch.set_linewidth(0.8)
    rng = np.random.default_rng(42)
    for index, (values_index, color) in enumerate(zip(distributions, colors)):
        jitter = rng.uniform(-0.15, 0.15, len(values_index))
        ax.scatter(index + jitter, values_index, s=11, alpha=0.55, color=color, edgecolor="none", zorder=4)
        ax.scatter(
            index,
            np.mean(values_index),
            marker="D",
            s=30,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
        )
    upper = max(float(np.max(np.concatenate(distributions))), 55.0)
    ax.set(
        xticks=np.arange(3),
        xticklabels=["Observations\nmixed (n = 50)", "Trajectories\nwithheld (n = 50)", "Studies\nwithheld (n = 16)"],
        xlabel="Level of experimental separation",
        ylabel="MAE (percentage points)",
        xlim=(-0.60, 3.02),
        ylim=(0, upper * 1.06),
    )
    for index, values_index in enumerate(distributions):
        ax.text(
            index + 0.32,
            float(np.mean(values_index)),
            f"mean {np.mean(values_index):.1f}",
            ha="left",
            va="center",
            fontsize=6.2,
            color=shade(colors[index], 0.2),
            fontweight="bold",
        )
    ax.text(
        1.0,
        1.015,
        "Diamonds mark group means; boxes show the median and interquartile range",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=SLATE,
        fontsize=6.0,
    )
    panel_label(ax, "B", dx=-0.115)

    save_figure(fig, "Figure_3_split_strictness_performance")


def figure_4() -> None:
    shift = pd.read_csv(
        ROOT / "results" / "metrics" / "study_level_shift_summary.csv",
        dtype={"study_id": str},
    ).sort_values("mae", ascending=False).reset_index(drop=True)
    rows = len(shift)
    y = np.arange(rows) + 0.5
    focus_colors = {"9": VERMILION, "19": DEEP_VERMILION, "12": GREEN, "14": BLUE}
    colors = [focus_colors.get(study, MUTED) for study in shift.study_id]

    fig = plt.figure(figsize=(8.4, 5.60))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.30, 3.00, 1.78],
        height_ratios=[1.0, 0.085],
        wspace=0.10,
        hspace=0.30,
    )
    ax_mae = fig.add_subplot(grid[0, 0])
    ax_shift = fig.add_subplot(grid[0, 1])
    ax_context = fig.add_subplot(grid[0, 2])
    ax_cbar = fig.add_subplot(grid[1, 1])

    def stripe(ax) -> None:
        for row in range(0, rows, 2):
            ax.axhspan(row, row + 1, color=ZEBRA, linewidth=0, zorder=0)

    # (A) Complete-study error --------------------------------------------------
    stripe(ax_mae)
    ax_mae.grid(True, axis="x", color=RULE, linewidth=0.55, zorder=1)
    ax_mae.set_axisbelow(True)
    ax_mae.barh(y, shift.mae, color=colors, height=0.62, zorder=3)
    ax_mae.set(
        yticks=y,
        yticklabels=[f"Study {study}  (n = {n})" for study, n in zip(shift.study_id, shift.n)],
        xlabel="Complete-study MAE\n(percentage points)",
        ylabel="Withheld source study",
        xlim=(0, 62),
        ylim=(rows, 0),
    )
    ax_mae.set_xticks([0, 20, 40, 60])
    label_barh(ax_mae, shift.mae, y, [f"{value:.1f}" for value in shift.mae], color=NAVY, fontsize=6.4)
    for tick, study in zip(ax_mae.get_yticklabels(), shift.study_id):
        if study in focus_colors:
            tick.set_fontweight("bold")
            tick.set_color(shade(focus_colors[study], 0.25))

    # (B) Environmental dissimilarity and response descriptors ------------------
    metric_columns = [
        "knn_distance_mean",
        "numeric_features_outside_training_range_count",
        "unseen_category_feature_count",
        "target_mean",
        "target_max",
        "time_max",
    ]
    metric_labels = [
        "Neighbor\ndistance",
        "Out-of-range\ndescriptors",
        "Unseen\nlevels",
        "Mean loss\n(%)",
        "Max loss\n(%)",
        "Max time\n(days)",
    ]
    raw = shift[metric_columns].to_numpy(float)
    ranked = shift[metric_columns].rank(pct=True, method="average").to_numpy(float)
    annotations = np.empty(raw.shape, dtype=object)
    annotations[:, 0] = [f"{value:.1f}" for value in raw[:, 0]]
    for column in range(1, raw.shape[1]):
        annotations[:, column] = [f"{value:.0f}" for value in raw[:, column]]
    sns.heatmap(
        ranked,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 6.6},
        cmap=sns.blend_palette(["#FFFFFF", tint(BLUE, 0.55), BLUE], as_cmap=True),
        vmin=0,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        xticklabels=metric_labels,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        cbar_kws={"orientation": "horizontal"},
        ax=ax_shift,
    )
    ax_shift.set(xlabel="Environmental dissimilarity and response descriptors", ylabel="")
    ax_shift.tick_params(axis="x", rotation=0, labelsize=6.0, pad=2, length=0)
    ax_shift.xaxis.set_ticks_position("top")
    ax_shift.xaxis.set_label_position("bottom")
    for spine in ax_shift.spines.values():
        spine.set_visible(False)
    ax_cbar.set_xlabel("Percentile within descriptor (cell colour); cell text is the raw value", fontsize=6.2, labelpad=2.5)
    ax_cbar.tick_params(labelsize=6.0, length=2.0, pad=1.5)
    ax_cbar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    for spine in ax_cbar.spines.values():
        spine.set_visible(True)
        spine.set_color(tint(NAVY, 0.6))
        spine.set_linewidth(0.5)

    # (C) Experimental context --------------------------------------------------
    environment_colors = {
        "soil": GREEN,
        "compost_organic_fertilizer": AMBER,
        "marine_aquatic": SKY,
        "laboratory_mineral_media": PURPLE,
    }
    environment_display = {
        "soil": "Soil",
        "compost_organic_fertilizer": "Compost",
        "marine_aquatic": "Marine",
        "laboratory_mineral_media": "Mineral",
    }
    ax_context.set_xlim(0, 2.40)
    ax_context.set_ylim(rows, 0)
    ax_context.axis("off")
    stripe(ax_context)
    ax_context.text(0.02, -0.62, "Environment", fontweight="bold", fontsize=6.8, color=NAVY, va="center")
    ax_context.text(1.02, -0.62, "Condition / scale", fontweight="bold", fontsize=6.8, color=NAVY, va="center")
    for index, row in shift.iterrows():
        environments = row.environment.split("; ")
        primary_environment = environments[0]
        display_environment = environment_display.get(primary_environment, "Mixed")
        swatch = environment_colors.get(primary_environment, MUTED)
        if len(environments) > 1:
            display_environment = "Mixed"
            swatch = MUTED
        ax_context.add_patch(
            mpl.patches.FancyBboxPatch(
                (0.02, index + 0.19),
                0.84,
                0.62,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=tint(swatch, 0.80),
                edgecolor=tint(swatch, 0.45),
                linewidth=0.5,
                zorder=2,
            )
        )
        ax_context.text(0.10, index + 0.50, display_environment, va="center", fontsize=6.4, color=shade(swatch, 0.35), zorder=3)
        condition = "Aerobic / anaerobic" if ";" in row.experimental_condition else row.experimental_condition.title()
        ax_context.text(1.02, index + 0.50, f"{condition} · {row.experimental_scale}", va="center", fontsize=6.3, color=SLATE, zorder=3)

    for ax, letter, dx, dy in [(ax_mae, "A", -0.36, 1.150), (ax_shift, "B", -0.055, 1.150), (ax_context, "C", -0.035, 1.150)]:
        panel_label(ax, letter, dx=dx, dy=dy, fontsize=10.0)

    save_figure(fig, "Figure_4_study_level_shift")


def figure_5() -> None:
    final = pd.read_csv(ROOT / "results" / "predictions" / "final_conformal_predictions.csv", dtype={"study_id": str})
    unconstrained = pd.read_csv(ROOT / "results" / "predictions" / "framework_ablation_predictions.csv", dtype={"study_id": str})
    unconstrained = unconstrained[unconstrained.model.eq("direct_unconstrained_row")][["row_id", "y_pred"]].rename(columns={"y_pred": "unconstrained_pred"})
    final = final.merge(unconstrained, on="row_id", validate="one_to_one")
    selected = {
        "study_9__instance_4": ("Study 9 · trajectory 4", "Endpoint-orientation conflict", VERMILION),
        "study_19__instance_2": ("Study 19 · trajectory 2", "Rapid protocol shift", DEEP_VERMILION),
        "study_12__instance_3": ("Study 12 · trajectory 3", "Lower-error transfer", GREEN),
        "study_14__instance_1": ("Study 14 · trajectory 1", "Lower-error transfer", GREEN),
    }

    fig = plt.figure(figsize=(7.2, 5.30))
    grid = fig.add_gridspec(2, 2, wspace=0.27, hspace=0.62)
    axes = [fig.add_subplot(grid[position // 2, position % 2]) for position in range(4)]

    for letter, ax, (curve_id, (display_id, diagnostic, diagnostic_color)) in zip(ascii_uppercase, axes, selected.items()):
        curve = final[final.curve_id.eq(curve_id)].sort_values(["degradation_time_days", "curve_position"])
        soft_grid(ax, axis="y")
        ax.fill_between(
            curve.degradation_time_days,
            curve.lower_90,
            curve.upper_90,
            color=BLUE,
            alpha=0.14,
            linewidth=0,
            zorder=2,
            label="90% empirical error band",
        )
        ax.plot(curve.degradation_time_days, curve.unconstrained_pred, ls=(0, (4, 2)), color="#6D7378", lw=1.2, zorder=4, label="Direct unconstrained")
        ax.plot(curve.degradation_time_days, curve.y_pred, color=BLUE, lw=1.7, zorder=5, label="Proposed framework")
        marker_size = 24 if len(curve) <= 16 else 11
        ax.scatter(
            curve.degradation_time_days,
            curve.y_true,
            facecolor="white",
            edgecolor=NAVY,
            linewidth=0.75,
            s=marker_size,
            zorder=6,
            label="Observed",
        )
        mae = float(np.mean(np.abs(curve.y_true - curve.y_pred)))
        span = float(curve.degradation_time_days.max() - curve.degradation_time_days.min())
        ax.set(
            xlabel="Degradation time (days)",
            ylabel="Mass loss (%)",
            ylim=(-4, 104),
            xlim=(
                float(curve.degradation_time_days.min()) - 0.05 * span,
                float(curve.degradation_time_days.max()) + 0.05 * span,
            ),
        )
        ax.set_yticks(np.arange(0, 101, 25))
        ax.text(0.0, 1.205, display_id, transform=ax.transAxes, ha="left", va="baseline", fontsize=7.6, fontweight="bold", color=NAVY)
        ax.text(1.0, 1.205, f"MAE = {mae:.1f} pp", transform=ax.transAxes, ha="right", va="baseline", fontsize=7.0, color=SLATE)
        pill(ax, 0.0, 1.075, diagnostic, diagnostic_color, fontsize=6.3, ha="left", va="bottom")
        panel_label(ax, letter, dx=-0.175, dy=1.185)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        handlelength=1.9,
        columnspacing=2.0,
    )
    save_figure(fig, "Figure_5_representative_heldout_trajectories")


def figure_6() -> None:
    dense = pd.read_csv(ROOT / "results" / "metrics" / "framework_dense_physical_metrics.csv")
    models = ["direct_unconstrained_row", "bounded_unconstrained_row", "direct_unconstrained_row_isotonic", "bounded_monotonic_study_weighted"]
    labels = ["Unconstrained", "Percentage-bounded", "Monotonic correction", "Physically constrained"]
    subset = dense.set_index("model").loc[models].reset_index()
    colors = [VERMILION, AMBER, GREEN, BLUE]
    y = np.arange(len(labels))
    metrics = [
        (100 * subset.negative_prediction_fraction, "Negative estimates (%)"),
        (100 * subset.monotonicity_violation_fraction, "Decreasing grid steps (%)"),
        (subset.mean_prediction_at_time_zero, "Mean estimate at day 0 (%)"),
    ]

    fig = plt.figure(figsize=(7.2, 3.05))
    grid = fig.add_gridspec(1, 3, wspace=0.16)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    for index, (ax, (values, xlabel)) in enumerate(zip(axes, metrics)):
        values = np.array([0.0 if abs(float(value)) < 0.05 else float(value) for value in values])
        for row in range(0, len(labels), 2):
            ax.axhspan(row - 0.5, row + 0.5, color=ZEBRA, linewidth=0, zorder=0)
        ax.grid(True, axis="x", color=RULE, linewidth=0.55, zorder=1)
        ax.set_axisbelow(True)
        upper = max(1.0, float(np.nanmax(values)))
        ax.set(ylim=(len(labels) - 0.5, -0.5), xlim=(-0.02 * upper, upper * 1.30))
        for row, (value, color) in enumerate(zip(values, colors)):
            ax.hlines(row, 0, value, color=color, lw=2.2, alpha=0.75, zorder=3)
            ax.scatter(value, row, s=42, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        label_barh(
            ax,
            values,
            y,
            [f"{value:.1f}" for value in values],
            color=NAVY,
            fontsize=6.6,
            gap=0.030,
            fontweight="bold",
        )
        ax.set_xlabel(xlabel)
        if index == 0:
            ax.set(yticks=y, yticklabels=labels, ylabel="Prediction formulation")
            for tick, color in zip(ax.get_yticklabels(), colors):
                tick.set_color(shade(color, 0.28))
            ax.get_yticklabels()[-1].set_fontweight("bold")
        else:
            ax.set_yticks(y)
            ax.tick_params(axis="y", left=False, labelleft=False)
        panel_label(ax, ascii_uppercase[index], dx=-0.70 if index == 0 else -0.09, dy=1.055)

    fig.text(
        0.995,
        1.015,
        "Zero is the physically admissible value in panels A and B",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=SLATE,
        transform=fig.transFigure,
    )
    fig.subplots_adjust(left=0.19, right=0.995, bottom=0.19, top=0.90, wspace=0.16)
    save_figure(fig, "Figure_6_physical_validity")


def figure_7() -> None:
    intervals = pd.read_csv(ROOT / "results" / "metrics" / "conformal_interval_metrics.csv")
    study90 = pd.read_csv(ROOT / "results" / "metrics" / "conformal_subgroup_metrics.csv", dtype={"group": str})
    study90 = study90[(study90.grouping == "study_id") & np.isclose(study90.nominal_coverage, 0.9)].copy()
    study90["order"] = pd.to_numeric(study90.group)
    study90 = study90.sort_values("order")

    colors = [VERMILION if group == "9" else BLUE for group in study90.group]
    pooled_width = float(intervals.loc[np.isclose(intervals.nominal_coverage, 0.9), "mean_interval_width"].iloc[0])

    fig = plt.figure(figsize=(7.2, 2.95))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.88, 1.32, 1.32], wspace=0.34)

    # (A) Calibration ------------------------------------------------------------
    ax = fig.add_subplot(grid[0, 0])
    soft_grid(ax, axis="both")
    ax.fill_between(
        intervals.nominal_coverage,
        intervals.nominal_coverage,
        intervals.empirical_coverage,
        color=BLUE,
        alpha=0.13,
        linewidth=0,
        zorder=2,
    )
    ax.plot([0.75, 1.0], [0.75, 1.0], ls=(0, (4, 2)), color=SLATE, lw=0.9, zorder=3, label="Nominal = empirical")
    ax.plot(intervals.nominal_coverage, intervals.empirical_coverage, marker="o", ms=4.6, color=BLUE, lw=1.5, zorder=4, markeredgecolor="white", markeredgewidth=0.7, label="Observed")
    for nominal, empirical in zip(intervals.nominal_coverage, intervals.empirical_coverage):
        ax.annotate(
            f"{empirical:.3f}",
            xy=(nominal, empirical),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=shade(BLUE, 0.2),
            fontweight="bold",
        )
    ax.set(
        xlabel="Nominal coverage",
        ylabel="Empirical point coverage",
        xlim=(0.775, 0.975),
        ylim=(0.775, 1.012),
    )
    ax.set_xticks([0.80, 0.85, 0.90, 0.95])
    ax.legend(loc="lower right", fontsize=6.0, handlelength=1.5, borderpad=0.2)
    panel_label(ax, "A", dx=-0.31)

    # (B) Band width by withheld study --------------------------------------------
    ax = fig.add_subplot(grid[0, 1])
    soft_grid(ax, axis="y")
    ax.bar(study90.group, study90.mean_width, color=colors, width=0.72, zorder=3)
    ax.axhline(pooled_width, color=NAVY, ls=(0, (4, 2)), lw=1.0, zorder=5, label=f"Pooled width = {pooled_width:.1f} pp")
    ax.set(xlabel="Withheld source study", ylabel="Mean 90% band width\n(percentage points)", ylim=(0, 118))
    ax.set_yticks(np.arange(0, 101, 25))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.145), fontsize=6.2, handlelength=1.9, borderpad=0.15)
    panel_label(ax, "B", dx=-0.185, dy=1.16)

    # (C) Coverage by withheld study ----------------------------------------------
    ax = fig.add_subplot(grid[0, 2])
    soft_grid(ax, axis="y")
    ax.bar(study90.group, study90.empirical_coverage, color=colors, width=0.72, zorder=3)
    ax.axhline(0.9, color=NAVY, ls=(0, (4, 2)), lw=1.0, zorder=5, label="Nominal coverage = 0.90")
    ax.set(xlabel="Withheld source study", ylabel="Empirical 90% coverage", ylim=(0, 1.22))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    study9 = study90.loc[study90.group.eq("9")].iloc[0]
    study9_index = list(study90.group).index("9")
    # The leader runs down the empty column above the study-9 bar, and the text
    # sits above every other bar, so neither overlaps the data.
    ax.annotate(
        f"Study 9: {study9.empirical_coverage:.3f}",
        xy=(study9_index, study9.empirical_coverage),
        xytext=(study9_index, 1.045),
        arrowprops={"arrowstyle": "-|>,head_width=0.12,head_length=0.26", "color": shade(VERMILION, 0.15), "lw": 0.8, "shrinkA": 2.0, "shrinkB": 2.0},
        color=shade(VERMILION, 0.15),
        ha="center",
        va="bottom",
        fontsize=6.3,
        fontweight="bold",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.145), fontsize=6.2, handlelength=1.9, borderpad=0.15)
    panel_label(ax, "C", dx=-0.185, dy=1.16)

    for ax in fig.axes[1:]:
        ax.tick_params(axis="x", labelsize=6.2)
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.20, top=0.86, wspace=0.34)
    save_figure(fig, "Figure_7_uncertainty_and_applicability")


def figure_s1() -> None:
    time_stage = pd.read_csv(ROOT / "results" / "metrics" / "time_stage_error.csv")
    time_order = ["0-<7 d", "7-<30 d", "30-<90 d", ">=90 d"]
    time_stage["time_stage"] = pd.Categorical(time_stage.time_stage, time_order, ordered=True)
    time_stage = time_stage.sort_values("time_stage")
    sensitivity = pd.read_csv(ROOT / "results" / "metrics" / "revision_sensitivity_metrics.csv")
    weights = pd.read_csv(ROOT / "results" / "metrics" / "weight_distribution_by_fold.csv")

    fig = plt.figure(figsize=(7.2, 2.95))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.20, 0.86], wspace=0.44)

    # (A) Error by elapsed-time interval ----------------------------------------
    ax = fig.add_subplot(grid[0, 0])
    soft_grid(ax, axis="y")
    values = time_stage.mae.to_numpy(float)
    ax.bar(np.arange(4), values, color=BLUE, width=0.64, zorder=3)
    # Interval sizes ride in the tick labels so that no annotation is wider
    # than the bar it describes.
    interval_labels = [
        f"{window}\n(n = {n})"
        for window, n in zip(["0–<7", "7–<30", "30–<90", "≥90"], time_stage.n)
    ]
    ax.set(
        xticks=np.arange(4),
        xticklabels=interval_labels,
        xlabel="Degradation time (days)",
        ylabel="LOSO point MAE\n(percentage points)",
        ylim=(0, values.max() * 1.22),
        xlim=(-0.62, 3.62),
    )
    ax.tick_params(axis="x", labelsize=6.4)
    label_bar(
        ax,
        values,
        np.arange(4),
        [f"{value:.1f}" for value in values],
        fontsize=6.8,
    )
    panel_label(ax, "A", dx=-0.31)

    # (B) Weighting and composition sensitivities --------------------------------
    ax = fig.add_subplot(grid[0, 1])
    selected_models = [
        "bounded_monotonic_row",
        "bounded_monotonic_study_only",
        "bounded_monotonic_study_curve",
        "bounded_monotonic_study_curve_capped95",
        "bounded_monotonic_hb_only",
        "bounded_monotonic_hv_only",
    ]
    labels = ["Row weighting", "Study only", "Study–curve", "Capped 95%", "HB ratio only", "HV ratio only"]
    subset = sensitivity.set_index("model").loc[selected_models]
    colors = [MUTED, AMBER, BLUE, GREEN, PURPLE, VERMILION]
    positions = np.arange(len(subset))
    for row in range(0, len(subset), 2):
        ax.axhspan(row - 0.5, row + 0.5, color=ZEBRA, linewidth=0, zorder=0)
    ax.grid(True, axis="x", color=RULE, linewidth=0.55, zorder=1)
    ax.set_axisbelow(True)
    values = subset.macro_study_mae.to_numpy(float)
    ax.barh(positions, values, color=colors, height=0.60, zorder=3)
    ax.set(
        yticks=positions,
        yticklabels=labels,
        xlabel="Study-equal MAE (percentage points)",
        ylim=(len(subset) - 0.5, -0.5),
        xlim=(0, values.max() * 1.20),
    )
    label_barh(ax, values, positions, [f"{value:.2f}" for value in values], color=SLATE, fontsize=6.2)
    panel_label(ax, "B", dx=-0.34)

    # (C) Fold-wise maximum row weight -------------------------------------------
    ax = fig.add_subplot(grid[0, 2])
    soft_grid(ax, axis="y")
    strategies = ["study_only", "study_curve", "study_curve_capped95"]
    weight_data = [weights.loc[weights.strategy.eq(strategy), "maximum"].to_numpy() for strategy in strategies]
    strategy_colors = [AMBER, BLUE, GREEN]
    boxes = ax.boxplot(
        weight_data,
        positions=np.arange(3),
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        manage_ticks=False,
        medianprops={"color": NAVY, "linewidth": 1.2},
        whiskerprops={"color": NAVY, "linewidth": 0.8},
        capprops={"color": NAVY, "linewidth": 0.8},
        zorder=3,
    )
    for patch, color in zip(boxes["boxes"], strategy_colors):
        patch.set_facecolor(tint(color, 0.72))
        patch.set_edgecolor(color)
        patch.set_linewidth(0.8)
    ax.set(
        xticks=np.arange(3),
        xticklabels=["Study\nonly", "Study–\ncurve", "Capped\n95%"],
        xlabel="Weighting strategy",
        ylabel="Maximum normalized\nrow weight",
        xlim=(-0.62, 2.62),
        ylim=(0, None),
    )
    panel_label(ax, "C", dx=-0.42)

    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.215, top=0.925, wspace=0.44)
    save_figure(fig, "Figure_S1_revision_sensitivities")


def figure_8() -> None:
    shap_imp = pd.read_csv(ROOT / "results" / "metrics" / "shap_grouped_importance.csv").head(10).sort_values("mean_absolute_shap_logit_scale")
    permutation = pd.read_csv(ROOT / "results" / "metrics" / "heldout_grouped_permutation_importance.csv").head(10).sort_values("mean_study_equal_mae_increase")
    counter = pd.read_csv(ROOT / "results" / "metrics" / "counterfactual_trajectories.csv")
    counter = counter[counter.scenario_type.eq("environment")]
    interactions = pd.read_csv(ROOT / "results" / "metrics" / "shap_grouped_interactions.csv").head(10).sort_values("mean_absolute_shap_interaction_logit_scale")

    feature_labels = {
        "degradation_time_days": "Time",
        "adjusted_hb_ratio_formulation_mol": "HB ratio",
        "adjusted_hv_ratio_formulation_mol": "HV ratio",
        "hydroxybutyrate_pct": "HB ratio",
        "hydroxyvalerate_pct": "HV ratio",
        "degradation_environment": "Environment",
        "degradation_condition": "Condition",
        "t_deg": "Temperature",
        "temperature_c": "Temperature",
        "sample_shape_morphology": "Morphology",
        "sample_morphology": "Morphology",
        "pha_degrading_microbes": "Microbes",
        "microorganism_presence": "Microbes",
        "experimental_scale": "Scale",
        "degradation_scale": "Scale",
        "degradation_mechanism": "Mechanism",
        "additive1_percentage_wt": "Additive 1",
        "additive2_percentage_wt": "Additive 2",
        "additive3_percentage_wt": "Additive 3",
        "additive_1_pct": "Additive 1",
        "additive_2_pct": "Additive 2",
        "additive_3_pct": "Additive 3",
        "additive_type_1": "Additive 1 type",
        "additive_type_2": "Additive 2 type",
        "additive_type_3": "Additive 3 type",
        "additive_1_type": "Additive 1 type",
        "additive_2_type": "Additive 2 type",
        "additive_3_type": "Additive 3 type",
        "additive1_concentration_reported": "Additive 1 amount reported",
        "additive2_concentration_reported": "Additive 2 amount reported",
        "additive3_concentration_reported": "Additive 3 amount reported",
        "additive1_concentration_ambiguous": "Additive 1 amount ambiguous",
        "additive2_concentration_ambiguous": "Additive 2 amount ambiguous",
        "additive3_concentration_ambiguous": "Additive 3 amount ambiguous",
        "temperature_was_reported": "Temperature reported",
        "additives": "Additives",
        "sample_type": "Sample type",
        "phbv_pct": "PHBV content",
    }

    def pretty(values: pd.Series) -> pd.Series:
        return values.map(feature_labels).fillna(values.str.replace("_", " ").str.title())

    fig = plt.figure(figsize=(7.2, 5.45))
    # The right-hand panels carry long categorical tick labels, so the column
    # gap is sized to hold them outside the neighbouring axes.
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.02], height_ratios=[1.0, 1.04], wspace=0.56, hspace=0.52)

    # (A) Full-data SHAP importance ---------------------------------------------
    ax = fig.add_subplot(grid[0, 0])
    names = pretty(shap_imp.original_feature).tolist()
    values = shap_imp.mean_absolute_shap_logit_scale.to_numpy(float)
    positions = np.arange(len(names))
    soft_grid(ax, axis="x")
    zebra_rows(ax, len(names))
    ax.barh(positions, values, color=BLUE, height=0.62, zorder=3)
    ax.set(yticks=positions, yticklabels=names, xlabel="Mean |SHAP| (logit scale)", ylabel="Feature group", ylim=(-0.5, len(names) - 0.5), xlim=(0, values.max() * 1.22))
    label_barh(ax, values, positions, [f"{value:.2f}" for value in values], color=SLATE, fontsize=6.1)
    panel_label(ax, "A", dx=-0.42)

    # (B) Held-out grouped permutation importance --------------------------------
    ax = fig.add_subplot(grid[0, 1])
    names = pretty(permutation.feature).tolist()
    values = permutation.mean_study_equal_mae_increase.to_numpy(float)
    positions = np.arange(len(names))
    soft_grid(ax, axis="x")
    zebra_rows(ax, len(names))
    ax.barh(positions, values, color=[GREEN if value >= 0 else VERMILION for value in values], height=0.62, zorder=3)
    ax.axvline(0, color=NAVY, lw=0.7, zorder=4)
    ax.set(yticks=positions, yticklabels=names, xlabel="Held-out MAE increase\n(percentage points)", ylim=(-0.5, len(names) - 0.5), xlim=(0, values.max() * 1.24))
    label_barh(ax, values, positions, [f"{value:.2f}" for value in values], color=SLATE, fontsize=6.1)
    panel_label(ax, "B", dx=-0.50)

    # (C) Environment counterfactual trajectories --------------------------------
    ax = fig.add_subplot(grid[1, 0])
    # Short keys keep the in-axes legend clear of the neighbouring panel; the
    # caption spells the environments out in full.
    environment_display = {
        "compost_organic_fertilizer": "Compost",
        "laboratory_mineral_media": "Mineral medium",
        "marine_aquatic": "Marine",
        "soil": "Soil",
    }
    environment_color = {
        "compost_organic_fertilizer": AMBER,
        "laboratory_mineral_media": VERMILION,
        "marine_aquatic": BLUE,
        "soil": GREEN,
    }
    line_styles = {
        "compost_organic_fertilizer": (0, (5, 2)),
        "laboratory_mineral_media": "-",
        "marine_aquatic": (0, (1, 1.4)),
        "soil": (0, (5, 1.6, 1, 1.6)),
    }
    soft_grid(ax, axis="y")
    groups = {label: group.sort_values("degradation_time_days") for label, group in counter.groupby("scenario_label")}
    time_max = float(counter.degradation_time_days.max())
    order = sorted(groups, key=lambda label: -float(groups[label].prediction_weight_loss_pct.iloc[-1]))
    for label in order:
        group = groups[label]
        colour = environment_color.get(label, MUTED)
        ax.plot(
            group.degradation_time_days,
            group.prediction_weight_loss_pct,
            color=colour,
            ls=line_styles.get(label, "-"),
            lw=1.5,
            zorder=3,
            label=environment_display.get(label, label.replace("_", " ").title()),
        )
    ax.set(xlabel="Degradation time (days)", ylabel="Predicted mass loss (%)", ylim=(0, 100), xlim=(0, time_max))
    ax.set_yticks(np.arange(0, 101, 25))
    # The lower-right quadrant stays empty for every trajectory, so the key can
    # sit inside the axes without covering any line.
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.005, -0.015),
        fontsize=6.1,
        handlelength=2.1,
        handletextpad=0.5,
        labelspacing=0.42,
        borderpad=0.3,
    )
    panel_label(ax, "C", dx=-0.235)

    # (D) Grouped SHAP interactions ----------------------------------------------
    ax = fig.add_subplot(grid[1, 1])
    names = (pretty(interactions.feature_1) + " × " + pretty(interactions.feature_2)).tolist()
    values = interactions.mean_absolute_shap_interaction_logit_scale.to_numpy(float)
    positions = np.arange(len(names))
    soft_grid(ax, axis="x")
    zebra_rows(ax, len(names))
    ax.barh(positions, values, color=AMBER, height=0.62, zorder=3)
    ax.set(yticks=positions, yticklabels=names, xlabel="Mean |interaction SHAP|\n(logit scale)", ylim=(-0.5, len(names) - 0.5), xlim=(0, values.max() * 1.24))
    ax.set_xticks(np.arange(0, values.max() * 1.05, 0.05))
    label_barh(ax, values, positions, [f"{value:.3f}" for value in values], color=SLATE, fontsize=6.1)
    panel_label(ax, "D", dx=-0.50)

    fig.subplots_adjust(left=0.128, right=0.995, bottom=0.115, top=0.955, wspace=0.56, hspace=0.52)
    save_figure(fig, "Figure_8_model_interpretation")


def figure_9() -> None:
    metrics = pd.read_csv(ROOT / "results" / "metrics" / "framework_ablation_metrics.csv")
    ci = pd.read_csv(ROOT / "results" / "statistics" / "model_macro_study_bootstrap_ci.csv")
    dense = pd.read_csv(ROOT / "results" / "metrics" / "framework_dense_physical_metrics.csv")
    selected = [
        "direct_unconstrained_row",
        "bounded_unconstrained_row",
        "bounded_monotonic_row",
        "bounded_monotonic_study_weighted",
        "bounded_monotonic_study_weighted_cleaned_target",
        "bounded_monotonic_study_weighted_cleaned_time",
        "bounded_monotonic_study_weighted_no_temperature",
    ]
    data = (
        metrics.set_index("model")
        .loc[selected]
        .reset_index()
        .merge(ci[["model", "bootstrap_ci_lower", "bootstrap_ci_upper"]], on="model")
        .merge(
            dense[["model", "monotonicity_violation_fraction"]].rename(
                columns={"monotonicity_violation_fraction": "dense_monotonicity_violation_fraction"}
            ),
            on="model",
        )
    )
    labels = data.model.map(MODEL_LABELS).fillna(data.model.str.replace("_", " "))
    y = np.arange(len(data))
    highlight = data.model.eq("bounded_monotonic_study_weighted").to_numpy()
    row_colors = [BLUE if flag else MUTED for flag in highlight]
    proposed = float(data.loc[highlight, "macro_study_mae"].iloc[0])

    fig = plt.figure(figsize=(8.2, 3.55))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.50, 0.88, 0.88], wspace=0.13)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    def stripe(ax) -> None:
        for row in range(0, len(data), 2):
            ax.axhspan(row - 0.5, row + 0.5, color=ZEBRA, linewidth=0, zorder=0)
        for row in np.flatnonzero(highlight):
            ax.axhspan(row - 0.5, row + 0.5, color=tint(BLUE, 0.90), linewidth=0, zorder=1)

    # (A) Study-equal MAE with bootstrap intervals ------------------------------
    ax = axes[0]
    stripe(ax)
    ax.grid(True, axis="x", color=RULE, linewidth=0.55, zorder=2)
    ax.set_axisbelow(True)
    ax.axvline(proposed, color=tint(BLUE, 0.4), lw=0.9, ls=(0, (4, 2)), zorder=2)
    xerr = np.vstack([data.macro_study_mae - data.bootstrap_ci_lower, data.bootstrap_ci_upper - data.macro_study_mae])
    for index, color in enumerate(row_colors):
        ax.errorbar(
            data.macro_study_mae.iloc[index],
            y[index],
            xerr=np.array([[xerr[0, index]], [xerr[1, index]]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.6,
            capthick=1.0,
            ms=5.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=4,
        )
    ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Study-equal MAE (percentage points)\nwith 95% study-bootstrap interval",
        ylabel="Analysis formulation",
        ylim=(len(data) - 0.5, -0.5),
    )
    for tick, flag in zip(ax.get_yticklabels(), highlight):
        if flag:
            tick.set_fontweight("bold")
            tick.set_color(shade(BLUE, 0.2))

    # (B) Observation-weighted MAE and (C) monotonicity violations ---------------
    for index, (ax, values, xlabel, formatter) in enumerate(
        [
            (axes[1], data.point_mae.to_numpy(float), "Point MAE\n(percentage points)", "{:.1f}"),
            (axes[2], 100 * data.dense_monotonicity_violation_fraction.to_numpy(float), "Decreasing grid\nsteps (%)", "{:.1f}"),
        ]
    ):
        stripe(ax)
        ax.grid(True, axis="x", color=RULE, linewidth=0.55, zorder=2)
        ax.set_axisbelow(True)
        ax.barh(y, values, color=row_colors, height=0.56, alpha=0.92, zorder=3)
        ax.set(yticks=y, yticklabels=[], xlabel=xlabel, ylim=(len(data) - 0.5, -0.5), xlim=(0, max(values.max(), 1.0) * 1.28))
        ax.tick_params(axis="y", left=False)
        label_barh(ax, values, y, [formatter.format(value) for value in values], color=SLATE, fontsize=6.2)

    for ax, letter, dx in [(axes[0], "A", -0.475), (axes[1], "B", -0.075), (axes[2], "C", -0.075)]:
        panel_label(ax, letter, dx=dx, dy=1.045)

    fig.subplots_adjust(left=0.325, right=0.995, bottom=0.235, top=0.925, wspace=0.13)
    save_figure(fig, "Figure_9_ablation_summary")


def graphical_abstract() -> None:
    """Preserve the author-selected original; create a fallback only if it is missing."""

    # The author-selected graphical abstract is a preserved repository asset.
    # Do not overwrite it when the analytical figures are regenerated.
    original = ROOT / "figures" / "TOC_graphical_abstract.png"
    if original.exists():
        return

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.set(xlim=(0, 100), ylim=(0, 50))
    ax.axis("off")

    panels = [
        (1.5, 9.0, 22.0, 37.5, "PRIMARY EXPERIMENTS"),
        (26.5, 9.0, 22.0, 37.5, "HIERARCHICAL CORPUS"),
        (51.5, 9.0, 20.5, 37.5, "EXPERIMENTAL INDEPENDENCE"),
        (75.0, 9.0, 23.5, 37.5, "PHYSICALLY FEASIBLE"),
    ]
    for x, y, width, height, title in panels:
        ax.add_patch(
            mpl.patches.FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.55,rounding_size=1.4",
                facecolor="#FBFCFD",
                edgecolor="#C6D2DA",
                linewidth=1.0,
            )
        )
        ax.text(x + width / 2, y + height - 3.0, title, ha="center", va="center", color=NAVY, fontsize=9, fontweight="bold")

    # Panel 1: provenance from primary experiments.
    environment_items = [
        ("Soil", OKABE_ITO[2]),
        ("Marine", OKABE_ITO[0]),
        ("Compost", OKABE_ITO[4]),
        ("Mineral medium", OKABE_ITO[3]),
    ]
    for index, (label, color) in enumerate(environment_items):
        y = 37.6 - index * 5.4
        ax.add_patch(mpl.patches.Circle((5.0, y), 1.05, facecolor=color, edgecolor="white", linewidth=0.8))
        ax.text(7.0, y, label, va="center", color=NAVY, fontsize=7.2)
    ax.plot([4.2, 20.4], [15.1, 15.1], color="#D5DEE4", lw=0.8)
    ax.text(12.5, 13.0, "17 publications  ·  1,546 observations", ha="center", color=NAVY, fontsize=7.2, fontweight="bold")
    ax.text(12.5, 10.8, "gravimetric mass loss  ≠  mineralization", ha="center", color=SLATE, fontsize=6.4)

    # Panel 2: nested evidence structure and repeated trajectories.
    hierarchy = [("17", "studies"), ("133", "trajectories"), ("1,546", "time points")]
    for index, (number, label) in enumerate(hierarchy):
        x = 30.2 + index * 6.9
        ax.add_patch(mpl.patches.Circle((x, 36.8), 2.15, facecolor=["#EAF2F8", "#EDF6F3", "#FFF3DE"][index], edgecolor=[OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[4]][index], linewidth=1.0))
        ax.text(x, 37.2, number, ha="center", va="center", color=NAVY, fontsize=8.2, fontweight="bold")
        ax.text(x, 34.0, label, ha="center", color=SLATE, fontsize=6.3)
        if index < 2:
            ax.annotate("", xy=(x + 4.6, 36.8), xytext=(x + 2.5, 36.8), arrowprops={"arrowstyle": "-|>", "color": "#9EABB5", "lw": 0.8})
    xgrid = np.linspace(29.5, 45.5, 60)
    for offset, color, style in [(0.0, OKABE_ITO[0], "-"), (2.0, OKABE_ITO[2], (0, (4, 2))), (4.0, OKABE_ITO[1], (0, (1, 1)))]:
        ygrid = 17.0 + offset + 8.5 * (1 - np.exp(-(xgrid - 29.5) / (5.5 + offset)))
        ax.plot(xgrid, ygrid, color=color, ls=style, lw=1.35)
        sample = np.linspace(0, len(xgrid) - 1, 6, dtype=int)
        ax.scatter(xgrid[sample], ygrid[sample], s=12, color=color, edgecolor="white", linewidth=0.35)
    ax.text(37.5, 12.0, "repeated measurements nested within trajectories", ha="center", color=SLATE, fontsize=6.4)

    # Panel 3: progressively stricter separation.
    split_rows = [
        ("Individual observations", "same-trajectory interpolation", OKABE_ITO[1]),
        ("Whole trajectories", "new trajectory", OKABE_ITO[4]),
        ("Whole studies", "new environmental experiment", OKABE_ITO[0]),
    ]
    for index, (label, role, color) in enumerate(split_rows):
        y = 36.6 - index * 8.5
        ax.text(54.0, y + 2.2, label, color=NAVY, fontsize=7.0, fontweight="bold")
        ax.text(54.0, y, role, color=SLATE, fontsize=6.0)
        for dot in range(8):
            dot_color = color if (dot < 5 or index == 0 and dot % 2 == 0) else "#D6DDE2"
            marker = "o" if dot % 2 == 0 else "s"
            ax.scatter(65.2 + (dot % 4) * 1.25, y + (dot // 4) * 1.5, s=15, marker=marker, color=dot_color, edgecolor="white", linewidth=0.35)
    ax.add_patch(mpl.patches.FancyBboxPatch((53.6, 11.4), 16.5, 3.3, boxstyle="round,pad=0.25", facecolor="#EAF2F8", edgecolor=OKABE_ITO[0], linewidth=0.8))
    ax.text(61.85, 13.05, "Primary estimand", ha="center", va="center", color=OKABE_ITO[0], fontsize=6.4, fontweight="bold")

    # Panel 4: bounded monotonic prediction with empirical uncertainty.
    xcurve = np.linspace(78.3, 95.6, 90)
    ycurve = 18.0 + 15.5 / (1 + np.exp(-(xcurve - 87.0) / 2.0))
    width = 3.0 + 1.8 * (xcurve - xcurve.min()) / (xcurve.max() - xcurve.min())
    ax.fill_between(xcurve, ycurve - width, ycurve + width, color=OKABE_ITO[0], alpha=0.14)
    ax.plot(xcurve, ycurve, color=OKABE_ITO[0], lw=2.0)
    ax.plot(xcurve, ycurve - 2 * width, color=OKABE_ITO[0], lw=0.8, ls=(0, (4, 2)))
    ax.plot(xcurve, ycurve + 2 * width, color=OKABE_ITO[0], lw=0.8, ls=(0, (4, 2)))
    ax.arrow(78.2, 15.2, 0, 21.0, width=0.08, head_width=0.7, head_length=0.9, color=NAVY, length_includes_head=True)
    ax.arrow(78.2, 15.2, 17.9, 0, width=0.08, head_width=0.7, head_length=0.9, color=NAVY, length_includes_head=True)
    badges = [
        ("bounded", OKABE_ITO[2], "#EAF7F3"),
        ("monotonic", OKABE_ITO[0], "#EAF2F8"),
        ("uncertainty", OKABE_ITO[3], "#F8EDF4"),
    ]
    for index, (label, color, fill) in enumerate(badges):
        ax.add_patch(mpl.patches.FancyBboxPatch((77.8 + index * 6.4, 10.7), 5.8, 2.8, boxstyle="round,pad=0.2", facecolor=fill, edgecolor=color, linewidth=0.7, zorder=5))
        ax.text(80.7 + index * 6.4, 12.1, label, ha="center", va="center", color=NAVY, fontsize=5.4, fontweight="bold", zorder=6)

    # Workflow arrows and evidence-first take-home strip.
    for x in [24.1, 49.1, 72.7]:
        ax.annotate("", xy=(x + 2.0, 27.0), xytext=(x, 27.0), arrowprops={"arrowstyle": "simple", "color": "#9BA8B2", "lw": 0.4})
    ax.add_patch(mpl.patches.FancyBboxPatch((1.5, 1.5), 97.0, 5.1, boxstyle="round,pad=0.35,rounding_size=1.2", facecolor=NAVY, edgecolor=NAVY))
    ax.text(18.0, 4.05, "Mixed-observation MAE  5.0 pp", color="white", ha="center", va="center", fontsize=8.0, fontweight="bold")
    ax.annotate("", xy=(39.2, 4.05), xytext=(30.5, 4.05), arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 1.0})
    ax.text(51.2, 4.05, "Complete-study MAE  21.3 pp", color="white", ha="center", va="center", fontsize=8.0, fontweight="bold")
    ax.plot([69.0, 69.0], [2.5, 5.6], color="#7D92A5", lw=0.8)
    ax.text(83.7, 4.05, "Physical violations eliminated", color="#9DE0CE", ha="center", va="center", fontsize=8.0, fontweight="bold")
    fig.tight_layout(pad=0.2)
    save_figure(fig, "TOC_graphical_abstract")


def generate_tables(table: pd.DataFrame) -> None:
    table1 = pd.DataFrame(
        [
            ["Source studies", 17, 16, "Study 8 absent after source cleaning"],
            ["Curves with time-target pairs", 133, 129, "Four curves / 79 observations excluded"],
            ["Time-target observations", 1546, 1467, "Raw decimal target reconstructed for all modeled rows"],
            ["Time range (days)", "0.081-364.172", "0.081-364.172", "Raw-decimal time is primary; truncated time is sensitivity"],
            ["Mass-loss range (%)", "0.00157-100.33613", "0-100 (source cleaned)", "Primary target retains raw decimal values"],
        ],
        columns=["Item", "Raw source", "Modeling data", "Definition / note"],
    )
    write_table(table1, "Table_1_dataset_hierarchy", "Dataset hierarchy and endpoint definition")

    table2 = pd.DataFrame(
        [
            ["A", "Random row", "80/20, seed 42", "Optimistic interpolation; curves may overlap"],
            ["B", "Unseen curve", "5-fold GroupKFold by curve_id", "No curve crosses folds"],
            ["C", "Unseen study (primary)", "Leave one complete study out", "All preprocessing/tuning uses remaining studies only"],
        ],
        columns=["Task", "Prediction setting", "Split", "Interpretation"],
    )
    write_table(table2, "Table_2_prediction_tasks", "Prediction tasks and split definitions")

    table3 = pd.DataFrame(
        [
            ["Numerical preprocessing", "Original missingness restored; fold-local median imputation and reporting indicators; scaling for linear/SVR only"],
            ["Categorical preprocessing", "Fold-local most-frequent imputation and OHE with unknown-category handling"],
            ["Primary models", "Linear, ridge, elastic net, SVR, RF, Extra Trees, HistGB, XGBoost, LightGBM, CatBoost"],
            ["Target", "Raw decimal mass loss; bounded logit epsilon=0.001 in proposed model"],
            ["Constraint", "Positive monotonic constraint on degradation time only"],
            ["Weights", "Row, curve-balanced, and study-balanced; normalized within training fold"],
            ["Uncertainty", "Post-selection study-grouped cross-validation residual intervals"],
        ],
        columns=["Component", "Implementation"],
    )
    write_table(table3, "Table_3_model_preprocessing", "Model and preprocessing summary")

    screening = pd.read_csv(ROOT / "results" / "metrics" / "screening_metrics.csv")
    table4 = screening[["task", "model", "point_mae", "point_rmse", "point_r2", "macro_curve_mae", "macro_study_mae"]].round(3)
    write_table(table4, "Table_4_split_performance", "Main performance under the three split strategies")

    study = pd.read_csv(ROOT / "results" / "metrics" / "framework_ablation_study_metrics.csv")
    table5 = study[study.model.eq("bounded_monotonic_study_weighted")][
        ["study_id", "n", "mae", "rmse", "median_absolute_error"]
    ].copy()
    table5["study_order"] = pd.to_numeric(table5.study_id)
    table5 = table5.sort_values("study_order").drop(columns="study_order").round(3)
    write_table(table5, "Table_5_primary_loso_by_study", "Primary leave-one-study-out performance by held-out study")

    dense = pd.read_csv(ROOT / "results" / "metrics" / "framework_dense_physical_metrics.csv")
    physical = dense[dense.model.eq("bounded_monotonic_study_weighted")].copy()
    intervals = pd.read_csv(ROOT / "results" / "metrics" / "conformal_interval_metrics.csv")
    table6 = pd.DataFrame(
        [
            ["Dense negative predictions (%)", 100 * physical.negative_prediction_fraction.iloc[0]],
            ["Dense predictions above 100% (%)", 100 * physical.above_100_prediction_fraction.iloc[0]],
            ["Dense monotonicity violations (%)", 100 * physical.monotonicity_violation_fraction.iloc[0]],
            ["Mean prediction at day 0 (%)", physical.mean_prediction_at_time_zero.iloc[0]],
            ["80% interval coverage / width (percentage points)", f"{intervals.iloc[0].empirical_coverage:.3f} / {intervals.iloc[0].mean_interval_width:.1f}"],
            ["90% interval coverage / width (percentage points)", f"{intervals.iloc[1].empirical_coverage:.3f} / {intervals.iloc[1].mean_interval_width:.1f}"],
            ["95% interval coverage / width (percentage points)", f"{intervals.iloc[2].empirical_coverage:.3f} / {intervals.iloc[2].mean_interval_width:.1f}"],
        ],
        columns=["Metric", "Value"],
    )
    write_table(table6, "Table_6_physical_uncertainty", "Physical validity and uncertainty metrics")

    ablation = pd.read_csv(ROOT / "results" / "metrics" / "framework_ablation_metrics.csv")
    table7 = ablation[["model", "point_mae", "point_rmse", "point_r2", "macro_curve_mae", "macro_study_mae", "monotonicity_violation_fraction"]].round(3)
    write_table(table7, "Table_7_ablation_results", "Ablation and sensitivity results")

    supplementary = ROOT / "supplementary" / "supplementary_tables"
    supplementary.mkdir(parents=True, exist_ok=True)

    source_metadata = pd.read_excel(
        ROOT / "origin_data" / "Weight-loss-model-main" / "scl_weight_loss_data_library_raw_LM.xlsx",
        sheet_name="metadata",
    )
    source_metadata = source_metadata.rename(columns=lambda name: str(name).strip())
    source_metadata["study_id"] = source_metadata["study_id"].astype(int).astype(str)
    corpus_studies = ["4", "5", "6", "8", "9", "10", "11", "12", "13", "14", "15", "16", "18", "19", "20", "21", "24"]
    source_metadata = source_metadata[source_metadata.study_id.isin(corpus_studies)].copy()
    source_metadata["workbook_title"] = source_metadata["Paper Title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    source_metadata["workbook_doi"] = source_metadata["DOI"].astype(str).str.replace("doi:", "", case=False, regex=False).str.replace(" ", "", regex=False).str.strip()
    doi_overrides = {
        "8": "10.1002/(SICI)1097-4628(20000620)76:12<1767::AID-APP6>3.0.CO;2-D",
        "15": "10.3390/polym15112481",
        "16": "10.3390/polym14040838",
    }
    source_metadata["verified_doi"] = source_metadata.apply(
        lambda row: doi_overrides.get(row.study_id, row.workbook_doi), axis=1
    )
    reference_keys = {
        "4": "seggiani2018", "5": "syahirah2021", "6": "chan2019", "8": "gordon2000",
        "9": "choi2003", "10": "liu2017", "11": "brunel2014", "12": "iqbal2025",
        "13": "oliveira2024", "14": "zaidi2019", "15": "feijoo2023", "16": "brdlik2022",
        "18": "imam1999", "19": "ramsay1993", "20": "avella2000", "21": "sanchezsafont2016",
        "24": "lafuentearias2024",
    }
    source_metadata["reference_key"] = source_metadata.study_id.map(reference_keys)
    bibliography_text = (ROOT / "manuscript" / "references.bib").read_text(encoding="utf-8")
    verified_titles: dict[str, str] = {}
    for block in re.split(r"\n(?=@)", bibliography_text):
        key_match = re.match(r"@\w+\{([^,]+),", block.strip())
        title_match = re.search(r"^\s*title\s*=\s*\{\{?(.*?)\}\}?,?\s*$", block, flags=re.MULTILINE)
        if key_match and title_match:
            verified_titles[key_match.group(1).strip()] = title_match.group(1).replace("{", "").replace("}", "").strip()
    source_metadata["verified_title"] = source_metadata.reference_key.map(verified_titles)
    if source_metadata["verified_title"].isna().any():
        missing = source_metadata.loc[source_metadata.verified_title.isna(), "reference_key"].tolist()
        raise ValueError(f"Missing verified bibliography titles for source studies: {missing}")
    source_metadata["analysis_role"] = np.where(source_metadata.study_id.eq("8"), "Expanded sensitivity only", "Primary and expanded analyses")
    source_metadata["source_type"] = "Primary experimental publication"
    source_metadata["verification_note"] = "DOI normalized against Crossref metadata on 1 August 2026"
    source_provenance = source_metadata[
        ["study_id", "workbook_title", "verified_title", "workbook_doi", "verified_doi", "reference_key", "analysis_role", "source_type", "verification_note"]
    ].sort_values("study_id", key=lambda values: values.astype(int))
    source_provenance.to_csv(ROOT / "results" / "source_study_provenance.csv", index=False)
    source_provenance.to_csv(supplementary / "Table_S27_source_study_provenance.csv", index=False)

    copies = {
        ROOT / "results" / "data_dictionary.csv": "Table_S1_data_dictionary.csv",
        ROOT / "results" / "data_audit.csv": "Table_S2_missingness_and_sheet_audit.csv",
        ROOT / "results" / "curve_level_summary.csv": "Table_S3_curve_summary.csv",
        ROOT / "results" / "raw_vs_cleaned_target_comparison.csv": "Table_S4_raw_cleaned_target_discrepancies.csv",
        ROOT / "results" / "tuning" / "all_trials.csv": "Table_S5_hyperparameter_trials.csv",
        ROOT / "results" / "metrics" / "framework_ablation_curve_metrics.csv": "Table_S6_all_curve_metrics.csv",
        ROOT / "results" / "statistics" / "model_comparisons.csv": "Table_S7_statistical_comparisons.csv",
        ROOT / "results" / "metrics" / "applicability_indicators.csv": "Table_S8_applicability_indicators.csv",
        ROOT / "results" / "excluded_raw_observations.csv": "Table_S9_excluded_observations.csv",
        ROOT / "results" / "metrics" / "framework_trajectory_metrics.csv": "Table_S10_trajectory_metrics.csv",
        ROOT / "results" / "category_crosswalk.csv": "Table_S11_category_crosswalk.csv",
        ROOT / "results" / "metrics" / "study_level_shift_summary.csv": "Table_S12_study_level_shift_summary.csv",
        ROOT / "results" / "metrics" / "focal_study_source_context.csv": "Table_S13_focal_study_source_context.csv",
        ROOT / "results" / "metrics" / "repeated_split_metrics.csv": "Table_S14_repeated_split_metrics.csv",
        ROOT / "results" / "metrics" / "time_stage_error.csv": "Table_S15_time_stage_error.csv",
        ROOT / "results" / "metrics" / "revision_sensitivity_metrics.csv": "Table_S16_revision_sensitivity_metrics.csv",
        ROOT / "results" / "metrics" / "weight_distribution_by_fold.csv": "Table_S17_weight_distribution_by_fold.csv",
        ROOT / "results" / "metrics" / "weight_by_study_summary.csv": "Table_S18_weight_by_study_summary.csv",
        ROOT / "results" / "metrics" / "composition_diagnostics.csv": "Table_S19_composition_diagnostics.csv",
    }
    for source, destination in copies.items():
        shutil.copy2(source, supplementary / destination)


def captions() -> None:
    figure_text = """# Figure captions

**Figure 1. Experimental-data provenance, decimal precision, and independence risk.** (A) Hierarchy from primary publications through the curated trajectory corpus to the primary analysis population; the side callout gives the observations removed upstream for incomplete descriptors. (B) Original decimal versus released truncated mass loss (n = 1,467); the dashed line is identity and the inset magnifies 39--43% to expose the truncation staircase. (C) Distribution of the absolute truncation error (bars, left axis) with its cumulative fraction (line, right axis) and the mean correction. (D) One marker per trajectory (n = 129) after random division of individual observations, coloured by whether the trajectory spans both portions or stays within one.

**Figure 2. Experimental-corpus composition and imbalance.** (A) Repeated observations by source study, with the trajectory count listed in the right-hand column. (B) Trajectory-length distribution with the median and the longest trajectory annotated. (C) Repeated observations by degradation environment, labelled with trajectory counts. (D) Distribution of observed decimal mass loss for the same environments (box: median and interquartile range; whiskers: 1.5 x IQR; points: individual observations). Counts are descriptive; repeated time points are not independent experiments.

**Figure 3. Performance under increasingly independent experimental separation.** (A) Point MAE for the same gradient-boosting relation when individual observations, complete trajectories, or complete source studies are withheld; the arrows give the step-wise inflation factors. (B) MAE distributions across 50 observation-level divisions, 50 trajectory-level divisions, and the 16 complete-study assessments; boxes show the median and interquartile range and diamonds the group means. The first two distributions contain assignment-level point MAE, whereas the final distribution contains study-level MAE; they address related but distinct environmental questions.

**Figure 4. Between-study environmental transferability and failure diagnosis.** Rows are aligned across all three panels. (A) Primary-formulation MAE by withheld study; labels include withheld observations. Study 9 and study 19 are highlighted as high-error cases and studies 12 and 14 as low-error comparators. (B) Environmental dissimilarity and response descriptors; cell colour is the within-column percentile and cell text is the raw value. The first column is the mean k-nearest-neighbour distance to the remaining studies. (C) Environment, degradation condition, and experimental scale. Out-of-range and previously unseen-category counts refer to affected descriptors, not observations.

**Figure 5. Focal held-out trajectories for failure analysis.** Studies 9 and 19 illustrate endpoint-orientation and rapid-kinetics failures; studies 12 and 14 are lower-error comparators. Each panel header carries the trajectory identifier, its MAE, and the diagnostic tag, so no annotation covers the data. Open points are raw-decimal observations, solid lines are primary outer-fold predictions, dashed lines are unconstrained predictions, and shaded regions are empirical 90% post-selection error bands. All panels use the 0--100% endpoint scale. The endpoint is degradation-induced mass loss, not mineralisation.

**Figure 6. Physical-validity comparison across interpolated exposure times.** Each withheld trajectory was evaluated at 101 exposure times with its material and environmental descriptors held fixed (13,029 estimates). Panels show negative estimates, decreasing adjacent steps, and the mean estimated mass loss at day 0. Percentage bounds and the positive time constraint preserve the expected cumulative endpoint behaviour.

**Figure 7. Empirical error-band performance.** (A) Nominal versus pooled empirical point coverage; the shaded region is the excess over nominal. (B) Mean 90% band width by withheld study, with the pooled width as the dashed reference. (C) Empirical 90% coverage by withheld study; study 9 is highlighted and the dashed line is nominal coverage. The same study-separated development assignments also informed method settings, so these are empirical error bands formed after selection rather than independent conditional guarantees.

**Figure 8. Associational model interpretation.** (A) Full-data SHAP importance on the bounded-logit scale. (B) Study-equal held-out grouped permutation importance. (C) Environment counterfactual trajectories for one observed reference formulation; the key abbreviates laboratory mineral medium, compost/organic fertilizer, and marine/aquatic exposure. (D) Top grouped SHAP interactions. These analyses describe model associations, not causal mechanisms.

**Figure 9. Analysis-formulation summary.** Rows are aligned across all three panels and the proposed formulation is shaded. (A) Study-equal mean absolute error (MAE) with 95% study-bootstrap intervals; the dashed reference marks the proposed value. (B) Observation-weighted MAE. (C) Decreasing-step frequency across interpolated times for seven formulations. No protocol-recorded paired difference remained family-wise significant after Holm correction across the 16 withheld studies.

**Supplementary Figure S1. Time, weighting, and composition sensitivity.** (A) Primary LOSO point MAE in four elapsed-time intervals, with interval sizes in the axis labels. (B) Macro-study MAE for row, study-only, study--curve, capped study--curve, HB-only, and HV-only variants. (C) Fold-wise maximum normalised row weight under the three study-equal strategies.

**Supplementary Figure S5. Curve-level endpoint integrity audit.** Classification of all 129 primary-corpus curves using the frozen orientation and integrity rules; cell text is the source instance and the two severe conflicts are study 9 instances 4 and 8.
"""
    (ROOT / "figures" / "figure_captions.md").write_text(figure_text, encoding="utf-8")
    table_text = """# Table captions

Tables 1-7 are generated directly from saved audit, prediction, metric, uncertainty, and statistical result files. Table 4 treats random-row results as optimistic interpolation. Table 5 uses held-out studies as the generalisation unit and omits unstable per-study R2 values. Supplementary Tables S1-S27 include the complete audit, prediction, failure-analysis, repeated-split, time-stage, weighting, composition, baseline, expanded-corpus, endpoint, source-traceability, source-study provenance, and missingness results.
"""
    (ROOT / "tables" / "table_captions.md").write_text(table_text, encoding="utf-8")


def write_figure_manifest() -> None:
    source_map = {
        "Figure_1_data_reconstruction_and_leakage": ["results/raw_vs_cleaned_target_comparison.csv", "splits/task_a_random_row.csv"],
        "Figure_2_dataset_composition": ["data/processed/phbv_weight_loss_decimal.csv"],
        "Figure_3_split_strictness_performance": ["results/metrics/screening_metrics.csv", "results/metrics/repeated_split_metrics.csv"],
        "Figure_4_study_level_shift": ["results/metrics/study_level_shift_summary.csv"],
        "Figure_5_representative_heldout_trajectories": ["results/predictions/final_conformal_predictions.csv", "results/predictions/framework_ablation_predictions.csv"],
        "Figure_6_physical_validity": ["results/metrics/framework_dense_physical_metrics.csv"],
        "Figure_7_uncertainty_and_applicability": ["results/metrics/conformal_interval_metrics.csv", "results/metrics/conformal_subgroup_metrics.csv"],
        "Figure_8_model_interpretation": ["results/metrics/shap_grouped_importance.csv", "results/metrics/heldout_grouped_permutation_importance.csv", "results/metrics/counterfactual_trajectories.csv"],
        "Figure_9_ablation_summary": ["results/metrics/framework_ablation_metrics.csv", "results/statistics/model_macro_study_bootstrap_ci.csv"],
        "TOC_graphical_abstract": ["Author-selected original graphical abstract preserved from repository history"],
    }
    entries = []
    for stem, sources in source_map.items():
        outputs = []
        suffixes = [".png"] if stem == "TOC_graphical_abstract" else [".pdf", ".svg", ".png"]
        for suffix in suffixes:
            path = ROOT / "figures" / f"{stem}{suffix}"
            outputs.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        entries.append({"figure": stem, "source_data": sources, "outputs": outputs})
    manifest = {
        "generated_by": "scripts/11_generate_figures_tables.py",
        "export": {"analytical_figure_png_dpi": 600, "pdf_fonttype": 42, "svg_text": "editable", "background": "opaque white"},
        "preserved_asset": {"figure": "TOC_graphical_abstract", "native_pixels": [1708, 921], "resampling": "none"},
        "palette": OKABE_ITO,
        "raw_data_policy": "Source files are read-only; all figures use saved derived tables and predictions.",
        "figures": entries,
    }
    (ROOT / "figures" / "figure_export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    configure_style()
    table = pd.read_csv(
        ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv",
        dtype={"study_id_normalized": str, "instance_normalized": str},
    )
    figure_1(table)
    figure_2(table)
    figure_3()
    figure_4()
    figure_5()
    figure_6()
    figure_7()
    figure_8()
    figure_9()
    figure_s1()
    graphical_abstract()
    generate_tables(table)
    captions()
    write_figure_manifest()


if __name__ == "__main__":
    main()
