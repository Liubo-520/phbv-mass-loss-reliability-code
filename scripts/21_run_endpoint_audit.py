"""Run the curve-level endpoint integrity and primary-source audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_io import ROOT
from src.endpoint_audit import audit_curves, study19_source_table, study9_source_table


CLASS_ORDER = [
    "endpoint_consistent",
    "minor_non_monotonic_noise",
    "uncertain_semantics",
    "severe_orientation_conflict",
]
CLASS_COLORS = ["#0072B2", "#8FD1C4", "#E69F00", "#D55E00"]
CLASS_DISPLAY = [
    "Endpoint consistent",
    "Minor non-monotonic noise",
    "Uncertain semantics",
    "Severe orientation conflict",
]
NAVY = "#13293D"
SLATE = "#4F6272"


def main() -> None:
    table = pd.read_csv(ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv")
    audit = audit_curves(table)
    audit.to_csv(ROOT / "results" / "curve_endpoint_integrity_audit.csv", index=False)
    study19_source_table(table).to_csv(
        ROOT / "results" / "study19_primary_source_curve_audit.csv", index=False
    )
    study9_source_table(table).to_csv(
        ROOT / "results" / "study9_primary_source_curve_audit.csv", index=False
    )

    rule = {
        "frozen_before_model_sensitivity_rerun": True,
        "severe_orientation_conflict": (
            "endpoint_minus_start_pct <= -10 and overall linear slope < 0"
        ),
        "uncertain_semantics": "endpoint_minus_start_pct < -2 or any value > 100",
        "minor_non_monotonic_noise": "at least one adjacent decrease not meeting earlier rules",
        "abnormal_jump": "absolute adjacent change > 25 percentage points",
        "study9_transform_100_minus_y": "not_run_without_primary_source_proof_of_remaining_mass",
    }
    (ROOT / "results" / "endpoint_integrity_rule.json").write_text(
        json.dumps(rule, indent=2) + "\n", encoding="utf-8"
    )

    ordered = audit.sort_values(
        ["study_id", "instance"],
        key=lambda x: x.map(lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value))),
    ).copy()
    ordered["curve_order_within_study"] = ordered.groupby("study_id").cumcount() + 1
    studies = ordered["study_id"].drop_duplicates().tolist()
    maximum_curves = int(ordered["curve_order_within_study"].max())
    values = np.full((len(studies), maximum_curves), np.nan)
    labels = np.full((len(studies), maximum_curves), "", dtype=object)
    study_position = {study: index for index, study in enumerate(studies)}
    for row in ordered.itertuples():
        y = study_position[row.study_id]
        x = int(row.curve_order_within_study) - 1
        values[y, x] = CLASS_ORDER.index(row.endpoint_integrity_class)
        labels[y, x] = str(row.instance)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.6,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "axes.edgecolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(CLASS_COLORS)
    cmap.set_bad("#F0F3F5")
    ax.imshow(values, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, maximum_curves, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(studies), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(np.arange(maximum_curves), np.arange(1, maximum_curves + 1), fontsize=6.0)
    ax.set_yticks(np.arange(len(studies)), [f"Study {study}" for study in studies], fontsize=6.8)
    ax.set_xlabel("Curve order within study", labelpad=4.0)
    ax.set_ylabel("Source study", labelpad=4.0)
    # Light class fills need dark text so every instance label stays readable.
    text_colors = ["white", NAVY, NAVY, "white"]
    for y in range(len(studies)):
        for x in range(maximum_curves):
            if labels[y, x]:
                ax.text(
                    x,
                    y,
                    labels[y, x],
                    ha="center",
                    va="center",
                    fontsize=4.4,
                    color=text_colors[int(values[y, x])],
                )
    ax.set_title(
        "Curve-level endpoint integrity audit (129 curves; cell text is the source instance)",
        loc="left",
        fontsize=8.0,
        color=NAVY,
        weight="bold",
        pad=7.0,
    )
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="none", color="none", markerfacecolor=color, markersize=7, label=label)
        for color, label in zip(CLASS_COLORS, CLASS_DISPLAY)
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(0.5, -0.22),
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=6.8,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    for suffix, dpi in [("png", 600), ("pdf", 600), ("svg", 600)]:
        fig.savefig(ROOT / "figures" / f"Figure_S5_endpoint_integrity_audit.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    summary = audit["endpoint_integrity_class"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    report = [
        "# Endpoint integrity audit",
        "",
        *[f"- `{key}`: {int(value)} curves" for key, value in summary.items()],
        "",
        "The severe-conflict rule was defined from endpoint orientation and local-change magnitude, not model error. "
        "It identifies study 9 instances 4 and 8. Their unchanged observations remain in the primary analysis; "
        "an exclusion analysis is an endpoint-harmonized sensitivity only.",
        "",
        "The accessible primary abstract for study 9 does not establish that the plotted quantity was remaining mass. "
        "Consequently, no 100-y transformation is performed. The full text of Ramsay et al. (1993), in contrast, "
        "labels the study 19 axes as Weight Loss (%), and its figure captions verify oxygen condition, thickness, "
        "starch fraction, and specimen geometry.",
    ]
    (ROOT / "results" / "endpoint_integrity_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
