"""
plotting.py

Figure generation for the artifact-cadence analysis. Each function saves a
PNG into the `figures/` directory and returns the file path.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

_PALETTE = {
    "code": "#4C72B0",
    "document": "#DD8452",
    "explanation": "#55A868",
    "guidance": "#C44E52",
    "app_website": "#8172B2",
    "marketing_content": "#937860",
    "Other": "#B0B0B0",
}


def plot_wage_quartile_stack(shares: pd.DataFrame, out_name: str = "fig1_artifact_by_wage_quartile.png") -> str:
    """Figure 1: stacked bar of artifact-type share by wage quartile.

    Args:
        shares: output of artifact_mix_by_group(df, "wage_quartile").
        out_name: output filename.

    Returns:
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cols = [c for c in shares.columns]
    colors = [_PALETTE.get(c, "#999999") for c in cols]
    shares[cols].plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.7)
    ax.set_xlabel("Wage quartile (Q1 = lowest, Q4 = highest)")
    ax.set_ylabel("Share of conversations")
    ax.set_title("Artifact mix by occupation wage quartile")
    ax.legend(title="Artifact type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, out_name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_hourly_heatmap(hourly_shares: pd.DataFrame, out_name: str = "fig2_hourly_heatmap.png") -> str:
    """Figure 2: heatmap of artifact-type share by hour (0-23), pooled weekday+weekend.

    Args:
        hourly_shares: output of hourly_artifact_share.
        out_name: output filename.

    Returns:
        Path to the saved PNG.
    """
    pooled = (
        hourly_shares.groupby(["hour", "artifact_type"], observed=True)["share"]
        .mean()
        .reset_index()
    )
    pivot = pooled.pivot(index="artifact_type", columns="hour", values="share").fillna(0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_title("Artifact-type share by hour of day")
    fig.colorbar(im, ax=ax, label="Share within hour")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, out_name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_weekday_weekend_lines(
    hourly_shares: pd.DataFrame,
    types_to_plot: list[str],
    out_name: str = "fig3_weekday_weekend_lines.png",
) -> str:
    """Figure 3: hourly share lines for selected artifact types, weekday vs weekend.

    Args:
        hourly_shares: output of hourly_artifact_share.
        types_to_plot: artifact type labels to include, e.g. ["code", "document", "explanation"].
        out_name: output filename.

    Returns:
        Path to the saved PNG.
    """
    fig, axes = plt.subplots(1, len(types_to_plot), figsize=(5 * len(types_to_plot), 4), sharey=True)
    if len(types_to_plot) == 1:
        axes = [axes]

    for ax, atype in zip(axes, types_to_plot):
        sub = hourly_shares[hourly_shares["artifact_type"] == atype]
        for weekend_flag, label, style in [(False, "Weekday", "-"), (True, "Weekend", "--")]:
            grp = sub[sub["is_weekend"] == weekend_flag].sort_values("hour")
            ax.plot(grp["hour"], grp["share"], style, label=label, color=_PALETTE.get(atype, "#333333"))
        ax.set_title(atype)
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(0, 24, 4))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Share within hour")
    fig.suptitle("Weekday vs. weekend hourly artifact share")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, out_name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_tokens_vs_wage_scatter(
    token_depth: pd.DataFrame, out_name: str = "fig4_tokens_vs_wage_scatter.png"
) -> str:
    """Figure 4 (optional): avg tokens-per-conversation vs avg wage, by artifact type.

    Args:
        token_depth: DataFrame with columns [artifact_type_norm, wage_quartile,
            mean_tokens, geo_mean_tokens, n] as produced by token_depth_by_group,
            plus a merged `mean_wage` column per wage_quartile.
        out_name: output filename.

    Returns:
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for atype, grp in token_depth.groupby("artifact_type_norm", observed=True):
        ax.scatter(
            grp["mean_wage"],
            grp["geo_mean_tokens"],
            s=grp["n"] / grp["n"].max() * 300 + 30,
            color=_PALETTE.get(atype, "#999999"),
            label=atype,
            alpha=0.8,
            edgecolor="white",
        )
    ax.set_xlabel("Mean occupation wage in wage quartile ($/hr)")
    ax.set_ylabel("Geometric-mean tokens per conversation")
    ax.set_title("Token depth vs. task wage, by artifact type")
    ax.legend(title="Artifact type", fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, out_name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
