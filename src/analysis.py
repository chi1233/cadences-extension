"""
analysis.py

Core analytical functions: artifact mix by group, Shannon entropy of the
artifact mix, hourly artifact shares (weekday vs. weekend), and token depth
by artifact type x wage quartile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def artifact_mix_by_group(
    df: pd.DataFrame, group_col: str, artifact_col: str = "artifact_type_norm"
) -> pd.DataFrame:
    """Compute artifact-type shares within each level of group_col.

    Args:
        df: conversation-level DataFrame.
        group_col: grouping column, e.g. "wage_quartile".
        artifact_col: normalized artifact-type column.

    Returns:
        Wide DataFrame indexed by group_col, columns = artifact types,
        values = share of conversations in that group with that artifact type.
    """
    counts = pd.crosstab(df[group_col], df[artifact_col])
    shares = counts.div(counts.sum(axis=1), axis=0)
    return shares


def compute_entropy(
    df: pd.DataFrame, group_col: str, artifact_col: str = "artifact_type_norm"
) -> pd.DataFrame:
    """Compute Shannon entropy (base 2, in bits) of the artifact mix per group.

    Higher entropy means a more evenly spread artifact mix (less concentrated
    on any one output type); lower entropy means output is dominated by one
    or two artifact types.

    Args:
        df: conversation-level DataFrame.
        group_col: grouping column, e.g. "wage_quartile".
        artifact_col: normalized artifact-type column.

    Returns:
        DataFrame with one row per group and an `entropy_bits` column.
    """
    shares = artifact_mix_by_group(df, group_col, artifact_col)
    probs = shares.values
    probs = np.where(probs > 0, probs, np.nan)
    entropy = -np.nansum(probs * np.log2(probs), axis=1)
    return pd.DataFrame({"entropy_bits": entropy}, index=shares.index)


def hourly_artifact_share(
    df: pd.DataFrame,
    artifact_col: str = "artifact_type_norm",
    hour_col: str = "hour",
    weekend_col: str = "is_weekend",
) -> pd.DataFrame:
    """Compute artifact-type shares by hour of day, split weekday vs. weekend.

    Args:
        df: conversation-level DataFrame with hour/weekend/artifact columns.
        artifact_col: normalized artifact-type column.
        hour_col: hour-of-day column (0-23).
        weekend_col: boolean weekend-flag column.

    Returns:
        Long-format DataFrame with columns [hour, is_weekend, artifact_type,
        share].
    """
    counts = (
        df.groupby([hour_col, weekend_col, artifact_col], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = counts.groupby([hour_col, weekend_col], observed=True)["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    counts = counts.rename(columns={artifact_col: "artifact_type"})
    return counts


def token_depth_by_group(
    df: pd.DataFrame, group_cols: list[str], token_col: str = "tokens_total"
) -> pd.DataFrame:
    """Compute mean and geometric-mean tokens per conversation within groups.

    Anthropic reports geometric means for token counts because the variable
    is heavily right-skewed; both are computed here so the arithmetic mean is
    available for comparison.

    Args:
        df: conversation-level DataFrame.
        group_cols: columns to group by, e.g. ["artifact_type_norm", "wage_quartile"].
        token_col: token-count column.

    Returns:
        DataFrame with one row per group combination and columns
        [mean_tokens, geo_mean_tokens, n].
    """
    def _geo_mean(x: pd.Series) -> float:
        x = x[x > 0]
        return float(np.exp(np.mean(np.log(x)))) if len(x) else np.nan

    grouped = df.groupby(group_cols, observed=True)[token_col]
    out = grouped.agg(mean_tokens="mean", n="count")
    out["geo_mean_tokens"] = grouped.apply(_geo_mean)
    return out.reset_index()


def peak_hour_by_type(
    hourly_shares: pd.DataFrame, artifact_types: list[str]
) -> pd.DataFrame:
    """Identify the peak hour for each artifact type, split weekday/weekend.

    Args:
        hourly_shares: output of hourly_artifact_share.
        artifact_types: list of artifact type labels to check.

    Returns:
        DataFrame with columns [artifact_type, is_weekend, peak_hour, peak_share].
    """
    rows = []
    for atype in artifact_types:
        sub = hourly_shares[hourly_shares["artifact_type"] == atype]
        for weekend_flag, grp in sub.groupby("is_weekend", observed=True):
            top = grp.loc[grp["share"].idxmax()]
            rows.append(
                {
                    "artifact_type": atype,
                    "is_weekend": weekend_flag,
                    "peak_hour": int(top["hour"]),
                    "peak_share": float(top["share"]),
                }
            )
    return pd.DataFrame(rows)
