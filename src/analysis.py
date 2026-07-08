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
    max_entropy = np.log2(shares.shape[1])
    return pd.DataFrame(
        {"entropy_bits": entropy, "entropy_normalized": entropy / max_entropy},
        index=shares.index,
    )


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


def test_weekday_weekend_shift(
    df: pd.DataFrame,
    artifact_types: list[str],
    artifact_col: str = "artifact_type_norm",
    hour_col: str = "hour",
    weekend_col: str = "is_weekend",
) -> pd.DataFrame:
    """Chi-square test of whether the hourly distribution shape differs
    between weekday and weekend for each artifact type.

    This tests the full 24-hour shape (via a 2 x 24 contingency table of
    weekday/weekend counts by hour), not just whether the single peak hour
    moved, so it is a stronger test of "does this artifact type's daily
    rhythm actually shift on weekends" than comparing peak_hour_by_type
    outputs alone.

    Args:
        df: conversation-level DataFrame.
        artifact_types: artifact type labels to test.
        artifact_col: normalized artifact-type column.
        hour_col: hour-of-day column (0-23).
        weekend_col: boolean weekend-flag column.

    Returns:
        DataFrame with columns [artifact_type, n_weekday, n_weekend,
        chi2, dof, p_value].
    """
    from scipy.stats import chi2_contingency

    rows = []
    for atype in artifact_types:
        sub = df[df[artifact_col] == atype]
        table = pd.crosstab(sub[weekend_col], sub[hour_col])
        # Ensure both weekday and weekend rows are present and all 24 hour
        # columns exist, filling any missing hour with 0 counts.
        table = table.reindex(columns=range(24), fill_value=0)
        table = table.reindex(index=[False, True], fill_value=0)
        chi2, p_value, dof, _ = chi2_contingency(table.values)
        rows.append(
            {
                "artifact_type": atype,
                "n_weekday": int(table.loc[False].sum()),
                "n_weekend": int(table.loc[True].sum()),
                "chi2": chi2,
                "dof": dof,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def decompose_token_gap(
    df: pd.DataFrame,
    low_group,
    high_group,
    group_col: str = "wage_quartile",
    artifact_col: str = "artifact_type_norm",
    token_col: str = "tokens_total",
) -> dict:
    """Oaxaca-style decomposition of the mean-token gap between two groups
    into a composition effect (different artifact-type mix) and a within-type
    effect (different token depth conditional on artifact type).

    total_gap = mean_tokens(high_group) - mean_tokens(low_group)
    composition_effect = sum_type[(share_high - share_low) * tokens_low]
    within_type_effect  = sum_type[share_high * (tokens_high - tokens_low)]
    total_gap == composition_effect + within_type_effect (exactly, by construction)

    Args:
        df: conversation-level DataFrame.
        low_group: value of group_col identifying the reference group (e.g. "Q1").
        high_group: value of group_col identifying the comparison group (e.g. "Q4").
        group_col: grouping column, e.g. "wage_quartile".
        artifact_col: normalized artifact-type column.
        token_col: token-count column.

    Returns:
        Dict with keys: total_gap, composition_effect, within_type_effect,
        composition_share (fraction of the gap explained by composition),
        low_mean, high_mean.
    """
    low = df[df[group_col] == low_group]
    high = df[df[group_col] == high_group]

    share_low = low[artifact_col].value_counts(normalize=True)
    share_high = high[artifact_col].value_counts(normalize=True)
    tokens_low = low.groupby(artifact_col, observed=True)[token_col].mean()
    tokens_high = high.groupby(artifact_col, observed=True)[token_col].mean()

    all_types = sorted(set(share_low.index) | set(share_high.index))
    share_low = share_low.reindex(all_types, fill_value=0.0)
    share_high = share_high.reindex(all_types, fill_value=0.0)
    tokens_low = tokens_low.reindex(all_types, fill_value=0.0)
    tokens_high = tokens_high.reindex(all_types, fill_value=0.0)

    composition_effect = float(((share_high - share_low) * tokens_low).sum())
    within_type_effect = float((share_high * (tokens_high - tokens_low)).sum())
    low_mean = float(low[token_col].mean())
    high_mean = float(high[token_col].mean())
    total_gap = high_mean - low_mean

    return {
        "low_mean": low_mean,
        "high_mean": high_mean,
        "total_gap": total_gap,
        "composition_effect": composition_effect,
        "within_type_effect": within_type_effect,
        "composition_share": composition_effect / total_gap if total_gap else float("nan"),
    }
