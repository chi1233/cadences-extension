"""
features.py

Feature construction for the artifact-cadence analysis: wage quartiles,
calendar/time features, and artifact-type normalization.
"""

from __future__ import annotations

import pandas as pd

TOP_ARTIFACT_TYPES = [
    "explanation",
    "document",
    "code",
    "guidance",
    "app_website",
    "marketing_content",
]


def assign_wage_quartile(df: pd.DataFrame, wage_col: str = "median_hourly_wage") -> pd.DataFrame:
    """Assign each conversation to a wage quartile (Q1 = lowest, Q4 = highest).

    Args:
        df: conversation-level DataFrame.
        wage_col: column holding the mapped occupation wage.

    Returns:
        Copy of df with a new `wage_quartile` categorical column.
    """
    out = df.copy()
    out["wage_quartile"] = pd.qcut(
        out[wage_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    return out


def extract_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp_utc") -> pd.DataFrame:
    """Derive hour-of-day, day-of-week, and weekend flag from a timestamp column.

    Args:
        df: conversation-level DataFrame.
        timestamp_col: column holding a datetime64 timestamp.

    Returns:
        Copy of df with `hour`, `day_of_week`, and `is_weekend` columns added.
    """
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek  # 0 = Monday ... 6 = Sunday
    out["is_weekend"] = out["day_of_week"] >= 5
    return out


def normalize_artifact_type(
    df: pd.DataFrame, artifact_col: str = "artifact_type", top_n: int = 6
) -> pd.DataFrame:
    """Collapse artifact_type into the top-N most frequent categories + "Other".

    Args:
        df: conversation-level DataFrame.
        artifact_col: column holding the raw artifact classifier label.
        top_n: number of top categories to retain individually.

    Returns:
        Copy of df with a normalized `artifact_type_norm` column.
    """
    out = df.copy()
    top_types = out[artifact_col].value_counts().head(top_n).index
    out["artifact_type_norm"] = out[artifact_col].where(
        out[artifact_col].isin(top_types), other="Other"
    )
    return out
