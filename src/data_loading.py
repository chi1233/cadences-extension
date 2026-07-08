"""
data_loading.py

Loading utilities for the Anthropic Economic Index "Cadences" release
(2026-06-26). This module is written against the schema documented in
Anthropic's public data_documentation for the EconomicIndex Hugging Face
dataset (https://huggingface.co/datasets/Anthropic/EconomicIndex), specifically
the artifact-classifier and hourly-aggregate tables added in the Cadences
release.

NOTE ON DATA ACCESS: this sandbox environment does not have network access to
huggingface.co (only pypi/npm/github mirrors are reachable). `load_index_data`
therefore falls back to a calibrated synthetic dataset when it cannot reach
the Hub. The synthetic generator is parameterized to match the summary
statistics Anthropic published in Cadences (93% artifact rate; explanation
17% / document 15% / guidance 11% of outputs; weekday-vs-weekend personal
share of ~35%/~50%; positive wage-tokens correlation; app/website and
code-fix/debug artifacts carrying the highest token counts; explanations
running near 0.2x the median tokens-per-conversation). Once pointed at a real
HF mirror or local parquet snapshot, `load_index_data` returns the same
schema and every downstream function runs unmodified.

Assumed columns (documented in Anthropic's data_documentation.md for the
2026-06-26 release):
    conversation_id      : str, unique conversation identifier
    timestamp_utc         : datetime64, conversation start time
    artifact_type          : str, top-level artifact classifier label
                              (e.g. "code", "document", "explanation", ...)
    onet_task_id            : str, mapped O*NET task identifier
    occupation_title         : str, mapped SOC/O*NET occupation title
    median_hourly_wage        : float, BLS OEWS May 2025 wage for the mapped
                                 occupation (USD/hour)
    tokens_total               : int, total tokens in the conversation
                                  (Anthropic reports geometric means of this
                                  variable because it is heavily right-skewed)
    surface                     : str, one of {"claude.ai", "cowork",
                                  "claude_code"}
    is_work                       : bool, work vs. personal/coursework flag
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CADENCES_WINDOW_START = "2026-04-10"
CADENCES_WINDOW_END = "2026-06-10"

# Artifact base rates calibrated to Cadences Chapter 2 summary stats.
# "none" (~7% of conversations) is dropped by filter_cadences_window before
# any artifact-mix analysis, consistent with the report's own framing of the
# 93% figure as the complement of "None".
_ARTIFACT_BASE_RATES = {
    "explanation": 0.183,      # ~17% of ALL conversations -> ~18.3% of the 93% with an artifact
    "document": 0.161,         # ~15% of all conversations
    "guidance": 0.118,         # ~11% of all conversations
    "code": 0.140,
    "app_website": 0.060,
    "marketing_content": 0.055,
    "misc": 0.283,
}

_OCCUPATIONS = [
    ("Marketing Managers", 80.0),
    ("Software Developers", 62.0),
    ("Financial Analysts", 51.0),
    ("Editors", 37.0),
    ("Registered Nurses", 47.0),
    ("Pharmacists", 66.0),
    ("Statistical Assistants", 29.0),
    ("Customer Service Representatives", 21.0),
    ("Paralegals", 32.0),
    ("K-12 Teachers", 34.0),
    ("Retail Salespersons", 17.0),
    ("Management Analysts (Consultants)", 55.0),
    ("Human Resources Specialists", 36.0),
    ("Data Scientists", 60.0),
    ("Administrative Assistants", 23.0),
]


def _simulate_cadences_dataset(n: int = 60_000, seed: int = 26) -> pd.DataFrame:
    """Generate a synthetic conversation-level dataset with the Cadences schema.

    The generator is a stand-in for the real Hugging Face release and is used
    only because this environment cannot reach huggingface.co. It is
    calibrated to reproduce, directionally, the following published Cadences
    findings so that the downstream analysis exercises realistic magnitudes:
      - artifact-type base rates (explanation/document/guidance as the top 3),
      - a positive wage <-> tokens relationship,
      - app/website and code artifacts carrying more tokens than explanations,
      - a weekday/weekend swing in personal-vs-work share (~35% -> ~50%),
      - suppressed weekday-evening/weekend "backend architecture" style coding.

    Args:
        n: number of synthetic conversations to generate.
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with one row per conversation, matching the assumed
        Cadences schema described in this module's docstring.
    """
    rng = np.random.default_rng(seed)

    occ_titles = [o[0] for o in _OCCUPATIONS]
    occ_wages = np.array([o[1] for o in _OCCUPATIONS])
    occ_idx = rng.integers(0, len(_OCCUPATIONS), size=n)
    occupation_title = np.array(occ_titles)[occ_idx]
    median_hourly_wage = occ_wages[occ_idx] * rng.lognormal(mean=0.0, sigma=0.08, size=n)

    # Timestamps spread across the Cadences sampling window, with weekday
    # weighting on work-hour mass and a modest late-night/weekend tail.
    start = pd.Timestamp(CADENCES_WINDOW_START)
    end = pd.Timestamp(CADENCES_WINDOW_END)
    span_seconds = int((end - start).total_seconds())
    raw_offsets = rng.integers(0, span_seconds, size=n)
    timestamp_utc = start + pd.to_timedelta(raw_offsets, unit="s")

    df = pd.DataFrame(
        {
            "conversation_id": [f"conv_{i:07d}" for i in range(n)],
            "timestamp_utc": timestamp_utc,
            "occupation_title": occupation_title,
            "median_hourly_wage": median_hourly_wage,
            "surface": rng.choice(
                ["claude.ai", "cowork", "claude_code"], size=n, p=[0.62, 0.23, 0.15]
            ),
        }
    )

    hour = df["timestamp_utc"].dt.hour
    dow = df["timestamp_utc"].dt.dayofweek
    is_weekend = dow >= 5

    # Work-hour mass on weekdays; a broader, flatter distribution on weekends,
    # consistent with Cadences' reported weekday/weekend usage rhythm.
    work_hours_boost = np.where((hour >= 9) & (hour <= 18) & (~is_weekend), 1.6, 1.0)
    keep_prob = work_hours_boost / work_hours_boost.max()
    keep_mask = rng.random(n) < keep_prob
    df = df[keep_mask].reset_index(drop=True)
    n = len(df)
    hour = df["timestamp_utc"].dt.hour
    dow = df["timestamp_utc"].dt.dayofweek
    is_weekend = (dow >= 5).values

    df["is_work"] = np.where(
        is_weekend,
        rng.random(n) > 0.50,   # ~50% personal on weekends
        rng.random(n) > 0.35,   # ~35% personal on weekdays
    )

    # --- artifact_type assignment ---
    base_labels = list(_ARTIFACT_BASE_RATES.keys())
    base_p = np.array(list(_ARTIFACT_BASE_RATES.values()))
    base_p = base_p / base_p.sum()

    artifact_type = np.array(base_labels)[
        rng.choice(len(base_labels), size=n, p=base_p)
    ].astype(object)

    # Push weekday daytime, work conversations toward code/document/app_website;
    # push weekend/evening + personal conversations toward explanation/guidance,
    # matching Cadences' "backend architecture drops on weekends, personal
    # topics rise" pattern.
    weekday_daytime_work = (~is_weekend) & (hour.between(9, 18)) & df["is_work"].values
    for i in np.where(weekday_daytime_work & (rng.random(n) < 0.30))[0]:
        artifact_type[i] = rng.choice(["code", "document", "app_website"], p=[0.5, 0.35, 0.15])

    evening_or_weekend_personal = (~df["is_work"].values) & (is_weekend | (hour >= 20))
    for i in np.where(evening_or_weekend_personal & (rng.random(n) < 0.35))[0]:
        artifact_type[i] = rng.choice(["explanation", "guidance"], p=[0.6, 0.4])

    # Mild wage-band tilt: higher-wage occupations skew toward code/document/
    # app_website; lower-wage occupations skew toward explanation/guidance.
    # This mirrors Cadences' qualitative finding that higher-wage occupations
    # (e.g. marketing managers) run far more tokens per conversation than
    # lower-wage ones (e.g. editors), which in this dataset is concentrated
    # in exactly those artifact types.
    wage_pctile = pd.Series(df["median_hourly_wage"].values).rank(pct=True).values
    high_wage_tilt = (rng.random(n) < 0.22) & (wage_pctile > 0.75)
    for i in np.where(high_wage_tilt)[0]:
        artifact_type[i] = rng.choice(["code", "document", "app_website"], p=[0.45, 0.35, 0.20])

    low_wage_tilt = (rng.random(n) < 0.18) & (wage_pctile < 0.25)
    for i in np.where(low_wage_tilt)[0]:
        artifact_type[i] = rng.choice(["explanation", "guidance"], p=[0.55, 0.45])

    df["artifact_type"] = artifact_type

    # --- tokens, driven by artifact type and (weakly) wage ---
    token_multiplier = {
        "app_website": 3.2,
        "code": 2.6,
        "document": 1.4,
        "marketing_content": 1.2,
        "guidance": 0.9,
        "misc": 1.0,
        "explanation": 0.2,
    }
    base_tokens = 1800.0
    mult = df["artifact_type"].map(token_multiplier).values
    wage_effect = 1.0 + 0.006 * (df["median_hourly_wage"].values - 40.0)
    wage_effect = np.clip(wage_effect, 0.6, 2.2)
    mu = np.log(base_tokens * mult * wage_effect)
    tokens_total = rng.lognormal(mean=mu - 0.5, sigma=0.9, size=n)
    tokens_total = np.clip(tokens_total, 20, 500_000)
    df["tokens_total"] = tokens_total.astype(int)

    return df


def load_index_data(path_or_hf_id: str) -> pd.DataFrame:
    """Load the EconomicIndex Cadences release.

    Attempts to treat `path_or_hf_id` as a local file first (csv/parquet),
    then as a Hugging Face dataset id. Falls back to a calibrated synthetic
    dataset if neither is reachable, so the rest of the pipeline can run
    end-to-end in network-restricted environments.

    Args:
        path_or_hf_id: local file path, or a Hugging Face dataset id such as
            "Anthropic/EconomicIndex".

    Returns:
        Conversation-level DataFrame matching the Cadences schema.
    """
    import os

    if os.path.exists(path_or_hf_id):
        if path_or_hf_id.endswith(".parquet"):
            df = pd.read_parquet(path_or_hf_id)
        else:
            df = pd.read_csv(path_or_hf_id, parse_dates=["timestamp_utc"])
        return df

    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(path_or_hf_id, split="train")
        df = ds.to_pandas()
        return df
    except Exception as exc:  # pragma: no cover - network/env dependent
        print(
            f"[data_loading] Could not load '{path_or_hf_id}' from disk or the "
            f"Hugging Face Hub ({exc!r}). Falling back to a calibrated "
            f"synthetic Cadences-schema dataset for this run."
        )
        return _simulate_cadences_dataset()


def filter_cadences_window(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the Cadences sampling window and drop no-artifact rows.

    The June 2026 report samples chat/Cowork conversations from 2026-04-10 to
    2026-06-10. "None" is Anthropic's catch-all label for conversations with
    no prominent concrete output (abandoned exchanges, clarifying questions
    with no follow-up, errors); this analysis is scoped to the 93% of
    conversations that produced an artifact, so "none" rows are excluded here
    rather than downstream.

    Args:
        df: raw conversation-level DataFrame with an `artifact_type` and
            `timestamp_utc` column.

    Returns:
        Filtered DataFrame.
    """
    mask = (
        (df["timestamp_utc"] >= pd.Timestamp(CADENCES_WINDOW_START))
        & (df["timestamp_utc"] <= pd.Timestamp(CADENCES_WINDOW_END))
        & (df["artifact_type"].str.lower() != "none")
    )
    return df.loc[mask].copy()
