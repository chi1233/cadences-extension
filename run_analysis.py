"""
run_analysis.py

End-to-end runner: load -> filter -> feature engineer -> analyze -> plot.
Run with: python3 run_analysis.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd

from data_loading import load_index_data, filter_cadences_window
from features import assign_wage_quartile, extract_time_features, normalize_artifact_type, TOP_ARTIFACT_TYPES
from analysis import (
    artifact_mix_by_group,
    compute_entropy,
    hourly_artifact_share,
    token_depth_by_group,
    peak_hour_by_type,
    test_weekday_weekend_shift,
    decompose_token_gap,
)
from plotting import (
    plot_wage_quartile_stack,
    plot_hourly_heatmap,
    plot_weekday_weekend_lines,
    plot_tokens_vs_wage_scatter,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    pd.set_option("display.width", 120)

    # 1. Load
    raw = load_index_data("Anthropic/EconomicIndex")
    print(f"[1/5] Loaded {len(raw):,} raw conversations.")

    df = filter_cadences_window(raw)
    print(f"[2/5] {len(df):,} conversations remain after Cadences-window + "
          f"artifact-produced filter ({len(df) / len(raw):.1%} of raw rows).")

    # 2. Feature construction
    df = assign_wage_quartile(df, wage_col="median_hourly_wage")
    df = extract_time_features(df, timestamp_col="timestamp_utc")
    df = normalize_artifact_type(df, artifact_col="artifact_type", top_n=6)
    print(f"[3/5] Feature engineering complete. Artifact categories: "
          f"{sorted(df['artifact_type_norm'].unique().tolist())}")

    # 3. Core analysis
    # Q1: artifact mix by wage quartile
    wage_mix = artifact_mix_by_group(df, "wage_quartile")
    wage_entropy = compute_entropy(df, "wage_quartile")
    wage_mix_with_entropy = wage_mix.join(wage_entropy)
    wage_mix_with_entropy.to_csv(os.path.join(OUT_DIR, "artifact_mix_by_wage_quartile.csv"))

    # Q2: hourly artifact share, weekday vs weekend
    hourly = hourly_artifact_share(df)
    hourly.to_csv(os.path.join(OUT_DIR, "hourly_artifact_share.csv"), index=False)
    peaks = peak_hour_by_type(hourly, ["code", "explanation", "document"])
    peaks.to_csv(os.path.join(OUT_DIR, "peak_hours_by_type.csv"), index=False)

    shift_test = test_weekday_weekend_shift(df, ["code", "explanation", "document", "guidance"])
    shift_test.to_csv(os.path.join(OUT_DIR, "weekday_weekend_shift_test.csv"), index=False)

    gap = decompose_token_gap(df, low_group="Q1", high_group="Q4")
    pd.DataFrame([gap]).to_csv(os.path.join(OUT_DIR, "token_gap_decomposition_q1_q4.csv"), index=False)

    # Q3 (optional): token depth by artifact type x wage quartile
    token_depth = token_depth_by_group(df, ["artifact_type_norm", "wage_quartile"])
    wage_lookup = df.groupby("wage_quartile", observed=True)["median_hourly_wage"].mean()
    token_depth["mean_wage"] = token_depth["wage_quartile"].map(wage_lookup)
    token_depth.to_csv(os.path.join(OUT_DIR, "token_depth_by_artifact_and_wage.csv"), index=False)

    print("[4/5] Analysis tables written to outputs/.")
    print("\n=== Artifact mix by wage quartile (with entropy) ===")
    print(wage_mix_with_entropy.round(3))
    print("\n=== Peak hours by artifact type (weekday vs weekend) ===")
    print(peaks)
    print("\n=== Weekday vs weekend hourly-shape chi-square test ===")
    print(shift_test.round(4))
    print("\n=== Q1 -> Q4 mean-token gap decomposition ===")
    for k, v in gap.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # 4. Figures
    f1 = plot_wage_quartile_stack(wage_mix)
    f2 = plot_hourly_heatmap(hourly)
    f3 = plot_weekday_weekend_lines(hourly, ["code", "document", "explanation"])
    f4 = plot_tokens_vs_wage_scatter(token_depth)
    print(f"\n[5/5] Figures saved:\n  {f1}\n  {f2}\n  {f3}\n  {f4}")


if __name__ == "__main__":
    main()
