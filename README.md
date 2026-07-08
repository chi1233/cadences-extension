# Artifacts and Cadence

Decomposing Claude's artifact mix by occupation, wage, and time of day, extending
the June 26, 2026 Anthropic Economic Index report, *Cadences*.

## Structure

```
economic_index_cadences/
  .github/workflows/run-analysis.yml  CI: installs deps, runs the pipeline, uploads figures/CSVs as artifacts
  src/
    data_loading.py   load_index_data, filter_cadences_window, synthetic fallback
    features.py        assign_wage_quartile, extract_time_features, normalize_artifact_type
    analysis.py         artifact_mix_by_group, compute_entropy (raw + normalized), hourly_artifact_share,
                          token_depth_by_group, peak_hour_by_type, test_weekday_weekend_shift,
                          decompose_token_gap
    plotting.py          Figures 1-4
  run_analysis.py         end-to-end pipeline
  figures/                 PNG outputs
  outputs/                  CSV analysis tables
  requirements.txt          pinned dependencies
  technical_note.md          2-3 page note draft
```

## Statistical additions beyond the original spec

- `test_weekday_weekend_shift`: chi-square test on the full 24-hour distribution shape (not just peak
  hour) per artifact type, so the "temporal patterns differ by weekday/weekend" claim in the note is
  backed by a p-value rather than an eyeballed peak-hour comparison.
- `decompose_token_gap`: a standard two-part (Oaxaca-style) decomposition of the wage-quartile token
  gap into a composition effect (different artifact-type mix) and a within-type effect (same artifact
  type running longer at higher wages). This replaced an earlier draft of the note that asserted
  composition "dominates," which turned out to be wrong once tested formally: composition explains
  about 43% of the Q1-to-Q4 gap in this run, within-type effects about 57%.
- `compute_entropy` now also returns `entropy_normalized` (entropy / max possible entropy for the
  category count), which is the version worth comparing across groups.

## Data access

This was built and run in a sandbox without network access to huggingface.co.
`load_index_data()` tries, in order: (1) a local file path, (2) the
`datasets` library against the Hugging Face Hub, (3) a calibrated synthetic
fallback with an identical schema. **All numbers in `technical_note.md` and
the figures in this run are therefore from the synthetic fallback, not the
real Anthropic/EconomicIndex release.** The synthetic generator
(`_simulate_cadences_dataset` in `src/data_loading.py`) is calibrated to
match the summary statistics Anthropic published in Cadences (93% artifact
rate; explanation 17% / document 15% / guidance 11% of outputs; ~35% to ~50%
weekday-to-weekend personal-use swing; positive wage-tokens correlation;
app/website and code carrying the most tokens, explanations the fewest) so
that the pipeline logic and figure shapes are directionally realistic, but
the exact quartile splits, entropy values, and peak hours reported in the
note are illustrative of the method, not a finding about real usage.

To run against the real release once Hub access is available:

```bash
pip install datasets
python3 run_analysis.py
```

No code changes are required; `load_index_data("Anthropic/EconomicIndex")`
will pull the real Hub dataset instead of falling back.

## Key assumption to revisit against the real schema

The real Cadences release publishes artifact labels and hourly aggregates,
but the exact column names for occupation wage mapping and per-conversation
artifact labels were not directly inspectable in this environment. This
pipeline assumes:

- `artifact_type`: top-level artifact classifier string per conversation
  (single label; if the real schema instead stores multiple labels per
  conversation, the "pick a consistent rule" step described in
  `normalize_artifact_type` needs a dominant-label or final-turn rule applied
  upstream, in `filter_cadences_window` or a new `resolve_artifact_label`
  function).
- `median_hourly_wage`: BLS OEWS wage joined via a mapped O*NET/SOC
  occupation field, one value per conversation.
- `tokens_total`: total conversation token count (Anthropic reports
  geometric means of this field because it is right-skewed; `analysis.py`
  follows that convention).

## Run

```bash
python3 run_analysis.py
```
