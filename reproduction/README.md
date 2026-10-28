# Frozen PVDAQ evaluation archive

This archive reproduces the manuscript's stored PVDAQ evaluation scores without model training. It contains all three neural seeds and the reported negative results.

From the repository root:

```sh
python -m pip install -r requirements-replay.txt
python reproduction/replay_scores.py --output replay_results.json
```

The scorer verifies the files against SHA-256 hashes, checks the archived all-horizon metrics, and reports each horizon and the mean/sample SD of per-seed scores. Its float64 reductions may differ from archived float32 reductions by less than 0.00001 percentage points. It does not recompute confidence intervals or train models.

## Array meaning

- Each NPZ has one row per increasing forecast issue time and 16 columns corresponding to 15 through 240 minutes.
- `target` and every prediction are already divided once by the site's fixed DC capacity: **P/K**, not kW. MAE is multiplied by 100; do not divide by capacity again.
- `daylight` is the solar-elevation-above-5-degrees scoring mask. `capacity` is the fixed DC research denominator in kW.
- `issue_utc_seconds` is the UTC issue key. `issue_local_assumed` stores the civil-clock label (date only at 2107).
- `tree` is fixed across neural seeds. Prefixes `2021_`, `2022_`, `2023_` identify neural training seeds. Reported means average scores, not predictions.
- At 2107, equal/global/persistence and some pair controls are primary-run controls unless per-seed arrays are explicitly present. The additional sites contain all seed-specific controls.

These are retrospective evaluations: their periods were previously examined. Scores are not pooled across sites. Timestamp assumptions, capacities and GFS initialization-plus-six-hour availability are research conventions. Raw weather and the full raw training datasets are obtained from their original providers; see `../DATA_SOURCES.md` and `../DATA_LICENSE.md`.
