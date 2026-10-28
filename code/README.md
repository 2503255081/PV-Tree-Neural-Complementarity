# Training and inference code

This directory contains the local LightGBM feature construction, N-HiTSx and
TimeMixer implementations, training loops, static convex fitting and scoring
used for the paper's two additional PVDAQ sites, 9069 and 9068. The neural
architectures and feature calculations are copied from the research source;
only docstrings, path handling and device selection were adapted for release.
`SOURCE_PROVENANCE.json` records the original and distributed source hashes.

The package is not an independent reimplementation of the published N-HiTS or
TimeMixer benchmarks. These are the study's local photovoltaic adaptations.

## What is executable

From the repository root, use Python 3.11 and install the training dependencies:

```text
python -m pip install -r code/requirements.txt
python code/run.py --help
python code/run.py smoke
```

The pinned versions reflect the locally available research environment during
release preparation. Paper runs used PyTorch 2.7.1 with CUDA 12.8 on an RTX A6000;
select the appropriate PyTorch wheel for your hardware. This is not a claim that
every dependency was frozen at the original training date. The smoke command
uses synthetic arrays on CPU, takes one neural optimization step per model,
checks 311 tree columns, and checks convex-weight recovery. If LightGBM is
installed, it also performs a tiny two-round synthetic tree fit. It never reads
the published evaluation arrays or any external dataset.

The full training commands require the prepared arrays described below. They
write only under the supplied work directory. All roles must be prepared before
feature construction and freezing. Run separately for 9069 and 9068:

```text
python code/run.py --work-dir runs features --site 9069
python code/run.py --work-dir runs features --site 9068
python code/run.py --work-dir runs freeze --sites 9069 9068
python code/run.py --work-dir runs tree --site 9069
python code/run.py --work-dir runs neural --site 9069 --model nhitsx --seed 2021
python code/run.py --work-dir runs neural --site 9069 --model timemixer --seed 2021
```

Repeat the neural commands for seeds 2022 and 2023 and repeat the tree/neural
commands for site 9068. Then seal component predictions and fit the static
combinations using the chronological inner and outer periods:

```text
python code/run.py --work-dir runs predict --site 9069
python code/run.py --work-dir runs predict --site 9068
python code/run.py --work-dir runs score --sites 9069 9068
```

The default device is CUDA and preserves bfloat16 autocasting and TF32 settings.
`--device cpu` is available for portability, but changes numerical execution and
is not a reproduction of the GPU timing measurements. No speed equivalence is
claimed. Training is not automatically started by installing or importing this
package. The archived budgets remain fixed: 12 epochs, batch 256, AdamW 3e-4,
weight decay 1e-4, L1 loss and gradient norm 1; TimeMixer then receives 2,400
updates at 3e-6. LightGBM uses 16 heads, up to 1,800 rounds and 120-round patience
on the separate early-stopping period. Existing model files are not overwritten.

## Prepared array contract

For each site the work directory has:

```text
runs/9069/normalizer.json
runs/9069/data/train/*.npy
runs/9069/data/early_stop/*.npy
runs/9069/data/inner/*.npy
runs/9069/data/outer/*.npy
runs/9069/data/evaluation/*.npy
```

Required arrays in every role (`N` is its number of issue times):

| File | Shape | Meaning |
| --- | --- | --- |
| past_power.npy | N,96,1 | Power divided once by fixed DC capacity |
| observed.npy | N,96,1 | Boolean history validity; complete at these two sites |
| past_A.npy / future_A.npy | N,96,13 / N,16,13 | Solar/time features |
| past_nwp.npy / future_nwp.npy | N,96,27 / N,16,27 | Raw ordered GFS feature columns |
| target.npy | N,16 | Capacity-normalized future power |
| p48.npy | N,16 | Causal 48-hour persistence, NaN if unavailable |
| daylight.npy | N,16 | Target solar elevation above 5 degrees |
| origins.npy | N | Unique ascending issue UTC seconds |
| coordinates.npy | N,2 | Latitude and longitude |
| capacity.npy | N | Fixed DC denominator in watts for feature construction |
| keys.csv | N rows | Must include issue_local_assumed for monthly/block analyses |

`normalizer.json` contains `mean` and `scale`, each 27 entries fitted only on
training future NWP, and records `fit_split: "train"`. Inputs in the arrays are
raw NWP values; the batch loader applies this normalization once. The 27-column
order is in `pv3/gfs_adapter.py`; solar feature names are in `pv3/common.py`.

The frozen evaluation NPZ score bundle elsewhere in this repository contains
targets and predictions, not these model input arrays. It supports metric replay
without training but cannot replace the prepared training corpus.

## Materialization from archived acquisition products

The original materializer is also included:

```text
python code/run.py --work-dir runs materialize --site 9069 --source-root /path/to/strict_acquisition
```

The input directory must already contain `9069_strict_power.csv`,
`9069_power_only_queue.csv`, `nwp_feature_schema.json`, and `gfs_9069_v3/` with
`acquisition_config.json`, per-key `receipts/YYYY/MM/*.json`,
`raw/YYYY/MM/*.nc` subsets and documented failures. Site 9068 uses the analogous
names. This command preserves source checks, the own-site weather grid,
initialization-plus-six-hour availability assumption, native statistical
intervals, 96 history/16 target completeness, and exclusions.

For relocation, a receipt's stored `local_path` is checked against the expected
relative `raw/year/month/key.nc` suffix, followed by its recorded hash and native
NetCDF checks. No fallback site, newer cycle, interpolation or future measured
weather is introduced. An arbitrary directory of downloaded GFS files is not
sufficient: the original receipt/configuration schema is required.

Acquiring the raw source data, reconstructing the strict power CSV/queue from
the provider's files, and recreating the original 2107 preprocessing pipeline
are not automated in this release. The original 2107 cohort permits some missing
history and has a different global-weight objective; do not apply these two-site
materialization or global fitting rules to claim exact 2107 reproduction.
The source descriptions and final evaluation bundle document those boundaries.

## Interpretation

All paper evaluation periods had already been examined. The results are
retrospective. Sites are trained separately; no pooled training or transferable
model is claimed. Global fitting at these two sites gives equal weight to each
horizon. The released scorer retains the original seven-day, nonoverlapping
block bootstrap and seed-mean procedure for these sites. It is not the original
2107 circular bootstrap and should not be presented as a harmonized reanalysis.

No full retraining or new evaluation experiment was performed to prepare this
release. Passing synthetic checks proves interface/shape consistency, not
bitwise recovery of historical checkpoints or final paper scores.
