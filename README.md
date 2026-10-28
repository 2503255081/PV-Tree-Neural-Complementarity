# Tree-neural complementarity in photovoltaic power forecasting

Code and evaluation data for **Complementarity between gradient-boosted trees and deep temporal models in short-term photovoltaic power forecasting**.

The study tests whether deep time-series forecasts improve a competitive tree forecast, how that benefit changes across sites and forecast horizons, and how much complexity is useful. It compares LightGBM, a local N-HiTS variant with weather inputs (N-HiTSx), TimeMixer, their fitted pairs and static three-model combinations. The evaluations are retrospective; this repository does not establish a new mixture-of-experts algorithm or state-of-the-art performance.

## Reproduce the stored evaluation scores

```sh
git clone https://github.com/2503255081/PV-Tree-Neural-Complementarity.git
cd PV-Tree-Neural-Complementarity
python -m pip install -r requirements-replay.txt
python reproduction/replay_scores.py --output replay_results.json
```

Python 3.12 and NumPy 2.3.5 were used to verify this command. It runs on CPU and does not require PyTorch, a GPU or raw-weather downloads. The approximately 76 MB evaluation archive is included in Git; no Git LFS setup is needed.

The script checks SHA-256 hashes, recomputes all 16 horizon errors on the original common forecast cases, checks stored scores and summarizes three neural training seeds. `results/replay_results.json` records the verified output. The maximum difference from archived scores is 0.00000591 percentage points, below the manuscript's displayed precision.

| PVDAQ site | LightGBM | Three-model ensemble, mean +/- sample SD |
| --- | ---: | ---: |
| 2107 | 4.5405 | 4.3538 +/- 0.0112 |
| 9069 | 6.2230 | 5.5726 +/- 0.0334 |
| 9068 | 9.2070 | 9.1862 +/- 0.0119 |

Values are daylight MAE as a percentage of fixed DC capacity, with equal weight for the 16 horizons from 15 to 240 minutes. The targets and predictions in the NPZ files are already P/K: **do not normalize by capacity a second time**. The small mean difference at 9068 is not statistically resolved by the manuscript's paired interval, and its fitted LightGBM/N-HiTSx pair has lower mean error than three models. Scores are not pooled across sites.

## Repository contents

- `reproduction/`: frozen targets, forecasts, issue identifiers, daylight masks, weights, expected scores and an independently runnable metric scorer.
- `code/`: portable model and training components, with their configuration, source provenance and usage instructions.
- `evidence/`: model/input specifications and numerical comparison summaries, including unfavorable cases.
- `results/`: score-replay output verified for this release.
- `DATA_SOURCES.md`: original PVDAQ and GFS sources, input meaning and acquisition boundaries.
- `MODEL_CARD.md`: chronology, data completeness, weight-fitting objectives and interpretation limits.

## Training and evaluation are different reproduction tasks

The included archive is sufficient to recompute the stored evaluation point metrics. Fresh training additionally requires the original raw power and archived forecast inputs, their preprocessing and the correct chronological partitions. Those full raw training inputs and the original model checkpoints are not bundled. See `code/README.md` for the prepared-array interface, executable commands and the parts that have actually been checked.

The training entry point covers the prepared-array workflow for 9069 and 9068. It includes the original local neural implementations and 311-column tree feature construction. Synthetic CPU checks verified both model parameter counts, forward/backward execution, a small tree fit and convex-weight fitting. The original 2107 preprocessing/training pipeline and raw-data acquisition are not fully automated in this release. The replay and training requirement files record separate checked environments; use separate virtual environments if installing both sets.

The release does not rerun full model training, claim bitwise checkpoint reproduction, recompute all block-bootstrap intervals or reproduce every adapted baseline in the paper. An exact score replay verifies the archived predictions; it is not evidence that a newly trained model has replicated the experiment.

## Data and code rights

Original source code in this repository is provided under the MIT license in `LICENSE`; applicable third-party notices are retained with their code. The processed PVDAQ evaluation archive and author-generated forecasts are provided under CC BY 4.0 with the attribution and modification notice in `DATA_LICENSE.md`. Cite the original PVDAQ dataset as well as this repository.

This repository is a research release, not a solar forecasting service. No private industrial data, credentials or manuscript submission correspondence are included.

## Citation

Use `CITATION.cff` for repository citation. A journal acceptance or article DOI is not claimed. Original dataset: Deline et al. (2021), *Photovoltaic Data Acquisition (PVDAQ) Public Datasets*, OEDI/NREL, https://doi.org/10.25984/1846021.
