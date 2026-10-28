# Study and evaluation scope

This repository accompanies **Complementarity between gradient-boosted trees and deep temporal models in short-term photovoltaic power forecasting**. The study compares a LightGBM model for each forecast horizon, a local N-HiTS variant with additional weather inputs, and TimeMixer. A static convex ensemble combines their independently trained forecasts.

## Forecasting task

- Target: measured PV power, normalized once by a fixed DC capacity.
- Forecast horizons: 16 horizons, 15 through 240 minutes.
- Main score: average of the 16 horizon-specific MAEs, with targets retained when solar elevation exceeds five degrees; scores are percentages of DC capacity.
- Sites: PVDAQ2107, PVDAQ9069, PVDAQ9068. Models are trained separately at each site, and errors are not pooled across sites.
- Neural training seeds: 2021, 2022, 2023. The tree is fixed across these neural seeds.
- Inputs: historical power, calendar and solar-position variables, archived GFS weather forecasts. The assumed GFS availability is initialization plus six hours; actual historical receipt times were unavailable.

## Chronological partitions

| Site | Model training | Early stopping for tree | Inner neural weights | Outer tree/pair weights | Evaluation |
| --- | --- | --- | --- | --- | --- |
| 2107 | 2019-2022 | See original training configuration | Jul-Sep 2023 | Oct-Dec 2023 | Jan-Oct 2024 |
| 9069 | 2022 | Jan-Feb 2023 | Mar-Apr 2023 | May-Jun 2023 | Jul-Oct 2023 |
| 9068 | 2022-2023 | Jan-Apr 2024 | May-Aug 2024 | Sep-Dec 2024 | Jan-Apr 2025 |

All evaluation periods had been examined during prior development. These results are retrospective replication, not an untouched confirmation or operational validation.

## Data completeness and normalization

Capacities are fixed research denominators: 893, 38,687 and 4,738 kW for 2107, 9069 and 9068. They do not certify AC operating limits. The additional sites require complete 96-slot historical and 16-slot future power windows, complete weather fields and a 15-minute power availability delay. Quarter-hour power means require all three five-minute readings. Daylight-saving transition days are excluded. The earlier 2107 cohort permits incomplete historical observations if at least 48 slots are valid.

Site 9069 uses the fixed revenue-meter channel, with negative readings invalid. Site 9068 sums the two fixed AC inverter channels after clipping negative readings of each channel to zero; both channels must be present. Missing raw data and documented official weather-source missing responses are not replaced with future observations.

## Weight fitting

The inner stage fits the N-HiTSx/TimeMixer weight, then the outer stage fits the LightGBM/neural-combination weight. Each pair fits one weight using the common outer interval. All weights are fixed during evaluation. The global-weight objective differs in the archived implementations: 2107 pools eligible forecast cases, while the additional sites weight horizons equally. This difference is retained and documented, not silently standardized.

## Interpretation and limitations

At 2107 the neural models have worse standalone average errors but improve the tree when combined. At 9069 both neural models already outperform the tree. At 9068 the tree is stronger, the ensemble improvement over it is not resolved by the reported paired interval, and the fitted tree/N-HiTSx pair has lower mean error than three models. A separate hourly GEFCom2014 comparison in the manuscript also fails to show an ensemble advantage.

Global weighting captures nearly all of the observed benefit of horizon-specific weights. The third-model increment is small at 2107 and is not individually resolved for every neural seed. Weather subgroups and some months favor the tree. Adapted deep baselines differ in features and training budgets; the study is not a strict ranking of their original architectures.

The published replay checks point metrics, horizons and training-seed summaries. It does not recompute the manuscript's block-bootstrap confidence intervals. The archived intervals use different resampling procedures at 2107 and the additional sites. Training seeds are repeated fits to the same data, not independent evaluation datasets.

Inference timings describe the tested implementation and CPU/GPU configuration, not intrinsic architecture efficiency or a production service. No new models or weights were fitted for this repository release.
