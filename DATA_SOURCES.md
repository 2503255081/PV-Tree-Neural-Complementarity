# Original sources and included data

The included NPZ files are the frozen evaluation archive for the manuscript's three PVDAQ sites. They contain targets, prediction arrays, capacity values, daylight masks and issue-time identifiers. They allow the released scorer to reproduce the reported point metrics. They contain neither the historical training inputs nor the archived GFS feature tensors.

## PVDAQ

- Dataset DOI: https://doi.org/10.25984/1846021
- Official OEDI record and license: https://data.openei.org/submissions/4568
- Official data layout and public download instructions: https://github.com/openEDI/documentation/blob/main/pvdaq.md
- Public object prefix: `s3://oedi-data-lake/pvdaq/2023-solar-data-prize/`

Metadata for the included systems:

| System | Official metadata | Frozen evaluation windows |
| --- | --- | ---: |
| 2107 | [2107_system_metadata.json](https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/2107_OEDI/metadata/2107_system_metadata.json) | 29,211 |
| 9069 | [9069_system_metadata.json](https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/9069_OEDI/metadata/9069_system_metadata.json) | 11,696 |
| 9068 | [9068_system_metadata.json](https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/9068_OEDI/metadata/9068_system_metadata.json) | 10,038 |

For example, the AWS CLI can list public source files without an account:

```sh
aws s3 ls --no-sign-request s3://oedi-data-lake/pvdaq/2023-solar-data-prize/2107_OEDI/ --recursive
```

Select the years and channels described in `MODEL_CARD.md` and the released preparation instructions. A source-file listing alone does not reproduce the paper's filtering and forecast-availability rules.

## GFS numerical weather forecasts

- NCAR GDEX dataset d084001: https://gdex.ucar.edu/datasets/d084001/
- DOI: https://doi.org/10.5065/D65D8PWK
- Citation instructions: https://gdex.ucar.edu/datasets/d084001/citation/
- Original data contributor: NOAA/NWS/NCEP. The current GDEX record displays CC BY 4.0.

The paper uses eight physical variables: 2 m temperature, 2 m relative humidity, 10 m wind u/v, surface pressure, entire-atmosphere total cloud cover, downward shortwave radiation, and surface precipitation accumulation. The 27-column schema adds native statistical interval spans, forecast lead, cycle age, endpoint-to-target offset and validity flags. See `evidence/nwp_feature_schema.json` for the original column order.

Reconstruct initialization time, valid time and native statistical interval for every forecast. The research assumes availability six hours after initialization; no historical delivery-time log validates this assumption. Do not substitute later cycles or realized future weather to fill missing inputs. Keep documented unavailable source requests missing, then exclude affected forecast windows.

Access dates in a new reproduction should describe the reproducer's actual downloads. The 2026-10-28 rights check is not the original experiments' acquisition date.

## Other manuscript comparisons

The hourly GEFCom2014 comparison and history-only DKASC pilot are discussed in the manuscript but their raw records are not redistributed in this release. Results from those tasks should not be presented as reproduced by the three-site PVDAQ score replay.
