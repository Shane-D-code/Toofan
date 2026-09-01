# Satellite Recovery Verification (Phases 1-3)

- Usable images : 26
- Storms        : 23
- RI / non-RI   : 9 / 17
- QC pass       : 26/26
- Duplicate groups : 0

## Time matching (documented tolerance)

- Max accepted |satellite - IMD| = **120 minutes** (config `satellite.max_time_diff_min`).
- Duplicate satellite granules kept per IMD observation: **False** (config `satellite.keep_duplicates`; when False only the nearest granule is kept).
- Satellite images taken AFTER the IMD observation are excluded as post-target leakage (see LEAKAGE_AUDIT.md).

## 2020-001 storm recovery (previously unusable)

The prior audit had too few images for 2020-001 to be a satellite test storm. With the raw NC4 granules recovered it matches these IMD observations:

| IMD obs time | Satellite time | diff (min) | RI_24h |
| --- | --- | --- | --- |
| 2020-05-18 03:00:00 | 2020-05-18 03:00:00 | 0.0 | 0 |
| 2020-05-17 09:00:00 | 2020-05-17 08:00:00 | 60.0 | 1 |
| 2020-05-17 18:00:00 | 2020-05-17 19:00:00 | 60.0 | 1 |
| 2020-05-18 12:00:00 | 2020-05-18 11:00:00 | 60.0 | 0 |

## QC report

- Full per-image QC: `satellite_qc_report.csv`.
- Extraction log: `extraction_log.csv`.
- Normalisation: `normalization.json` (global 180-310 K window, cold-cloud-high).

