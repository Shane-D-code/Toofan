# Leakage Audit

Automatic data-leakage checks for the RI pipeline (Phase 5). Each rule either passes or fails; the pipeline aborts on any failure.

## 1. no storm in multiple splits  [PASS]
- no violations

## 2. no future observation used  [PASS]
- no violations

## 3. no target-time variable in predictors  [PASS]
- no violations

## 4. no satellite image after prediction time  [PASS]
- no violations

## 8-9. split-before-resample + duplicate granules  [PASS]
- no violations

## 10-11. scaler fit on training data only  [PASS]
- XGBoost branches (IMD / ERA5 / Combined) use native `missing` handling —
  no external scaler fit on the full dataset.
- Satellite CNN leg: per-fold MinMaxScaler (180–310 K normalization) is fit on
  **training storms only**, never on the full or test set — verified by
  `src/leakage.check_preprocessing_leakage`.
- SMOTE is NOT used by default (temporal correlation makes synthetic samples
  physically implausible); when present it is ablation-only, applied inside
  train folds after the storm split, never before it.

## CNN tabular branch (real 11 IMD features)  [PASS]
- The hybrid CNN tabular head consumes the **11 contemporaneous IMD features** joined to each satellite image at its observation time `t` (`CN_TAB_FEATURES`).
- Strict join: all 11 features present at `t`; rows with any missing feature are **removed, never zero-padded / imputed**.
- `imd_p'`-type target-time / future variables are never used as predictors; `RI_24h` is only the label.
- Per-fold MinMaxScaler fitted on **training storms only**; never fit globally.
- ERA5 variables are **not** part of this 11-feature head; ERA5 stays on its own branch until the multimodal-fusion stage.
- Resulting hybrid set: 9 rows / 7 storms (6 RI / 3 non-RI).

## Summary
- 0 of 7 rule group(s) FAILED.

## ERA5 expansion (satellite-overlap) leakage audit  [PASS]
- **Source:** 20 new Copernicus CDS `reanalysis-era5-pressure-levels` NetCDF files
  (`ERA5_expanded/`), one per date needed to cover the 22 satellite
  observations that previously lacked ERA5.
- **Temporal matching:** each file is fetched for the exact satellite
  observation time `t` (`era5_delta_minutes = 0`). No ERA5 field from after `t`
  is ever used — the reanalysis is interpolated at, not after, the observation
  instant. This respects the project's no-future-information bar.
- **Matching rule:** the three-way table is built on exact
  `(storm_id, datetime_utc)` joins — no filename-only matching, no nearest-time
  fallback beyond `era5_delta_minutes = 0`.
- **No target leakage:** `RI_24h` is only ever the label, never a predictor.
  The appended rows reuse IMD's `RI_24h` label at the same timestamp.
- **No global normalization:** the extracted fields are raw ERA5 values written
  directly into the canonical feature table format; no scaler is fit on the
  combined or test data. (Any downstream scaler must be fit on training storms
  only, as in the existing pipeline.)
- **No duplicate rows:** appended rows are de-duplicated against the canonical
  ERA5 table on `(storm_id, datetime_utc)` (keep last); no storm/time row is
  duplicated.
- **Preserved data:** the canonical `models/RI_ERA5_features_MVP.csv` is NOT
  modified; expansion lives in `results/RI_ERA5_features_expanded.csv`.
- **Physical sanity:** new-row RH / temperature / shear / divergence ranges are
  consistent with the existing ERA5 rows (e.g. r_850 84.6–109.9 %, t_850
  290.9–296.1 K, shear 2.1–30.9 m/s). RH > 100 % is expected supersaturation in
  ERA5, matching the original data's range (up to 102.3 %).
- Resulting three-way overlap: **26 observations / 23 storms / 9 RI / 17 non-RI**
  (up from 4 obs / 4 storms / 3 RI before the expansion).

## TCIR CNN (global multimodal) leakage audit  [PASS]
- **Source:** `results/tcir_oof_predictions.csv`, `results/tcir_embeddings.npy`,
  `results/tcir_embeddings_meta.csv` (global TC storm set, 2003-2016).
- **Out-of-fold only:** every TCIR probability in `P_RI` is a genuine
  out-of-fold prediction — no TCIR row is ever scored by a model that saw its
  storm during training (no leakage into `tcirOof`).
- **No target leakage:** `RI_24h` is used strictly as the label, never as a
  predictor; the TCIR embedding/probability carry no future-target information.
- **Exact-time matching only:** TCIR↔IMD alignment is on exact shared 3-hourly
  `datetime_utc`; the ≥30-timestamp / single-IMD-storm rule guarantees
  unambiguous matches. No nearest-time fallback, no future fields.
- **No global normalization:** embeddings are used as-is; any downstream model
  (XGBoost) is fit on training storms only, thresholds tuned on validation only.
- **No duplicate rows:** 0 duplicate `(storm_id, datetime_utc)`; OOF, metadata
  and embeddings are row-aligned (audit asserts `oof_meta_aligned`,
  `ri_meta_aligned`, `emb_n_matches_meta` all true).
- **Coverage honesty:** TCIR (2003-2016) and ERA5 (1982-2000) share **0**
  timestamps; the three-way `IMD+ERA5+TCIR` model is reported NOT EVALUABLE
  rather than given a made-up number.
- **Note (documented conflict, not leakage):** for the aligned IO storms, TCIR
  and IMD `RI_24h` labels disagree with 0 shared RI cases; this is a label
  inconsistency between the two source datasets, not a leakage violation — the
  IMD+TCIR fusion is therefore reported as data-limited MVP, not fabricated.
