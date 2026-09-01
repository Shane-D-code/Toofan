# Satellite-IR CNN for Tropical Cyclone Rapid Intensification (RI) — SIH

> **Integration status (updated):** This package is the **origin** of the
> satellite CNN, but the **authoritative, single canonical implementation now
> lives in the repo root pipeline at [`src/satellite_cnn.py`](../src/satellite_cnn.py)**
> (class `RICNNFusion`, `FocalLoss`, the valid-pixel-mask encoder, storm-safe
> OOF harness, embeddings and Grad-CAM). `run_pipeline.py` and the official
> demo notebook `TC_RI_CNN_Demo.ipynb` import **that** module and nothing else —
> there is **one** satellite CNN implementation, not a duplicate.
>
> The files here are retained as the proven source / reference for the model
> architecture, patch extraction and MERG-IR download. Do **not** add a second
> training driver; change `src/satellite_cnn.py` instead.

This extends the RI pipeline (IMD best-track + ERA5 + XGBoost) with a
**satellite infrared imagery CNN**, and a **hybrid CNN + tabular fusion model**
that combines both — this fusion model is the one to present as your "best
model" at SIH.

## What's here

```
tc_ri_cnn/
├── data/
│   ├── build_ri_dataset.py     # rebuilds IMD best-track RI labels (RI_24h) — WORKS OFFLINE
│   ├── extract_ir_patches.py   # crops MERG-IR Tb patches centered on each storm fix
│   └── download_mergir.py      # downloads the FULL IR archive (run on Colab/laptop, needs internet)
├── models/
│   └── cnn_model.py            # hybrid CNN(IR) + MLP(tabular) fusion architecture + focal loss
├── train.py                    # end-to-end training + evaluation driver
├── outputs/                    # demo dataset, trained demo weights, plots, metrics
└── README.md                   # this file
```

Everything under `data/build_ri_dataset.py` and `extract_ir_patches.py` and
`train.py` **already runs and was verified in this session** using your
11 uploaded `.nc4` files and the `imdtrack` package (which ships IMD
best-track data offline, no network needed for the tabular half).

## Why a hybrid model is your "best" option for SIH

| Approach | What it sees | Weakness alone |
|---|---|---|
| XGBoost on tabular (your current model) | wind/pressure history, position | Blind to storm *structure* — can't see eyewall formation, convective bursts, symmetry |
| Pure IR-CNN | cloud-top temperature imagery | Blind to storm history/persistence, which is one of the strongest known RI predictors |
| **Hybrid CNN+tabular (this repo)** | both | Best of both — matches the approach in published RI-CNN literature (e.g. Combinido et al. 2018, Su et al. 2020), and lets you tell judges a clear story: *"physics-based persistence features + deep-learning structural features"* |

The architecture (`models/cnn_model.py`):
- **IR branch**: small 4-block CNN (not a deep ResNet) — RI-positive samples
  are rare (~5% of fixes), so a compact model + heavy regularization
  generalizes better than a large one that will just overfit/memorize.
- **Tabular branch**: MLP on lat/lon, current wind & pressure, and
  6h/12h/24h wind-trend features (your existing engineered features).
- **Fusion**: concatenate learned embeddings → small classifier head.
- **Focal loss** (α=0.75, γ=2) instead of plain BCE — down-weights the
  huge number of easy "no-RI" negatives instead of naively oversampling
  the same rare storms over and over.
- A **valid-pixel mask channel** goes in alongside Tb — patches near the
  edge of the global grid or with sensor gaps are filled with a neutral
  280K value, and the mask tells the network which pixels are real.

## What was demonstrated with your 11 uploaded files

All 11 files matched real IMD Bay-of-Bengal fixes (1998-008, 1999-001,
1999-007, 2000-004, 2000-005, 2000-006, and 2020-001 = Cyclone **AMPHAN**,
captured mid-RI as it went from 45kt → 130kt in ~48h). Running the pipeline:

```bash
python3 data/build_ri_dataset.py       # rebuilds full IMD RI dataset (425 storms, 5.47% RI rate in BoB)
python3 data/extract_ir_patches.py     # matches your 11 files -> 11 labeled 400km IR patches
python3 train.py --epochs 25           # trains hybrid model, saves weights + metrics to outputs/
```

See `outputs/ir_patches_demo_grid.png` — you can visibly see Amphan's eye
sharpen and the eyewall cold-cloud ring tighten between the pre-RI and
RI-onset snapshots. That visual is a great SIH slide on its own.

**Important honesty check**: 11 samples is nowhere near enough to *train*
a CNN that generalizes — the `train.py` run on just these files is a
**pipeline-correctness demo**, proving the code works end-to-end, not a
validated model. The reported metrics on 3 held-out samples are not
meaningful. To get a real, presentable model you need the full archive.

## Getting more images ("access more images")

I could not download more `.nc4` files myself — NASA's GES DISC servers
are not reachable from this sandboxed environment (network allowlist
blocks `*.gesdisc.eosdis.nasa.gov`). Run `data/download_mergir.py`
somewhere with normal internet access (Google Colab is easiest):

```bash
pip install earthaccess
python3 data/download_mergir.py --basin BOB --out_dir ./mergir_archive
```

This targets **only the ~3,000 hours actually needed** (every valid RI
fix + its -6h/-12h/-24h lag times), not the full 40-year global archive,
so it's a manageable download (a few thousand files, each ~25-35 MB).
You'll need a free Earthdata login: https://urs.earthdata.nasa.gov/

Once downloaded, just re-run `train.py --nc_folder ./mergir_archive` — no
other code changes needed, since `extract_ir_patches.py` automatically
scans whatever folder you point it at.

## Suggested SIH-day improvements (if you have time before the demo)

1. **More storms = more positives.** Prioritize downloading IR for the
   169 BoB RI-positive fixes first, then a matched sample of negatives,
   rather than the full dataset, if download time is limited.
2. **Multi-frame input.** Stack Tb(t), Tb(t-6h), Tb(t-12h) as 3 channels
   instead of 1 — lets the CNN see cloud-top *cooling trend*, which is
   itself a strong RI signal (this is already scaffolded via the
   `ir_channels` parameter in `RICNNFusion`).
3. **Storm-level cross-validation**, not just a single split — with only
   ~40 RI-positive storms, a single random split can be misleading;
   report mean ± std over 5 storm-grouped folds.
4. **Report POD / FAR / CSI**, not just accuracy — these are the metrics
   real RI-forecast verification (NHC, IMD) uses, and `train.py` already
   computes them. Accuracy alone is meaningless at 5% positive rate
   (94%+ "accuracy" from a model that always predicts "no RI").
5. **Add SHAP/Grad-CAM visualizations** of what the CNN is looking at in
   the IR patch — a heatmap over the eyewall on RI cases is a very strong
   visual for judges.
