# ============================================================
# Bay of Bengal Cyclone RI Detection — src package
# ============================================================
# Modular pipeline:
#   data.py        -> load canonical CSVs / npy, merge, storm-safe split,
#                     RI label construction audit + end-of-storm censoring
#   features.py    -> clean feature sets + physically meaningful derived
#                     predictors + land interaction features
#   models.py      -> IMD, ERA5, combined XGBoost trainers
#   evaluate.py    -> storm-safe metrics, threshold tuning, comparisons,
#                     storm-bootstrap CI, probability calibration,
#                     preprocessing leakage guards
#   baselines.py   -> persistence / trend / climatology baselines
#   event_metrics.py -> event-level (episode) detection metrics
#   satellite_qc.py -> satellite image QC + duplicate detection
#   satellite.py    -> data-limited satellite branch gate
#   satellite_cnn.py-> CANONICAL satellite CNN (hybrid RICNNFusion + focal
#                      loss + valid-mask + storm-safe OOF + Grad-CAM)
#   satellite_bridge.py -> ingests CNN artifacts for fusion
#   leakage.py       -> automatic data-leakage audit (fail-fast)
#
# The pipeline is driven by run_pipeline.py and configured via config.yaml.
# ============================================================

from .config import get_config

__all__ = ["get_config"]
