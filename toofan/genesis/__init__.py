"""TOOFAN Genesis module.

24-hour tropical cyclone genesis prediction.

Only three models are integrated:

    LightGBM     (PRIMARY / PRODUCTION)   weight 0.40
    XGBoost      (ensemble)               weight 0.35
    RandomForest (ensemble)               weight 0.25

Threshold: 0.24 (documented optimized, NOT 0.50).
Target:    genesis_24h  (0 = non-genesis, 1 = genesis)
Input:     34 features.

IMPORTANT SCIENTIFIC CAVEAT
    This system is a PROTOTYPE (300 samples, 191 storms, 2015-2024). The
    report states synthetic SST/SST-anomaly and TCHP/OHC700 were used, and
    storm-aware CV was lower than the final held-out test. Do NOT claim
    production validation / operational proof without new evidence.
"""
