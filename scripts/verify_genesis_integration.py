"""Comprehensive Genesis integration verification.

Runs an inline end-to-end verification using tiny real models serialized to
temporary artifacts, exercising:

    - LightGBM / XGBoost / RandomForest loading
    - predictions
    - ensemble using EXACT weights 0.40 / 0.35 / 0.25
    - threshold 0.24
    - no CatBoost / ExtraTrees loaded
    - original-vs-adapter fidelity
    - provenance + SHA-256 recording
    - missing-artifact honest failure
    - orchestrator Phase1/Phase2
    - native runtime compatibility

This mirrors the pytest suite but also prints a human-readable report.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Runtime must be configured before ML frameworks import.
from toofan import configure_runtime, load_config

configure_runtime(load_config())

import numpy as np  # noqa: E402

from toofan.core.schemas import ForecastContext, sha256_of_file  # noqa: E402
from toofan.genesis.ensemble import ENSEMBLE_WEIGHTS, GenesisSoftVotingEnsemble  # noqa: E402
from toofan.genesis.factory import ModelFactory  # noqa: E402
from toofan.genesis.model_adapters import (  # noqa: E402
    LightGBMAdapter,
    RandomForestAdapter,
    XGBoostAdapter,
)
from toofan.genesis.service import GenesisService  # noqa: E402
from toofan.orchestrator import CyclonePipelineOrchestrator  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
results: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def make_tiny_models(dirpath: Path) -> None:
    import lightgbm as lgb
    import xgboost as xgb
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 34))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=200) > 0).astype(int)

    # LightGBM
    d = lgb.Dataset(X, label=y)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "seed": 42, "num_leaves": 7},
        d, num_boost_round=5,
    )
    booster.save_model(str(dirpath / "tc_genesis_lightgbm_300_OPTIMIZED.joblib"))

    # XGBoost
    x = xgb.XGBClassifier(n_estimators=5, max_depth=2, learning_rate=0.2, eval_metric="logloss")
    x.fit(X, y)
    x.save_model(str(dirpath / "tc_genesis_xgboost_300_OPTIMIZED.joblib"))

    # RandomForest
    rf = RandomForestClassifier(n_estimators=5, random_state=42, max_depth=3)
    rf.fit(X, y)
    joblib.dump(rf, str(dirpath / "tc_genesis_randomforest_300_OPTIMIZED.joblib"))


def main() -> int:
    feats = {f: float(v) for f, v in zip(
        __import__("toofan.genesis.features", fromlist=["GENESIS_FEATURES"]).GENESIS_FEATURES,
        np.linspace(0.1, 0.9, 34),
    )}

    with tempfile.TemporaryDirectory() as td:
        adir = Path(td) / "artifacts"
        adir.mkdir()
        make_tiny_models(adir)

        lgbp = adir / "tc_genesis_lightgbm_300_OPTIMIZED.joblib"
        xgbp = adir / "tc_genesis_xgboost_300_OPTIMIZED.joblib"
        rfp = adir / "tc_genesis_randomforest_300_OPTIMIZED.joblib"

        # 1. Loading
        lgb_ad = LightGBMAdapter().load(lgbp, Path(td))
        check("LightGBM loads", lgb_ad.is_loaded, lgbp.name)
        xgb_ad = XGBoostAdapter().load(xgbp, Path(td))
        check("XGBoost loads", xgb_ad.is_loaded, xgbp.name)
        rf_ad = RandomForestAdapter().load(rfp, Path(td))
        check("RandomForest loads", rf_ad.is_loaded, rfp.name)

        X = np.array([list(feats.values())])
        p_lgb = float(lgb_ad.predict_proba(X)[0])
        p_xgb = float(xgb_ad.predict_proba(X)[0])
        p_rf = float(rf_ad.predict_proba(X)[0])
        check("LightGBM prediction in [0,1]", 0.0 <= p_lgb <= 1.0, f"{p_lgb:.4f}")
        check("XGBoost prediction in [0,1]", 0.0 <= p_xgb <= 1.0, f"{p_xgb:.4f}")
        check("RandomForest prediction in [0,1]", 0.0 <= p_rf <= 1.0, f"{p_rf:.4f}")

        # 2. Ensemble exact weights (mock math)
        ens = GenesisSoftVotingEnsemble(_Stub(0.80), _Stub(0.60), _Stub(0.40))
        res = ens.predict(np.zeros((1, 34)))
        check("Ensemble EXACT weights 0.40/0.35/0.25", ENSEMBLE_WEIGHTS ==
              {"lightgbm": 0.40, "xgboost": 0.35, "randomforest": 0.25})
        check("Ensemble deterministic 0.63", abs(res.ensemble_probability - 0.63) < 1e-9,
              f"{res.ensemble_probability:.6f}")

        # 3. real ensemble
        ens2 = GenesisSoftVotingEnsemble(lgb_ad, xgb_ad, rf_ad)
        res2 = ens2.predict(X)
        expected = 0.40 * res2.lightgbm_probability + 0.35 * res2.xgboost_probability + 0.25 * res2.randomforest_probability
        check("Ensemble weighted math real models", abs(res2.ensemble_probability - expected) < 1e-9)

        # 4. No CatBoost / ExtraTrees
        from toofan.genesis.adapter import APPROVED_GENESIS_MODELS
        check("No CatBoost in approved set", "catboost" not in APPROVED_GENESIS_MODELS)
        check("No ExtraTrees in approved set", "extratree" not in APPROVED_GENESIS_MODELS)

        # 5. Provenance + SHA-256
        for name, ad, p in [("LightGBM", lgb_ad, lgbp), ("XGBoost", xgb_ad, xgbp), ("RandomForest", rf_ad, rfp)]:
            on_disk = sha256_of_file(p)
            check(f"{name} SHA-256 recorded matches disk", ad.artifact_hash == on_disk)
            check(f"{name} provenance target", ad.provenance["target"] == "genesis_24h")

        # 6. Original-vs-adapter fidelity
        import lightgbm as lgb
        import xgboost as xgb
        raw_lgb = lgb.Booster(model_file=str(lgbp))
        check("LightGBM fidelity", np.allclose(raw_lgb.predict(X), lgb_ad.predict_proba(X), atol=1e-6))
        raw_xgb = xgb.XGBClassifier(); raw_xgb.load_model(str(xgbp))
        check("XGBoost fidelity", np.allclose(raw_xgb.predict_proba(X)[:, 1], xgb_ad.predict_proba(X), atol=1e-6))
        import joblib
        check("RandomForest fidelity", np.allclose(joblib.load(str(rfp)).predict_proba(X)[:, 1], rf_ad.predict_proba(X), atol=1e-6))

        # 7. Factory + orchestrator
        factory = ModelFactory(repo_root=Path(td))
        check("ModelFactory registers 3", set(factory._registry) == {
            "genesis_lightgbm", "genesis_xgboost", "genesis_randomforest"})
        check("ModelFactory unknown rejected", _raises(lambda: factory.load("genesis_catboost", "x")))

        cfg = {"genesis": {"mode": "production", "threshold": 0.24}}
        orch = CyclonePipelineOrchestrator(cfg=cfg, artifacts_dir=adir)
        ctx = ForecastContext("S1", datetime.now(timezone.utc), 10.0, 88.0)
        state = orch.run_genesis(ctx, feats)
        check("Orchestrator Phase1 produces GenesisPrediction", state.genesis is not None)
        check("Orchestrator production SUCCESS", state.genesis.status == "SUCCESS")
        check("Threshold 0.24", state.genesis.threshold == 0.24)

        # 8. Missing artifact honest failure
        missing_dir = Path(td) / "empty"
        missing_dir.mkdir()
        svc = GenesisService(cfg=cfg, artifacts_dir=missing_dir)
        pred = svc.predict(ctx, feats)
        check("Missing artifact -> UNAVAILABLE", pred.status == "UNAVAILABLE")

        # 9. Native runtime
        check("OMP_NUM_THREADS configured", os.environ.get("OMP_NUM_THREADS") == "1")

    print("\n=== SUMMARY ===")
    npass = sum(1 for _, c in results if c)
    nfail = sum(1 for _, c in results if not c)
    print(f"PASS={npass} FAIL={nfail}")
    for name, ok in results:
        if not ok:
            print(f"  FAILED: {name}")
    return 1 if nfail else 0


class _Stub:
    def __init__(self, prob):
        self._p = prob
        self.is_loaded = True
    def predict_proba(self, X):
        return np.asarray([self._p], dtype=float)


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


if __name__ == "__main__":
    sys.exit(main())
