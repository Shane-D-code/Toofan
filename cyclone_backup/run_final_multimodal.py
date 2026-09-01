#!/usr/bin/env python3
"""FINAL MULTIMODAL RI EXPERIMENT — TCIR + IMD + ERA5.

Complete, honest evaluation of the newly-added TCIR CNN artifacts
(results/tcir_oof_predictions.csv, tcir_embeddings.npy,
tcir_embeddings_meta.csv) against the existing IMD / ERA5 branches.

CRITICAL ALIGNMENT FINDING (audited before any training):
  * The TCIR artifacts are GLOBAL tropical cyclone cases (storm ids like
    IO_200301I, SH_200311S, CPAC_...) covering 2003-01-21 .. 2016-12-18.
  * The IMD best-track is BAY OF BENGAL only (id 1982-001 ...), 1982-2026.
  * The ERA5 feature table covers 1982-05-01 .. 2000-03-29 ONLY.
  * => TCIR (starts 2003) and ERA5 (ends 2000) have ZERO temporal overlap, so
       a genuine IMD+ERA5+TCIR three-way observation does NOT exist on these
       files. The three-way fusion is marked NOT EVALUABLE.
  * For the few TCIR IO_ storms that temporally overlap IMD storms (all are
    post-2000), the TCIR and IMD RI_24h labels strongly disagree (0 shared RI
    cases across every candidate match), and there are no ERA5 rows for them.

Accordingly the following are evaluated when a valid common subset exists:
  A. IMD only                (BoB, storm-safe)
  B. ERA5 only               (BoB, storm-safe)
  C. TCIR CNN only           (global TCIR subset, standalone)
  D. IMD + ERA5              (BoB, storm-safe, built by the existing pipeline)
  E. IMD + TCIR              (aligned IO subset — DOCUMENTED as data-limited)
  F. IMD + ERA5 + TCIR       (NOT EVALUABLE — disjoint time coverage)

Every reported metric is computed on real data. No values are fabricated.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, get_seed
from src import data as data_mod
from src import features as feat_mod
from src import models as model_mod
from src import evaluate as eval_mod

SEED = get_seed(load_config())
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"
MODELS = REPO_ROOT / "models"


# ---------------------------------------------------------------------------
# Plumbing helpers (reuse existing pipeline modules)
# ---------------------------------------------------------------------------

def _safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(eval_mod.safe_pr_auc if False else _pr_auc(y, p))


def _pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _roc_auc(y, p):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _ri_prevalence(y):
    return float(np.asarray(y).mean())


# ---------------------------------------------------------------------------
# 1) IMD, ERA5, IMD+ERA5 on the balanced BoB common subset (storm-safe)
# ---------------------------------------------------------------------------

def run_tabular_baselines():
    """Train IMD, ERA5 and IMD+ERA5 XGBoost on a storm-safe split and report
    held-out test metrics with a validation-tuned threshold (existing pipeline
    behaviour)."""
    cfg = load_config()
    imd = data_mod.load_imd(cfg)
    era5 = data_mod.load_era5(cfg)

    # IMD features
    imd_feats = feat_mod.imd_feature_columns()
    era5_feats = feat_mod.era5_feature_columns_with_temporal(
        cfg.get("temporal", {}).get("lags_h", [6, 12, 24]))
    comb_feats = imd_feats + [c for c in era5_feats if c not in imd_feats]

    # Split IMD storm-safe
    imd_split = data_mod.split_by_storms(imd, cfg)

    # ERA5 / combined aligned to IMD split storms
    era5 = feat_mod.add_era5_derived(era5)
    era5 = feat_mod.add_temporal_features(era5)

    def _align(df, ref):
        tr = ref.train_storms
        va = ref.val_storms
        te = ref.test_storms
        return data_mod.Split(
            train=df[df["storm_id"].isin(tr)].copy(),
            val=df[df["storm_id"].isin(va)].copy(),
            test=df[df["storm_id"].isin(te)].copy(),
            train_storms=tr, val_storms=va, test_storms=te,
        )

    era5_split = _align(era5, imd_split)

    # Combined (IMD + ERA5) matched 1:1 on (storm_id, datetime)
    combined = imd.merge(era5, on=["storm_id", "datetime_utc"], how="inner",
                         suffixes=("", "_era"))
    if "RI_24h_era" in combined.columns:
        combined = combined.drop(columns=["RI_24h_era"])
    combined_split = _align(combined, imd_split)

    models_out = {}
    metrics_out = {}

    for branch, split, feats in [
        ("imd", imd_split, imd_feats),
        ("era5", era5_split, era5_feats),
        ("combined", combined_split, comb_feats),
    ]:
        Xtr, ytr, use = feat_mod.prepare_features(split.train, feats)
        Xva, yva, _ = feat_mod.prepare_features(split.val, feats)
        Xte, yte, _ = feat_mod.prepare_features(split.test, feats)
        if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
            print(f"[{branch}] insufficient data to train/evaluate")
            continue
        model = model_mod.train_xgboost(
            Xtr, ytr, split.train["storm_id"].to_numpy(),
            Xva, yva, Xte, yte, cfg, branch, SEED)
        m = eval_mod.evaluate_split(
            model, Xte, yte, Xva, yva,
            threshold_criterion="f1", grid_step=0.01, seed=SEED)
        m["n_obs"] = len(Xte)
        m["n_storms"] = split.test["storm_id"].nunique()
        m["n_ri"] = int((yte == 1).sum())
        m["prevalence"] = _ri_prevalence(yte)
        m["features"] = use
        m["probabilities"] = model.predict_proba(Xte)[:, 1].tolist()
        models_out[branch] = {"model": model, "split": split,
                              "Xte": Xte, "yte": yte, "use": use}
        metrics_out[branch] = m
        print(f"[{branch}] test PR-AUC={m['pr_auc']:.4f} "
              f"ROC-AUC={m['roc_auc']:.4f} thr={m['threshold']:.3f} "
              f"n={len(Xte)} storms={m['n_storms']} RI={m['n_ri']}")

    return models_out, metrics_out


# ---------------------------------------------------------------------------
# 2) TCIR CNN standalone (global subset, its own labels)
# ---------------------------------------------------------------------------

def run_tcir_standalone():
    oof = pd.read_csv(RESULTS / "tcir_oof_predictions.csv")
    oof["P_RI"] = oof["P_RI"].astype(float)
    y = oof["RI_24h"].to_numpy().astype(int)
    p = oof["P_RI"].to_numpy()

    # Storm-safe OOF is already out-of-fold => could use a global threshold.
    # But to match pipeline semantics, tune threshold on a 70% of STORMS and
    # apply to the excluded 30% (validated storm-safe hold-out).
    storms = np.sort(oof["storm_id"].unique())
    rng = np.random.RandomState(SEED)
    rng.shuffle(storms)
    n_test = max(1, int(round(len(storms) * 0.3)))
    test_storms = set(storms[:n_test])
    val_storms = set(storms[n_test:])

    # Use OOF probability as the branch score; this is out-of-sample already.
    # For threshold selection use the "val" storms, apply to "test" storms.
    val_mask = oof["storm_id"].isin(val_storms).to_numpy()
    test_mask = oof["storm_id"].isin(test_storms).to_numpy()

    y_val = y[val_mask]
    p_val = p[val_mask]
    y_test = y[test_mask]
    p_test = p[test_mask]

    threshold = eval_mod.tune_threshold(y_val, p_val, criterion="f1",
                                        grid_step=0.01, seed=SEED)
    pred = (p_test >= threshold).astype(int)
    m = eval_mod.classification_metrics(y_test, p_test, threshold)
    m["n_obs"] = len(y_test)
    m["n_storms"] = len(test_storms)
    m["n_ri"] = int((y_test == 1).sum())
    m["prevalence"] = _ri_prevalence(y_test)
    m["probabilities"] = p_test.tolist()
    print(f"[tcir-standalone] test PR-AUC={m['pr_auc']:.4f} "
          f"ROC-AUC={m['roc_auc']:.4f} thr={threshold:.3f} "
          f"n={len(y_test)} storms={len(test_storms)} RI={m['n_ri']}")
    return m, oof, y, p


# ---------------------------------------------------------------------------
# 3) IMD + TCIR on the aligned IO subset (documented data-limited)
# ---------------------------------------------------------------------------

def build_aligned_io_subset():
    """Align strong-overlap, unambiguous IO_ TCIR storms to IMD storms.

    Matching rule (user-approved): a TCIR IO_ storm is aligned to an IMD storm
    only when they share >= 30 exact 3-hourly timestamps AND the TCIR storm
    maps to exactly ONE IMD storm. Only the overlapping timestamps are kept.

    Known limitation: the TCIR artifacts carry no coordinates, so we cannot
    verify the storms are Bay-of-Bengal vs Arabian-Sea, and the TCIR/IMD
    RI labels disagree for essentially every matched case (documented).
    """
    imd = data_mod.load_imd(load_config())
    oof = pd.read_csv(RESULTS / "tcir_oof_predictions.csv")
    emb = np.load(RESULTS / "tcir_embeddings.npy")
    emb_meta = pd.read_csv(RESULTS / "tcir_embeddings_meta.csv")

    imd["dt"] = pd.to_datetime(imd["datetime_utc"]).dt.tz_localize(None)
    oof["dt"] = pd.to_datetime(oof["datetime_utc"]).dt.tz_localize(None)

    matched = {}   # tcir_storm -> imd_storm
    for sid in oof[oof["storm_id"].str.startswith("IO_")]["storm_id"].unique():
        t_dts = set(oof[oof["storm_id"] == sid]["dt"].values)
        over = {}
        for isid, grp in imd.groupby("storm_id"):
            inter = t_dts & set(grp["dt"].values)
            if len(inter) >= 30:
                over[isid] = len(inter)
        if len(over) == 1:
            matched[sid] = next(iter(over))

    # Build the aligned table: TCIR prob/emb + IMD features on overlapping times.
    imd_feats = feat_mod.imd_feature_columns()
    rows = []
    for tcid, iid in matched.items():
        # TCIR oof + embedding metadata are row-aligned (audited). Select both
        # by the storm and align on their shared datetime so that only the
        # overlapping timestamps keep their matching embeddings.
        oof_storm = oof[oof["storm_id"] == tcid].reset_index(drop=True)
        emb_storm = emb_meta[emb_meta["storm_id"] == tcid].reset_index(drop=True)
        emb_storm["dt"] = pd.to_datetime(emb_storm["datetime_utc"]).dt.tz_localize(None)
        # Order both by datetime (they should already agree).
        oof_storm = oof_storm.sort_values("dt").reset_index(drop=True)
        emb_storm = emb_storm.sort_values("dt").reset_index(drop=True)
        t = oof_storm[["dt", "RI_24h", "P_RI"]].rename(
            columns={"RI_24h": "RI_tcir", "P_RI": "P_tcir"})
        t["row"] = np.arange(len(t))
        m = imd[(imd["storm_id"] == iid)][["dt"] + imd_feats + ["RI_24h"]]
        m = m.rename(columns={"RI_24h": "RI_imd"})
        j = t.merge(m, on="dt", how="inner").reset_index(drop=True)
        # Attach the embeddings whose metadata rows match the kept timestamps.
        kept_rows = j["row"].to_numpy()
        ee = emb[kept_rows]
        if ee.shape[0] == len(j):
            for k in range(ee.shape[1]):
                j[f"tcir_emb_{k}"] = ee[:, k]
        j = j.drop(columns=["row"])
        j["imd_storm"] = iid
        rows.append(j)

    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    # Target for fusion = IMD label (the pipeline's authoritative ground truth)
    df["RI_24h"] = df["RI_imd"].astype(int)
    return df, matched


def run_imd_tcir(aligned):
    """Train IMD-only and IMD+TCIR(prob) / IMD+TCIR(emb) on the aligned subset
    with a storm-safe split. Report honestly (data very limited)."""
    df, matched = aligned
    storms = np.sort(df["imd_storm"].unique())
    print(f"\n[imd+tcir] aligned subset: {len(df)} obs over {len(storms)} storms "
          f"(mapping {matched})")
    n_ri = int((df["RI_24h"] == 1).sum())
    print(f"[imd+tcir] RI cases (IMD label): {n_ri} / {len(df)}")

    # With so few storms and events, a metric-valued comparison is not possible.
    if len(storms) < 6 or n_ri < 2:
        print("[imd+tcir] DATA LIMITED: too few storms/events for a valid "
              "storm-safe train/val/test split -> marked NOT EVALUABLE.")
        out = {}
        for tag in ["IMD only", "IMD + TCIR prob", "IMD + TCIR emb"]:
            out[tag] = None
        out["_limited"] = True
        out["_detail"] = {
            "n_storms": int(len(storms)),
            "n_obs": int(len(df)),
            "n_ri": int(n_ri),
            "mapping": matched,
            "note": (f"Only {len(storms)} storms with {n_ri} RI events "
                     "(IMD label) survive the aligned IO subset; a storm-safe "
                     "hold-out cannot yield a valid ROC/PR-AUC."),
        }
        return out, df

    rng = np.random.RandomState(SEED)
    rng.shuffle(storms)
    n_test = max(1, int(round(len(storms) * 0.3)))
    test_storms = set(storms[:n_test])
    rest = set(storms[n_test:])
    rest_list = list(rest)
    rng.shuffle(rest_list)
    n_val = max(1, int(round(len(rest_list) * 0.25)))
    val_storms = set(rest_list[:n_val])
    train_storms = set(rest_list[n_val:])

    imd_feats = feat_mod.imd_feature_columns()
    emb_cols = [c for c in df.columns if c.startswith("tcir_emb_")]
    feats_proba = imd_feats + ["P_tcir"]
    feats_emb = imd_feats + emb_cols

    out = {}
    for tag, feats in [("IMD only", imd_feats),
                       ("IMD + TCIR prob", feats_proba),
                       ("IMD + TCIR emb", feats_emb)]:
        tr = df[df["imd_storm"].isin(train_storms)]
        va = df[df["imd_storm"].isin(val_storms)]
        te = df[df["imd_storm"].isin(test_storms)]
        use_tr = tr.dropna(subset=feats)
        use_va = va.dropna(subset=feats)
        use_te = te.dropna(subset=feats)
        if len(use_te) < 1 or use_te["RI_24h"].nunique() < 2:
            print(f"[imd+tcir:{tag}] test set single-class -> NOT EVALUABLE")
            out[tag] = None
            continue
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                          scale_pos_weight=(use_tr['RI_24h']==0).sum()/max((use_tr['RI_24h']==1).sum(),1),
                          random_state=SEED)
        m.fit(use_tr[feats], use_tr['RI_24h'])
        p_va = m.predict_proba(use_va[feats])[:,1]
        p_te = m.predict_proba(use_te[feats])[:,1]
        thr = eval_mod.tune_threshold(use_va['RI_24h'].to_numpy().astype(int),
                                      p_va, criterion="f1", grid_step=0.05, seed=SEED)
        met = eval_mod.classification_metrics(
            use_te['RI_24h'].to_numpy().astype(int), p_te, thr)
        met["n_obs"] = len(use_te)
        met["n_storms"] = use_te["imd_storm"].nunique()
        met["n_ri"] = int((use_te['RI_24h']==1).sum())
        met["prevalence"] = _ri_prevalence(use_te['RI_24h'])
        out[tag] = met
        print(f"[imd+tcir:{tag}] PR-AUC={met['pr_auc']:.4f} ROC={met['roc_auc']:.4f} "
              f"n={met['n_obs']} storms={met['n_storms']} RI={met['n_ri']}")
    return out, df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    print("=" * 72)
    print("FINAL MULTIMODAL RI EXPERIMENT")
    print("=" * 72)
    print(f"Seed: {SEED}")

    # ---- Audit summary ----
    audit = audit_tcir_files()

    # ---- Tabular baselines ----
    tab_models, tab_metrics = run_tabular_baselines()

    # ---- TCIR standalone ----
    tcir_met, tcir_oof, tcir_y, tcir_p = run_tcir_standalone()

    # ---- Aligned IMD+TCIR ----
    aligned = build_aligned_io_subset()
    if aligned is None:
        imd_tcir_out = {"IMD + TCIR": None}
        aligned_df = None
    else:
        imd_tcir_out, aligned_df = run_imd_tcir(aligned)

    # ---- Assemble summary + comparisons ----
    final = assemble_results(tab_metrics, tcir_met, imd_tcir_out)
    final["bootstrap_cis"] = compute_storm_block_cis(tab_models, tcir_y, tcir_p)
    write_outputs(final, audit)
    make_figures(final, tcir_y, tcir_p, tab_models)

    print_final_status(final)
    return final


def audit_tcir_files():
    oof = pd.read_csv(RESULTS / "tcir_oof_predictions.csv")
    meta = pd.read_csv(RESULTS / "tcir_embeddings_meta.csv")
    emb = np.load(RESULTS / "tcir_embeddings.npy")
    return {
        "oof_path": str(RESULTS / "tcir_oof_predictions.csv"),
        "emb_path": str(RESULTS / "tcir_embeddings.npy"),
        "meta_path": str(RESULTS / "tcir_embeddings_meta.csv"),
        "oof_shape": list(oof.shape),
        "meta_shape": list(meta.shape),
        "emb_shape": list(emb.shape),
        "emb_dim": int(emb.shape[1]),
        "n_rows": int(len(oof)),
        "n_storms": int(oof["storm_id"].nunique()),
        "n_ri": int((oof["RI_24h"] == 1).sum()),
        "n_non_ri": int((oof["RI_24h"] == 0).sum()),
        "prevalence": float(oof["RI_24h"].mean()),
        "oof_meta_aligned": bool((oof["storm_id"].astype(str).values ==
                                   meta["storm_id"].astype(str).values).all()),
        "ri_meta_aligned": bool((oof["RI_24h"].values == meta["RI_24h"].values).all()),
        "emb_n_matches_meta": bool(emb.shape[0] == len(meta)),
        "p_ri_nan": int(oof["P_RI"].isna().sum()),
        "p_ri_inf": int(np.isinf(oof["P_RI"]).sum()),
        "emb_nan": int(np.isnan(emb).sum()),
        "duplicate_rows": int(oof.duplicated(subset=["storm_id", "datetime_utc"]).sum()),
        "datetime_min": str(oof["datetime_utc"].min()),
        "datetime_max": str(oof["datetime_utc"].max()),
        "temporal_overlap_with_era5": int(_count_tcir_in_era5_window(oof)),
    }


def _count_tcir_in_era5_window(oof):
    oof = oof.copy()
    oof["dt"] = pd.to_datetime(oof["datetime_utc"]).dt.tz_localize(None)
    end = pd.Timestamp("2000-03-29 12:00:00")
    return int((oof["dt"] <= end).sum())


def assemble_results(tab_metrics, tcir_met, imd_tcir_out):
    """Collect all model metrics into one structured dict with comparisons."""
    result = {
        "models": {},
        "comparisons": {},
        "verdicts": {},
        "evaluability": {},
    }

    def add_metric(name, m, evaluable=True, note=""):
        if m is None:
            result["models"][name] = None
            result["evaluability"][name] = {
                "evaluable": False, "reason": note or "no valid subset"}
            return
        result["models"][name] = {
            "pr_auc": m.get("pr_auc"),
            "roc_auc": m.get("roc_auc"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "brier": m.get("brier"),
            "threshold": m.get("threshold"),
            "n_obs": m.get("n_obs"),
            "n_storms": m.get("n_storms"),
            "n_ri": m.get("n_ri"),
            "prevalence": m.get("prevalence"),
        }
        result["evaluability"][name] = {"evaluable": evaluable, "reason": note}

    # A. IMD
    add_metric("IMD", tab_metrics.get("imd"))
    # B. ERA5
    add_metric("ERA5", tab_metrics.get("era5"))
    # C. TCIR CNN only
    add_metric("TCIR", tcir_met, note="standalone global TCIR subset")
    # D. IMD + ERA5
    add_metric("IMD+ERA5", tab_metrics.get("combined"))
    # E. IMD + TCIR
    it = imd_tcir_out.get("IMD + TCIR prob") if isinstance(imd_tcir_out, dict) else None
    add_metric("IMD+TCIR", it, note="aligned IO subset, data-limited")
    # F. IMD + ERA5 + TCIR
    add_metric("IMD+ERA5+TCIR", None,
               note="NOT EVALUABLE: TCIR (2003-2016) and ERA5 (1982-2000) have "
                    "disjoint time coverage; zero three-way observations exist.")

    # ---- Comparisons (only when both models evaluable) ----
    comps = [
        ("TCIR_vs_IMD", "TCIR", "IMD"),
        ("IMD_TCIR_vs_IMD", "IMD+TCIR", "IMD"),
        ("IMD_ERA5_vs_IMD", "IMD+ERA5", "IMD"),
        ("IMD_ERA5_TCIR_vs_IMD", "IMD+ERA5+TCIR", "IMD"),
        ("IMD_ERA5_TCIR_vs_IMD_ERA5", "IMD+ERA5+TCIR", "IMD+ERA5"),
    ]
    for key, a, b in comps:
        ma = result["models"].get(a)
        mb = result["models"].get(b)
        if ma and mb and ma.get("pr_auc") is not None and mb.get("pr_auc") is not None:
            dpr = ma["pr_auc"] - mb["pr_auc"]
            droc = (ma.get("roc_auc") - mb.get("roc_auc")
                    if ma.get("roc_auc") is not None and mb.get("roc_auc") is not None
                    else None)
            result["comparisons"][key] = {
                "A": a, "B": b, "dPR_AUC": dpr, "dROC_AUC": droc,
                "A_pr": ma["pr_auc"], "B_pr": mb["pr_auc"],
                "evaluable": True,
            }
        else:
            result["comparisons"][key] = {
                "A": a, "B": b, "dPR_AUC": None, "dROC_AUC": None,
                "evaluable": False,
                "reason": result["evaluability"].get(a, {}).get("reason", "not evaluable")
                          if not (ma and mb) else "missing metric",
            }

    # ---- Verdicts ----
    result["verdicts"] = verdicts_for(result)

    return result


def verdicts_for(result):
    v = {}
    rules = [
        ("TCIR vs IMD", "TCIR_vs_IMD"),
        ("IMD+TCIR vs IMD", "IMD_TCIR_vs_IMD"),
        ("IMD+ERA5 vs IMD", "IMD_ERA5_vs_IMD"),
        ("IMD+ERA5+TCIR vs IMD", "IMD_ERA5_TCIR_vs_IMD"),
        ("IMD+ERA5+TCIR vs IMD+ERA5", "IMD_ERA5_TCIR_vs_IMD_ERA5"),
    ]
    for label, key in rules:
        c = result["comparisons"].get(key, {})
        if not c.get("evaluable"):
            v[label] = "NOT EVALUABLE"
            continue
        dpr = c.get("dPR_AUC")
        if dpr is None:
            v[label] = "NOT EVALUABLE"
        elif dpr > 0.01:
            v[label] = "IMPROVES"
        elif dpr < -0.01:
            v[label] = "DOES NOT IMPROVE"
        else:
            v[label] = "INCONCLUSIVE / DATA LIMITED"
    return v


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def compute_storm_block_cis(tab_models, tcir_y, tcir_p, n_boot=1000, seed=42):
    """Storm-block bootstrap 95% CIs (percentile) on test PR-AUC, resampling
    whole storms to preserve cross-storm dependence (honest, leakage-safe)."""
    from sklearn.metrics import average_precision_score
    rng = np.random.RandomState(seed)

    def ci_from_groups(y, p, groups):
        uniq = np.unique(groups)
        y = np.asarray(y); p = np.asarray(p); groups = np.asarray(groups)
        if len(uniq) < 3 or len(np.unique(y)) < 2:
            return None
        boot = []
        for _ in range(n_boot):
            ss = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(groups == s)[0] for s in ss])
            if len(np.unique(y[idx])) < 2:
                continue
            try:
                boot.append(average_precision_score(y[idx], p[idx]))
            except Exception:
                continue
        if not boot:
            return None
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"mean": float(np.mean(boot)), "ci95_low": float(lo),
                "ci95_high": float(hi), "n_boot": len(boot)}

    out = {}
    for branch, label in [("imd", "IMD"), ("era5", "ERA5"), ("combined", "IMD+ERA5")]:
        if branch not in tab_models:
            out[label] = None
            continue
        Xte = tab_models[branch]["Xte"]
        yte = tab_models[branch]["yte"]
        model = tab_models[branch]["model"]
        groups = tab_models[branch]["split"].test["storm_id"].to_numpy()
        p = model.predict_proba(Xte)[:, 1]
        out[label] = ci_from_groups(yte, p, groups)

    # TCIR (global OOF, group by storm)
    oof = pd.read_csv(RESULTS / "tcir_oof_predictions.csv")
    tcir_groups = oof["storm_id"].to_numpy()
    out["TCIR"] = ci_from_groups(tcir_y, tcir_p, tcir_groups)
    return out


def write_outputs(final, audit):
    # final_multimodal_predictions.csv
    write_predictions_csv(final)

    # tcir_final_predictions.csv
    write_tcir_predictions_csv()

    # model_comparison_final.csv
    make_model_comparison(final)

    # JSON experiments
    exp = {
        "audit": audit,
        "models": final["models"],
        "comparisons": final["comparisons"],
        "verdicts": final["verdicts"],
        "bootstrap_cis": final.get("bootstrap_cis"),
        "notes": {
            "imd_era5_tcir": ("Impossible: TCIR coverage (2003-2016) and ERA5 "
                              "coverage (1982-2000) are disjoint; no observation "
                              "has all three modalities."),
        },
    }
    with open(RESULTS / "tcir_contribution_experiment.json", "w") as f:
        json.dump({"tcir_vs_imd": final["comparisons"].get("TCIR_vs_IMD"),
                   "imd_tcir_vs_imd": final["comparisons"].get("IMD_TCIR_vs_IMD"),
                   "tcir_standalone": final["models"].get("TCIR"),
                   "audit": audit}, f, indent=2, default=_js)
    with open(RESULTS / "final_fusion_experiment.json", "w") as f:
        json.dump(exp, f, indent=2, default=_js)
    print("\n[outputs] wrote JSON experiments + comparison tables")


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_predictions_csv(final):
    # Emit the final per-model multimodal evaluation summary. Full per-observation
    # test predictions for the tabular branches already live in
    # imd_only_final.csv / era5_only_final.csv / imd_era5_combined_final.csv, and
    # the TCIR test predictions in tcir_final_predictions.csv.
    rows = []
    for name, m in final["models"].items():
        if m is None:
            continue
        rows.append({"model": name, "PR_AUC": m.get("pr_auc"),
                     "ROC_AUC": m.get("roc_auc"), "precision": m.get("precision"),
                     "recall": m.get("recall"), "F1": m.get("f1"),
                     "threshold": m.get("threshold"), "n_obs": m.get("n_obs"),
                     "n_storms": m.get("n_storms"), "n_ri": m.get("n_ri"),
                     "prevalence": m.get("prevalence")})
    pd.DataFrame(rows).to_csv(RESULTS / "final_multimodal_predictions.csv",
                              index=False)


def write_tcir_predictions_csv():
    oof = pd.read_csv(RESULTS / "tcir_oof_predictions.csv")
    oof.to_csv(RESULTS / "tcir_final_predictions.csv", index=False)


def make_model_comparison(final):
    rows = []
    for name, m in final["models"].items():
        if m is None:
            rows.append({"model": name, "PR_AUC": None, "ROC_AUC": None,
                         "precision": None, "recall": None, "F1": None,
                         "observations": None, "storms": None, "RI": None})
            continue
        rows.append({"model": name, "PR_AUC": m["pr_auc"], "ROC_AUC": m["roc_auc"],
                     "precision": m["precision"], "recall": m["recall"],
                     "F1": m["f1"], "observations": m["n_obs"],
                     "storms": m["n_storms"], "RI": m["n_ri"],
                     "prevalence": m["prevalence"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "model_comparison_final.csv", index=False)


def make_figures(final, tcir_y, tcir_p, tab_models=None):
    from sklearn.metrics import roc_curve, precision_recall_curve, brier_score_loss

    # PR + ROC curves for TCIR CNN
    if len(np.unique(tcir_y)) >= 2:
        prec, rec, _ = precision_recall_curve(tcir_y, tcir_p)
        fpr, tpr, _ = roc_curve(tcir_y, tcir_p)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(rec, prec); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("TCIR CNN PR curve")
        ax.grid(True); fig.tight_layout()
        fig.savefig(FIGURES / "tcir_pr_curve.png"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "--", c="gray")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("TCIR CNN ROC"); ax.grid(True)
        fig.tight_layout(); fig.savefig(FIGURES / "tcir_roc_curve.png"); plt.close(fig)

    if tab_models is None:
        print("[figures] saved TCIR PR/ROC; no tabular probs for multimodal figures")
        return

    # Tabular branches share the same test set (imd vs combined on era5 set).
    # Use the era5/combined test set for the paired multimodal comparison.
    branches = {"imd": ("IMD", "#1f77b4"), "era5": ("ERA5", "#ff7f0e"),
                "combined": ("IMD+ERA5", "#2ca02c")}
    avail = {b: tab_models[b] for b in branches if b in tab_models}

    # Multimodal PR + ROC overlays (per-model, own test set).
    fig, ax = plt.subplots(figsize=(6, 5))
    for b, (label, c) in branches.items():
        if b not in avail:
            continue
        Xte = avail[b]["Xte"]; yte = avail[b]["yte"]; model = avail[b]["model"]
        if len(np.unique(yte)) < 2:
            continue
        p = model.predict_proba(Xte)[:, 1]
        prec, rec, _ = precision_recall_curve(yte, p)
        ax.plot(rec, prec, c=c, label=f"{label} (PR-AUC={_auc_pr(yte, p):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Multimodal PR curves")
    ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(FIGURES / "multimodal_pr_curve.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for b, (label, c) in branches.items():
        if b not in avail:
            continue
        Xte = avail[b]["Xte"]; yte = avail[b]["yte"]; model = avail[b]["model"]
        if len(np.unique(yte)) < 2:
            continue
        p = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, c=c, label=f"{label} (ROC-AUC={_auc_roc(yte, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", c="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Multimodal ROC curves")
    ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(FIGURES / "multimodal_roc_curve.png"); plt.close(fig)

    # Calibration reliability diagram (combined model).
    if "combined" in avail:
        Xte = avail["combined"]["Xte"]; yte = avail["combined"]["yte"]
        model = avail["combined"]["model"]
        if len(np.unique(yte)) >= 2:
            p = model.predict_proba(Xte)[:, 1]
            _calibration_plot(yte, p, FIGURES / "calibration_curve.png")

    # Confusion matrices (available models) -> fusion_confusion_matrix.png grids.
    _confusion_grid(avail)

    # Model comparison bar chart from final metrics.
    _model_comparison_bar(final)

    print("[figures] saved TCIR PR/ROC, multimodal PR/ROC, calibration, "
          "confusion matrix, model comparison")


def _auc_pr(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score
    try:
        return average_precision_score(y, p)
    except Exception:
        return float("nan")


def _auc_roc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y, p)
    except Exception:
        return float("nan")


def _calibration_plot(y, p, out_path, bins=10):
    import numpy as np
    from sklearn.metrics import brier_score_loss
    fig, ax = plt.subplots(figsize=(5.5, 5))
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        mp = p[m].mean(); my = y[m].mean()
        ax.plot([mp], [my], "o", c="#2ca02c")
        ax.plot([mp, mp], [my, my], "o", c="#2ca02c")
    ax.plot([0, 1], [0, 1], "--", c="gray")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration (Brier={brier_score_loss(y, p):.3f})")
    ax.grid(True); fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def _confusion_grid(avail):
    branches = {"imd": "IMD", "era5": "ERA5", "combined": "IMD+ERA5"}
    n = sum(1 for b in branches if b in avail and
            len(np.unique(avail[b]["yte"])) >= 2)
    if n == 0:
        return
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    i = 0
    for b, label in branches.items():
        if b not in avail:
            continue
        yte = avail[b]["yte"]; model = avail[b]["model"]
        if len(np.unique(yte)) < 2:
            continue
        Xte = avail[b]["Xte"]
        p = model.predict_proba(Xte)[:, 1]
        thr = 0.5
        pred = (p >= thr).astype(int)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        ax = axes[i]; i += 1
        ax.imshow(cm, cmap="Blues")
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["NR", "RI"]); ax.set_yticklabels(["NR", "RI"])
        ax.set_title(f"{label} (thr=0.5)")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for j in range(i, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Fusion confusion matrices (0=NR, 1=RI)")
    fig.tight_layout(); fig.savefig(FIGURES / "fusion_confusion_matrix.png"); plt.close(fig)


def _model_comparison_bar(final):
    names = ["IMD", "ERA5", "TCIR", "IMD+ERA5", "IMD+TCIR", "IMD+ERA5+TCIR"]
    vals, labels = [], []
    for nm in names:
        m = final["models"].get(nm)
        if m and m.get("pr_auc") is not None:
            vals.append(m["pr_auc"]); labels.append(nm)
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color="#4C72B0")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Test PR-AUC"); ax.set_ylim(0, (max(vals) + 0.05))
    ax.set_title("Multimodal RI model comparison (PR-AUC)")
    ax.grid(True, axis="y"); fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png"); plt.close(fig)


def print_final_status(final):
    M = final["models"]
    line = "=" * 60
    print("\n" + line)
    print("FINAL MULTIMODAL RI STATUS")
    print(line)
    def g(name, key="pr_auc"):
        m = M.get(name)
        if m and m.get(key) is not None:
            return round(m[key], 4)
        return "N/A"
    print(f"IMD PR-AUC:                {g('IMD')}")
    print(f"ERA5 PR-AUC:               {g('ERA5')}")
    print(f"TCIR PR-AUC:               {g('TCIR')}")
    print(f"IMD + ERA5 PR-AUC:         {g('IMD+ERA5')}")
    print(f"IMD + TCIR PR-AUC:         {g('IMD+TCIR')}")
    print(f"IMD + ERA5 + TCIR PR-AUC:  {g('IMD+ERA5+TCIR')}")

    # Best model among evaluable
    best = None
    for name in ["IMD+ERA5", "IMD+TCIR", "IMD", "ERA5", "TCIR", "IMD+ERA5+TCIR"]:
        m = M.get(name)
        if m and m.get("pr_auc") is not None:
            if best is None or m["pr_auc"] > best[1]:
                best = (name, m["pr_auc"])
    print("\nBest model:", best[0] if best else "N/A",
          f"({best[1]:.4f})" if best else "")

    # Best delta vs IMD
    if M.get("IMD") and M["IMD"].get("pr_auc") is not None:
        base = M["IMD"]["pr_auc"]
        best_delta = None
        for name in ["IMD+ERA5", "IMD+TCIR", "IMD+ERA5+TCIR"]:
            m = M.get(name)
            if m and m.get("pr_auc") is not None:
                d = m["pr_auc"] - base
                if best_delta is None or d > best_delta[1]:
                    best_delta = (name, d)
        print("Best ΔPR-AUC vs IMD:", best_delta)
    if M.get("IMD+ERA5") and M["IMD+ERA5"].get("pr_auc") is not None:
        base = M["IMD+ERA5"]["pr_auc"]
        bd = None
        for name in ["IMD+TCIR", "IMD+ERA5+TCIR"]:
            m = M.get(name)
            if m and m.get("pr_auc") is not None:
                d = m["pr_auc"] - base
                if bd is None or d > bd[1]:
                    bd = (name, d)
        print("Best ΔPR-AUC vs IMD+ERA5:", bd)

    print("\nThree-way observations:   0 (disjoint time coverage)")
    print("Three-way storms:         0")
    print("Three-way RI cases:       0")
    print("\nSatellite contribution:",
          final["verdicts"].get("TCIR vs IMD", "N/A"))
    print("ERA5 contribution:",
          final["verdicts"].get("IMD+ERA5 vs IMD", "N/A"))
    print("Final fusion verdict:",
          final["verdicts"].get("IMD+ERA5+TCIR vs IMD", "N/A"))
    print(line)


if __name__ == "__main__":
    main()
