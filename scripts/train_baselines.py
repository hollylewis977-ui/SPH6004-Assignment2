"""train_baselines.py — 经典生存分析基线模型（Cox PH / XGBoost / RSF）

在与深度模型相同的 5 折划分和动态样本上训练，仅使用 107 维静态特征。
产出逐折 metrics JSON 到 artifacts/results/<experiment>/ 用于对比分析。

Usage:
    python scripts/train_baselines.py                # 所有模型, 全部折
    python scripts/train_baselines.py --model cox    # 仅 Cox PH
    python scripts/train_baselines.py --model xgb    # 仅 XGBoost
    python scripts/train_baselines.py --folds 0 1    # 指定折
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.options import (
    DEFAULT_NUM_BINS,
    DEFAULT_NUM_FOLDS,
    DYNAMIC_DIR,
    RESULTS_DIR,
    static_feature_path,
)
from utils.data_utils import discretize_residual_los
from utils.evaluate import evaluate_survival_predictions
from utils.util import ensure_dir, read_json, resolve_dynamic_sample_path, set_seed, write_json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fold_data(fold_idx: int, split_name: str, static_bundle: dict):
    """Build X matrix and structured y array for one fold split."""
    dynamic_path = resolve_dynamic_sample_path(DYNAMIC_DIR, fold_idx, split_name)
    df = pd.read_csv(dynamic_path)

    static_data = static_bundle["static_data"]
    feature_names = static_bundle["feature_names"]

    # Build feature matrix — one row per dynamic sample
    X_rows = []
    valid_mask = []
    for _, row in df.iterrows():
        stay_id = int(row["stay_id"])
        if stay_id in static_data:
            X_rows.append(static_data[stay_id])
            valid_mask.append(True)
        else:
            X_rows.append(np.zeros(len(feature_names), dtype=np.float32))
            valid_mask.append(False)

    X = np.vstack(X_rows)
    valid_mask = np.array(valid_mask)

    # Also add landmark_h_norm as feature — it's an important predictor
    landmark_feat = df["landmark_h_norm"].to_numpy(dtype=np.float32).reshape(-1, 1)
    X = np.hstack([X, landmark_feat])

    # scikit-survival needs structured array: (event_bool, time_float)
    event = df["event"].to_numpy(dtype=bool)       # event=1 means alive discharge (uncensored)
    duration = df["residual_los_hours"].to_numpy(dtype=np.float64)

    # Clamp tiny durations for numerical stability
    duration = np.clip(duration, 1e-4, None)

    return X, event, duration, df, valid_mask


def _build_structured_y(event: np.ndarray, duration: np.ndarray) -> np.ndarray:
    """Create the structured array that scikit-survival expects."""
    from sksurv.util import Surv
    return Surv.from_arrays(event, duration)


def _predict_survival_matrix(model, X_test: np.ndarray, times: np.ndarray, model_type: str):
    """Predict S(t) for each sample at each time in `times`."""
    if model_type == "xgb":
        return _xgb_predict_survival_matrix(model, X_test, times)

    surv_funcs = model.predict_survival_function(X_test)
    n_samples = len(surv_funcs)
    surv_matrix = np.zeros((n_samples, len(times)), dtype=np.float64)
    for i, sf in enumerate(surv_funcs):
        # Clip query times to the valid domain of this step function
        t_lo, t_hi = sf.domain
        clipped = np.clip(times, t_lo, t_hi)
        surv_matrix[i, :] = sf(clipped)
    return surv_matrix


# ---------------------------------------------------------------------------
# XGBoost survival (Cox PH objective + Breslow baseline hazard)
# ---------------------------------------------------------------------------

def train_xgb_survival(X_train, y_train, X_val, y_val, duration_train, event_train):
    """Train XGBoost with survival:cox objective on the FULL training data.

    Returns a dict with the trained model and baseline hazard info needed to
    reconstruct S(t) = exp(-H0(t) * exp(risk_score)).
    """
    import xgboost as xgb
    from sksurv.metrics import concordance_index_censored

    # XGBoost survival:cox label convention: positive for event, negative for censored
    # event=1 (alive discharge, uncensored) -> +duration
    # event=0 (ICU death, censored)        -> -duration
    labels_train = np.where(event_train, duration_train, -duration_train).astype(np.float32)

    dtrain = xgb.DMatrix(X_train, label=labels_train)
    # Validation DMatrix (for early stopping via Harrell's C-index on training set partial likelihood)
    duration_val = np.array([t for (_, t) in y_val], dtype=np.float64)
    event_val = np.array([e for (e, _) in y_val], dtype=bool)
    labels_val = np.where(event_val, duration_val, -duration_val).astype(np.float32)
    dval = xgb.DMatrix(X_val, label=labels_val)

    best_score = -np.inf
    best_model = None
    best_cfg = None

    configs = [
        {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8, "num_boost_round": 500},
        {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8, "num_boost_round": 500},
    ]

    for cfg in configs:
        num_boost_round = cfg.pop("num_boost_round")
        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "tree_method": "hist",
            "seed": 42,
            "nthread": 4,
            **cfg,
        }
        label = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"    XGB [{label}] ... ", end="", flush=True)

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        # Evaluate via Harrell's C-index on validation set
        risk_val = booster.predict(dval)
        try:
            c_val, _, _, _, _ = concordance_index_censored(event_val, duration_val, risk_val)
        except Exception:
            c_val = float("nan")
        print(f"best_iter={booster.best_iteration} val C-index={c_val:.4f}")

        if c_val > best_score:
            best_score = c_val
            best_model = booster
            best_cfg = {**cfg, "best_iteration": booster.best_iteration}

    print(f"    -> Best XGB: val C-index={best_score:.4f}")

    # Fit Breslow baseline hazard on training set using the BEST model
    risk_train = best_model.predict(dtrain)
    baseline = _fit_breslow_baseline(duration_train, event_train, risk_train)

    return {"booster": best_model, "baseline": baseline}, {"best_params": best_cfg, "val_c_index": best_score}


def _fit_breslow_baseline(durations: np.ndarray, events: np.ndarray, risk_scores: np.ndarray):
    """Fit a Breslow-style cumulative baseline hazard H0(t).

    Centres risk_scores before exp() to prevent numerical overflow for extreme trees.
    Returns a dict with times, cum_hazard, and the centering offset.
    """
    # Centre risk scores to avoid exp overflow; this is mathematically equivalent because
    # S(t|x) = exp(-H0(t) * exp(r(x))) with H0 scaled by exp(-centre) gives the same curves
    # when we add the same centre back to r(x) at prediction time.
    centre = float(np.mean(risk_scores))
    r_centred = risk_scores - centre

    order = np.argsort(durations)
    durations = durations[order]
    events = events[order].astype(bool)
    # Tighter clip for high-dim feature spaces where XGBoost can produce extreme risk scores
    risk_exp = np.exp(np.clip(r_centred[order], -8, 8))

    # Running reverse cumulative sum of exp(risk) = sum over the risk set
    n = len(durations)
    risk_set_sum = np.zeros(n)
    cum = np.sum(risk_exp)
    for i in range(n):
        risk_set_sum[i] = cum
        cum -= risk_exp[i]

    # Aggregate hazard contributions per unique event time (Breslow tie-handling)
    uniq_event_times, inv = np.unique(durations[events], return_inverse=True)
    event_hazards = np.zeros(len(uniq_event_times), dtype=np.float64)
    event_rows = np.where(events)[0]
    for k, row in enumerate(event_rows):
        event_hazards[inv[k]] += 1.0 / max(risk_set_sum[row], 1e-12)

    H0 = np.cumsum(event_hazards)
    return {
        "times": uniq_event_times.astype(np.float64),
        "cum_hazard": H0.astype(np.float64),
        "centre": centre,
    }


def _xgb_predict_survival_matrix(model_dict, X_test: np.ndarray, times: np.ndarray):
    """S(t | x) = exp(-H0(t) * exp(risk(x) - centre)) with clipping for stability."""
    import xgboost as xgb

    booster = model_dict["booster"]
    baseline = model_dict["baseline"]

    dtest = xgb.DMatrix(X_test)
    risk_test = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    r_centred = np.clip(risk_test - baseline["centre"], -8, 8)
    exp_risk = np.exp(r_centred).reshape(-1, 1)

    # Interpolate cumulative hazard at query times via step function (left-continuous)
    bl_times = baseline["times"]
    bl_haz = baseline["cum_hazard"]
    idx = np.searchsorted(bl_times, times, side="right") - 1
    idx = np.clip(idx, -1, len(bl_haz) - 1)
    H0_at_t = np.where(idx >= 0, bl_haz[idx], 0.0)  # shape (len(times),)

    # S(t|x) = exp(-H0(t) * exp(r(x) - centre))
    surv_matrix = np.exp(-H0_at_t[np.newaxis, :] * exp_risk)
    return np.clip(surv_matrix, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Cox PH
# ---------------------------------------------------------------------------

def train_cox_ph(X_train, y_train, X_val, y_val, alpha_candidates=None):
    """Train CoxPH via scikit-survival with regularisation tuning on val set."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    if alpha_candidates is None:
        alpha_candidates = [0.001, 0.01, 0.1, 1.0]

    best_score = -np.inf
    best_model = None
    best_alpha = None

    for alpha in alpha_candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = CoxPHSurvivalAnalysis(alpha=alpha, n_iter=200, tol=1e-7)
                model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            print(f"    CoxPH alpha={alpha:.4f} -> val C-index={score:.4f}")
            if score > best_score:
                best_score = score
                best_model = model
                best_alpha = alpha
        except Exception as e:
            print(f"    CoxPH alpha={alpha:.4f} -> FAILED: {e}")

    print(f"    -> Best CoxPH: alpha={best_alpha}, val C-index={best_score:.4f}")
    return best_model, {"best_alpha": best_alpha, "val_c_index": best_score}


# ---------------------------------------------------------------------------
# Random Survival Forest
# ---------------------------------------------------------------------------

def train_rsf(X_train, y_train, X_val, y_val, subsample_frac: float = 0.15):
    """Train RSF on a subsample. RSF is O(n^2) per split so we subsample to keep it tractable."""
    from sksurv.ensemble import RandomSurvivalForest

    # Subsample training data — RSF has quadratic cost per log-rank split
    n_total = len(X_train)
    n_sub = int(n_total * subsample_frac)
    rng = np.random.RandomState(42)
    idx = rng.choice(n_total, size=n_sub, replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]
    print(f"    RSF training on {n_sub}/{n_total} samples ({subsample_frac:.0%} subsample)")

    cfg = {"n_estimators": 50, "max_depth": 6, "min_samples_leaf": 30, "max_features": "sqrt"}
    label = ", ".join(f"{k}={v}" for k, v in cfg.items())
    print(f"    RSF [{label}] ... ", end="", flush=True)

    model = RandomSurvivalForest(random_state=42, n_jobs=1, **cfg)
    model.fit(X_sub, y_sub)
    score = model.score(X_val, y_val)
    print(f"val C-index={score:.4f}")

    return model, {"best_params": cfg, "val_c_index": score, "train_subsample": subsample_frac}


# ---------------------------------------------------------------------------
# Evaluation (reuse project's pycox-based pipeline)
# ---------------------------------------------------------------------------

def evaluate_baseline(
    model,
    model_type: str,
    X_test: np.ndarray,
    df_test: pd.DataFrame,
    num_bins: int,
):
    """Evaluate a baseline model using the same metrics as the deep model.

    For fair comparison, we compute per-landmark metrics on their OWN time_grid
    (avoids the aggregate-grid resampling bottleneck that can bias continuous
    baselines like Cox/XGB at short landmarks).  The aggregate metrics are also
    computed on a pooled grid.
    """
    from utils.evaluate import evaluate_survival_curves, make_eval_time_grid, _compute_evalsurv_metrics

    duration_test = df_test["residual_los_hours"].to_numpy(dtype=np.float64)
    censor_test = df_test["censored"].to_numpy(dtype=np.float64)
    event_test = df_test["event"].to_numpy(dtype=np.float64)

    # Aggregate time_grid for pooled metrics and expected_time
    t_min = max(duration_test.min(), 1e-4)
    t_max = duration_test.max()
    agg_time_grid = np.linspace(t_min, t_max, 200, dtype=np.float64)

    # Aggregate surv_matrix (for pooled C-index/IBS/NBLL and expected_time)
    agg_surv_matrix = _predict_survival_matrix(model, X_test, agg_time_grid, model_type)
    agg_surv_matrix = np.clip(agg_surv_matrix, 0.0, 1.0)

    metadata = pd.DataFrame({
        "sample_id": df_test["sample_id"].astype(str).values,
        "stay_id": df_test["stay_id"].astype(int).values,
        "landmark_h": df_test["landmark_h"].astype(int).values,
        "landmark_h_norm": df_test["landmark_h_norm"].astype(np.float32).values,
        "residual_los_hours": duration_test.astype(np.float32),
        "censored": censor_test.astype(np.float32),
        "event": event_test.astype(np.float32),
    })

    # Compute pooled metrics (ignore project's per-landmark path; we redo it below with own grids)
    pooled_metrics = _compute_evalsurv_metrics(metadata, agg_surv_matrix, agg_time_grid)
    metrics = {"c_index": pooled_metrics["c_index"], "ibs": pooled_metrics["ibs"], "nbll": pooled_metrics["nbll"]}

    # Per-landmark: each gets its own time_grid and freshly-predicted surv_matrix for max resolution
    landmark_values = sorted(int(v) for v in pd.Series(metadata["landmark_h"]).dropna().unique().tolist())
    macro_acc = {"c_index": [], "ibs": [], "nbll": []}
    for lm in landmark_values:
        mask = (metadata["landmark_h"].to_numpy(dtype=np.int64) == lm)
        if not mask.any():
            continue
        subset_df = metadata.loc[mask].reset_index(drop=True)
        d_sub = subset_df["residual_los_hours"].to_numpy(dtype=np.float64)
        sub_time_grid = make_eval_time_grid(d_sub)
        # Predict fresh on this landmark's own time grid
        sub_surv = _predict_survival_matrix(model, X_test[mask], sub_time_grid, model_type)
        sub_surv = np.clip(sub_surv, 0.0, 1.0)
        sub_metrics = _compute_evalsurv_metrics(subset_df, sub_surv, sub_time_grid)
        for name, val in sub_metrics.items():
            metrics[f"{name}_{lm}h"] = val
            if np.isfinite(val):
                macro_acc[name].append(val)

    for name, vals in macro_acc.items():
        metrics[f"macro_{name}"] = float(np.mean(vals)) if vals else float("nan")

    # Build pred_df with expected_time for KM-style plots
    from utils.evaluate import _expected_time_from_grid
    pred_df = metadata.copy()
    pred_df["expected_time"] = _expected_time_from_grid(agg_surv_matrix, agg_time_grid)
    pred_df["risk_score"] = -pred_df["expected_time"]

    return pred_df, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_fold(fold_idx: int, model_type: str, num_bins: int, output_dir: Path):
    """Train + evaluate one baseline model on one fold."""
    import pickle

    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx} — {model_type.upper()}")
    print(f"{'='*60}")

    # Load preprocessed static features
    sf_path = static_feature_path(fold_idx)
    with open(sf_path, "rb") as f:
        static_bundle = pickle.load(f)
    print(f"  Loaded static features: dim={static_bundle['static_dim']}")

    # Load dynamic samples
    X_train, event_train, dur_train, df_train, _ = _load_fold_data(fold_idx, "train", static_bundle)
    X_val, event_val, dur_val, df_val, _ = _load_fold_data(fold_idx, "val", static_bundle)
    X_test, event_test, dur_test, df_test, _ = _load_fold_data(fold_idx, "test", static_bundle)
    print(f"  Samples — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

    # Build structured y
    y_train = _build_structured_y(event_train, dur_train)
    y_val = _build_structured_y(event_val, dur_val)

    # Train
    t0 = time.time()
    if model_type == "cox":
        model, train_info = train_cox_ph(X_train, y_train, X_val, y_val)
    elif model_type == "rsf":
        model, train_info = train_rsf(X_train, y_train, X_val, y_val)
    elif model_type == "xgb":
        model, train_info = train_xgb_survival(X_train, y_train, X_val, y_val, dur_train, event_train)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    if model is None:
        print(f"  ERROR: All training attempts failed for fold {fold_idx} {model_type}")
        return

    # Evaluate on test set
    pred_df, metrics = evaluate_baseline(model, model_type, X_test, df_test, num_bins)

    # Save results
    fold_dir = ensure_dir(output_dir)
    metrics_payload = {
        "fold": fold_idx,
        "model_name": f"baseline_{model_type}",
        "model_type": model_type,
        "train_info": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in train_info.items()},
        "train_time_seconds": round(train_time, 1),
        "metrics": metrics,
    }
    metrics_path = fold_dir / f"fold_{fold_idx}_metrics.json"
    write_json(metrics_payload, metrics_path)
    print(f"  Saved metrics -> {metrics_path}")

    # Save predictions CSV
    pred_csv_path = fold_dir / f"fold_{fold_idx}_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"  Saved predictions -> {pred_csv_path}")

    # Print key metrics
    print(f"\n  Test metrics:")
    for k in ["macro_c_index", "macro_ibs", "macro_nbll", "c_index_24h", "c_index_48h", "c_index_72h"]:
        if k in metrics:
            print(f"    {k}: {metrics[k]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train classical survival baselines.")
    parser.add_argument("--model", choices=["cox", "rsf", "xgb", "all"], default="cox")
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_NUM_FOLDS)
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    fold_indices = args.folds if args.folds is not None else list(range(args.n_folds))
    models_to_run = ["cox", "rsf", "xgb"] if args.model == "all" else [args.model]

    print("=" * 60)
    print(f"  Baseline Training: {', '.join(models_to_run)}")
    print(f"  Folds: {fold_indices}")
    print("=" * 60)

    for model_type in models_to_run:
        exp_name = f"baseline_{model_type}"
        output_dir = RESULTS_DIR / exp_name

        for fold_idx in fold_indices:
            # Check if already exists
            metrics_path = output_dir / f"fold_{fold_idx}_metrics.json"
            if metrics_path.exists():
                print(f"\n  [SKIP] {exp_name} fold {fold_idx} — already exists ({metrics_path})")
                continue
            run_fold(fold_idx, model_type, args.num_bins, output_dir)

    # Print summary
    print("\n\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for model_type in models_to_run:
        exp_name = f"baseline_{model_type}"
        output_dir = RESULTS_DIR / exp_name
        c_indices = []
        for fold_idx in fold_indices:
            mp = output_dir / f"fold_{fold_idx}_metrics.json"
            if mp.exists():
                data = read_json(mp)
                m = data.get("metrics", data)
                mc = m.get("macro_c_index", float("nan"))
                c_indices.append(mc)
                print(f"  {exp_name} fold {fold_idx}: macro_c_index = {mc:.4f}")

        if c_indices:
            arr = np.array(c_indices)
            print(f"  {exp_name} MEAN: {arr.mean():.4f} ± {arr.std():.4f}")
        print()


if __name__ == "__main__":
    main()
