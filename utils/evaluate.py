"""evaluate.py — 生存分析评估指标：时间依赖 C-index、IBS、NBLL

基于 pycox.EvalSurv 计算 Antolini 时间依赖一致性指数、积分 Brier 分数和
积分负二项对数似然。支持按 landmark 时间分层计算并汇总宏平均值。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.integrate

if not hasattr(scipy.integrate, "simps"):  # pragma: no cover
    scipy.integrate.simps = scipy.integrate.simpson

try:
    from pycox.evaluation import EvalSurv
except ImportError:  # pragma: no cover
    EvalSurv = None


def _empty_metrics() -> dict[str, float]:
    return {"c_index": float("nan"), "ibs": float("nan"), "nbll": float("nan")}


def expected_time_from_survival(survs: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    if survs.size == 0:
        return np.empty((0,), dtype=np.float32)

    edges = np.asarray(bin_edges, dtype=np.float32).copy()
    if not np.isfinite(edges[-1]):
        finite_widths = np.diff(edges[:-1])
        tail_width = float(np.median(finite_widths)) if len(finite_widths) else 1.0
        edges[-1] = edges[-2] + tail_width
    widths = np.diff(edges)
    start_surv = np.concatenate([np.ones((survs.shape[0], 1), dtype=np.float32), survs[:, :-1]], axis=1)
    rmst = (start_surv * widths.reshape(1, -1)).sum(axis=1)
    return rmst.astype(np.float32)


def make_eval_time_grid(times: np.ndarray, num_points: int = 100) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    times = times[np.isfinite(times)]
    if times.size == 0:
        return np.empty((0,), dtype=np.float64)

    t_min = float(times.min())
    t_max = float(times.max())
    if t_max <= t_min:
        return np.array([t_min, t_min + 1e-6], dtype=np.float64)
    return np.linspace(t_min, t_max, num_points, dtype=np.float64)


def _survival_at_times(survs: np.ndarray, bin_edges: np.ndarray, eval_times: np.ndarray) -> np.ndarray:
    values = np.zeros((survs.shape[0], len(eval_times)), dtype=np.float64)
    finite_edges = np.asarray(bin_edges[1:], dtype=np.float64).copy()
    finite_edges[~np.isfinite(finite_edges)] = np.finfo(np.float64).max
    for idx, eval_time in enumerate(eval_times):
        bin_idx = int(np.searchsorted(finite_edges, eval_time, side="right"))
        bin_idx = min(bin_idx, survs.shape[1] - 1)
        values[:, idx] = survs[:, bin_idx]
    return values


def _expected_time_from_grid(surv_values: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    if surv_values.size == 0 or len(time_grid) == 0:
        return np.empty((0,), dtype=np.float32)
    return np.trapezoid(surv_values, x=time_grid, axis=1).astype(np.float32)


def _resample_survival_grid(
    surv_values: np.ndarray,
    source_time_grid: np.ndarray,
    target_time_grid: np.ndarray,
) -> np.ndarray:
    if surv_values.size == 0 or len(source_time_grid) == 0 or len(target_time_grid) == 0:
        return np.empty((surv_values.shape[0], 0), dtype=np.float64)

    source_time_grid = np.asarray(source_time_grid, dtype=np.float64)
    target_time_grid = np.asarray(target_time_grid, dtype=np.float64)
    indices = np.searchsorted(source_time_grid, target_time_grid, side="right") - 1
    indices = np.clip(indices, 0, len(source_time_grid) - 1)
    return surv_values[:, indices]


def _compute_evalsurv_metrics(
    metadata: pd.DataFrame,
    surv_values: np.ndarray,
    time_grid: np.ndarray,
) -> dict[str, float]:
    metrics = _empty_metrics()
    if EvalSurv is None or len(metadata) == 0 or len(time_grid) < 2:
        return metrics

    sample_ids = (
        metadata["sample_id"].astype(str).tolist()
        if "sample_id" in metadata.columns
        else [str(idx) for idx in range(len(metadata))]
    )
    surv_df = pd.DataFrame(
        np.clip(surv_values, 0.0, 1.0).T,
        index=time_grid,
        columns=sample_ids,
    )

    durations = metadata["residual_los_hours"].to_numpy(dtype=np.float64)
    events = metadata["event"].to_numpy(dtype=np.int64)
    try:
        ev = EvalSurv(surv_df, durations, events, censor_surv="km")
        metrics["c_index"] = float(ev.concordance_td("adj_antolini"))
        metrics["ibs"] = float(ev.integrated_brier_score(time_grid))
        metrics["nbll"] = float(ev.integrated_nbll(time_grid))
    except Exception as exc:  # pragma: no cover
        import warnings
        warnings.warn(f"EvalSurv metrics computation failed: {exc}")
    return metrics


def evaluate_survival_curves(
    metadata: pd.DataFrame,
    surv_values: np.ndarray,
    time_grid: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred_df = metadata.copy().reset_index(drop=True)
    surv_values = np.asarray(surv_values, dtype=np.float64)
    time_grid = np.asarray(time_grid, dtype=np.float64)

    if surv_values.ndim != 2:
        raise ValueError("surv_values must be a 2D array with shape [n_samples, n_times].")
    if surv_values.shape[0] != len(pred_df):
        raise ValueError("surv_values row count must match metadata length.")
    if surv_values.shape[1] != len(time_grid):
        raise ValueError("surv_values column count must match time_grid length.")

    pred_df["expected_time"] = _expected_time_from_grid(surv_values, time_grid)
    pred_df["risk_score"] = -pred_df["expected_time"]

    metrics = _compute_evalsurv_metrics(pred_df, surv_values, time_grid)
    if len(pred_df) == 0:
        return pred_df, metrics

    if "landmark_h" in pred_df.columns:
        landmark_values = sorted(
            int(value)
            for value in pd.Series(pred_df["landmark_h"]).dropna().unique().tolist()
        )
        macro_accumulator: dict[str, list[float]] = {"c_index": [], "ibs": [], "nbll": []}
        landmark_array = pred_df["landmark_h"].to_numpy(dtype=np.int64)

        for landmark_h in landmark_values:
            mask = landmark_array == landmark_h
            if not mask.any():
                continue

            subset_df = pred_df.loc[mask].reset_index(drop=True)
            subset_time_grid = make_eval_time_grid(
                subset_df["residual_los_hours"].to_numpy(dtype=np.float64)
            )
            subset_surv_values = _resample_survival_grid(
                surv_values[mask],
                source_time_grid=time_grid,
                target_time_grid=subset_time_grid,
            )
            subset_metrics = _compute_evalsurv_metrics(subset_df, subset_surv_values, subset_time_grid)

            suffix = f"{landmark_h}h"
            for metric_name, metric_value in subset_metrics.items():
                metrics[f"{metric_name}_{suffix}"] = metric_value
                if np.isfinite(metric_value):
                    macro_accumulator[metric_name].append(metric_value)

        for metric_name, values in macro_accumulator.items():
            metrics[f"macro_{metric_name}"] = (
                float(np.mean(values)) if values else float("nan")
            )

    return pred_df, metrics


def evaluate_survival_predictions(
    prediction_bundle: dict[str, Any],
    bin_edges: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:

    metadata = pd.DataFrame(
        {
            "sample_id": prediction_bundle["sample_id"].astype(str),
            "stay_id": prediction_bundle["stay_id"].astype(int),
            "landmark_h": prediction_bundle["landmark_h"].astype(int),
            "landmark_h_norm": prediction_bundle["landmark_h_norm"].astype(np.float32),
            "residual_los_hours": prediction_bundle["residual_los_hours"].astype(np.float32),
            "censored": prediction_bundle["censored"].astype(np.float32),
            "event": prediction_bundle["event"].astype(np.float32),
            "text_mask": prediction_bundle["text_mask"].astype(np.float32),
            "time_bin": prediction_bundle["time_bin"].astype(np.int64),
        }
    )
    survs = prediction_bundle["survs"].astype(np.float64)
    time_grid = make_eval_time_grid(metadata["residual_los_hours"].to_numpy(dtype=np.float64))
    surv_at_times = _survival_at_times(survs, bin_edges, time_grid)

    pred_df, metrics = evaluate_survival_curves(metadata, surv_at_times, time_grid)
    pred_df["expected_time"] = expected_time_from_survival(survs.astype(np.float32), bin_edges)
    pred_df["risk_score"] = -pred_df["expected_time"]
    return pred_df, metrics
