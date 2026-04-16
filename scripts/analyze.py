"""analyze.py — 训练后评估与 MAE 点预测精度分析

两种分析模式:
  --mode plots  : 聚合各折 metrics JSON，生成对比图表（C-index / IBS / NBLL / KM 曲线）
  --mode mae    : 计算全队列 / LOS 1-21d / LOS 1-4d 三个窗口下的 MAE（小时）
  --mode all    : 同时运行两种分析（默认）

用法:
    python scripts/analyze.py --mode plots --result-dir artifacts/results/multimodal_xxx
    python scripts/analyze.py --mode mae
    python scripts/analyze.py --mode all --result-dir artifacts/results/multimodal_xxx
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

# Windows 控制台 UTF-8 兼容
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.options import RESULTS_DIR
from utils.util import ensure_dir

try:
    from lifelines import KaplanMeierFitter
except ImportError:  # pragma: no cover
    KaplanMeierFitter = None


# ═══════════════════════════════════════════════════════════════════════
# Part 1: 指标聚合 + 可视化 (plots)
# ═══════════════════════════════════════════════════════════════════════

METRIC_LABELS = {"c_index": "C-index", "ibs": "IBS", "nbll": "NBLL"}


def _infer_model_name(path: Path, result_dir: Path, payload: dict[str, Any] | None = None) -> str:
    if payload is not None and "model_name" in payload:
        return str(payload["model_name"])
    rel_parts = path.relative_to(result_dir).parts
    return rel_parts[0] if len(rel_parts) > 1 else result_dir.name


def _collect_metrics(result_dir: Path) -> pd.DataFrame:
    """收集 result_dir 下所有折的 metrics JSON 并合并为 DataFrame。"""
    rows = []
    for mp in sorted(result_dir.glob("**/fold_*_metrics.json")):
        payload = json.loads(mp.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", payload.get("test_metrics"))
        if metrics is None:
            continue
        rows.append({"fold": payload["fold"], "model_name": _infer_model_name(mp, result_dir, payload), **metrics})
    if not rows:
        raise FileNotFoundError(f"未找到 metrics JSON: {result_dir}")
    return pd.DataFrame(rows)


def _collect_predictions(result_dir: Path) -> pd.DataFrame:
    """收集所有折的 predictions CSV 并合并。"""
    frames = []
    for pp in sorted(result_dir.glob("**/fold_*_predictions.csv")):
        frame = pd.read_csv(pp)
        frame["model_name"] = _infer_model_name(pp, result_dir)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"未找到 predictions CSV: {result_dir}")
    return pd.concat(frames, ignore_index=True)


def _plot_model_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """按模型分组的指标柱状图。"""
    metrics = [m for m in METRIC_LABELS if m in metrics_df.columns]
    if not metrics or metrics_df["model_name"].nunique() <= 1:
        return
    mean_df = metrics_df.groupby("model_name")[metrics].mean()
    x = np.arange(len(mean_df))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2.0) * width
        ax.bar(x + offset, mean_df[m].to_numpy(dtype=float), width=width, label=METRIC_LABELS[m])
    ax.set_xticks(x, mean_df.index.astype(str).tolist())
    ax.set_title("Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "model_metrics.png", dpi=160)
    plt.close(fig)


def _plot_fold_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """逐折指标折线图。"""
    metrics = [m for m in METRIC_LABELS if m in metrics_df.columns]
    if not metrics:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = metrics_df.sort_values(["model_name", "fold"]).reset_index(drop=True)
    for m in metrics:
        ax.plot(ordered["fold"].to_numpy(dtype=int), ordered[m].to_numpy(dtype=float),
                marker="o", linewidth=2.0, label=METRIC_LABELS[m])
    ax.set_xlabel("Fold")
    ax.set_title("Fold-wise Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fold_metrics.png", dpi=160)
    plt.close(fig)


def _plot_residual_distribution(pred_df: pd.DataFrame, output_dir: Path) -> None:
    """残余住院时间分布直方图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pred_df["residual_los_hours"].to_numpy(dtype=float), bins=50, color="#4C78A8", alpha=0.85)
    ax.set_xlabel("Residual LOS (hours)")
    ax.set_ylabel("Count")
    ax.set_title("Residual LOS Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_los_distribution.png", dpi=160)
    plt.close(fig)


def _plot_risk_group_km(pred_df: pd.DataFrame, output_dir: Path) -> None:
    """按预测风险分组的 Kaplan-Meier 出院曲线。"""
    if KaplanMeierFitter is None or "risk_score" not in pred_df.columns:
        return
    risk = pred_df["risk_score"].to_numpy(dtype=float)
    groups = pd.qcut(risk, q=3, labels=["Low risk", "Mid risk", "High risk"], duplicates="drop")
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    for label in pd.Series(groups).dropna().unique():
        mask = groups == label
        kmf.fit(pred_df.loc[mask, "residual_los_hours"], event_observed=pred_df.loc[mask, "event"], label=str(label))
        kmf.plot_survival_function(ax=ax)
    ax.set_title("Risk-group Kaplan-Meier")
    ax.set_xlabel("Residual LOS (hours)")
    ax.set_ylabel("Survival probability")
    fig.tight_layout()
    fig.savefig(output_dir / "risk_group_km.png", dpi=160)
    plt.close(fig)


def run_plots(result_dir: Path) -> None:
    """聚合指标并生成可视化图表。"""
    print("=" * 60)
    print("  指标聚合与可视化 (Plots)")
    print("=" * 60)
    plot_dir = ensure_dir(result_dir / "plots")
    metrics_df = _collect_metrics(result_dir)
    pred_df = _collect_predictions(result_dir)
    single_model = metrics_df["model_name"].nunique() == 1

    # 保存汇总 CSV
    if metrics_df["model_name"].nunique() > 1:
        mean_df = metrics_df.groupby("model_name").mean(numeric_only=True).reset_index()
        std_df = metrics_df.groupby("model_name").std(numeric_only=True).reset_index()
    else:
        mean_df = metrics_df.mean(numeric_only=True).to_frame().T
        std_df = metrics_df.std(numeric_only=True).to_frame().T
    mean_df.to_csv(result_dir / "summary_metrics_mean.csv", index=False)
    std_df.to_csv(result_dir / "summary_metrics_std.csv", index=False)

    # 绘图
    _plot_model_metrics(metrics_df, plot_dir)
    _plot_fold_metrics(metrics_df, plot_dir)
    if single_model:
        _plot_residual_distribution(pred_df, plot_dir)
        _plot_risk_group_km(pred_df, plot_dir)
    print(f"  图表已保存到 {plot_dir}")


# ═══════════════════════════════════════════════════════════════════════
# Part 2: MAE 点预测精度分析
# ═══════════════════════════════════════════════════════════════════════

# 需要评估的全部模型配置
MODELS = [
    ("Baseline",    "baseline_cox",                                     "Cox PH"),
    ("Baseline",    "baseline_xgb",                                     "XGBoost Cox"),
    ("Deep single", "ablation_static_only",                             "Static"),
    ("Deep single", "ablation_ts_only",                                 "TS"),
    ("Deep single", "ablation_text_only",                               "Text"),
    ("Deep dual",   "ablation_static_ts",                               "Static+TS"),
    ("Deep dual",   "ablation_static_text",                             "Static+Text"),
    ("Deep dual",   "ablation_text_ts",                                 "TS+Text"),
    ("Deep full",   "multimodal_textsmart_weighted_5fold_20260402",     "Static+Text+TS"),
]

HOURS_PER_DAY = 24.0
WINDOW_21D = 21.0 * HOURS_PER_DAY   # 504 h
WINDOW_4D  =  4.0 * HOURS_PER_DAY   #  96 h
N_FOLDS    = 5


def _compute_fold_mae(pred_df: pd.DataFrame) -> dict[str, float] | None:
    """对单折预测结果计算三种 MAE 窗口指标。"""
    unc = pred_df[pred_df["censored"] == 0].copy()
    if unc.empty:
        return None

    pred = unc["expected_time"].to_numpy(dtype=np.float64)
    actual = unc["residual_los_hours"].to_numpy(dtype=np.float64)
    abs_err = np.abs(pred - actual)

    # 重建总住院时间 = landmark 观测时间 + 残余住院时间
    total_los = unc["landmark_h"].to_numpy(dtype=np.float64) + actual
    mask_1_21d = (total_los >= HOURS_PER_DAY) & (total_los <= WINDOW_21D)
    mask_1_4d  = (total_los >= HOURS_PER_DAY) & (total_los <= WINDOW_4D)

    def _s(e: np.ndarray) -> float:
        return float(e.mean()) if e.size else float("nan")

    return {
        "mae_all": _s(abs_err), "mae_1_21d": _s(abs_err[mask_1_21d]), "mae_1_4d": _s(abs_err[mask_1_4d]),
    }


def _aggregate_model_mae(exp_name: str) -> dict[str, float] | None:
    """加载 5 折预测并聚合 MAE（mean +/- std）。"""
    exp_dir = RESULTS_DIR / exp_name
    fold_results = []
    for fold in range(N_FOLDS):
        pp = exp_dir / f"fold_{fold}_predictions.csv"
        if not pp.exists():
            continue
        r = _compute_fold_mae(pd.read_csv(pp))
        if r is not None:
            fold_results.append(r)
    if not fold_results:
        return None
    agg: dict[str, float] = {"n_folds": len(fold_results)}
    for key in ("mae_all", "mae_1_21d", "mae_1_4d"):
        vals = np.array([r[key] for r in fold_results], dtype=np.float64)
        agg[f"{key}_mean"] = float(np.nanmean(vals))
        agg[f"{key}_std"]  = float(np.nanstd(vals))
    return agg


def run_mae() -> None:
    """计算所有模型在三种 LOS 窗口下的 MAE 并打印汇总表。"""
    fmt = lambda m, s: f"{m:.1f} \u00b1 {s:.1f}"

    print("=" * 60)
    print("  MAE 点预测精度分析 (hours)")
    print("=" * 60)
    print(f"{'Model':<22} | {'Full cohort':<18} | {'LOS 1-21d':<18} | {'LOS 1-4d':<18}")
    print("-" * 82)

    rows: list[dict] = []
    cur_cat = None
    for cat, exp, name in MODELS:
        agg = _aggregate_model_mae(exp)
        if agg is None:
            print(f"  [SKIP] {name}")
            continue
        rows.append({"category": cat, "exp": exp, "display": name, **agg})
        if cat != cur_cat:
            cur_cat = cat
            print(f"  [{cur_cat}]")
        print(f"    {name:<20} | {fmt(agg['mae_all_mean'], agg['mae_all_std']):<18} | "
              f"{fmt(agg['mae_1_21d_mean'], agg['mae_1_21d_std']):<18} | "
              f"{fmt(agg['mae_1_4d_mean'], agg['mae_1_4d_std']):<18}")

    out = RESULTS_DIR / "mae_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"  已保存 -> {out}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练后评估: 指标可视化 + MAE 分析")
    parser.add_argument("--mode", choices=["plots", "mae", "all"], default="all",
                        help="运行模式 (默认: all)")
    parser.add_argument("--result-dir", type=Path, default=None,
                        help="plots 模式所需的结果目录（mae 模式自动遍历所有模型）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in ("all", "plots"):
        if args.result_dir is None:
            print("[WARN] --result-dir 未指定，跳过 plots 模式")
        else:
            run_plots(args.result_dir)
    if args.mode in ("all", "mae"):
        run_mae()
    print("\n评估分析完成。")


if __name__ == "__main__":
    main()
