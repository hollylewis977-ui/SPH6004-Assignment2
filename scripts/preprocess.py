"""preprocess.py — 三模态特征预处理（静态 / 时序 / 文本）

对每个交叉验证折，分别处理三种模态的特征：
  - 静态特征: KNN 填补 + 标准化 + 独热编码（仅在训练集上拟合）
  - 时序特征: 构建全局时序字典 + 逐折 Z-score 归一化
  - 文本特征: 放射科报告清洗 + landmark 时间安全过滤

用法:
    python scripts/preprocess.py                        # 全部三模态
    python scripts/preprocess.py --modality static       # 只处理静态
    python scripts/preprocess.py --modality timeseries   # 只处理时序
    python scripts/preprocess.py --modality text         # 只处理文本
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.options import (
    COHORT_MODEL_PATH, COHORT_RAW_PATH, DEFAULT_NUM_FOLDS, DYNAMIC_DIR,
    MAX_LANDMARK, SPLITS_DIR, TEXT_CSV, TIMESERIES_CSV, TS_BANK_PATH,
    static_feature_path, text_meta_path, ts_scaler_path,
)
from utils.data_utils import (
    build_dynamic_text_table, build_ts_bank, fit_ts_scaler,
    load_cohort_artifacts, load_text_df, prepare_timeseries_frame,
    preprocess_static_fold, save_ts_scaler,
)
from utils.util import read_json, resolve_dynamic_sample_path, save_pickle, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三模态特征预处理")
    parser.add_argument("--modality", choices=["static", "timeseries", "text", "all"], default="all",
                        help="要处理的模态 (默认: all)")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_NUM_FOLDS)
    return parser.parse_args()


# ── 静态特征: KNN 填补 + StandardScaler + OneHotEncoder ─────────────
def run_static(n_folds: int) -> None:
    """对每折单独拟合 KNN 填补器、标准化器和编码器（避免数据泄漏）。"""
    print("=" * 60)
    print("  静态特征预处理 (Static)")
    print("=" * 60)
    _, df_model = load_cohort_artifacts(COHORT_RAW_PATH, COHORT_MODEL_PATH)
    rows = []
    for fold in range(n_folds):
        split = read_json(SPLITS_DIR / f"fold_{fold}.json")
        artifact = preprocess_static_fold(
            df_model, split["train_stay_ids"], split["val_stay_ids"], split["test_stay_ids"],
        )
        out = static_feature_path(fold)
        save_pickle(artifact, out)
        rows.append({"fold": fold, "static_dim": artifact["static_dim"]})
        print(f"  fold {fold}: {artifact['static_dim']}维 -> {out}")
    pd.DataFrame(rows).to_csv(
        PROJECT_ROOT / "artifacts" / "features" / "static" / "summary.csv", index=False,
    )


# ── 时序特征: 构建时序字典 + 逐折 Z-score 归一化 ────────────────────
def run_timeseries(n_folds: int) -> None:
    """构建全局时序字典（按 stay_id 索引），然后对每折用训练集统计量做标准化。"""
    print("=" * 60)
    print("  时序特征预处理 (Timeseries)")
    print("=" * 60)
    df_raw, _ = load_cohort_artifacts(COHORT_RAW_PATH, COHORT_MODEL_PATH)

    # 构建全局时序 DataFrame，过滤到 [-1h, 72h) 相对入院时间窗
    df_ts = prepare_timeseries_frame(df_raw, ts_path=TIMESERIES_CSV, max_landmark=MAX_LANDMARK)
    # 转为按 stay_id 索引的字典，供训练时 O(1) 查找
    ts_bank = build_ts_bank(df_ts, stay_ids=df_raw["stay_id"].astype(int).tolist(), max_len=MAX_LANDMARK)
    save_pickle(ts_bank, TS_BANK_PATH)
    print(f"  时序字典: {len(ts_bank)} stays -> {TS_BANK_PATH}")

    rows = []
    for fold in range(n_folds):
        split = read_json(SPLITS_DIR / f"fold_{fold}.json")
        ts_mean, ts_std = fit_ts_scaler(ts_bank, split["train_stay_ids"])
        out = ts_scaler_path(fold)
        save_ts_scaler(out, ts_mean, ts_std)
        rows.append({"fold": fold, "feature_dim": int(ts_mean.shape[0])})
        print(f"  fold {fold}: {ts_mean.shape[0]}维 scaler -> {out}")

    pd.DataFrame(rows).to_csv(
        PROJECT_ROOT / "artifacts" / "features" / "timeseries" / "summary.csv", index=False,
    )
    write_json(
        {"n_rows": int(len(df_ts)), "n_stays": int(df_ts["stay_id"].nunique()), "max_landmark": MAX_LANDMARK},
        PROJECT_ROOT / "artifacts" / "features" / "timeseries" / "meta.json",
    )


# ── 文本特征: 放射科报告清洗 + landmark 安全过滤 ─────────────────────
def run_text(n_folds: int) -> None:
    """清洗放射科报告，仅保留 note_time <= intime + landmark_h 的文本（避免未来信息泄漏）。"""
    print("=" * 60)
    print("  文本特征预处理 (Text)")
    print("=" * 60)
    df_raw, _ = load_cohort_artifacts(COHORT_RAW_PATH, COHORT_MODEL_PATH)
    df_text = load_text_df(TEXT_CSV)
    encoder = "clinicalbert"
    rows = []

    for fold in range(n_folds):
        # 加载该折的 train/val/test 动态样本
        parts = []
        for split_name in ("train", "val", "test"):
            dyn = pd.read_csv(resolve_dynamic_sample_path(DYNAMIC_DIR, fold, split_name))
            text_df = build_dynamic_text_table(dyn, df_text, df_raw)
            parts.append(text_df)

        all_text = pd.concat(parts, ignore_index=True)
        out = text_meta_path(fold, encoder)
        all_text.to_csv(out, index=False)

        n_safe = int((parts[0]["text_safe"] == 1).sum())  # 训练集中有安全文本的样本数
        rows.append({"fold": fold, "n_samples": len(all_text), "safe_train": n_safe})
        print(f"  fold {fold}: {len(all_text)} samples, {n_safe} safe train texts -> {out}")

    pd.DataFrame(rows).to_csv(
        PROJECT_ROOT / "artifacts" / "features" / "text" / f"summary_{encoder}.csv", index=False,
    )


def main() -> None:
    args = parse_args()
    modality = args.modality
    if modality in ("all", "static"):
        run_static(args.n_folds)
    if modality in ("all", "timeseries"):
        run_timeseries(args.n_folds)
    if modality in ("all", "text"):
        run_text(args.n_folds)
    print("\n预处理完成。")


if __name__ == "__main__":
    main()
