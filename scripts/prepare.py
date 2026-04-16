"""prepare.py — 数据准备流水线（队列构建 → 交叉验证划分 → 动态样本展开）

将原始 MIMIC-IV 静态 CSV 加工为可供模型训练的动态样本表。
三个阶段按顺序执行，中间产物保存到 artifacts/ 目录。

用法:
    python scripts/prepare.py                         # 默认参数，全量运行
    python scripts/prepare.py --n-folds 3 --seed 123  # 自定义折数和种子
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
    COHORT_MODEL_PATH, COHORT_RAW_PATH, COHORT_SUMMARY_PATH,
    DEFAULT_NUM_FOLDS, DEFAULT_SEED, DYNAMIC_DIR, LANDMARK_HOURS,
    SPLITS_DIR, STATIC_CSV,
)
from utils.data_utils import (
    build_cohort, build_dynamic_sample_table, load_cohort_artifacts,
    make_subject_splits, save_cohort_artifacts, save_split_artifacts,
    save_split_balance_artifacts,
)
from utils.util import ensure_dir, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="数据准备: 队列 → 划分 → 动态样本")
    parser.add_argument("--static-path",  type=Path, default=STATIC_CSV)
    parser.add_argument("--splits-dir",   type=Path, default=SPLITS_DIR)
    parser.add_argument("--dynamic-dir",  type=Path, default=DYNAMIC_DIR)
    parser.add_argument("--n-folds",      type=int,  default=DEFAULT_NUM_FOLDS)
    parser.add_argument("--seed",         type=int,  default=DEFAULT_SEED)
    parser.add_argument("--landmarks",    type=int,  nargs="+", default=LANDMARK_HOURS)
    return parser.parse_args()


# ── 第一步：构建队列 ────────────────────────────────────────────────
def step_build_cohort(args: argparse.Namespace) -> pd.DataFrame:
    """从原始静态 CSV 构建队列，去除泄漏列和高缺失列，保存 pickle。"""
    print("=" * 60)
    print("  Step 1/3: 构建队列 (build cohort)")
    print("=" * 60)
    df_raw, df_model, summary = build_cohort(args.static_path)
    save_cohort_artifacts(
        df_raw, df_model, summary,
        COHORT_RAW_PATH, COHORT_MODEL_PATH, COHORT_SUMMARY_PATH,
    )
    print(f"  队列人数: {summary['n_stays']} stays, {summary['n_subjects']} subjects")
    print(f"  存活出院率: {summary['event_rate_alive_discharge']:.1%}")
    return df_raw


# ── 第二步：交叉验证划分 ────────────────────────────────────────────
def step_make_splits(args: argparse.Namespace, df_raw: pd.DataFrame) -> list:
    """按 subject_id 进行分层分组 K 折划分，保证同一患者不跨折。"""
    print("\n" + "=" * 60)
    print("  Step 2/3: 交叉验证划分 (subject-level splits)")
    print("=" * 60)
    split_records = make_subject_splits(df_raw, n_splits=args.n_folds, seed=args.seed)
    summary_df = save_split_artifacts(split_records, args.splits_dir)
    save_split_balance_artifacts(df_raw, split_records, args.splits_dir)
    print(f"  已保存 {args.n_folds} 折划分到 {args.splits_dir}")
    print(summary_df.to_string(index=False))
    return split_records


# ── 第三步：展开动态样本 ────────────────────────────────────────────
def step_build_dynamic_samples(args: argparse.Namespace, df_raw: pd.DataFrame) -> None:
    """对每个 stay，按 landmark={24,48,72}h 展开为多行动态样本。"""
    print("\n" + "=" * 60)
    print("  Step 3/3: 展开动态样本 (dynamic samples)")
    print("=" * 60)
    output_dir = ensure_dir(args.dynamic_dir)
    summary_rows = []

    for fold_idx in range(args.n_folds):
        split = read_json(args.splits_dir / f"fold_{fold_idx}.json")
        for split_name in ("train", "val", "test"):
            dynamic_df = build_dynamic_sample_table(
                df_raw, split[f"{split_name}_stay_ids"],
                landmark_hours=args.landmarks,
            )
            out = output_dir / f"fold_{fold_idx}_{split_name}.csv"
            dynamic_df.to_csv(out, index=False)
            summary_rows.append({"fold": fold_idx, "split": split_name, "n": len(dynamic_df)})
            print(f"  fold {fold_idx} {split_name}: {len(dynamic_df)} samples -> {out}")

    pd.DataFrame(summary_rows).to_csv(output_dir / "dynamic_sample_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    df_raw = step_build_cohort(args)
    step_make_splits(args, df_raw)
    step_build_dynamic_samples(args, df_raw)
    print("\n数据准备完成。")


if __name__ == "__main__":
    main()
