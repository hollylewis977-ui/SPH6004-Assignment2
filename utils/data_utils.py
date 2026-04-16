"""data_utils.py — 数据处理核心：队列构建、交叉验证划分、三模态预处理、时间离散化

包含的主要功能：
  - 队列构建与保存 (build_cohort / save_cohort_artifacts)
  - 被试级分层分组 K 折划分 (make_subject_splits)
  - 静态特征预处理：KNN 填补 + StandardScaler + OneHotEncoder (preprocess_static_fold)
  - 时序特征：DataFrame 构建、按 stay_id 索引的字典、逐折 Z-score (build_ts_bank / fit_ts_scaler)
  - 文本特征：放射科报告清洗、landmark 安全过滤 (clean_radiology_text / build_dynamic_text_table)
  - 残余住院时间离散化为 K 个区间 (discretize_residual_los)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.options import (
    DEFAULT_NUM_BINS,
    LANDMARK_HOURS,
    MAX_LANDMARK,
    RACE_MAP,
    STATIC_BINARY_COLS,
    STATIC_CATEGORICAL_COLS,
    STATIC_CSV,
    STATIC_FULL_MISSING_COLS,
    STATIC_HIGH_MISSING_COLS,
    STATIC_ID_COLS,
    STATIC_LEAK_COLS,
    STATIC_TIME_COLS,
    TEXT_CSV,
    TIMESERIES_CSV,
    TS_DROP_COLS,
    TS_FEATURES,
)
from utils.util import ensure_dir, load_pickle, save_pickle, write_json


def load_static_df(path: Path | str = STATIC_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def load_text_df(path: Path | str = TEXT_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def load_timeseries_df(path: Path | str = TIMESERIES_CSV) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["NULL"])


def build_cohort(static_path: Path | str = STATIC_CSV) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df_static_raw = load_static_df(static_path)
    df_static_raw["intime"] = pd.to_datetime(df_static_raw["intime"])
    df_static_raw = df_static_raw.sort_values(["subject_id", "intime", "stay_id"]).reset_index(drop=True)

    if not df_static_raw["stay_id"].is_unique:
        raise ValueError("stay_id must be unique in the static table.")

    df_static_model = df_static_raw.drop(
        columns=STATIC_LEAK_COLS + STATIC_ID_COLS + STATIC_TIME_COLS + STATIC_FULL_MISSING_COLS + STATIC_HIGH_MISSING_COLS,
        errors="ignore",
    ).copy()
    df_static_model["gender"] = (df_static_model["gender"] == "M").astype(int)
    df_static_model["race"] = df_static_model["race"].map(RACE_MAP).fillna("Other")
    for col in ["first_careunit", "insurance", "language"]:
        df_static_model[col] = df_static_model[col].fillna("UNKNOWN")
    df_static_model["marital_status"] = df_static_model["marital_status"].fillna("UNKNOWN")
    df_static_model = df_static_model.set_index("stay_id")

    summary = {
        "n_stays": int(len(df_static_raw)),
        "n_subjects": int(df_static_raw["subject_id"].nunique()),
        "event_rate_alive_discharge": float((df_static_raw["icu_death_flag"] == 0).mean()),
        "eligible_counts": {
            str(hour): int((df_static_raw["icu_los_hours"] > hour).sum()) for hour in LANDMARK_HOURS
        },
    }
    return df_static_raw, df_static_model, summary


def save_cohort_artifacts(
    df_static_raw: pd.DataFrame,
    df_static_model: pd.DataFrame,
    summary: dict[str, Any],
    raw_path: Path | str,
    model_path: Path | str,
    summary_path: Path | str,
) -> None:
    save_pickle(df_static_raw, raw_path)
    save_pickle(df_static_model, model_path)
    write_json(summary, summary_path)


def load_cohort_artifacts(raw_path: Path | str, model_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_pickle(raw_path), load_pickle(model_path)


def make_subject_splits(
    df_static_raw: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    df_split = df_static_raw[["stay_id", "subject_id", "icu_death_flag"]].copy()
    df_split["event"] = (df_split["icu_death_flag"] == 0).astype(int)

    outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    split_records: list[dict[str, Any]] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(
        outer_cv.split(df_split, y=df_split["event"], groups=df_split["subject_id"])
    ):
        train_val_df = df_split.iloc[train_val_idx].reset_index(drop=True)
        test_df = df_split.iloc[test_idx].reset_index(drop=True)

        inner_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + fold_idx + 123)
        train_idx_inner, val_idx_inner = next(
            inner_cv.split(train_val_df, y=train_val_df["event"], groups=train_val_df["subject_id"])
        )

        train_df = train_val_df.iloc[train_idx_inner].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx_inner].reset_index(drop=True)

        split_records.append(
            {
                "fold": fold_idx,
                "train_stay_ids": train_df["stay_id"].astype(int).tolist(),
                "val_stay_ids": val_df["stay_id"].astype(int).tolist(),
                "test_stay_ids": test_df["stay_id"].astype(int).tolist(),
                "train_subject_ids": train_df["subject_id"].astype(int).tolist(),
                "val_subject_ids": val_df["subject_id"].astype(int).tolist(),
                "test_subject_ids": test_df["subject_id"].astype(int).tolist(),
            }
        )
    return split_records


def save_split_artifacts(split_records: list[dict[str, Any]], output_dir: Path | str) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    summary_rows = []
    for record in split_records:
        fold_idx = record["fold"]
        write_json(record, Path(output_dir) / f"fold_{fold_idx}.json")
        summary_rows.append(
            {
                "fold": fold_idx,
                "train_stays": len(record["train_stay_ids"]),
                "val_stays": len(record["val_stay_ids"]),
                "test_stays": len(record["test_stay_ids"]),
                "train_subjects": len(set(record["train_subject_ids"])),
                "val_subjects": len(set(record["val_subject_ids"])),
                "test_subjects": len(set(record["test_subject_ids"])),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(Path(output_dir) / "split_summary.csv", index=False)
    return summary_df


def _summarize_split_slice(df_part: pd.DataFrame) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "n_stays": int(len(df_part)),
        "n_subjects": int(df_part["subject_id"].nunique()),
        "alive_discharge_rate": float((df_part["icu_death_flag"] == 0).mean()),
        "mean_los_hours": float(df_part["icu_los_hours"].mean()),
        "median_los_hours": float(df_part["icu_los_hours"].median()),
    }
    for hour in LANDMARK_HOURS:
        summary[f"eligible_{hour}h"] = int((df_part["icu_los_hours"] > hour).sum())
    return summary


def save_split_balance_artifacts(
    df_static_raw: pd.DataFrame,
    split_records: list[dict[str, Any]],
    output_dir: Path | str,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    balance_rows = []
    for record in split_records:
        for split_name in ("train", "val", "test"):
            stay_ids = record[f"{split_name}_stay_ids"]
            df_part = df_static_raw[df_static_raw["stay_id"].isin(stay_ids)].copy()
            balance_rows.append(
                {
                    "fold": int(record["fold"]),
                    "split": split_name,
                    **_summarize_split_slice(df_part),
                }
            )

    balance_df = pd.DataFrame(balance_rows)
    balance_df.to_csv(Path(output_dir) / "split_balance_summary.csv", index=False)
    return balance_df


def build_dynamic_sample_table(
    df_static_raw: pd.DataFrame,
    stay_ids: list[int] | np.ndarray,
    landmark_hours: list[int] | None = None,
) -> pd.DataFrame:
    if landmark_hours is None:
        landmark_hours = LANDMARK_HOURS

    cohort = df_static_raw[df_static_raw["stay_id"].isin(stay_ids)].copy()
    rows: list[dict[str, Any]] = []

    for row in cohort.itertuples(index=False):
        for t_obs in landmark_hours:
            if row.icu_los_hours > t_obs:
                rows.append(
                    {
                        "sample_id": f"{int(row.stay_id)}_t{t_obs}",
                        "stay_id": int(row.stay_id),
                        "subject_id": int(row.subject_id),
                        "landmark_h": int(t_obs),
                        "landmark_h_norm": float(t_obs / max(landmark_hours)),
                        "residual_los_hours": float(row.icu_los_hours - t_obs),
                        "censored": int(row.icu_death_flag),
                        "event": int(row.icu_death_flag == 0),
                    }
                )
    return pd.DataFrame(rows)


def preprocess_static_fold(
    df_static_model: pd.DataFrame,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
) -> dict[str, Any]:
    model_df = df_static_model.copy()
    continuous_cols = [col for col in model_df.columns if col not in STATIC_CATEGORICAL_COLS + STATIC_BINARY_COLS]

    for col in STATIC_CATEGORICAL_COLS:
        model_df[col] = model_df[col].fillna("UNKNOWN")

    df_train = model_df.loc[model_df.index.intersection(train_ids)].copy()
    df_val = model_df.loc[model_df.index.intersection(val_ids)].copy()
    df_test = model_df.loc[model_df.index.intersection(test_ids)].copy()

    knn_imputer = KNNImputer(n_neighbors=10, weights="distance", metric="nan_euclidean")
    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    x_train_num = scaler.fit_transform(knn_imputer.fit_transform(df_train[continuous_cols]))
    x_val_num = scaler.transform(knn_imputer.transform(df_val[continuous_cols]))
    x_test_num = scaler.transform(knn_imputer.transform(df_test[continuous_cols]))

    x_train_cat = ohe.fit_transform(df_train[STATIC_CATEGORICAL_COLS])
    x_val_cat = ohe.transform(df_val[STATIC_CATEGORICAL_COLS])
    x_test_cat = ohe.transform(df_test[STATIC_CATEGORICAL_COLS])

    x_train_bin = df_train[STATIC_BINARY_COLS].to_numpy(dtype=np.float32)
    x_val_bin = df_val[STATIC_BINARY_COLS].to_numpy(dtype=np.float32)
    x_test_bin = df_test[STATIC_BINARY_COLS].to_numpy(dtype=np.float32)

    x_train = np.hstack([x_train_num, x_train_bin, x_train_cat]).astype(np.float32)
    x_val = np.hstack([x_val_num, x_val_bin, x_val_cat]).astype(np.float32)
    x_test = np.hstack([x_test_num, x_test_bin, x_test_cat]).astype(np.float32)
    feature_names = (
        continuous_cols
        + STATIC_BINARY_COLS
        + ohe.get_feature_names_out(STATIC_CATEGORICAL_COLS).tolist()
    )

    static_data: dict[int, np.ndarray] = {}
    for arr, ids in ((x_train, df_train.index), (x_val, df_val.index), (x_test, df_test.index)):
        for row_idx, stay_id in enumerate(ids):
            static_data[int(stay_id)] = arr[row_idx]

    return {
        "static_data": static_data,
        "static_dim": int(x_train.shape[1]),
        "continuous_cols": continuous_cols,
        "categorical_cols": STATIC_CATEGORICAL_COLS,
        "binary_cols": STATIC_BINARY_COLS,
        "feature_names": feature_names,
        "train_ids": [int(x) for x in train_ids],
        "val_ids": [int(x) for x in val_ids],
        "test_ids": [int(x) for x in test_ids],
        "scaler": scaler,
        "knn_imputer": knn_imputer,
        "ohe": ohe,
    }


def prepare_timeseries_frame(
    df_static_raw: pd.DataFrame,
    ts_path: Path | str = TIMESERIES_CSV,
    max_landmark: int = MAX_LANDMARK,
) -> pd.DataFrame:
    df_ts = load_timeseries_df(ts_path)
    df_ts = df_ts.drop(columns=TS_DROP_COLS, errors="ignore").copy()
    df_ts = df_ts[df_ts["stay_id"].isin(df_static_raw["stay_id"])].copy()
    df_ts = df_ts.drop_duplicates(subset=["stay_id", "hour_ts"])
    df_ts = df_ts.merge(df_static_raw[["stay_id", "intime"]], on="stay_id", how="left")
    df_ts["hour_ts"] = pd.to_datetime(df_ts["hour_ts"])
    df_ts["intime"] = pd.to_datetime(df_ts["intime"])
    df_ts["rel_hour"] = (df_ts["hour_ts"] - df_ts["intime"]).dt.total_seconds() / 3600.0
    df_ts = df_ts[(df_ts["rel_hour"] > -1) & (df_ts["rel_hour"] < max_landmark)].copy()
    df_ts = df_ts.sort_values(["stay_id", "rel_hour"]).reset_index(drop=True)
    return df_ts


def build_ts_bank(
    df_ts: pd.DataFrame,
    stay_ids: list[int] | None = None,
    ts_features: list[str] | None = None,
    max_len: int = MAX_LANDMARK,
) -> dict[int, dict[str, np.ndarray]]:
    if ts_features is None:
        ts_features = TS_FEATURES
    if stay_ids is None:
        stay_ids = sorted(df_ts["stay_id"].unique().tolist())

    grouped = {
        int(stay_id): group[["rel_hour"] + ts_features].reset_index(drop=True)
        for stay_id, group in df_ts.groupby("stay_id", sort=False)
    }
    feat_dim = len(ts_features)
    ts_bank: dict[int, dict[str, np.ndarray]] = {}

    for stay_id in stay_ids:
        group = grouped.get(int(stay_id))
        if group is None:
            rel_hours = np.empty((0,), dtype=np.float32)
            raw_values = np.empty((0, feat_dim), dtype=np.float32)
            obs_mask = np.empty((0, feat_dim), dtype=np.float32)
        else:
            rel_hours = group["rel_hour"].to_numpy(dtype=np.float32)[:max_len]
            values_df = group[ts_features].iloc[:max_len]
            raw_values = values_df.to_numpy(dtype=np.float32)
            obs_mask = values_df.notna().to_numpy(dtype=np.float32)

        ts_bank[int(stay_id)] = {
            "raw_value_full": raw_values,
            "obs_mask_full": obs_mask,
            "rel_hour_full": rel_hours,
            "available_len": np.array([len(rel_hours)], dtype=np.int32),
        }
    return ts_bank


def fit_ts_scaler(ts_bank: dict[int, dict[str, np.ndarray]], train_stay_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    all_values = []
    all_masks = []
    for stay_id in train_stay_ids:
        bundle = ts_bank[int(stay_id)]
        values = bundle["raw_value_full"]
        masks = bundle["obs_mask_full"]
        if len(values) == 0:
            continue
        all_values.append(values)
        all_masks.append(masks)

    if not all_values:
        feat_dim = next(iter(ts_bank.values()))["raw_value_full"].shape[1]
        return np.zeros((feat_dim,), dtype=np.float32), np.ones((feat_dim,), dtype=np.float32)

    stacked_values = np.vstack(all_values)
    stacked_masks = np.vstack(all_masks)
    masked_values = np.where(stacked_masks == 1, stacked_values, 0.0)
    mask_count = stacked_masks.sum(axis=0) + 1e-8
    ts_mean = masked_values.sum(axis=0) / mask_count
    ts_var = np.where(stacked_masks == 1, (stacked_values - ts_mean) ** 2, 0.0).sum(axis=0) / mask_count
    ts_std = np.sqrt(ts_var) + 1e-8
    return ts_mean.astype(np.float32), ts_std.astype(np.float32)


def save_ts_scaler(path: Path | str, ts_mean: np.ndarray, ts_std: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, ts_mean=ts_mean.astype(np.float32), ts_std=ts_std.astype(np.float32))


def load_ts_scaler(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    bundle = np.load(path)
    return bundle["ts_mean"].astype(np.float32), bundle["ts_std"].astype(np.float32)


REPORT_SEPARATOR_PATTERN = re.compile(r"\n?\s*-{5,}\s*\n?")
REPORT_SUMMARY_PATTERN = re.compile(r"(?im)^\s*(IMPRESSION|CONCLUSION|SUMMARY)\s*:?\s*")


def _collapse_text_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _take_last_words(text: str, max_words: int) -> str:
    words = text.split()
    if max_words <= 0 or not words:
        return ""
    return " ".join(words[-max_words:])


def _head_tail_window(text: str, head_words: int, tail_words: int) -> str:
    words = text.split()
    if len(words) <= head_words + tail_words:
        return " ".join(words)
    return " ".join(words[:head_words] + words[-tail_words:])


def _select_latest_report_block(text: str) -> str:
    blocks = [
        _collapse_text_whitespace(block)
        for block in REPORT_SEPARATOR_PATTERN.split(text)
        if block and block.strip()
    ]
    if not blocks:
        return _collapse_text_whitespace(text)
    return blocks[-1]


def _prioritize_report_summary(text: str, context_words: int = 80) -> str:
    matches = list(REPORT_SUMMARY_PATTERN.finditer(text))
    if not matches:
        return ""

    match = matches[-1]
    summary_text = _collapse_text_whitespace(text[match.start() :])
    context_tail = _take_last_words(text[: match.start()], max_words=context_words)
    if not context_tail:
        return summary_text
    return _collapse_text_whitespace(f"{summary_text}\n{context_tail}")


def clean_radiology_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("___", "[DEID]")
    latest_report = _select_latest_report_block(text)
    if not latest_report:
        return ""

    prioritized_summary = _prioritize_report_summary(latest_report)
    if prioritized_summary:
        return _head_tail_window(prioritized_summary, head_words=192, tail_words=64)

    return _head_tail_window(latest_report, head_words=96, tail_words=160)


def build_dynamic_text_table(
    dynamic_df: pd.DataFrame,
    df_text: pd.DataFrame,
    df_static_raw: pd.DataFrame,
) -> pd.DataFrame:
    text_df = df_text[["stay_id", "radiology_note_time_max", "radiology_note_text"]].copy()
    merged = dynamic_df.merge(df_static_raw[["stay_id", "intime"]], on="stay_id", how="left")
    merged = merged.merge(text_df, on="stay_id", how="left")

    merged["intime"] = pd.to_datetime(merged["intime"])
    merged["radiology_note_time_max"] = pd.to_datetime(merged["radiology_note_time_max"])
    merged["cutoff_time"] = merged["intime"] + pd.to_timedelta(merged["landmark_h"], unit="h")
    merged["text_safe"] = (
        merged["radiology_note_text"].notna()
        & (merged["radiology_note_time_max"] <= merged["cutoff_time"])
    ).astype(int)
    merged.loc[merged["text_safe"] == 0, "radiology_note_text"] = ""
    merged["clean_text"] = merged["radiology_note_text"].apply(clean_radiology_text)
    return merged[["sample_id", "stay_id", "landmark_h", "text_safe", "clean_text"]].copy()


def discretize_residual_los(
    residual_los_hours: np.ndarray,
    censor: np.ndarray,
    num_bins: int = DEFAULT_NUM_BINS,
    bin_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    residual_los_hours = np.asarray(residual_los_hours, dtype=np.float32)
    censor = np.asarray(censor)

    if bin_edges is None:
        uncensored = residual_los_hours[censor == 0]
        if len(uncensored) == 0:
            uncensored = residual_los_hours
        bin_edges = np.quantile(uncensored, np.linspace(0.0, 1.0, num_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = np.inf

    labels = np.digitize(residual_los_hours, bin_edges[1:], right=False)
    labels = np.clip(labels, 0, num_bins - 1).astype(np.int64)
    return labels, np.asarray(bin_edges, dtype=np.float32)
