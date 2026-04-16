"""icu_dataset.py — ICU 动态生存分析数据集

提供 PyTorch Dataset 和 collate 函数：
  - ICUDynamicDataset: 处理变长时序、按需 BERT 分词、landmark 时间感知
  - dynamic_collate_fn: 自定义 batch 拼接（tensor 堆叠 + 列表收集）
  - get_ts_prefix: 按 landmark 截取时序前缀并做 Z-score 归一化
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.options import DEFAULT_TEXT_MAX_LENGTH, MAX_LANDMARK


def _smart_truncate_token_ids(token_ids: list[int], max_tokens: int) -> list[int]:
    if len(token_ids) <= max_tokens:
        return token_ids
    head_tokens = max_tokens // 2
    tail_tokens = max_tokens - head_tokens
    return token_ids[:head_tokens] + token_ids[-tail_tokens:]


def get_ts_prefix(
    ts_bank: dict[int, dict[str, np.ndarray]],
    stay_id: int,
    landmark_h: int,
    ts_mean: np.ndarray,
    ts_std: np.ndarray,
    max_len: int = MAX_LANDMARK,
) -> dict[str, Any]:
    full = ts_bank[int(stay_id)]
    valid_mask = full["rel_hour_full"] < landmark_h
    raw_values = full["raw_value_full"][valid_mask][:max_len]
    obs_mask = full["obs_mask_full"][valid_mask][:max_len]
    prefix_len = int(raw_values.shape[0])

    feat_dim = int(ts_mean.shape[0])
    values = np.zeros((max_len, feat_dim), dtype=np.float32)
    masks = np.zeros((max_len, feat_dim), dtype=np.float32)

    if prefix_len > 0:
        scaled = np.where(obs_mask == 1, (raw_values - ts_mean) / ts_std, 0.0)
        values[:prefix_len] = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        masks[:prefix_len] = obs_mask.astype(np.float32)

    return {"value": values, "obs_mask": masks, "length": prefix_len}


class ICUDynamicDataset(Dataset):
    def __init__(
        self,
        dynamic_df: pd.DataFrame,
        static_data: dict[int, np.ndarray],
        ts_bank: dict[int, dict[str, np.ndarray]] | None,
        ts_mean: np.ndarray | None,
        ts_std: np.ndarray | None,
        text_masks: dict[str, float] | None,
        time_bins_dict: dict[str, int],
        residual_los_dict: dict[str, float],
        censor_dict: dict[str, int],
        include_ts: bool = True,
        include_text: bool = True,
        text_mode: str = "content",
        sample_weight_dict: dict[str, float] | None = None,
        text_lookup: dict[str, str] | None = None,
        tokenizer: Any | None = None,
        text_max_length: int = DEFAULT_TEXT_MAX_LENGTH,
    ) -> None:
        self.dynamic_df = dynamic_df.reset_index(drop=True)
        self.static_data = static_data
        self.ts_bank = ts_bank
        self.ts_mean = ts_mean
        self.ts_std = ts_std
        self.text_masks = text_masks or {}
        self.time_bins_dict = time_bins_dict
        self.residual_los_dict = residual_los_dict
        self.censor_dict = censor_dict
        self.include_ts = include_ts
        self.include_text = include_text
        self.text_mode = text_mode
        self.sample_weight_dict = sample_weight_dict or {}
        self.text_lookup = {str(key): str(value) for key, value in (text_lookup or {}).items()}
        self.tokenizer = tokenizer
        self.text_max_length = int(text_max_length)
        self.text_dim = 0

        self.input_ids = np.zeros((len(self.dynamic_df), 0), dtype=np.uint16)
        self.attention_mask = np.zeros((len(self.dynamic_df), 0), dtype=np.uint8)
        if self.include_text:
            if self.tokenizer is None:
                raise ValueError("tokenizer is required when include_text=True.")
            sample_ids = self.dynamic_df["sample_id"].astype(str).tolist()
            texts = [
                self.text_lookup.get(sample_id, "")
                if self.text_mode != "mask_only" and float(self.text_masks.get(sample_id, 0.0)) > 0
                else ""
                for sample_id in sample_ids
            ]
            encoded = self._tokenize_texts(texts)
            self.input_ids = encoded["input_ids"].cpu().numpy().astype(np.uint16, copy=False)
            self.attention_mask = encoded["attention_mask"].cpu().numpy().astype(np.uint8, copy=False)

    def __len__(self) -> int:
        return len(self.dynamic_df)

    def _tokenize_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise ValueError("tokenizer is required when include_text=True.")

        special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        content_max_length = max(self.text_max_length - special_tokens, 1)
        raw_encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )

        input_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for token_ids in raw_encoded["input_ids"]:
            truncated_ids = _smart_truncate_token_ids(list(token_ids), max_tokens=content_max_length)
            prepared = self.tokenizer.prepare_for_model(
                truncated_ids,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=True,
            )
            input_rows.append(prepared["input_ids"])
            mask_rows.append(prepared["attention_mask"])

        return self.tokenizer.pad(
            {"input_ids": input_rows, "attention_mask": mask_rows},
            padding="max_length",
            max_length=self.text_max_length,
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.dynamic_df.iloc[idx]
        sample_id = str(row["sample_id"])
        stay_id = int(row["stay_id"])
        landmark_h = int(row["landmark_h"])

        static_vec = self.static_data[stay_id]
        if self.include_ts and self.ts_bank is not None and self.ts_mean is not None and self.ts_std is not None:
            ts = get_ts_prefix(self.ts_bank, stay_id, landmark_h, self.ts_mean, self.ts_std)
            ts_input = np.concatenate([ts["value"], ts["obs_mask"]], axis=-1).astype(np.float32)
            seq_len = ts["length"]
        else:
            ts_input = np.zeros((MAX_LANDMARK, 0), dtype=np.float32)
            seq_len = 0

        if self.include_text:
            input_ids = torch.tensor(self.input_ids[idx].astype(np.int64), dtype=torch.long)
            attention_mask = torch.tensor(self.attention_mask[idx].astype(np.int64), dtype=torch.long)
            text_mask = float(self.text_masks.get(sample_id, 0.0))
        else:
            input_ids = torch.zeros((0,), dtype=torch.long)
            attention_mask = torch.zeros((0,), dtype=torch.long)
            text_mask = 0.0

        censored = int(self.censor_dict[sample_id])
        return {
            "sample_id": sample_id,
            "stay_id": stay_id,
            "landmark_h": landmark_h,
            "landmark_h_norm": torch.tensor([float(row["landmark_h_norm"])], dtype=torch.float32),
            "static": torch.tensor(static_vec, dtype=torch.float32),
            "ts": torch.tensor(ts_input, dtype=torch.float32),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_mask": torch.tensor([text_mask], dtype=torch.float32),
            "time_bin": torch.tensor(int(self.time_bins_dict[sample_id]), dtype=torch.long),
            "residual_los_hours": torch.tensor(float(self.residual_los_dict[sample_id]), dtype=torch.float32),
            "censored": torch.tensor(float(censored), dtype=torch.float32),
            "event": torch.tensor(float(1 - censored), dtype=torch.float32),
            "sample_weight": torch.tensor(float(self.sample_weight_dict.get(sample_id, 1.0)), dtype=torch.float32),
        }


def dynamic_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    tensor_keys = {
        "landmark_h_norm",
        "static",
        "ts",
        "seq_len",
        "input_ids",
        "attention_mask",
        "text_mask",
        "time_bin",
        "residual_los_hours",
        "censored",
        "event",
        "sample_weight",
    }
    for key in batch[0]:
        if key in tensor_keys:
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    collated["seq_len"] = collated["seq_len"].view(-1)
    collated["time_bin"] = collated["time_bin"].view(-1)
    collated["residual_los_hours"] = collated["residual_los_hours"].view(-1)
    collated["censored"] = collated["censored"].view(-1)
    collated["event"] = collated["event"].view(-1)
    collated["sample_weight"] = collated["sample_weight"].view(-1)
    return collated
