"""engine.py — 训练引擎：损失函数、训练循环、模型预测

包含：
  - nll_survival_loss: 离散时间生存分析负对数似然损失（支持右删失）
  - TrainConfig: 训练超参数配置
  - fit_model: AdamW + ReduceLROnPlateau + 早停训练循环
  - predict_model: 模型推断，输出风险概率和生存曲线
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import contextlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.util import ensure_dir


_nullcontext = contextlib.nullcontext


def nll_survival_loss(
    hazards: torch.Tensor,
    survs: torch.Tensor | None,
    time_bins: torch.Tensor,
    censor: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    alpha: float = 0.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    batch_size = len(time_bins)
    time_bins = time_bins.view(batch_size, 1)
    censor = censor.view(batch_size, 1).float()

    if survs is None:
        survs = torch.cumprod(1.0 - hazards, dim=1)

    padded_surv = torch.cat([torch.ones_like(censor), survs], dim=1)
    uncensored_loss = -(1.0 - censor) * (
        torch.log(torch.gather(padded_surv, 1, time_bins).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, time_bins).clamp(min=eps))
    )
    censored_loss = -censor * torch.log(torch.gather(padded_surv, 1, time_bins + 1).clamp(min=eps))
    neg_log_likelihood = censored_loss + uncensored_loss
    loss = ((1.0 - alpha) * neg_log_likelihood + alpha * uncensored_loss).view(-1)
    if sample_weight is None:
        return loss.mean()

    weights = sample_weight.view(-1).float()
    weights = weights / weights.sum().clamp(min=eps)
    return torch.sum(loss * weights)


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    alpha: float = 0.0
    device: str = "cpu"
    num_workers: int = 0


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    alpha: float,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    context = torch.no_grad() if not is_train else _nullcontext()
    with context:
        for batch in loader:
            batch = _move_batch(batch, device)
            if is_train:
                optimizer.zero_grad()

            hazards, survs = model(
                x_static=batch["static"],
                x_ts=batch["ts"] if batch["ts"].shape[-1] > 0 else None,
                text_mask=batch["text_mask"],
                landmark_h_norm=batch["landmark_h_norm"],
                ts_lengths=batch["seq_len"],
                input_ids=batch["input_ids"] if batch["input_ids"].shape[-1] > 0 else None,
                attention_mask=batch["attention_mask"] if batch["attention_mask"].shape[-1] > 0 else None,
            )
            sample_weight = batch["sample_weight"] if is_train and "sample_weight" in batch else None
            loss = nll_survival_loss(
                hazards,
                survs,
                batch["time_bin"],
                batch["censored"],
                sample_weight=sample_weight,
                alpha=alpha,
            )

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = int(batch["time_bin"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    checkpoint_path: Path | str,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    ensure_dir(checkpoint_path.parent)
    device = torch.device(config.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(model, train_loader, device, optimizer, config.alpha)
        val_loss = _run_epoch(model, val_loader, device, None, config.alpha)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            wait = 0
            improved = "*"
        else:
            wait += 1
            improved = ""

        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"[train] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best_epoch={best_epoch:03d} best_val={best_val_loss:.6f} lr={current_lr:.2e}{improved}",
            flush=True,
        )

        if wait >= config.patience:
            print(f"[train] early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_loss": best_val_loss, "history": history}


def predict_model(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, Any]:
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    sample_ids: list = []
    stay_ids: list = []
    landmark_h: list = []
    landmark_h_norm_list: list[np.ndarray] = []
    residual_los_list: list[np.ndarray] = []
    censored_list: list[np.ndarray] = []
    event_list: list[np.ndarray] = []
    text_mask_list: list[np.ndarray] = []
    time_bin_list: list[np.ndarray] = []
    hazards_list: list[np.ndarray] = []
    survs_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            moved = _move_batch(batch, torch_device)
            hazards, survs = model(
                x_static=moved["static"],
                x_ts=moved["ts"] if moved["ts"].shape[-1] > 0 else None,
                text_mask=moved["text_mask"],
                landmark_h_norm=moved["landmark_h_norm"],
                ts_lengths=moved["seq_len"],
                input_ids=moved["input_ids"] if moved["input_ids"].shape[-1] > 0 else None,
                attention_mask=moved["attention_mask"] if moved["attention_mask"].shape[-1] > 0 else None,
            )

            sample_ids.extend(batch["sample_id"])
            stay_ids.extend(batch["stay_id"])
            landmark_h.extend(batch["landmark_h"])
            landmark_h_norm_list.append(batch["landmark_h_norm"].cpu().numpy().reshape(-1))
            residual_los_list.append(batch["residual_los_hours"].cpu().numpy().reshape(-1))
            censored_list.append(batch["censored"].cpu().numpy().reshape(-1))
            event_list.append(batch["event"].cpu().numpy().reshape(-1))
            text_mask_list.append(batch["text_mask"].cpu().numpy().reshape(-1))
            time_bin_list.append(batch["time_bin"].cpu().numpy().reshape(-1))
            hazards_list.append(hazards.cpu().numpy().astype(np.float32))
            survs_list.append(survs.cpu().numpy().astype(np.float32))

    return {
        "sample_id": np.array(sample_ids, dtype=object),
        "stay_id": np.array(stay_ids, dtype=np.int64),
        "landmark_h": np.array(landmark_h, dtype=np.int64),
        "landmark_h_norm": np.concatenate(landmark_h_norm_list).astype(np.float32) if landmark_h_norm_list else np.empty((0,), dtype=np.float32),
        "residual_los_hours": np.concatenate(residual_los_list).astype(np.float32) if residual_los_list else np.empty((0,), dtype=np.float32),
        "censored": np.concatenate(censored_list).astype(np.float32) if censored_list else np.empty((0,), dtype=np.float32),
        "event": np.concatenate(event_list).astype(np.float32) if event_list else np.empty((0,), dtype=np.float32),
        "text_mask": np.concatenate(text_mask_list).astype(np.float32) if text_mask_list else np.empty((0,), dtype=np.float32),
        "time_bin": np.concatenate(time_bin_list).astype(np.int64) if time_bin_list else np.empty((0,), dtype=np.int64),
        "hazards": np.vstack(hazards_list) if hazards_list else np.empty((0, 0), dtype=np.float32),
        "survs": np.vstack(survs_list) if survs_list else np.empty((0, 0), dtype=np.float32),
    }


def save_prediction_bundle(predictions: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **predictions)


def load_prediction_bundle(path: Path | str) -> dict[str, Any]:
    bundle = np.load(path, allow_pickle=True)
    return {key: bundle[key] for key in bundle.files}
