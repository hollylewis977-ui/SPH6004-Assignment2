from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.options import (
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BERT_MODEL,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LR,
    DEFAULT_NUM_BINS,
    DEFAULT_NUM_FOLDS,
    DEFAULT_PATIENCE,
    DEFAULT_TEXT_MAX_LENGTH,
    DEFAULT_WEIGHT_DECAY,
    DYNAMIC_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    TS_BANK_PATH,
    static_feature_path,
    text_meta_path,
    ts_scaler_path,
)
from utils.data_utils import (
    discretize_residual_los,
    load_ts_scaler,
)
from datasets.icu_dataset import ICUDynamicDataset, dynamic_collate_fn
from utils.evaluate import evaluate_survival_predictions
from models.network import MultiModalSurvival
from models.engine import TrainConfig, fit_model, predict_model, save_prediction_bundle
from utils.util import ensure_dir, load_pickle, resolve_dynamic_sample_path, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multimodal dynamic survival model.")
    parser.add_argument("--dynamic-dir", type=Path, default=DYNAMIC_DIR)
    parser.add_argument("--experiment-name", type=str, default="multimodal_main")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_NUM_FOLDS)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--attention-heads", type=int, default=DEFAULT_ATTENTION_HEADS)
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-static", action="store_true")
    parser.add_argument("--disable-ts", action="store_true")
    parser.add_argument("--disable-text", action="store_true")
    parser.add_argument("--text-mode", choices=["content", "mask_only"], default="content")
    parser.add_argument("--bert-model-name", type=str, default=DEFAULT_BERT_MODEL)
    parser.add_argument("--text-max-length", type=int, default=DEFAULT_TEXT_MAX_LENGTH)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--freeze-bert", dest="freeze_bert", action="store_true")
    parser.add_argument("--unfreeze-bert", dest="freeze_bert", action="store_false")
    parser.add_argument("--train-subject-fraction", type=float, default=1.0)
    parser.add_argument("--exclude-icu-deaths", action="store_true")
    parser.add_argument("--subject-weighting", action="store_true", default=True)
    parser.add_argument("--no-subject-weighting", dest="subject_weighting", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cross-attention", dest="use_cross_attention", action="store_false",
                        help="Replace cross-attention fusion with simple concatenation of modality summaries.")
    parser.set_defaults(freeze_bert=True, use_cross_attention=True)
    return parser.parse_args()


def _run_config_payload(
    args: argparse.Namespace,
    use_static: bool,
    use_ts: bool,
    use_text: bool,
    text_mode: str,
) -> dict[str, object]:
    return {
        "use_static": use_static,
        "use_ts": use_ts,
        "use_text": use_text,
        "text_mode": text_mode,
        "text_encoder": "clinicalbert",
        "bert_model_name": args.bert_model_name,
        "text_max_length": args.text_max_length,
        "freeze_bert": args.freeze_bert,
        "attention_heads": args.attention_heads,
        "hidden_dim": args.hidden_dim,
        "num_bins": args.num_bins,
        "subject_weighting": args.subject_weighting,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "local_files_only": args.local_files_only,
        "train_subject_fraction": args.train_subject_fraction,
        "exclude_icu_deaths": args.exclude_icu_deaths,
        "use_cross_attention": args.use_cross_attention,
    }


def _checkpoint_config_path(model_root: Path, fold_idx: int) -> Path:
    return model_root / f"fold_{fold_idx}_config.json"


def _load_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_result_summary(result_root: Path) -> None:
    summary_rows: list[dict[str, object]] = []
    for metrics_path in sorted(result_root.glob("fold_*_metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics")
        if metrics is None:
            continue
        summary_rows.append({"fold": int(payload["fold"]), **metrics})

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("fold").reset_index(drop=True)
    else:
        summary_df = pd.DataFrame()
    summary_df.to_csv(result_root / "summary.csv", index=False)
    if summary_df.empty:
        pd.DataFrame().to_csv(result_root / "summary_mean.csv", index=False)
    else:
        summary_df.mean(numeric_only=True).to_frame().T.to_csv(result_root / "summary_mean.csv", index=False)


def _make_lookup_tables(
    train_dynamic: pd.DataFrame,
    val_dynamic: pd.DataFrame,
    test_dynamic: pd.DataFrame,
    num_bins: int,
) -> tuple[dict[str, int], dict[str, float], dict[str, int], np.ndarray]:
    _, bin_edges = discretize_residual_los(
        train_dynamic["residual_los_hours"].to_numpy(),
        train_dynamic["censored"].to_numpy(),
        num_bins=num_bins,
    )
    all_dynamic = pd.concat([train_dynamic, val_dynamic, test_dynamic], ignore_index=True)
    all_bins, _ = discretize_residual_los(
        all_dynamic["residual_los_hours"].to_numpy(),
        all_dynamic["censored"].to_numpy(),
        num_bins=num_bins,
        bin_edges=bin_edges,
    )
    time_bins_dict = dict(zip(all_dynamic["sample_id"].astype(str), all_bins.astype(int).tolist()))
    residual_los_dict = dict(
        zip(all_dynamic["sample_id"].astype(str), all_dynamic["residual_los_hours"].astype(float).tolist())
    )
    censor_dict = dict(zip(all_dynamic["sample_id"].astype(str), all_dynamic["censored"].astype(int).tolist()))
    return time_bins_dict, residual_los_dict, censor_dict, bin_edges


def _make_subject_weight_dict(train_dynamic: pd.DataFrame) -> dict[str, float]:
    counts = train_dynamic.groupby("subject_id").size()
    weights = 1.0 / train_dynamic["subject_id"].map(counts).astype(float)
    weights = weights / weights.mean()
    return dict(zip(train_dynamic["sample_id"].astype(str), weights.astype(float)))


def _subsample_train_dynamic_by_subject(
    train_dynamic: pd.DataFrame,
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--train-subject-fraction must be in the interval (0, 1].")

    train_dynamic = train_dynamic.reset_index(drop=True)
    unique_subjects = train_dynamic["subject_id"].drop_duplicates().to_numpy()
    full_subject_count = int(len(unique_subjects))
    full_sample_count = int(len(train_dynamic))

    if fraction >= 1.0 or full_subject_count == 0:
        return train_dynamic, {
            "requested_fraction": float(fraction),
            "used_fraction": 1.0,
            "full_subject_count": full_subject_count,
            "used_subject_count": full_subject_count,
            "full_sample_count": full_sample_count,
            "used_sample_count": full_sample_count,
        }

    rng = np.random.default_rng(seed)
    keep_subject_count = max(1, int(np.ceil(full_subject_count * fraction)))
    keep_subjects = rng.choice(unique_subjects, size=keep_subject_count, replace=False)
    keep_subjects = set(int(subject_id) for subject_id in keep_subjects.tolist())

    sampled_dynamic = train_dynamic[train_dynamic["subject_id"].isin(keep_subjects)].copy().reset_index(drop=True)
    used_subject_count = int(sampled_dynamic["subject_id"].nunique())
    used_sample_count = int(len(sampled_dynamic))

    return sampled_dynamic, {
        "requested_fraction": float(fraction),
        "used_fraction": float(used_subject_count / max(full_subject_count, 1)),
        "full_subject_count": full_subject_count,
        "used_subject_count": used_subject_count,
        "full_sample_count": full_sample_count,
        "used_sample_count": used_sample_count,
    }


def _exclude_icu_death_rows(dynamic_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    dynamic_df = dynamic_df.reset_index(drop=True)
    full_sample_count = int(len(dynamic_df))
    full_subject_count = int(dynamic_df["subject_id"].nunique()) if "subject_id" in dynamic_df.columns else 0
    filtered_df = dynamic_df[dynamic_df["censored"].astype(int) == 0].copy().reset_index(drop=True)
    kept_sample_count = int(len(filtered_df))
    kept_subject_count = int(filtered_df["subject_id"].nunique()) if "subject_id" in filtered_df.columns else 0
    return filtered_df, {
        "full_sample_count": full_sample_count,
        "kept_sample_count": kept_sample_count,
        "dropped_sample_count": int(full_sample_count - kept_sample_count),
        "full_subject_count": full_subject_count,
        "kept_subject_count": kept_subject_count,
        "dropped_subject_count": int(full_subject_count - kept_subject_count),
    }


def _load_clinicalbert_text_metadata(fold_idx: int) -> tuple[dict[str, str], dict[str, float]]:
    path = text_meta_path(fold_idx, "clinicalbert")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/preprocess.py --modality text before training."
        )
    text_meta = pd.read_csv(path)
    text_lookup = dict(
        zip(
            text_meta["sample_id"].astype(str),
            text_meta["clean_text"].fillna("").astype(str),
        )
    )
    text_masks = dict(
        zip(
            text_meta["sample_id"].astype(str),
            text_meta["text_safe"].astype(float),
        )
    )
    return text_lookup, text_masks


def _selected_folds(args: argparse.Namespace) -> list[int]:
    if args.folds is None or len(args.folds) == 0:
        return list(range(args.n_folds))
    folds = sorted(set(int(fold) for fold in args.folds))
    invalid = [fold for fold in folds if fold < 0 or fold >= args.n_folds]
    if invalid:
        raise ValueError(f"Invalid fold indices: {invalid}. Expected values in [0, {args.n_folds - 1}].")
    return folds


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_subject_fraction <= 1.0:
        raise ValueError("--train-subject-fraction must be in the interval (0, 1].")
    use_static = not args.disable_static
    use_ts = not args.disable_ts
    use_text = not args.disable_text
    text_mode = args.text_mode if use_text else "content"
    tokenizer = None

    result_root = ensure_dir(RESULTS_DIR / args.experiment_name)
    model_root = ensure_dir(MODELS_DIR / args.experiment_name)
    folds_to_run = _selected_folds(args)

    ts_bank = load_pickle(TS_BANK_PATH) if use_ts else None
    run_config = _run_config_payload(args, use_static=use_static, use_ts=use_ts, use_text=use_text, text_mode=text_mode)

    if use_text:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for ClinicalBERT text encoding.") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            args.bert_model_name,
            local_files_only=args.local_files_only,
        )

    for fold_idx in folds_to_run:
        set_seed(args.seed + fold_idx)
        print(f"[fold {fold_idx}] loading artifacts", flush=True)
        train_dynamic = pd.read_csv(resolve_dynamic_sample_path(args.dynamic_dir, fold_idx, "train"))
        val_dynamic = pd.read_csv(resolve_dynamic_sample_path(args.dynamic_dir, fold_idx, "val"))
        test_dynamic = pd.read_csv(resolve_dynamic_sample_path(args.dynamic_dir, fold_idx, "test"))
        static_artifact = load_pickle(static_feature_path(fold_idx))
        checkpoint_path = model_root / f"fold_{fold_idx}.pt"
        checkpoint_config_path = _checkpoint_config_path(model_root, fold_idx)
        prediction_npz_path = result_root / f"fold_{fold_idx}_predictions.npz"
        prediction_csv_path = result_root / f"fold_{fold_idx}_predictions.csv"
        metrics_json_path = result_root / f"fold_{fold_idx}_metrics.json"
        bin_edges_path = result_root / f"fold_{fold_idx}_bin_edges.npy"

        if args.resume and metrics_json_path.exists() and prediction_npz_path.exists() and prediction_csv_path.exists():
            print(f"[fold {fold_idx}] skipping because outputs already exist", flush=True)
            continue

        if args.exclude_icu_deaths:
            train_dynamic, train_exclusion_summary = _exclude_icu_death_rows(train_dynamic)
            val_dynamic, val_exclusion_summary = _exclude_icu_death_rows(val_dynamic)
            test_dynamic, test_exclusion_summary = _exclude_icu_death_rows(test_dynamic)
        else:
            train_exclusion_summary = None
            val_exclusion_summary = None
            test_exclusion_summary = None

        train_dynamic, train_subset_summary = _subsample_train_dynamic_by_subject(
            train_dynamic,
            fraction=args.train_subject_fraction,
            seed=args.seed + fold_idx,
        )
        if use_ts:
            ts_mean, ts_std = load_ts_scaler(ts_scaler_path(fold_idx))
        else:
            ts_mean, ts_std = None, None

        if use_text:
            text_lookup, text_masks = _load_clinicalbert_text_metadata(fold_idx)
        else:
            text_masks = {}
            text_lookup = {}

        time_bins_dict, residual_los_dict, censor_dict, bin_edges = _make_lookup_tables(
            train_dynamic,
            val_dynamic,
            test_dynamic,
            num_bins=args.num_bins,
        )
        train_sample_weight_dict = _make_subject_weight_dict(train_dynamic) if args.subject_weighting else None
        print(
            f"[fold {fold_idx}] prepared tables: "
            f"train={len(train_dynamic)} val={len(val_dynamic)} test={len(test_dynamic)} "
            f"(subjects {train_subset_summary['used_subject_count']}/{train_subset_summary['full_subject_count']}, "
            f"requested_fraction={train_subset_summary['requested_fraction']:.2f})",
            flush=True,
        )
        if args.exclude_icu_deaths:
            print(
                f"[fold {fold_idx}] excluded ICU deaths: "
                f"train_drop={train_exclusion_summary['dropped_sample_count']} "
                f"val_drop={val_exclusion_summary['dropped_sample_count']} "
                f"test_drop={test_exclusion_summary['dropped_sample_count']}",
                flush=True,
            )

        train_ds = ICUDynamicDataset(
            train_dynamic,
            static_artifact["static_data"],
            ts_bank,
            ts_mean,
            ts_std,
            text_masks,
            time_bins_dict,
            residual_los_dict,
            censor_dict,
            include_ts=use_ts,
            include_text=use_text,
            text_mode=text_mode,
            sample_weight_dict=train_sample_weight_dict,
            text_lookup=text_lookup,
            tokenizer=tokenizer,
            text_max_length=args.text_max_length,
        )
        val_ds = ICUDynamicDataset(
            val_dynamic,
            static_artifact["static_data"],
            ts_bank,
            ts_mean,
            ts_std,
            text_masks,
            time_bins_dict,
            residual_los_dict,
            censor_dict,
            include_ts=use_ts,
            include_text=use_text,
            text_mode=text_mode,
            text_lookup=text_lookup,
            tokenizer=tokenizer,
            text_max_length=args.text_max_length,
        )
        test_ds = ICUDynamicDataset(
            test_dynamic,
            static_artifact["static_data"],
            ts_bank,
            ts_mean,
            ts_std,
            text_masks,
            time_bins_dict,
            residual_los_dict,
            censor_dict,
            include_ts=use_ts,
            include_text=use_text,
            text_mode=text_mode,
            text_lookup=text_lookup,
            tokenizer=tokenizer,
            text_max_length=args.text_max_length,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=dynamic_collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=dynamic_collate_fn,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=dynamic_collate_fn,
        )

        ts_input_dim = int(train_ds[0]["ts"].shape[1]) if use_ts and len(train_ds) > 0 else 0

        model = MultiModalSurvival(
            static_dim=int(static_artifact["static_dim"]),
            ts_input_dim=ts_input_dim,
            hidden_dim=args.hidden_dim,
            num_time_bins=args.num_bins,
            use_static=use_static,
            use_ts=use_ts,
            use_text=use_text,
            bert_model_name=args.bert_model_name,
            local_files_only=args.local_files_only,
            freeze_bert=args.freeze_bert,
            attention_heads=args.attention_heads,
            use_cross_attention=args.use_cross_attention,
        )
        model.text_mode = text_mode

        if args.resume and checkpoint_path.exists():
            saved_run_config = _load_json_if_exists(checkpoint_config_path)
            if saved_run_config is not None and saved_run_config != run_config:
                raise ValueError(
                    f"Checkpoint config mismatch for fold {fold_idx}. "
                    f"Saved config at {checkpoint_config_path} does not match current arguments."
                )
            state_dict = torch.load(checkpoint_path, map_location=args.device)
            model.load_state_dict(state_dict)
            fit_info = {"resumed_from_checkpoint": True}
            print(f"[fold {fold_idx}] loaded checkpoint from {checkpoint_path}", flush=True)
        else:
            write_json(run_config, checkpoint_config_path)
            print(f"[fold {fold_idx}] starting training", flush=True)
            fit_info = fit_model(
                model,
                train_loader,
                val_loader,
                TrainConfig(
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    device=args.device,
                    num_workers=args.num_workers,
                ),
                checkpoint_path,
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[fold {fold_idx}] starting evaluation", flush=True)
        prediction_bundle = predict_model(model, test_loader, device=args.device)
        prediction_df, metrics = evaluate_survival_predictions(
            prediction_bundle,
            bin_edges=bin_edges,
        )

        np.save(bin_edges_path, bin_edges.astype(np.float32))
        save_prediction_bundle(prediction_bundle, prediction_npz_path)
        prediction_df.to_csv(prediction_csv_path, index=False)
        write_json(
            {
                "fold": fold_idx,
                "fit_info": fit_info,
                "metrics": metrics,
                "config": run_config,
                "train_subset": train_subset_summary,
                "death_exclusion": {
                    "train": train_exclusion_summary,
                    "val": val_exclusion_summary,
                    "test": test_exclusion_summary,
                },
            },
            metrics_json_path,
        )

        print(f"[fold {fold_idx}] saved outputs to {result_root}", flush=True)
        _refresh_result_summary(result_root)

        del (
            train_dynamic,
            val_dynamic,
            test_dynamic,
            static_artifact,
            ts_mean,
            ts_std,
            text_masks,
            text_lookup,
            time_bins_dict,
            residual_los_dict,
            censor_dict,
            train_sample_weight_dict,
            train_ds,
            val_ds,
            test_ds,
            train_loader,
            val_loader,
            test_loader,
            model,
            prediction_bundle,
            prediction_df,
            bin_edges,
            fit_info,
            train_subset_summary,
            train_exclusion_summary,
            val_exclusion_summary,
            test_exclusion_summary,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _refresh_result_summary(result_root)


if __name__ == "__main__":
    main()
