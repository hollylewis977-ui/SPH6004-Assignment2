# Multi-Modal Deep Learning for ICU Time-to-Discharge Prediction

A landmark dynamic survival analysis framework integrating three clinical data modalities to predict ICU time-to-discharge using the MIMIC-IV (v3.1) database.

## Overview

This project implements a **multi-modal discrete-time survival model** that combines:
- **Static features** (107-dim): demographics, vitals, labs, SOFA scores
- **Time-series** (22 channels): hourly physiological measurements via BiGRU
- **Clinical text**: radiology reports via frozen ClinicalBERT

Predictions are updated at three landmark times (24h, 48h, 72h) using **pairwise cross-attention fusion** across modalities.

## Project Structure

```
├── models/                  # Model definitions
│   ├── network.py           # MultiModalSurvival (Static/TS/Text encoders + CrossAttention)
│   └── engine.py            # Loss function, training loop, prediction
├── datasets/
│   └── icu_dataset.py       # PyTorch Dataset with on-the-fly BERT tokenization
├── utils/
│   ├── options.py           # Paths, hyperparameters, feature constants
│   ├── data_utils.py        # Cohort building, preprocessing, text cleaning
│   ├── evaluate.py          # C-index, IBS, NBLL (pycox EvalSurv)
│   └── util.py              # IO helpers, seed management
├── scripts/
│   ├── prepare.py           # Data preparation (cohort → splits → dynamic samples)
│   ├── preprocess.py        # Feature preprocessing (--modality static|timeseries|text|all)
│   ├── train_baselines.py   # Cox PH / XGBoost baselines
│   ├── train_multimodal.py  # Multi-modal deep model training
│   ├── run_sequential.py    # Fold-by-fold training orchestrator
│   ├── analyze.py           # Evaluation metrics + MAE analysis
│   └── plot_bin_sensitivity.py  # Time-bin granularity sensitivity
├── report/
│   ├── report.tex           # LaTeX report
│   ├── generate_figures.py  # Figure generation from experiment results
│   └── figures/             # Report figures
├── artifacts/               # Generated data (splits, samples, models, results)
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Data preparation
python scripts/prepare.py

# 2. Feature preprocessing
python scripts/preprocess.py

# 3. Train baselines
python scripts/train_baselines.py --model cox
python scripts/train_baselines.py --model xgb

# 4. Train multi-modal model
python scripts/train_multimodal.py --experiment-name multimodal_main --device cuda

# 5. Ablation experiments
python scripts/train_multimodal.py --disable-text --experiment-name ablation_no_text
python scripts/train_multimodal.py --disable-ts --experiment-name ablation_no_ts
python scripts/train_multimodal.py --disable-ts --disable-text --experiment-name ablation_static_only

# 6. Evaluation
python scripts/analyze.py --mode mae
python scripts/analyze.py --mode plots --result-dir artifacts/results/<experiment>
```

## Key Results

| Model | Macro C-index | Macro IBS | Macro NBLL |
|-------|:---:|:---:|:---:|
| Static Only | 0.7208 | 0.0301 | 0.1119 |
| TS Only | 0.7358 | 0.0296 | 0.1141 |
| Text Only | 0.6489 | 0.0337 | 0.1276 |
| **Full (S+T+TS)** | **0.7810** | **0.0276** | **0.1020** |

## Data

This project uses the [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) database (not included in this repository due to access restrictions). Approved users can download the data from PhysioNet.

## Dependencies

- PyTorch >= 2.0
- Transformers (ClinicalBERT: `emilyalsentzer/Bio_ClinicalBERT`)
- scikit-survival, lifelines, pycox
- pandas, numpy, scikit-learn, matplotlib

## Course

SPH6004 — Advanced Statistical Learning in Healthcare (AY2025-26)
