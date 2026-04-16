"""options.py — 全局配置：路径常量、超参数默认值、特征列名映射

所有脚本和模块通过 from utils.options import ... 获取统一配置。
"""
from __future__ import annotations

from pathlib import Path

# ── 项目路径 ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "Assignment2_mimic dataset" / "data"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
COHORT_DIR = ARTIFACTS_DIR / "cohort"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
DYNAMIC_DIR = ARTIFACTS_DIR / "dynamic_samples"
FEATURES_DIR = ARTIFACTS_DIR / "features"
FEATURE_STATIC_DIR = FEATURES_DIR / "static"
FEATURE_TS_DIR = FEATURES_DIR / "timeseries"
FEATURE_TEXT_DIR = FEATURES_DIR / "text"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"

STATIC_CSV = RAW_DATA_DIR / "MIMIC-IV-static(Group Assignment).csv"
TEXT_CSV = RAW_DATA_DIR / "MIMIC-IV-text(Group Assignment).csv"
TIMESERIES_CSV = RAW_DATA_DIR / "MIMIC-IV-time_series(Group Assignment).csv"

COHORT_RAW_PATH = COHORT_DIR / "static_raw.pkl"
COHORT_MODEL_PATH = COHORT_DIR / "static_model.pkl"
COHORT_SUMMARY_PATH = COHORT_DIR / "summary.json"
TS_BANK_PATH = FEATURE_TS_DIR / "ts_bank.pkl"

LANDMARK_HOURS = [24, 48, 72]
MAX_LANDMARK = 72
DEFAULT_SEED = 42
DEFAULT_NUM_FOLDS = 5
DEFAULT_NUM_BINS = 16
DEFAULT_TEXT_DIM = 128
DEFAULT_TEXT_MAX_LENGTH = 256
DEFAULT_HIDDEN_DIM = 128
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
DEFAULT_BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

STATIC_LEAK_COLS = [
    "outtime",
    "los",
    "deathtime",
    "last_careunit",
    "hospital_expire_flag",
    "radiology_note_count",
    "icu_los_hours",
    "icu_death_flag",
]

STATIC_ID_COLS = ["subject_id", "hadm_id"]
STATIC_TIME_COLS = ["intime"]

STATIC_FULL_MISSING_COLS = [
    "nrbc_min",
    "nrbc_max",
    "inr_min",
    "inr_max",
    "ast_min",
    "ast_max",
    "so2_min",
    "so2_max",
]

STATIC_HIGH_MISSING_COLS = [
    "globulin_min",
    "globulin_max",
    "total_protein_min",
    "total_protein_max",
    "bilirubin_direct_min",
    "bilirubin_direct_max",
    "bilirubin_indirect_min",
    "bilirubin_indirect_max",
    "d_dimer_min",
    "d_dimer_max",
    "ggt_min",
    "ggt_max",
    "ck_mb_min",
    "ck_mb_max",
    "amylase_min",
    "amylase_max",
    "atyps_min",
    "atyps_max",
    "bands_min",
    "bands_max",
    "metas_min",
    "metas_max",
    "imm_granulocytes_min",
    "imm_granulocytes_max",
    "abs_basophils_min",
    "abs_basophils_max",
    "abs_eosinophils_min",
    "abs_eosinophils_max",
    "abs_lymphocytes_min",
    "abs_lymphocytes_max",
    "abs_monocytes_min",
    "abs_monocytes_max",
    "abs_neutrophils_min",
    "abs_neutrophils_max",
]

STATIC_CATEGORICAL_COLS = [
    "first_careunit",
    "race",
    "insurance",
    "marital_status",
    "language",
]
STATIC_BINARY_COLS = ["gender"]

RACE_MAP = {
    "WHITE": "White",
    "WHITE - OTHER EUROPEAN": "White",
    "WHITE - RUSSIAN": "White",
    "WHITE - BRAZILIAN": "White",
    "WHITE - EASTERN EUROPEAN": "White",
    "BLACK/AFRICAN AMERICAN": "Black",
    "BLACK/AFRICAN": "Black",
    "BLACK/CARIBBEAN ISLAND": "Black",
    "BLACK/CAPE VERDEAN": "Black",
    "ASIAN": "Asian",
    "ASIAN - CHINESE": "Asian",
    "ASIAN - SOUTH EAST ASIAN": "Asian",
    "ASIAN - ASIAN INDIAN": "Asian",
    "ASIAN - KOREAN": "Asian",
    "HISPANIC/LATINO - PUERTO RICAN": "Hispanic",
    "HISPANIC/LATINO - DOMINICAN": "Hispanic",
    "HISPANIC/LATINO - MEXICAN": "Hispanic",
    "HISPANIC/LATINO - GUATEMALAN": "Hispanic",
    "HISPANIC/LATINO - CUBAN": "Hispanic",
    "HISPANIC/LATINO - SALVADORAN": "Hispanic",
    "HISPANIC/LATINO - CENTRAL AMERICAN": "Hispanic",
    "HISPANIC/LATINO - COLOMBIAN": "Hispanic",
    "HISPANIC/LATINO - HONDURAN": "Hispanic",
    "SOUTH AMERICAN": "Hispanic",
}

TS_DROP_COLS = ["spo2", "inr"]
TS_FEATURES = [
    "hr",
    "map",
    "rr",
    "temp",
    "gcs",
    "lactate",
    "creatinine",
    "bilirubin",
    "platelets",
    "wbc",
    "sodium",
    "bun",
    "urine_output",
    "vasopressor_dose",
    "fluid_input",
    "ventilation_flag",
    "sofa_resp",
    "sofa_cardio",
    "sofa_renal",
    "sofa_liver",
    "sofa_coag",
    "sofa_cns",
]


def split_path(fold_idx: int) -> Path:
    return SPLITS_DIR / f"fold_{fold_idx}.json"


def dynamic_sample_path(fold_idx: int, split_name: str) -> Path:
    return DYNAMIC_DIR / f"fold_{fold_idx}_{split_name}.csv"


def static_feature_path(fold_idx: int) -> Path:
    return FEATURE_STATIC_DIR / f"static_fold_{fold_idx}.pkl"


def ts_scaler_path(fold_idx: int) -> Path:
    return FEATURE_TS_DIR / f"ts_scaler_fold_{fold_idx}.npz"


def text_meta_path(fold_idx: int, encoder: str) -> Path:
    return FEATURE_TEXT_DIR / f"text_fold_{fold_idx}_{encoder}_meta.csv"
