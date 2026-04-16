"""Generate all figures for the SPH6004 Assignment 2 report.

Reads results DIRECTLY from artifacts/ -- no hardcoded numbers.
"""

import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

REPORT_DIR = Path(__file__).resolve().parent
PROJECT    = REPORT_DIR.parent
RESULTS    = PROJECT / "artifacts" / "results"
OUT        = REPORT_DIR / "figures"
OUT.mkdir(exist_ok=True)

# ── Model configurations to evaluate ─────────────────────────────────
MODEL_DIRS = {
    "Text Only":      "ablation_text_only",
    "Static Only":    "ablation_static_only",
    "TS Only":        "ablation_ts_only",
    "Static+TS":      "ablation_static_ts",
    "Text+TS":        "ablation_text_ts",
    "Static+Text":    "ablation_static_text",
    "Full (S+T+TS)":  "multimodal_textsmart_weighted_5fold_20260402",
}

# ── Read all metrics from JSON files ─────────────────────────────────
def load_fold_metrics(model_dir_name, metric_key="macro_c_index"):
    """Load per-fold metric values from JSON files."""
    d = RESULTS / model_dir_name
    vals = []
    for fold in range(5):
        fp = d / f"fold_{fold}_metrics.json"
        if not fp.exists():
            print(f"  WARNING: {fp} not found, skipping")
            continue
        with open(fp) as f:
            data = json.load(f)
        # metrics may be nested under "metrics" key or at top level
        m = data.get("metrics", data)
        vals.append(m[metric_key])
    return vals


def load_all_metrics():
    """Load all metrics for all models, all folds."""
    all_data = {}
    for name, dirn in MODEL_DIRS.items():
        all_data[name] = {}
        for key in ["macro_c_index", "macro_ibs", "macro_nbll",
                     "c_index_24h", "c_index_48h", "c_index_72h",
                     "c_index", "ibs", "nbll"]:
            vals = load_fold_metrics(dirn, key)
            all_data[name][key] = vals
    return all_data


# ── Colors ───────────────────────────────────────────────────────────
PER_LANDMARK_COLORS = {
    "Text Only": "#4E79A7",
    "Static Only": "#F28E2B",
    "TS Only": "#59A14F",
    "Static+Text": "#B07AA1",
    "Static+TS": "#76B7B2",
    "Text+TS": "#E15759",
    "Full (S+T+TS)": "#D62728",
}

def make_risk_score_from_surv(survs, bin_edges, t_horizon=48.0):
    """Compute risk score as S(t_horizon) * 100.

    S(t) = probability of still being in the ICU at *t* hours after the
    observation landmark.  Higher value  ⇒  higher risk of prolonged stay.
    """
    survs = np.asarray(survs, dtype=float)
    edges = np.asarray(bin_edges, dtype=float).copy()
    if not np.isfinite(edges[-1]):
        edges[-1] = edges[-2] + float(np.median(np.diff(edges[:-1])))
    bin_idx = min(int(np.searchsorted(edges[1:], t_horizon, side="right")),
                  survs.shape[1] - 1)
    return np.clip(survs[:, bin_idx] * 100.0, 0.0, 100.0)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Per-landmark C-index grouped bar chart
# ═══════════════════════════════════════════════════════════════════════
def fig_per_landmark(all_data):
    models_show = ["Text Only", "Static Only", "TS Only",
                   "Static+Text", "Static+TS", "Text+TS", "Full (S+T+TS)"]
    landmarks = ["24 h", "48 h", "72 h"]
    lm_keys = ["c_index_24h", "c_index_48h", "c_index_72h"]
    n_models = len(models_show)
    width = 0.10
    x = np.arange(3)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(models_show):
        vals = [np.mean(all_data[m][k]) for k in lm_keys]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width * 0.9,
            label=m,
            color=PER_LANDMARK_COLORS[m],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.92,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(landmarks)
    ax.set_ylabel("C-index")
    ax.set_ylim(0.60, 0.82)
    ax.set_xlabel("Landmark Hour")
    ax.set_title("Per-Landmark C-index by Model Configuration")
    ax.legend(fontsize=6.5, ncol=3, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_landmark.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_per_landmark.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] fig_per_landmark")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Modality contribution (delta)
# ═══════════════════════════════════════════════════════════════════════
def fig_modality_delta(all_data):
    pairs = [
        ("Static Only",   "Static+Text",   "+Text"),
        ("Static Only",   "Static+TS",     "+TimeSeries"),
        ("TS Only",       "Text+TS",       "+Text"),
        ("TS Only",       "Static+TS",     "+Static"),
        ("Text Only",     "Static+Text",   "+Static"),
        ("Text Only",     "Text+TS",       "+TimeSeries"),
        ("Static+Text",   "Full (S+T+TS)", "+TimeSeries"),
        ("Static+TS",     "Full (S+T+TS)", "+Text"),
        ("Text+TS",       "Full (S+T+TS)", "+Static"),
    ]
    color_map = {"+Text": "#FF7043", "+TimeSeries": "#26A69A", "+Static": "#5C6BC0"}
    labels, deltas, colors_d = [], [], []
    for base, target, added in pairs:
        delta = np.mean(all_data[target]["macro_c_index"]) - np.mean(all_data[base]["macro_c_index"])
        labels.append(f"{base} {added}")
        deltas.append(delta)
        colors_d.append(color_map[added])

    fig, ax = plt.subplots(figsize=(6.5, 4))
    y = np.arange(len(labels))
    ax.barh(y, deltas, color=colors_d, edgecolor="black", linewidth=0.4, height=0.6)
    for i, d in enumerate(deltas):
        ax.text(d + 0.001, i, f"+{d:.4f}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("$\\Delta$ Macro C-index")
    ax.set_title("Incremental Modality Contribution")
    ax.axvline(0, color="black", lw=0.5)
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=v, edgecolor="black", label=k)
                       for k, v in color_map.items()],
              loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_modality_delta.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_modality_delta.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] fig_modality_delta")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Risk score distribution by outcome
# ═══════════════════════════════════════════════════════════════════════
def fig_risk_distribution():
    """Predicted risk score distribution, split by discharged vs censored (died)."""
    pred_file = (RESULTS /
                 "multimodal_textsmart_weighted_5fold_20260402" /
                 "fold_0_predictions.csv")
    if not pred_file.exists():
        print("  [SKIP] risk dist: prediction file not found")
        return

    df = pd.read_csv(pred_file)
    survs = np.load(pred_file.with_suffix(".npz"))["survs"]
    be = np.load(pred_file.parent / "fold_0_bin_edges.npy")
    risk = make_risk_score_from_surv(survs, be)

    fig, ax = plt.subplots(figsize=(7, 4))
    mask_alive = df["event"] == 1
    mask_dead  = df["event"] == 0
    bin_edges = np.linspace(float(risk.min()), float(risk.max()), 41)

    ax.hist(risk[mask_alive], bins=bin_edges, alpha=0.6, color="#2ecc71",
            label=f"Discharged (n={mask_alive.sum()})", density=True)
    ax.hist(risk[mask_dead], bins=bin_edges, alpha=0.6, color="#e74c3c",
            label=f"Died in ICU (n={mask_dead.sum()})", density=True)

    ax.set_xlabel("Predicted Risk Score (0-100)")
    ax.set_ylabel("Density")
    ax.set_title("Risk Score Distribution by Outcome (Full Model, Fold 0)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_risk_dist.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_risk_dist.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] fig_risk_dist")


# ═══════════════════════════════════════════════════════════════════════
# Print summary table for verification
# ═══════════════════════════════════════════════════════════════════════
def print_summary(all_data):
    print("\n" + "=" * 80)
    print(f"{'Model':<18} {'macro_C-idx':>12} {'IBS':>12} {'NBLL':>12} "
          f"{'C24h':>10} {'C48h':>10} {'C72h':>10}")
    print("=" * 80)
    for name in MODEL_DIRS:
        d = all_data[name]
        mc = d["macro_c_index"]
        ibs = d["macro_ibs"]
        nbll = d["macro_nbll"]
        c24 = d["c_index_24h"]
        c48 = d["c_index_48h"]
        c72 = d["c_index_72h"]
        print(f"{name:<18} "
              f"{np.mean(mc):.4f}+/-{np.std(mc,ddof=1):.4f} "
              f"{np.mean(ibs):.4f}+/-{np.std(ibs,ddof=1):.4f} "
              f"{np.mean(nbll):.4f}+/-{np.std(nbll,ddof=1):.4f} "
              f"{np.mean(c24):.4f} {np.mean(c48):.4f} {np.mean(c72):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    print("Loading metrics from JSON files...")
    all_data = load_all_metrics()
    print_summary(all_data)

    print("\nGenerating figures...")
    fig_per_landmark(all_data)
    fig_modality_delta(all_data)
    fig_risk_distribution()
    print(f"\nAll figures saved to {OUT}/")
