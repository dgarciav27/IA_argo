"""
Threshold-based anomaly detector 

It's built on top of the (already trained) profile reconstructor.

It consumes what the reconstructor already produced:
    - reconstructor_metrics.pkl  (per-profile RMSE_temp / RMSE_psal per split)
and combines them into a single anomaly score, picks a decision threshold
on the VALIDATION set, and reports standard classification metrics.

--------------------
Runs the pipeline independently for each ocean basin

Each basin is treated as a fully separate experiment, just looped over for convenience (each ocean has its own
PREPROC_DIR / OUTPUT_DIR, its own threshold selection).

Required inputs (per ocean):
  - reconstructor_metrics.pkl        (from the reconstruction script)
  - <split>_meta.parquet             (for is_bad labels + QC severity)

Outputs (per ocean, under that ocean's OUTPUT_DIR):
  - threshold_metrics.pkl            all threshold-selection + evaluation results
  - figures_seuil/score_distribution_val.png
  - figures_seuil/roc_pr_val.png
  - figures_seuil/confusion_matrices.png

Additionally, cross-ocean summary is printed and saved as summary_all_oceans.pkl and summary_all_oceans.csv.
"""

import os

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score,
)

# 0. Config per ocean
# Adjust the base paths / split_time_... subfolder name per ocean as needed.
YEARS_RANGE = "2018_2022"  # Years of data to use for training/validation/testing
OCEANS = {
    "atlantic": {
        "preproc_dir": f"/work/drgarcia/Dataset/DL_datasets/atlantic_ocean/{YEARS_RANGE}/split_time_ascA_dmodeD_masked_grid100",
        "output_dir":  f"/work/drgarcia/Models_and_results/Autoencoder/reconstructor_atlantic_{YEARS_RANGE}",
    },
    "pacific": {
        "preproc_dir": f"/work/drgarcia/Dataset/DL_datasets/pacific_ocean/{YEARS_RANGE}/split_time_ascA_dmodeD_masked_grid100",
        "output_dir":  f"/work/drgarcia/Models_and_results/Autoencoder/reconstructor_pacific_{YEARS_RANGE}",
    },
    "indian": {
        "preproc_dir": f"/work/drgarcia/Dataset/DL_datasets/indian_ocean/{YEARS_RANGE}/split_time_ascA_dmodeD_masked_grid100",
        "output_dir":  f"/work/drgarcia/Models_and_results/Autoencoder/reconstructor_indian_{YEARS_RANGE}",
    },
}

# Top-level dir where the cross-ocean summary is saved
SUMMARY_DIR = f"/work/drgarcia/Models_and_results/Autoencoder/reconstructor_alloceans_{YEARS_RANGE}"
os.makedirs(SUMMARY_DIR, exist_ok=True)

SEVERITY_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
SEVERITY_NAMES = {v: k for k, v in SEVERITY_ORDER.items()}

THRESHOLD_STRATEGY = "f1_macro_val"
PERCENTILE = 99.0  # only used if THRESHOLD_STRATEGY == "percentile_val_normal"
N_THRESHOLD_CANDIDATES = 500  # grid size for the f1_macro search


# Helpers

def load_labels_and_severity(preproc_dir, name):
    meta = pd.read_parquet(os.path.join(preproc_dir, f"{name}_meta.parquet"))
    labels = meta["is_bad"].values.astype(np.int8)
    t = meta["PROFILE_TEMP_QC"].map(SEVERITY_ORDER)
    s = meta["PROFILE_PSAL_QC"].map(SEVERITY_ORDER)
    severity = np.where(t.notna() & s.notna(), np.maximum(t.values, s.values), np.nan)
    return labels, severity


def combined_score(rmse_t, rmse_s, mu_t, sd_t, mu_s, sd_s):
    z_t = (rmse_t - mu_t) / sd_t
    z_s = (rmse_s - mu_s) / sd_s
    return z_t + z_s


def clean(score, labels, severity):
    ok = ~np.isnan(score)
    return score[ok], labels[ok], severity[ok]


def find_best_threshold_f1_macro(y_true, score, n_candidates=N_THRESHOLD_CANDIDATES):
    """
    Grid-search the threshold that maximizes F1 macro (average of F1 for
    the normal class and F1 for the anomaly class).
    """
    candidates = np.unique(score)
    if len(candidates) > n_candidates:
        candidates = np.quantile(score, np.linspace(0, 1, n_candidates))

    best_t, best_f1m = candidates[0], -1.0
    for t in candidates:
        y_pred = (score >= t).astype(int)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1m > best_f1m:
            best_f1m, best_t = f1m, t

    return float(best_t), float(best_f1m)


def evaluate_split(name, score, y_true, severity, threshold):
    y_pred = (score >= threshold).astype(int)

    report = classification_report(y_true, y_pred, target_names=["normal", "anomaly"],
                                     output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=["normal", "anomaly"], zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    roc_auc = roc_auc_score(y_true, score) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc  = average_precision_score(y_true, score) if len(np.unique(y_true)) > 1 else np.nan
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    sev_detection = {}
    for sev_val_, sev_name in SEVERITY_NAMES.items():
        m = (y_true == 1) & (severity == sev_val_)
        if m.sum() == 0:
            continue
        sev_detection[sev_name] = {
            "n": int(m.sum()),
            "detected": int(y_pred[m].sum()),
            "detection_rate": float(y_pred[m].mean()),
        }

    print(f"\n=== [{name}] threshold = {threshold:.4f} ===")
    print(report_str)
    print("Confusion matrix [rows=true, cols=pred] (0=normal, 1=anomaly):")
    print(cm)
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1 macro: {f1_macro:.4f}")

    return {
        "y_pred": y_pred, "classification_report": report, "classification_report_str": report_str,
        "confusion_matrix": cm, "roc_auc": roc_auc, "pr_auc": pr_auc, "f1_macro": f1_macro,
        "severity_detection": sev_detection,
    }


def run_for_ocean(ocean_name, preproc_dir, output_dir):
    print("\n" + "=" * 70)
    print(f"OCEAN = {ocean_name.upper()}")
    print("=" * 70)

    metrics_path           = os.path.join(output_dir, "reconstructor_metrics.pkl")
    threshold_metrics_path = os.path.join(output_dir, "threshold_metrics.pkl")
    fig_dir = os.path.join(output_dir, "figures_seuil")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load reconstructor outputs (per-profile RMSE) and labels
    print("Loading reconstructor metrics ...")
    results = joblib.load(metrics_path)

    labels_train, sev_train = load_labels_and_severity(preproc_dir, "train")
    labels_val,   sev_val   = load_labels_and_severity(preproc_dir, "val")
    labels_test,  sev_test  = load_labels_and_severity(preproc_dir, "test")

    rmse_t_train, rmse_s_train = results["train"]["ae_rmse_temp"], results["train"]["ae_rmse_psal"]
    rmse_t_val,   rmse_s_val   = results["val"]["ae_rmse_temp"],   results["val"]["ae_rmse_psal"]
    rmse_t_test,  rmse_s_test  = results["test"]["ae_rmse_temp"],  results["test"]["ae_rmse_psal"]

    assert len(labels_train) == len(rmse_t_train), f"[{ocean_name}] train: labels/rmse length mismatch"
    assert len(labels_val)   == len(rmse_t_val),   f"[{ocean_name}] val: labels/rmse length mismatch"
    assert len(labels_test)  == len(rmse_t_test),  f"[{ocean_name}] test: labels/rmse length mismatch"

    # 2. Build a single anomaly score per profile — normalized on THIS ocean's
    #    own train-normal distribution (never mixed with other oceans)
    idx_normal_train = (labels_train == 0)
    mu_t = np.nanmean(rmse_t_train[idx_normal_train]); sd_t = np.nanstd(rmse_t_train[idx_normal_train]) + 1e-8
    mu_s = np.nanmean(rmse_s_train[idx_normal_train]); sd_s = np.nanstd(rmse_s_train[idx_normal_train]) + 1e-8

    score_train = combined_score(rmse_t_train, rmse_s_train, mu_t, sd_t, mu_s, sd_s)
    score_val   = combined_score(rmse_t_val,   rmse_s_val,   mu_t, sd_t, mu_s, sd_s)
    score_test  = combined_score(rmse_t_test,  rmse_s_test,  mu_t, sd_t, mu_s, sd_s)

    score_train, labels_train_c, sev_train_c = clean(score_train, labels_train, sev_train)
    score_val,   labels_val_c,   sev_val_c   = clean(score_val,   labels_val,   sev_val)
    score_test,  labels_test_c,  sev_test_c  = clean(score_test,  labels_test,  sev_test)

    #  Diagnostic: does temp-only or psal-only discriminate better than
    #     the combined score? If one channel's AUC is much higher than the
    #     combined one, the sum is diluting signal from the stronger channel.
    rmse_t_val_ok = rmse_t_val[~np.isnan(rmse_t_val)]
    rmse_s_val_ok = rmse_s_val[~np.isnan(rmse_s_val)]
    labels_val_t = labels_val[~np.isnan(rmse_t_val)]
    labels_val_s = labels_val[~np.isnan(rmse_s_val)]

    auc_temp_only = roc_auc_score(labels_val_t, rmse_t_val_ok) if len(np.unique(labels_val_t)) > 1 else np.nan
    auc_psal_only = roc_auc_score(labels_val_s, rmse_s_val_ok) if len(np.unique(labels_val_s)) > 1 else np.nan

    # 3. Threshold selection on VALIDATION only (this ocean's own val set)
    print(f"\nSelecting decision threshold on validation set [{ocean_name}] ...")

    roc_auc_val = roc_auc_score(labels_val_c, score_val)
    pr_auc_val  = average_precision_score(labels_val_c, score_val)
    print(f"  Val ROC-AUC (combined): {roc_auc_val:.4f} | Val PR-AUC (combined): {pr_auc_val:.4f}")
    print(f"  Val ROC-AUC (temp only): {auc_temp_only:.4f} | Val ROC-AUC (psal only): {auc_psal_only:.4f}")
    if not np.isnan(auc_temp_only) and not np.isnan(auc_psal_only):
        best_single = max(auc_temp_only, auc_psal_only)
        if best_single > roc_auc_val + 0.02:
            print(f"  [WARNING] a single channel (AUC={best_single:.4f}) discriminates noticeably "
                  f"better than the combined score ({roc_auc_val:.4f}). Consider using "
                  f"max(z_temp, z_psal) instead of the sum, or investigate which channel "
                  f"is adding noise.")

    fpr, tpr, roc_thresh = roc_curve(labels_val_c, score_val)
    precision, recall, pr_thresh = precision_recall_curve(labels_val_c, score_val)

    f1_scores = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    best_f1_idx = np.nanargmax(f1_scores[:-1])
    threshold_f1 = pr_thresh[best_f1_idx]

    youden_idx = np.argmax(tpr - fpr)
    threshold_youden = roc_thresh[youden_idx]

    threshold_percentile = np.percentile(score_val[labels_val_c == 0], PERCENTILE)

    threshold_f1_macro, best_f1_macro_val = find_best_threshold_f1_macro(labels_val_c, score_val)

    thresholds = {
        "f1_val": float(threshold_f1),
        "f1_macro_val": threshold_f1_macro,
        "youden_val": float(threshold_youden),
        "percentile_val_normal": float(threshold_percentile),
    }
    print("  Candidate thresholds:", {k: round(v, 3) for k, v in thresholds.items()})
    print(f"  (f1_macro_val achieves F1 macro = {best_f1_macro_val:.4f} on val)")

    chosen_threshold = thresholds[THRESHOLD_STRATEGY]
    print(f"  -> Using strategy '{THRESHOLD_STRATEGY}': threshold = {chosen_threshold:.4f}")

    # 4. Evaluate on train / val / test with the chosen threshold
    eval_results = {}
    for split_name, score, y_true, severity in [
        ("train", score_train, labels_train_c, sev_train_c),
        ("val",   score_val,   labels_val_c,   sev_val_c),
        ("test",  score_test,  labels_test_c,  sev_test_c),
    ]:
        eval_results[split_name] = evaluate_split(split_name, score, y_true, severity, chosen_threshold)

    # Diagnostic: does detection improve a lot if we restrict "anomaly"
    #     to severe QC flags only (D/E/F)? If yes, the reconstructor mostly
    #     struggles with mild (A/B) cases?
    sev_severe_val = sev_val_c >= SEVERITY_ORDER["D"]
    mask_severe = (labels_val_c == 0) | sev_severe_val
    if mask_severe.sum() > 0 and len(np.unique(labels_val_c[mask_severe])) > 1:
        auc_severe_only = roc_auc_score(labels_val_c[mask_severe], score_val[mask_severe])
        print(f"\n  Val ROC-AUC restricted to severe anomalies only (QC>=D): {auc_severe_only:.4f} "
              f"(vs {roc_auc_val:.4f} overall)")

    # 5. Plots
    print(f"\nSaving figures [{ocean_name}] ...")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(score_val[labels_val_c == 0], bins=60, alpha=0.6, label="normal", density=True)
    ax.hist(score_val[labels_val_c == 1], bins=60, alpha=0.6, label="anomaly", density=True)
    ax.axvline(chosen_threshold, color="k", linestyle="--", label=f"threshold ({THRESHOLD_STRATEGY})")
    ax.set_xlabel("combined anomaly score (z-summed RMSE)")
    ax.set_ylabel("density")
    ax.set_title(f"Validation score distribution — {ocean_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "score_distribution_val.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_val:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC curve (val) — {ocean_name}")
    axes[0].legend()

    axes[1].plot(recall, precision, label=f"PR-AUC = {pr_auc_val:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall curve (val) — {ocean_name}")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "roc_pr_val.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, split_name in zip(axes, ["train", "val", "test"]):
        cm = eval_results[split_name]["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(split_name)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "anomaly"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "anomaly"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle(f"Confusion matrices — {ocean_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "confusion_matrices.png"), dpi=150)
    plt.close(fig)

    print(f"  Figures saved -> {fig_dir}")

    # 6. Save per-ocean results
    threshold_results = {
        "ocean": ocean_name,
        "strategy_used": THRESHOLD_STRATEGY,
        "chosen_threshold": chosen_threshold,
        "candidate_thresholds": thresholds,
        "score_normalization": {"mu_temp": mu_t, "sd_temp": sd_t, "mu_psal": mu_s, "sd_psal": sd_s},
        "val_roc_auc": roc_auc_val, "val_pr_auc": pr_auc_val,
        "val_roc_auc_temp_only": auc_temp_only, "val_roc_auc_psal_only": auc_psal_only,
        "scores": {"train": score_train, "val": score_val, "test": score_test},
        "eval": eval_results,
    }
    joblib.dump(threshold_results, threshold_metrics_path)
    print(f"Saved threshold metrics -> {threshold_metrics_path}")

    return threshold_results


# Main: run independently for each ocean, then build a small comparison table
if __name__ == "__main__":
    all_results = {}
    for ocean_name, cfg in OCEANS.items():
        all_results[ocean_name] = run_for_ocean(ocean_name, cfg["preproc_dir"], cfg["output_dir"])

    # Cross-ocean summary (comparison only — thresholds/scores stay per-ocean)
    summary_rows = []
    for ocean_name, res in all_results.items():
        test_report = res["eval"]["test"]["classification_report"]
        summary_rows.append({
            "ocean": ocean_name,
            "strategy": res["strategy_used"],
            "chosen_threshold": res["chosen_threshold"],
            "val_roc_auc": res["val_roc_auc"],
            "val_pr_auc": res["val_pr_auc"],
            "test_roc_auc": res["eval"]["test"]["roc_auc"],
            "test_pr_auc": res["eval"]["test"]["pr_auc"],
            "test_f1_macro": res["eval"]["test"]["f1_macro"],
            "test_precision_anomaly": test_report["anomaly"]["precision"],
            "test_recall_anomaly": test_report["anomaly"]["recall"],
            "test_f1_anomaly": test_report["anomaly"]["f1-score"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS OCEANS")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    joblib.dump(all_results, os.path.join(SUMMARY_DIR, "summary_all_oceans.pkl"))
    summary_df.to_csv(os.path.join(SUMMARY_DIR, "summary_all_oceans.csv"), index=False)
    print(f"\nSaved cross-ocean summary -> {SUMMARY_DIR}")