"""
Train RF script - 3 oceans (Atlantic, Indian, Pacific)

Loads preprocessed splits from argo_preprocess.py output.
For each ocean:
  - Optimizes F1-macro with HalvingRandomSearchCV (CV on train only)
  - Tunes the threshold on val
  - Evaluates on test
  - Saves everything in its own results folder and its own MLflow experiment
  
-To run after: /.../preprocessing.ipynb
"""

import os, gc, json, warnings
import numpy as np
import pandas as pd
import joblib
import mlflow

from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score,
)

warnings.filterwarnings("ignore")

# SET UP
# one ocean per input, each with its own preprocessing folder, output folder, and plot color
OCEANS = {
    "Atlantic": {
        "preprocess_dir": "/work/drgarcia/Dataset/atlantic_ocean/2018-2022/2.preprocessed/split_time_ascA_dmodeD",
        "model_dir":      "/work/drgarcia/Models_and_results/RF/2018_2022/atlantic_f1macro_time_asc_nounder",
        "color":          "#1f77b4",
    },
    "Indian": {
        "preprocess_dir": "/work/drgarcia/Dataset/indian_ocean/2018-2022/2.preprocessed/split_time_ascA_dmodeD",
        "model_dir":      "/work/drgarcia/Models_and_results/RF/2018_2022/indian_f1macro_time_asc_nounder",
        "color":          "#2ca02c",
    },
    "Pacific": {
        "preprocess_dir": "/work/drgarcia/Dataset/pacific_ocean/2018-2022/2.preprocessed/split_time_ascA_dmodeD",
        "model_dir":      "/work/drgarcia/Models_and_results/RF/2018_2022/pacific_f1macro_time_asc_nounder",
        "color":          "#ff7f0e",
    },
}

MLFLOW_DIR = "/home/drgarcia/Argo_ml_code/ML_flow"

# MLflow: a single shared tracking URI, but a separate EXPERIMENT
# per ocean (so each ocean is kept separate in the MLflow UI)
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")


def _load_split(preprocess_dir, feature_cols, name):
    df = pd.read_parquet(os.path.join(preprocess_dir, f"{name}.parquet"))
    X  = df[feature_cols].astype(np.float32)
    y  = df["is_bad"]
    return X, y


def train_one_ocean(ocean_name, cfg):
    preprocess_dir = cfg["preprocess_dir"]
    output_dir     = cfg["model_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print(f"#  OCEANO: {ocean_name}")
    print("#" * 70)

    #  LOAD PREPROCESSED SPLITS
    feature_cols = joblib.load(os.path.join(preprocess_dir, "feature_cols.pkl"))

    X_train, y_train = _load_split(preprocess_dir, feature_cols, "train")
    X_val,   y_val   = _load_split(preprocess_dir, feature_cols, "val")
    X_test,  y_test  = _load_split(preprocess_dir, feature_cols, "test")
    df_test_meta      = pd.read_parquet(os.path.join(preprocess_dir, "test_meta.parquet"))

    print(f"  Train : {len(X_train):,} profiles | {y_train.mean():.2%} anomalous")
    print(f"  Val   : {len(X_val):,}   profiles | {y_val.mean():.2%} anomalous")
    print(f"  Test  : {len(X_test):,}  profiles | {y_test.mean():.2%} anomalous")
    print(f"  Features: {len(feature_cols)}  :  {feature_cols}")

    #  MLFLOW SETUP
    experiment_name = f"RF_{ocean_name}_2018_2022_TimeSplit_Dmode"
    experiment      = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=f"file://{MLFLOW_DIR}/mlruns",
        )
        print(f"\nExperiment '{experiment_name}' created (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"\nUsing experiment '{experiment_name}' (id={experiment_id})")

    mlflow.set_experiment(experiment_name)

    #  TRAINING
    with mlflow.start_run(run_name=f"RF_f1macro_time_ascD_nounder_Dmode_{ocean_name.lower()}"):

        mlflow.log_params({
            "ocean":           ocean_name,
            "split_type":      "time",
            "d_mode":          "D",
            "Min_levels":      "10",
            "direction":       "ascending_cycles_only",
            "undersample":     "false",
            "preprocess_dir":  preprocess_dir,
            "n_features":      len(feature_cols),
            "train_profiles":  len(X_train),
            "val_profiles":    len(X_val),
            "test_profiles":   len(X_test),
            "train_pct_bad":   float(y_train.mean()),
            "scoring":         "f1_macro",
        })

        #  BASELINE
        print("\n" + "=" * 70)
        print("Baseline RF (100 trees, default threshold 0.5)")
        print("=" * 70)

        rf_base = RandomForestClassifier(
            n_estimators=100, random_state=42,
            n_jobs=4, class_weight="balanced_subsample")
        rf_base.fit(X_train, y_train)

        y_proba_base = rf_base.predict_proba(X_test)[:, 1]
        y_pred_base  = (y_proba_base >= 0.5).astype(int)
        baseline_auc = roc_auc_score(y_test, y_proba_base)
        baseline_f1  = f1_score(y_test, y_pred_base, average="macro")

        print(f"  Baseline ROC-AUC (test) : {baseline_auc:.4f}")
        print(f"  Baseline F1-macro (test): {baseline_f1:.4f}")
        mlflow.log_metrics({
            "baseline_roc_auc":  baseline_auc,
            "baseline_f1_macro": baseline_f1,
        })

        #  HYPERPARAMETER SEARCH (F1-MACRO)
        print("\n" + "=" * 70)
        print("HalvingRandomSearchCV — scoring=f1_macro  (CV on train only)")
        print("=" * 70)

        param_dist = {
            "n_estimators":          [1000],
            "max_depth":             [10, 20, 30, 50, None],
            "min_samples_split":     randint(2, 20),
            "min_samples_leaf":      randint(1, 10),
            "max_features":          ["sqrt", "log2", 0.3, 0.5],
            "class_weight":          ["balanced", "balanced_subsample"],
            "criterion":             ["gini", "entropy"],
            "bootstrap":             [True],
            "max_samples":           [0.6, 0.7, 0.8, 0.9, None],
            "min_impurity_decrease": [0.0, 0.001, 0.005],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        search = HalvingRandomSearchCV(
            estimator=RandomForestClassifier(random_state=42, n_jobs=4),
            param_distributions=param_dist,
            factor=3,
            n_candidates=30,
            min_resources="exhaust",
            cv=cv,
            scoring="f1_macro",          #  F1-macro or recall
            n_jobs=-1,
            verbose=2,
            random_state=42,
        )

        search.fit(X_train, y_train)

        cv_results    = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False)
        best_rf_model = search.best_estimator_
        best_params   = search.best_params_

        print(f"\nBest F1-macro (CV on train): {search.best_score_:.4f}")
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1_macro", search.best_score_)

        #  THRESHOLD TUNING ON VAL
        print("\n" + "=" * 70)
        print("Threshold tuning on VALIDATION set")
        print("=" * 70)

        probs_val  = best_rf_model.predict_proba(X_val)[:, 1]
        thresholds = np.round(np.arange(0.05, 0.95, 0.05), 2)

        records = []
        for t in thresholds:
            yp = (probs_val >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, yp, labels=[0, 1]).ravel()
            records.append({
                "threshold":      t,
                "f1_macro":       f1_score(y_val, yp, average="macro",   zero_division=0),
                "f1_anomaly":     f1_score(y_val, yp, pos_label=1,        zero_division=0),
                "recall_anom":    recall_score(y_val, yp, pos_label=1,    zero_division=0),
                "precision_anom": precision_score(y_val, yp, pos_label=1, zero_division=0),
                "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            })

        df_th = pd.DataFrame(records)
        df_th["youden"] = df_th["tpr"] - df_th["fpr"]

        # Primary: best F1-macro on val
        best_f1_macro_row  = df_th.loc[df_th["f1_macro"].idxmax()]
        # Secondary candidates (logged, not used for final eval)
        best_f1_anom_row   = df_th.loc[df_th["f1_anomaly"].idxmax()]
        best_youden_row    = df_th.loc[df_th["youden"].idxmax()]
        high_recall_rows   = df_th[df_th["recall_anom"] >= 0.90].sort_values("precision_anom", ascending=False)
        best_recall90_row  = high_recall_rows.iloc[0] if not high_recall_rows.empty else df_th.loc[df_th["recall_anom"].idxmax()]

        #chosen_threshold = float(best_recall90_row["threshold"])
        chosen_threshold   = float(best_f1_macro_row["threshold"])

        print(f"  Best F1-macro  threshold (val): {chosen_threshold:.2f}  "
              f"→ f1_macro={best_f1_macro_row['f1_macro']:.4f}  "
              f"recall_anom={best_f1_macro_row['recall_anom']:.4f}")
        print(f"  Best F1-anomaly threshold (val): {best_f1_anom_row['threshold']:.2f}")
        print(f"  Best Youden   threshold (val): {best_youden_row['threshold']:.2f}")
        print(f"  Best recall≥90 threshold (val): {best_recall90_row['threshold']:.2f}")

        mlflow.log_metrics({
            "val_f1_macro_at_chosen":    float(best_f1_macro_row["f1_macro"]),
            "val_recall_anom_at_chosen": float(best_f1_macro_row["recall_anom"]),
            "val_precision_at_chosen":   float(best_f1_macro_row["precision_anom"]),
            "chosen_threshold":          chosen_threshold,
        })

        #  FINAL EVALUATION ON TEST
        print("\n" + "=" * 70)
        print(f"Final evaluation on TEST  (threshold={chosen_threshold:.2f} from val)")
        print("=" * 70)

        probs        = best_rf_model.predict_proba(X_test)[:, 1]
        y_pred_final = (probs >= chosen_threshold).astype(int)

        fpr_roc, tpr_roc, _ = roc_curve(y_test, probs)
        test_roc_auc         = auc(fpr_roc, tpr_roc)
        test_f1_macro        = f1_score(y_test, y_pred_final, average="macro")
        test_f1_anom         = f1_score(y_test, y_pred_final, pos_label=1, zero_division=0)
        test_recall_anom     = recall_score(y_test, y_pred_final, pos_label=1, zero_division=0)
        test_precision_anom  = precision_score(y_test, y_pred_final, pos_label=1, zero_division=0)

        mlflow.log_metrics({
            "test_roc_auc":        test_roc_auc,
            "test_f1_macro":       test_f1_macro,
            "test_f1_anomaly":     test_f1_anom,
            "test_recall_anomaly": test_recall_anom,
            "test_precision_anom": test_precision_anom,
        })

        print("\nBASELINE  (test, threshold=0.5)")
        print("=" * 70)
        print(classification_report(y_test, y_pred_base, target_names=["Good", "Bad"], digits=4))

        print(f"\nOPTIMIZED  (test, threshold={chosen_threshold:.2f} tuned on val, scoring=f1_macro)")
        print("=" * 70)
        print(classification_report(y_test, y_pred_final, target_names=["Good", "Bad"], digits=4))

        #  SAVE ARTIFACTS
        print("\n" + "=" * 70)
        print("Saving artifacts")
        print("=" * 70)

        def _save(label, fn):
            try:
                fn()
                print(f"  - {label}")
            except Exception as e:
                print(f"  x {label}: {e}")

        _save("Model",
              lambda: joblib.dump(best_rf_model,
                                  os.path.join(output_dir, "RF_optimized.pkl")))

        plot_bundle = {
            "ocean":               ocean_name,
            "color":               cfg["color"],
            "cv_results":          cv_results,
            "feature_importances": best_rf_model.feature_importances_,
            "feature_cols":        feature_cols,
            "df_th":               df_th,
            "y_val":               y_val.values,
            "probs_val":           probs_val,
            "y_test":              y_test.values,
            "probs":               probs,
            "y_proba_base":        y_proba_base,
            "y_pred_base":         y_pred_base,
            "y_pred_final":        y_pred_final,
            "chosen_threshold":    chosen_threshold,
            "best_f1_macro_row":   best_f1_macro_row.to_dict(),
            "best_f1_anom_row":    best_f1_anom_row.to_dict(),
            "best_youden_row":     best_youden_row.to_dict(),
            "best_recall90_row":   best_recall90_row.to_dict(),
            "fpr_roc":             fpr_roc,
            "tpr_roc":             tpr_roc,
            "variables_qc": {
                "PRES": df_test_meta["PRES_is_bad"].values if "PRES_is_bad" in df_test_meta.columns else None,
                "TEMP": df_test_meta["TEMP_is_bad"].values if "TEMP_is_bad" in df_test_meta.columns else None,
                "PSAL": df_test_meta["PSAL_is_bad"].values if "PSAL_is_bad" in df_test_meta.columns else None,
            },
            "split_info": {
                "ocean":          ocean_name,
                "type":           "time",
                "direction":      "ascending_only",
                "undersample":    "false",
                "scoring":        "f1_macro",
                "d_mode":         "D",
                "preprocess_dir": preprocess_dir,
            },
        }
        _save("Plot bundle",
              lambda: joblib.dump(plot_bundle,
                                  os.path.join(output_dir, "plot_data_bundle.pkl")))

        def _save_test_results():
            out = df_test_meta.copy()
            out["y_pred"]       = y_pred_final
            out["y_pred_proba"] = probs
            out.to_parquet(os.path.join(output_dir, "test_results.parquet"), index=False)
        _save("Test results parquet", _save_test_results)

        threshold_info = {
            "ocean":                ocean_name,
            "chosen_threshold":     chosen_threshold,
            "chosen_by":            "f1",
            "th_f1_macro":          float(best_f1_macro_row["threshold"]),
            "th_f1_anomaly":        float(best_f1_anom_row["threshold"]),
            "th_youden":            float(best_youden_row["threshold"]),
            "th_recall90":          float(best_recall90_row["threshold"]),
            "best_params":          best_params,
            "tuned_on":             "validation",
        }
        _save("Threshold info",
              lambda: joblib.dump(threshold_info,
                                  os.path.join(output_dir, "threshold_info.pkl")))

        def _save_json():
            d = {**{str(k): str(v) for k, v in best_params.items()}, **threshold_info}
            with open(os.path.join(output_dir, "best_params.json"), "w") as f:
                json.dump(d, f, indent=4)
        _save("Best params JSON", _save_json)

        _save("HP search CSV",
              lambda: cv_results.to_csv(
                  os.path.join(output_dir, "hyperparameter_search_results.csv"), index=False))

        _save("Threshold CSV",
              lambda: df_th.to_csv(
                  os.path.join(output_dir, "threshold_analysis.csv"), index=False))

        def _save_report():
            path = os.path.join(output_dir, "classification_report.txt")
            with open(path, "w") as f:
                f.write(f"RF — {ocean_name} | Time split | Ascending only | Not undersampled | scoring=f1_macro \n")
                f.write("=" * 70 + "\n\n")
                f.write("BASELINE  (test, threshold=0.5)\n")
                f.write(classification_report(y_test, y_pred_base,
                                              target_names=["Good", "Bad"], digits=4))
                f.write(f"\n\nOPTIMIZED  (threshold={chosen_threshold:.2f}, tuned on val)\n")
                f.write(classification_report(y_test, y_pred_final,
                                              target_names=["Good", "Bad"], digits=4))
        _save("Classification report", _save_report)

    # free memory before the next ocean
    del X_train, y_train, X_val, y_val, X_test, y_test, df_test_meta
    gc.collect()


if __name__ == "__main__":
    for ocean_name, cfg in OCEANS.items():
        train_one_ocean(ocean_name, cfg)

    print("Train completed for 3 oceans (Atlantic, Indian, Pacific)")
    print("Results saved in each ocean's model_dir (see OCEANS config)")