"""
02_train_lgbm.py

Entraîne LightGBM comme classificateur d'anomalies Argo en utilisant les caractéristiques
générées par 01_build_features.py (erreur de reconstruction de l'auto-encodeur + caractéristiques
supplémentaires : résidus WOA, dérive, voisin, inversion de densité, résidu T-S, etc.).

Ce script NE relance PAS l'auto-encodeur ni ne reconstruit les caractéristiques :
il lit directement les fichiers parquet que 01_build_features.py a déjà enregistrés dans
`output_dir` (même modèle pour chaque océan/plage d'années).

Sélection des caractéristiques

Au lieu d'utiliser toutes les caractéristiques manuellement, un LGBM exploratoire est entraîné
avec TOUTES les colonnes disponibles, le 'gain' de chacune est mesuré, et le
sous-ensemble minimum accumulant `FEATURE_GAIN_THRESHOLD` (ex. 0.90 = 90%) de l'importance totale
est automatiquement sélectionné, avec un seuil minimum `MIN_FEATURES` au cas où le gain
serait trop concentré sur peu de caractéristiques.

Utilisation :
    python 02_train_lgbm.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.lightgbm
import lightgbm as lgb

from scipy.stats import randint, uniform
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_halving_search_cv  # noqa: F401 (active HalvingRandomSearchCV)
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

# CONFIGURATION

OCEANS = ["atlantic", "pacific", "indian"]
YEARS_RANGES = ["2018_2022"]

BASE_RECON_DIR = "/work/drgarcia/Models_and_results/Autoencoder"
MLFLOW_DIR = os.path.join(BASE_RECON_DIR, "mlflow")

SEED = 42

# Sélection automatique des caractéristiques par gain cumulé
FEATURE_GAIN_THRESHOLD = 0.90   # 0.90 = 90% de l'importance cumulée. Essayer aussi 0.80.
MIN_FEATURES = 10               # seuil minimum, au cas où le gain serait très concentré

# Méthode pour choisir le seuil final de classification
CHOSEN_THRESHOLD_METHOD = "f1"  # "f1" | "youden" | "recall90"
TARGET_RECALL = 0.90            # utilisé uniquement si CHOSEN_THRESHOLD_METHOD == "recall90"

SEARCH_SCORING = "average_precision"

# Option A : utilise toutes les colonnes et exclut celles ci-dessous. Laisser vide pour toutes les utiliser.
EXCLUDE_FEATURES = [
    "neighbor_resid_temp_mean",
    "neighbor_resid_psal_mean",
    "neighbor_resid_temp_max",
    "neighbor_resid_psal_max",
    "neighbor_n_used",
    "prevcycle_dev_temp",
    "prevcycle_dev_psal",
    "drift_slope",
    "drift_median_dev",
    "drift_slope_long",
    "drift_same_month_last_year_dev",
    # "LATITUDE", "LONGITUDE",
]

# colonnes qui ne sont JAMAIS des caractéristiques (métadonnées / cible / ids), + celles exclues ci-dessus
NON_FEATURE_COLS_BASE = ["is_bad", "severity", "PLATFORM_NUMBER", "CYCLE_NUMBER"]
NON_FEATURE_COLS = NON_FEATURE_COLS_BASE + EXCLUDE_FEATURES


def get_paths(ocean, years_range):
    output_dir = os.path.join(BASE_RECON_DIR, f"LightGBM_{ocean}_{years_range}")
    return {
        "output_dir": output_dir,
        "ts_reg_path": os.path.join(output_dir, "ts_reg.joblib"),
    }


# Chargement des données (déjà construites par 01_build_features.py)

def load_data_for_ocean(ocean, years_range):
    paths = get_paths(ocean, years_range)
    output_dir = paths["output_dir"]

    df_train = pd.read_parquet(os.path.join(output_dir, "anomaly_features_train.parquet"))
    df_val = pd.read_parquet(os.path.join(output_dir, "anomaly_features_val.parquet"))
    df_test = pd.read_parquet(os.path.join(output_dir, "anomaly_features_test.parquet"))

    meta_extra_path = os.path.join(output_dir, "test_meta_extra.parquet")
    df_test_meta_extra = pd.read_parquet(meta_extra_path) if os.path.exists(meta_extra_path) else None

    ts_reg = joblib.load(paths["ts_reg_path"]) if os.path.exists(paths["ts_reg_path"]) else None
    if ts_reg is None:
        print(f"  AVERTISSEMENT : ts_reg.joblib introuvable dans {output_dir} (enregistré comme None)")

    return df_train, df_val, df_test, df_test_meta_extra, ts_reg, paths


# Sélection des caractéristiques par gain cumulé

def select_features_by_cumulative_gain(model, feature_names, threshold=FEATURE_GAIN_THRESHOLD,
                                       min_features=MIN_FEATURES):
    """
    Étant donné un LGBMClassifier déjà entraîné avec TOUTES les caractéristiques, calcule le
    'gain' de chacune, les trie de la plus grande à la plus petite, et renvoie le sous-ensemble
    minimum de caractéristiques dont l'importance cumulée (normalisée à 1) atteint
    `threshold` (ex. 0.90 = 90%).

    Ne renvoie jamais moins de `min_features` (plancher de sécurité, au cas où le gain
    serait très concentré sur 2-3 caractéristiques).

    Returns

    selected_features : list[str]
        Sous-ensemble de caractéristiques sélectionnées, par ordre d'importance décroissant.
    fi_df : pd.DataFrame
        DataFrame [feature, gain, gain_norm, gain_cum] pour TOUTES les
        caractéristiques (utile pour les graphiques d'importance / gain cumulé).
    """
    # nous utilisons l'ordre que le booster lui-même rapporte, pour éviter
    # les désalignements si feature_names ne correspond pas exactement
    booster_names = list(model.booster_.feature_name())
    if booster_names != list(feature_names):
        feature_names = booster_names

    gain = model.booster_.feature_importance(importance_type="gain")
    fi_df = pd.DataFrame({"feature": feature_names, "gain": gain})
    fi_df = fi_df.sort_values("gain", ascending=False).reset_index(drop=True)

    total_gain = fi_df["gain"].sum()
    if total_gain <= 0:
        # cas dégénéré (ne devrait pas arriver) : nous les renvoyons toutes
        fi_df["gain_norm"] = 0.0
        fi_df["gain_cum"] = 0.0
        return fi_df["feature"].tolist(), fi_df

    fi_df["gain_norm"] = fi_df["gain"] / total_gain
    fi_df["gain_cum"] = fi_df["gain_norm"].cumsum()

    n_needed = int(np.searchsorted(fi_df["gain_cum"].values, threshold) + 1)
    n_selected = min(max(n_needed, min_features, 1), len(fi_df))

    selected_features = fi_df["feature"].iloc[:n_selected].tolist()
    return selected_features, fi_df


# Seuils et évaluation

def best_f1_threshold(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best_idx = np.nanargmax(f1[:-1])
    return float(thr[best_idx]), float(f1[best_idx])


def youden_threshold(y_true, proba):
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    best_idx = np.argmax(j)
    return float(thr[best_idx]), float(j[best_idx])


def recall_target_threshold(y_true, proba, target_recall=TARGET_RECALL):
    """Seuil le plus élevé (= meilleure précision) qui garantit encore un rappel >= target_recall."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    valid = np.where(rec[:-1] >= target_recall)[0]
    if len(valid) == 0:
        return float(thr[-1]) if len(thr) else 0.5, float(rec[-1] if len(rec) else 0.0)
    best_idx = valid[np.argmax(thr[valid])]
    return float(thr[best_idx]), float(rec[best_idx])


def evaluate(y_true, proba, threshold, label=""):
    y_pred = (proba >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=[0, 1],
                                   target_names=["Normal", "Anomaly"], digits=3)
    print(f"\n  [{label}] threshold={threshold:.4f}")
    print(f"    ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f"    Confusion matrix:\n{cm}")
    print(report)
    return {"y_pred": y_pred, "roc_auc": roc_auc, "pr_auc": pr_auc,
            "confusion_matrix": cm, "report": report, "threshold": threshold}


# Entraînement par ocean/years_range

def train_lgbm_for_ocean(ocean, years_range, df_train, df_val, df_test, df_test_meta_extra, ts_reg, paths):
    output_dir = paths["output_dir"]

    all_available = [c for c in df_train.columns if c not in NON_FEATURE_COLS]
    X_train_full, y_train = df_train[all_available], df_train["is_bad"]
    X_val_full, y_val = df_val[all_available], df_val["is_bad"]
    X_test_full, y_test = df_test[all_available], df_test["is_bad"]

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
    experiment_name = f"LGBM_ReconAE_MultiOcean_{years_range}"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=f"file://{MLFLOW_DIR}/mlruns")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"LGBM_AE_{ocean}_{years_range}"):
        mlflow.log_params({
            "ocean": ocean, "years_range": years_range,
            "n_features_available": len(all_available),
            "feature_gain_threshold": FEATURE_GAIN_THRESHOLD,
            "min_features": MIN_FEATURES,
            "chosen_threshold_method": CHOSEN_THRESHOLD_METHOD,
            "n_train": len(df_train), "n_val": len(df_val), "n_test": len(df_test),
        })

        # 1) Ajustement exploratoire avec TOUTES les caractéristiques, uniquement pour mesurer le gain
        print(f"\n [{ocean}] Ajustement exploratoire (toutes les caractéristiques) pour mesurer le gain ")
        clf_explore = lgb.LGBMClassifier(
            n_estimators=500, random_state=SEED, n_jobs=4,
            class_weight="balanced", verbosity=-1,
        )
        clf_explore.fit(X_train_full, y_train)

        proba_test_base = clf_explore.predict_proba(X_test_full)[:, 1]
        y_pred_base = (proba_test_base >= 0.5).astype(int)
        print(f"  [référence, toutes les caractéristiques, thr=0.5] "
              f"ROC-AUC={roc_auc_score(y_test, proba_test_base):.4f}  "
              f"PR-AUC={average_precision_score(y_test, proba_test_base):.4f}")

        # 2) Sélection automatique des caractéristiques par gain cumulé
        FEATURE_COLS, fi_full = select_features_by_cumulative_gain(clf_explore, all_available)
        print(f"  Caractéristiques sélectionnées : {len(FEATURE_COLS)}/{len(all_available)} "
              f"(objectif {FEATURE_GAIN_THRESHOLD:.0%} de gain cumulé)")
        mlflow.log_text("\n".join(FEATURE_COLS), "features_selected.txt")
        mlflow.log_metric("n_features_selected", len(FEATURE_COLS))
        fi_full.to_csv(os.path.join(output_dir, "feature_importance_full.csv"), index=False)
        mlflow.log_artifact(os.path.join(output_dir, "feature_importance_full.csv"))

        X_train, X_val, X_test = X_train_full[FEATURE_COLS], X_val_full[FEATURE_COLS], X_test_full[FEATURE_COLS]

        # 3) HalvingRandomSearchCV UNIQUEMENT sur le sous-ensemble sélectionné
        print(f"\n [{ocean}] HalvingRandomSearchCV sur {len(FEATURE_COLS)} caractéristiques ")
        param_dist = {
            "n_estimators": [3000],
            "num_leaves": randint(15, 90),
            "max_depth": [3, 5, 7, -1],
            "learning_rate": [0.02, 0.05, 0.07, 0.1],
            "min_child_samples": randint(20, 100),
            "subsample": uniform(0.6, 0.35),
            "subsample_freq": [1],
            "colsample_bytree": uniform(0.5, 0.4),
            "reg_alpha": [0.5, 1, 2, 5, 10],
            "reg_lambda": [0.5, 1, 2, 5, 10],
            "class_weight": ["balanced"],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        search = HalvingRandomSearchCV(
            estimator=lgb.LGBMClassifier(random_state=SEED, n_jobs=4, verbosity=-1),
            param_distributions=param_dist, factor=3, n_candidates=50,
            min_resources="exhaust", cv=cv, scoring=SEARCH_SCORING,
            n_jobs=-1, verbose=1, random_state=SEED,
        )
        search.fit(X_train, y_train)
        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        mlflow.log_metric("cv_best_score", search.best_score_)

        best_params_final = {k: v for k, v in search.best_params_.items() if k != "n_estimators"}
        clf_opt = lgb.LGBMClassifier(**best_params_final, n_estimators=3000,
                                      random_state=SEED, n_jobs=4, verbosity=-1)
        clf_opt.fit(
            X_train, y_train, eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)],
        )
        mlflow.log_metric("best_iteration", clf_opt.best_iteration_)

        feat_importance = pd.DataFrame({
            "feature": clf_opt.booster_.feature_name(),
            "gain": clf_opt.booster_.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False).reset_index(drop=True)
        feat_importance.to_csv(os.path.join(output_dir, "feature_importance_selected.csv"), index=False)
        mlflow.log_artifact(os.path.join(output_dir, "feature_importance_selected.csv"))

        # 4) Baseline LogisticRegression (pour comparer avec LGBM)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train.fillna(0))
        X_test_sc = scaler.transform(X_test.fillna(0))
        logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
        logreg.fit(X_train_sc, y_train)
        proba_test_lr = logreg.predict_proba(X_test_sc)[:, 1]

        # 5) Probabilités du modèle final
        proba_train = clf_opt.predict_proba(X_train)[:, 1]
        proba_val = clf_opt.predict_proba(X_val)[:, 1]
        proba_test = clf_opt.predict_proba(X_test)[:, 1]

        # 6) Seuils (calculés sur val)
        thr_f1, f1_val = best_f1_threshold(y_val, proba_val)
        thr_youden, _ = youden_threshold(y_val, proba_val)
        thr_recall90, _ = recall_target_threshold(y_val, proba_val, TARGET_RECALL)

        chosen_threshold = {"f1": thr_f1, "youden": thr_youden, "recall90": thr_recall90}[CHOSEN_THRESHOLD_METHOD]
        mlflow.log_metric("thr_best_f1", thr_f1)
        mlflow.log_metric("thr_youden", thr_youden)
        mlflow.log_metric("thr_recall90", thr_recall90)
        mlflow.log_metric("chosen_threshold", chosen_threshold)

        # 7) Évaluation + classification_report sur train / val / test
        metrics_train = evaluate(y_train, proba_train, chosen_threshold, f"{ocean} / train")
        metrics_val = evaluate(y_val, proba_val, chosen_threshold, f"{ocean} / val")
        metrics_test = evaluate(y_test, proba_test, chosen_threshold, f"{ocean} / test")
        y_pred_final = metrics_test["y_pred"]

        mlflow.log_metric("test_roc_auc", metrics_test["roc_auc"])
        mlflow.log_metric("test_pr_auc", metrics_test["pr_auc"])
        mlflow.log_text(metrics_train["report"], "train_classification_report.txt")
        mlflow.log_text(metrics_val["report"], "val_classification_report.txt")
        mlflow.log_text(metrics_test["report"], "test_classification_report.txt")
        mlflow.lightgbm.log_model(clf_opt, artifact_path="model")

        # 8) Méta de test pour les graphiques (aligné par position avec df_test)
        df_test_meta_out = df_test[[
            "PLATFORM_NUMBER", "CYCLE_NUMBER", "LATITUDE", "LONGITUDE",
            "severity", "is_bad", "mse_total", "rmse_temp_overall", "rmse_psal_overall",
        ]].reset_index(drop=True).copy()
        if df_test_meta_extra is not None and len(df_test_meta_extra) == len(df_test_meta_out):
            extra_only = [c for c in df_test_meta_extra.columns if c not in df_test_meta_out.columns]
            df_test_meta_out = pd.concat(
                [df_test_meta_out, df_test_meta_extra[extra_only].reset_index(drop=True)], axis=1
            )
        else:
            print("  AVERTISSEMENT : test_meta_extra ne correspond pas en longueur à df_test, fusion ignorée")
        df_test_meta_out["proba_test"] = proba_test
        df_test_meta_out["y_pred_final"] = y_pred_final

        # 9) Sauvegarde pour analyse / graphiques ultérieurs
        joblib.dump({
            "ocean": ocean, "years_range": years_range,
            "feature_cols": FEATURE_COLS,
            "feature_importance_full": fi_full,
            "feature_importance": feat_importance,
            "X_test": X_test,
            "z_test": None,  # z n'est pas recalculé ici ; utiliser 01_build_features si nécessaire
            "mse_total_test": df_test["mse_total"].values,
            "rmse_temp_test": df_test["rmse_temp_overall"].values,
            "rmse_psal_test": df_test["rmse_psal_overall"].values,
            "y_test": y_test.values,
            "y_pred_final": y_pred_final,
            "y_pred_base": y_pred_base,
            "proba_train": proba_train,
            "proba_val": proba_val,
            "proba_test": proba_test,
            "proba_test_base": proba_test_base,
            "proba_test_logreg": proba_test_lr,
            "best_model": clf_opt,
            "df_test_meta": df_test_meta_out,
            "cv_results": pd.DataFrame(search.cv_results_),
            "best_params": search.best_params_,
            "thr_best_f1": thr_f1,
            "thr_youden": thr_youden,
            "thr_recall90": thr_recall90,
            "chosen_threshold": chosen_threshold,
            "metrics_train": metrics_train,
            "metrics_val": metrics_val,
            "metrics_test": metrics_test,
            "split_policy": "temporal",
            "ts_reg": ts_reg,
        }, os.path.join(output_dir, "analysis_data.pkl"))

        print(f"\n✓ [{ocean}/{years_range}] Test ROC-AUC={metrics_test['roc_auc']:.4f}  "
              f"PR-AUC={metrics_test['pr_auc']:.4f}  "
              f"(caractéristiques utilisées : {len(FEATURE_COLS)}/{len(all_available)}, "
              f"threshold={CHOSEN_THRESHOLD_METHOD}={chosen_threshold:.4f})")

        return {
            "ocean": ocean, "years_range": years_range,
            "n_features_selected": len(FEATURE_COLS),
            "n_features_available": len(all_available),
            "metrics_test": metrics_test,
        }


# PRINCIPAL — boucle sur oceans x years_range

def main():
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    summary = []

    for years_range in YEARS_RANGES:
        for ocean in OCEANS:
            paths = get_paths(ocean, years_range)
            train_path = os.path.join(paths["output_dir"], "anomaly_features_train.parquet")
            if not os.path.exists(train_path):
                print(f"\n AVERTISSEMENT : {train_path} n'existe pas, on saute {ocean}/{years_range} "
                      f"(avez-vous exécuté 01_build_features.py ?)")
                continue

            try:
                df_train, df_val, df_test, df_test_meta_extra, ts_reg, paths = load_data_for_ocean(
                    ocean, years_range
                )
                result = train_lgbm_for_ocean(
                    ocean, years_range, df_train, df_val, df_test, df_test_meta_extra, ts_reg, paths
                )
                summary.append(result)
            except Exception as e:
                print(f"\n ERREUR dans {ocean}/{years_range} : {e}")
                continue

    print(f"\n\nRÉSUMÉ FINAL\n")
    for r in summary:
        print(f"  {r['ocean']:<10} {r['years_range']:<12} "
              f"features={r['n_features_selected']}/{r['n_features_available']:<3}  "
              f"test ROC-AUC={r['metrics_test']['roc_auc']:.4f}  "
              f"PR-AUC={r['metrics_test']['pr_auc']:.4f}")


if __name__ == "__main__":
    main()