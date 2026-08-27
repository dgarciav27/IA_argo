"""
Training-speed benchmark on Argo data: Random Forest vs XGBoost vs LightGBM

Fixes comparable hyperparameters (same n_estimators, same max_depth, same n_jobs,
same random_state) across the 3 models and measures wall-clock training + predict
time, plus F1(anomalies)/ROC-AUC, for each ocean.

"""

import os, time
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

OCEANS = {
    "Atlantic": {
        "preprocess_dir": "/work/drgarcia/Dataset/atlantic_ocean/2018-2022/2.preprocessed/split_time_ascA_dmodeD",
    },
    "Indian": {
        "preprocess_dir": "/work/drgarcia/Dataset/indian_ocean/2018-2022/2.preprocessed/split_time_ascA_dmodeD",
    },
    "Pacific": {
        "preprocess_dir": "/work/drgarcia/Dataset/pacific_ocean/2018-2022/2.preprocessed/split_time_ascA_dmodeD",
    },
}

OUTPUT_CSV = "/work/drgarcia/Models_and_results/benchmark_train_time_comparison.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 200     # fixed and identical for all 3 models for fair comparison
MAX_DEPTH = 6           # fixed and identical for all 3 models
N_JOBS = 4              # fixed and identical for all 3 models


def _load_split(preprocess_dir, feature_cols, name):
    df = pd.read_parquet(os.path.join(preprocess_dir, f"{name}.parquet"))
    X = df[feature_cols].astype(np.float32)
    y = df["is_bad"]
    return X, y


def time_model(name, model, X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = model.predict_proba(X_test)[:, 1]
    predict_time = time.perf_counter() - t0

    pred = (proba >= 0.5).astype(int)
    f1_anom = f1_score(y_test, pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_test, proba)

    print(f"    {name:15s} | train: {train_time:7.2f}s | predict: {predict_time:6.3f}s "
          f"| F1(anom): {f1_anom:.3f} | F1(macro): {f1_macro:.3f} | ROC-AUC: {auc:.3f}")

    return dict(train_time=train_time, predict_time=predict_time,
                f1_anom=f1_anom, f1_macro=f1_macro, auc=auc)


def benchmark_ocean(ocean_name, preprocess_dir):
    print(f"\n{'#'*70}\n#  OCEAN: {ocean_name}\n{'#'*70}")

    feature_cols = joblib.load(os.path.join(preprocess_dir, "feature_cols.pkl"))
    X_train, y_train = _load_split(preprocess_dir, feature_cols, "train")
    X_test, y_test = _load_split(preprocess_dir, feature_cols, "test")

    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | "
          f"Features: {len(feature_cols)} | Positive rate: {y_train.mean():.1%} | "
          f"scale_pos_weight={imbalance_ratio:.2f}\n")

    results = {}

    # RANDOM FOREST
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        n_jobs=N_JOBS, random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    results["Random Forest"] = time_model("Random Forest", rf, X_train, y_train, X_test, y_test)

    # XGBOOST
    xgb_model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=imbalance_ratio,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=N_JOBS,
        tree_method="hist", verbosity=0,
    )
    results["XGBoost"] = time_model("XGBoost", xgb_model, X_train, y_train, X_test, y_test)

    # LIGHTGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=imbalance_ratio,
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
        verbosity=-1,
    )
    results["LightGBM"] = time_model("LightGBM", lgb_model, X_train, y_train, X_test, y_test)

    rows = []
    for model_name, r in results.items():
        rows.append({"ocean": ocean_name, "model": model_name, **r})
    return rows


def main():
    all_rows = []
    for ocean_name, cfg in OCEANS.items():
        rows = benchmark_ocean(ocean_name, cfg["preprocess_dir"])
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    print("\n" + "=" * 90)
    print("FULL RESULTS (all oceans)")
    print("=" * 90)
    print(df.to_string(index=False))

    print("\n" + "=" * 90)
    print("AVERAGE TRAINING TIME ACROSS THE 3 OCEANS")
    print("=" * 90)
    summary = df.groupby("model")[["train_time", "predict_time", "f1_anom", "f1_macro", "auc"]].mean()
    summary = summary.sort_values("train_time")
    summary["speedup_vs_slowest"] = summary["train_time"].max() / summary["train_time"]
    print(summary.round(3).to_string())

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved detailed results to: {OUTPUT_CSV}")

    fastest = summary.index[0]
    print(f"\n- Fastest to train on average: {fastest} "
          f"({summary.loc[fastest, 'speedup_vs_slowest']:.1f}x faster than the slowest)")


if __name__ == "__main__":
    main()