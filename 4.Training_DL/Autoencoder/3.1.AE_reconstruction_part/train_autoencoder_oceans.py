"""
Standalone profile reconstructor (autoencoder) — multi-ocean, parallel, MLflow-tracked.

Same model/training logic as the single-ocean version, but:
  - wrapped in train_one_ocean(...) so it can run independently per ocean
  - the 3 oceans run in parallel processes (ProcessPoolExecutor)
  - every run (per ocean) is logged to MLflow: params, per-epoch metrics,
    final RMSE/baseline metrics, and artifacts (checkpoint, norm stats, metrics pkl)

Each ocean keeps its own independent:
  - normalization stats (fit on that ocean's train-normal profiles)
  - model checkpoint
  - reconstructor_metrics.pkl
  - MLflow run (same experiment, run_name = f"AE-reconstructor-{ocean}")

Edit OCEAN_CONFIGS below to match your actual folder names/paths.
"""

import os
import json
import random
import time
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow


# Per-ocean paths 
OCEAN_CONFIGS = {
    "atlantic": {
        "preproc_dir": "/work/drgarcia/Dataset/DL_datasets/atlantic_ocean/2021_2025/split_time_ascA_dmodeD_masked_grid100",
        "output_dir":  "/work/drgarcia/Models_and_results/Autoencoder/reconstructor_atlantic_2021_2025",
    },
    "pacific": {
        "preproc_dir": "/work/drgarcia/Dataset/DL_datasets/pacific_ocean/2021_2025/split_time_ascA_dmodeD_masked_grid100",
        "output_dir":  "/work/drgarcia/Models_and_results/Autoencoder/reconstructor_pacific_2021_2025",
    },
    "indian": {
        "preproc_dir": "/work/drgarcia/Dataset/DL_datasets/indian_ocean/2021_2025/split_time_ascA_dmodeD_masked_grid100",
        "output_dir":  "/work/drgarcia/Models_and_results/Autoencoder/reconstructor_indian_2021_2025",
    },
}


# MLflow config
MLFLOW_DIR = "/home/drgarcia/Argo_ml_code/ML_flow"
EXPERIMENT_NAME = "AutoEncoder_Profile_Reconstructor_Oceans_2021_2025"
MLFLOW_LOG_RETRIES = 5          # retries on sqlite "database is locked"
MLFLOW_RETRY_SLEEP_S = 2.0

# Shared hyperparameters (same across oceans; kept as module-level
# constants so every run logs identical params unless overridden)
SEED          = 42
BATCH_SIZE    = 128
EPOCHS        = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
PATIENCE      = 20
LATENT_DIM    = 32
POLY_DEGREE   = 4
DEPTH_BANDS_TEMPLATE = [("surface", 0, 200), ("mid", 200, 1000), ("deep", 1000, 100000)]
SEVERITY_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


# Small MLflow helpers with retry (sqlite backend can lock under
# concurrent writes from parallel processes)
def mlflow_log_with_retry(fn, *args, **kwargs):
    last_err = None
    for attempt in range(MLFLOW_LOG_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if "locked" in str(e).lower():
                time.sleep(MLFLOW_RETRY_SLEEP_S)
                continue
            raise
    print(f"  [mlflow] giving up after {MLFLOW_LOG_RETRIES} retries: {last_err}")


def mlf_log_param(k, v):
    mlflow_log_with_retry(mlflow.log_param, k, v)

def mlf_log_metric(k, v, step=None):
    mlflow_log_with_retry(mlflow.log_metric, k, v, step=step)

def mlf_log_artifact(path):
    mlflow_log_with_retry(mlflow.log_artifact, path)


# ------------------------------------------------------------------
# Model (identical to single-ocean version)
# ------------------------------------------------------------------
class ProfileDataset(Dataset):
    def __init__(self, profiles, target, mask, labels):
        self.profiles = torch.from_numpy(profiles)
        self.target   = torch.from_numpy(target)
        self.mask     = torch.from_numpy(mask.astype(np.bool_))
        self.labels   = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        return self.profiles[idx], self.target[idx], self.mask[idx], self.labels[idx]


class ProfileReconstructor(nn.Module):
    """Pure 1D-CNN autoencoder. No auxiliary or classifier-only inputs -
    single objective: reconstruct T/S profiles well."""

    def __init__(self, n_channels_in=8, n_channels_out=2, seq_len=100, latent_dim=32):
        super().__init__()
        assert seq_len % 4 == 0
        self.seq_len_4 = seq_len // 4

        self.enc1 = nn.Conv1d(n_channels_in, 16, kernel_size=5, padding=2)
        self.bn_e1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn_e2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn_e3 = nn.BatchNorm1d(64)
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.fc_enc = nn.Linear(64, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 64 * self.seq_len_4)
        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn_d1 = nn.BatchNorm1d(32)
        self.deconv2 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn_d2 = nn.BatchNorm1d(16)
        self.out_conv = nn.Conv1d(16, n_channels_out, kernel_size=3, padding=1)

    def encode(self, x):
        h = torch.relu(self.bn_e1(self.enc1(x)))
        h = self.pool1(h)
        h = torch.relu(self.bn_e2(self.enc2(h)))
        h = self.pool2(h)
        h = torch.relu(self.bn_e3(self.enc3(h)))
        pooled = self.gpool(h).squeeze(-1)
        return self.fc_enc(pooled)

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 64, self.seq_len_4)
        h = torch.relu(self.bn_d1(self.deconv1(h)))
        h = torch.relu(self.bn_d2(self.deconv2(h)))
        return self.out_conv(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def recon_error_mean(recon, target, mask):
    diff = (recon - target) ** 2
    mask_f = mask.unsqueeze(1).float()
    diff = diff * mask_f
    valid_counts = mask.sum(dim=1).float().clamp(min=1) * 2.0
    return diff.sum(dim=(1, 2)) / valid_counts


def evaluate_loader(model, loader, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for x, target, mask, _ in loader:
            x, target, mask = x.to(device), target.to(device), mask.to(device)
            recon, _ = model(x)
            err = recon_error_mean(recon, target, mask)
            total_loss += err.sum().item()
            total_n += x.size(0)
    return total_loss / max(total_n, 1)


# ------------------------------------------------------------------
# Preprocessing helpers (parameterized, no module-level globals)
# ------------------------------------------------------------------
def load_split_raw(preproc_dir, name):
    X_temp = np.load(os.path.join(preproc_dir, f"{name}_X_temp.npy"))
    X_psal = np.load(os.path.join(preproc_dir, f"{name}_X_psal.npy"))
    mask   = np.load(os.path.join(preproc_dir, f"{name}_mask.npy"))
    meta   = pd.read_parquet(os.path.join(preproc_dir, f"{name}_meta.parquet"))
    labels = meta["is_bad"].values.astype(np.int8)
    return X_temp, X_psal, mask, meta, labels


def compute_profile_severity(meta):
    t = meta["PROFILE_TEMP_QC"].map(SEVERITY_ORDER)
    s = meta["PROFILE_PSAL_QC"].map(SEVERITY_ORDER)
    return np.where(t.notna() & s.notna(), np.maximum(t.values, s.values), np.nan)


def fit_per_level_stats(X, mask, idx_subset):
    Xs, Ms = X[idx_subset], mask[idx_subset]
    count = Ms.sum(axis=0).astype(np.float64)
    sum_  = np.where(Ms, Xs, 0.0).sum(axis=0, dtype=np.float64)
    mean  = sum_ / np.clip(count, 1, None)
    sq_sum = np.where(Ms, Xs.astype(np.float64) ** 2, 0.0).sum(axis=0)
    var = sq_sum / np.clip(count, 1, None) - mean ** 2
    std = np.sqrt(np.clip(var, 1e-12, None)) + 1e-6
    weak = count < 30
    if weak.any():
        global_mean = Xs[Ms].mean()
        global_std  = Xs[Ms].std() + 1e-6
        mean[weak] = global_mean
        std[weak]  = global_std
    return mean.astype(np.float32), std.astype(np.float32)


def compute_gradient_channels(X, mask, P_grid):
    dP = np.gradient(P_grid).astype(np.float32)
    dP = np.where(dP == 0, 1e-3, dP)
    dX = np.gradient(X, axis=1) / dP[None, :]
    d2X = np.gradient(dX, axis=1) / dP[None, :]
    dX = np.where(mask, dX, 0.0).astype(np.float32)
    d2X = np.where(mask, d2X, 0.0).astype(np.float32)
    return dX, d2X


def fit_gradient_stats(dX, d2X, mask, idx_subset):
    m = mask[idx_subset]
    d1, d2 = dX[idx_subset][m], d2X[idx_subset][m]
    return float(d1.mean()), float(d1.std() + 1e-6), float(d2.mean()), float(d2.std() + 1e-6)


def build_profile_tensor(X_temp, X_psal, mask, dT, d2T, dS, d2S,
                          P_norm_row, T_mean_L, T_std_L, S_mean_L, S_std_L,
                          dT_mu, dT_sd, d2T_mu, d2T_sd, dS_mu, dS_sd, d2S_mu, d2S_sd):
    T_norm = np.where(mask, (X_temp - T_mean_L[None, :]) / T_std_L[None, :], 0.0).astype(np.float32)
    S_norm = np.where(mask, (X_psal - S_mean_L[None, :]) / S_std_L[None, :], 0.0).astype(np.float32)
    P_norm = np.broadcast_to(P_norm_row, T_norm.shape).astype(np.float32)
    M_chan = mask.astype(np.float32)
    dT_norm  = np.where(mask, (dT  - dT_mu)  / dT_sd,  0.0).astype(np.float32)
    d2T_norm = np.where(mask, (d2T - d2T_mu) / d2T_sd, 0.0).astype(np.float32)
    dS_norm  = np.where(mask, (dS  - dS_mu)  / dS_sd,  0.0).astype(np.float32)
    d2S_norm = np.where(mask, (d2S - d2S_mu) / d2S_sd, 0.0).astype(np.float32)
    profiles = np.stack([T_norm, S_norm, P_norm, M_chan, dT_norm, d2T_norm, dS_norm, d2S_norm], axis=1)
    target = np.stack([T_norm, S_norm], axis=1)
    return profiles.astype(np.float32), target.astype(np.float32)


def denorm(vals, mean_L, std_L):
    return vals * std_L + mean_L


def rmse_denorm_by_class(recon, target, mask, y_true, T_mean_L, T_std_L, S_mean_L, S_std_L):
    temp_true = np.apply_along_axis(denorm, 1, target[:, 0, :], T_mean_L, T_std_L)
    temp_pred = np.apply_along_axis(denorm, 1, recon[:, 0, :], T_mean_L, T_std_L)
    psal_true = np.apply_along_axis(denorm, 1, target[:, 1, :], S_mean_L, S_std_L)
    psal_pred = np.apply_along_axis(denorm, 1, recon[:, 1, :], S_mean_L, S_std_L)
    m = mask.astype(bool)

    def per_profile_rmse(true_dn, pred_dn):
        diff2 = np.where(m, (true_dn - pred_dn) ** 2, np.nan)
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.nanmean(diff2, axis=1))

    rmse_temp = per_profile_rmse(temp_true, temp_pred)
    rmse_psal = per_profile_rmse(psal_true, psal_pred)

    def summarize(arr, cls_mask):
        vals = arr[cls_mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return {"mean": np.nan, "median": np.nan, "std": np.nan, "n": 0}
        return {"mean": float(vals.mean()), "median": float(np.median(vals)), "std": float(vals.std()), "n": int(len(vals))}

    summary = {
        "TEMP": {"normal": summarize(rmse_temp, y_true == 0), "anomaly": summarize(rmse_temp, y_true == 1)},
        "PSAL": {"normal": summarize(rmse_psal, y_true == 0), "anomaly": summarize(rmse_psal, y_true == 1)},
    }
    return rmse_temp, rmse_psal, summary


def rmse_by_depth_band(recon, target, mask, y_true, bands, pressure_grid):
    rows = []
    diff = (recon - target) ** 2
    for name, lo, hi in bands:
        band_level = (pressure_grid >= lo) & (pressure_grid < hi)
        band_mask = mask & band_level[None, :]
        band_mask_2ch = np.broadcast_to(band_mask[:, None, :], diff.shape)
        diff_band = np.where(band_mask_2ch, diff, 0.0)
        valid = band_mask.sum(axis=1).astype(np.float32) * 2.0
        err = diff_band.sum(axis=(1, 2)) / np.clip(valid, 1, None)
        err = np.where(valid > 0, err, np.nan)
        for cls_name, cls_mask in [("normal", y_true == 0), ("anomaly", y_true == 1)]:
            vals = err[cls_mask]
            vals = vals[~np.isnan(vals)]
            rows.append({"band": name, "class": cls_name,
                         "mse_mean": float(vals.mean()) if len(vals) else np.nan, "n": int(len(vals))})
    return pd.DataFrame(rows)


def level_mean_baseline_rmse(target, mask, y_true, T_mean_L, T_std_L, S_mean_L, S_std_L):
    zero_recon = np.zeros_like(target)
    return rmse_denorm_by_class(zero_recon, target, mask, y_true, T_mean_L, T_std_L, S_mean_L, S_std_L)


def poly_baseline_rmse(target, mask, y_true, P_norm_row, T_mean_L, T_std_L, S_mean_L, S_std_L, degree=POLY_DEGREE):
    N, _, L = target.shape
    recon = np.zeros_like(target)
    p_norm = P_norm_row
    for i in range(N):
        m = mask[i]
        if m.sum() < degree + 2:
            continue
        for ch in range(2):
            coeffs = np.polyfit(p_norm[m], target[i, ch, m], deg=degree)
            recon[i, ch, :] = np.polyval(coeffs, p_norm)
    return rmse_denorm_by_class(recon, target, mask, y_true, T_mean_L, T_std_L, S_mean_L, S_std_L)


def variance_explained_by_level(recon, target, mask, y_true, pressure_grid):
    idx = (y_true == 0)
    diff2 = (recon[idx] - target[idx]) ** 2
    m = mask[idx]
    rows = []
    for ch, ch_name in [(0, "TEMP"), (1, "PSAL")]:
        for lvl in range(target.shape[2]):
            valid = m[:, lvl]
            if valid.sum() < 30:
                rows.append({"channel": ch_name, "level_idx": lvl, "pressure": float(pressure_grid[lvl]), "r2": np.nan})
                continue
            mse = diff2[valid, ch, lvl].mean()
            var = target[idx][valid, ch, lvl].var()
            r2 = 1.0 - mse / max(var, 1e-12)
            rows.append({"channel": ch_name, "level_idx": lvl, "pressure": float(pressure_grid[lvl]), "r2": float(r2)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Main per-ocean training routine (runs in its own process)
# ------------------------------------------------------------------
def train_one_ocean(ocean_name, preproc_dir, output_dir, device_str, mlflow_tracking_uri, experiment_name):
    try:
        print(f"[{ocean_name}] starting on device={device_str}")

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seed_generator = torch.Generator()
        seed_generator.manual_seed(SEED)

        device = torch.device(device_str)
        os.makedirs(output_dir, exist_ok=True)
        norm_stats_path = os.path.join(output_dir, "norm_stats_per_level.npz")
        metrics_path    = os.path.join(output_dir, "reconstructor_metrics.pkl")
        best_model_path = os.path.join(output_dir, "best_reconstructor.pth")
        loss_curve_path = os.path.join(output_dir, "loss_curve.png")

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        exp = mlflow.get_experiment_by_name(experiment_name)
        exp_id = (
            mlflow.create_experiment(name=experiment_name, artifact_location=f"file://{MLFLOW_DIR}/mlruns")
            if exp is None else exp.experiment_id
        )
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"AE-reconstructor-{ocean_name}"):
            mlf_log_param("ocean", ocean_name)
            mlf_log_param("device", device_str)
            mlf_log_param("batch_size", BATCH_SIZE)
            mlf_log_param("epochs_max", EPOCHS)
            mlf_log_param("learning_rate", LEARNING_RATE)
            mlf_log_param("weight_decay", WEIGHT_DECAY)
            mlf_log_param("patience", PATIENCE)
            mlf_log_param("latent_dim", LATENT_DIM)
            mlf_log_param("poly_degree", POLY_DEGREE)
            mlf_log_param("seed", SEED)
            mlf_log_param("preproc_dir", preproc_dir)

            # 1. Load preprocessed partitions
            with open(os.path.join(preproc_dir, "norm_stats.json")) as f:
                norm_stats_original = json.load(f)
            PRESSURE_GRID = np.load(os.path.join(preproc_dir, "pressure_grid.npy"))
            N_VERT_LEVELS = len(PRESSURE_GRID)
            GRID_MAX_PRES = norm_stats_original["PRES"]["grid_max"]
            P_NORM_ROW = (PRESSURE_GRID / GRID_MAX_PRES).astype(np.float32)
            DEPTH_BANDS = [(name, lo, min(hi, float(PRESSURE_GRID.max()) + 1.0)) for name, lo, hi in DEPTH_BANDS_TEMPLATE]

            X_temp_train, X_psal_train, mask_train, meta_train, labels_train = load_split_raw(preproc_dir, "train")
            X_temp_val,   X_psal_val,   mask_val,   meta_val,   labels_val   = load_split_raw(preproc_dir, "val")
            X_temp_test,  X_psal_test,  mask_test,  meta_test,  labels_test  = load_split_raw(preproc_dir, "test")

            idx_normal_train = (labels_train == 0)
            idx_normal_val   = (labels_val == 0)

            print(f"[{ocean_name}] Train: {len(labels_train):,} ({idx_normal_train.sum():,} normal) | "
                  f"Val: {len(labels_val):,} ({idx_normal_val.sum():,} normal) | "
                  f"Test: {len(labels_test):,} ({(labels_test == 0).sum():,} normal)")
            mlf_log_param("n_train", int(len(labels_train)))
            mlf_log_param("n_val", int(len(labels_val)))
            mlf_log_param("n_test", int(len(labels_test)))

            sev_train = compute_profile_severity(meta_train)
            sev_val   = compute_profile_severity(meta_val)
            sev_test  = compute_profile_severity(meta_test)

            # 2. Normalization stats (train, normal only)
            T_mean_L, T_std_L = fit_per_level_stats(X_temp_train, mask_train, idx_normal_train)
            S_mean_L, S_std_L = fit_per_level_stats(X_psal_train, mask_train, idx_normal_train)

            dT_train, d2T_train = compute_gradient_channels(X_temp_train, mask_train, PRESSURE_GRID)
            dS_train, d2S_train = compute_gradient_channels(X_psal_train, mask_train, PRESSURE_GRID)
            dT_val,   d2T_val   = compute_gradient_channels(X_temp_val,   mask_val,   PRESSURE_GRID)
            dS_val,   d2S_val   = compute_gradient_channels(X_psal_val,   mask_val,   PRESSURE_GRID)
            dT_test,  d2T_test  = compute_gradient_channels(X_temp_test,  mask_test,  PRESSURE_GRID)
            dS_test,  d2S_test  = compute_gradient_channels(X_psal_test,  mask_test,  PRESSURE_GRID)

            dT_mu, dT_sd, d2T_mu, d2T_sd = fit_gradient_stats(dT_train, d2T_train, mask_train, idx_normal_train)
            dS_mu, dS_sd, d2S_mu, d2S_sd = fit_gradient_stats(dS_train, d2S_train, mask_train, idx_normal_train)

            np.savez_compressed(
                norm_stats_path,
                pressure_grid=PRESSURE_GRID,
                T_mean_L=T_mean_L, T_std_L=T_std_L, S_mean_L=S_mean_L, S_std_L=S_std_L,
                grid_max_pres=np.array([GRID_MAX_PRES]),
                dT_mu=dT_mu, dT_sd=dT_sd, d2T_mu=d2T_mu, d2T_sd=d2T_sd,
                dS_mu=dS_mu, dS_sd=dS_sd, d2S_mu=d2S_mu, d2S_sd=d2S_sd,
            )
            print(f"[{ocean_name}] Saved norm stats -> {norm_stats_path}")

            build_kwargs = dict(P_norm_row=P_NORM_ROW, T_mean_L=T_mean_L, T_std_L=T_std_L,
                                 S_mean_L=S_mean_L, S_std_L=S_std_L,
                                 dT_mu=dT_mu, dT_sd=dT_sd, d2T_mu=d2T_mu, d2T_sd=d2T_sd,
                                 dS_mu=dS_mu, dS_sd=dS_sd, d2S_mu=d2S_mu, d2S_sd=d2S_sd)

            profiles_train, target_train = build_profile_tensor(X_temp_train, X_psal_train, mask_train, dT_train, d2T_train, dS_train, d2S_train, **build_kwargs)
            profiles_val,   target_val   = build_profile_tensor(X_temp_val,   X_psal_val,   mask_val,   dT_val,   d2T_val,   dS_val,   d2S_val,   **build_kwargs)
            profiles_test,  target_test  = build_profile_tensor(X_temp_test,  X_psal_test,  mask_test,  dT_test,  d2T_test,  dS_test,  d2S_test,  **build_kwargs)

            # 3. Datasets / loaders
            train_dataset = ProfileDataset(profiles_train[idx_normal_train], target_train[idx_normal_train],
                                            mask_train[idx_normal_train], labels_train[idx_normal_train])
            val_normal_dataset = ProfileDataset(profiles_val[idx_normal_val], target_val[idx_normal_val],
                                                 mask_val[idx_normal_val], labels_val[idx_normal_val])
            val_all_dataset = ProfileDataset(profiles_val, target_val, mask_val, labels_val)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                       num_workers=2, pin_memory=True, generator=seed_generator)
            val_normal_loader = DataLoader(val_normal_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            val_all_loader    = DataLoader(val_all_dataset,    batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

            model = ProfileReconstructor(n_channels_in=8, n_channels_out=2, seq_len=N_VERT_LEVELS, latent_dim=LATENT_DIM).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[{ocean_name}] Model on {device} - {n_params:,} params")
            mlf_log_param("n_params", int(n_params))

            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

            train_losses, val_normal_losses, val_all_losses = [], [], []
            best_val_normal_loss = float("inf")
            patience_counter = 0
            stopped_epoch = EPOCHS

            # 4. Training loop
            for epoch in range(EPOCHS):
                model.train()
                total_loss, total_n = 0.0, 0
                for x, target, mask, _ in train_loader:
                    x, target, mask = x.to(device), target.to(device), mask.to(device)
                    optimizer.zero_grad()
                    recon, _ = model(x)
                    loss = recon_error_mean(recon, target, mask).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item() * x.size(0)
                    total_n += x.size(0)

                train_loss = total_loss / max(total_n, 1)
                val_normal_loss = evaluate_loader(model, val_normal_loader, device)
                val_all_loss = evaluate_loader(model, val_all_loader, device)

                train_losses.append(train_loss)
                val_normal_losses.append(val_normal_loss)
                val_all_losses.append(val_all_loss)

                current_lr = optimizer.param_groups[0]["lr"]
                mlf_log_metric("train_mse", train_loss, step=epoch)
                mlf_log_metric("val_normal_mse", val_normal_loss, step=epoch)
                mlf_log_metric("val_all_mse_diagnostic", val_all_loss, step=epoch)
                mlf_log_metric("lr", current_lr, step=epoch)

                if (epoch + 1) % 5 == 0:
                    print(f"[{ocean_name}] Epoch {epoch+1:3d}/{EPOCHS} | Train {train_loss:.5f} | "
                          f"Val-normal {val_normal_loss:.5f} | Val-all(diag) {val_all_loss:.5f} | LR {current_lr:.2e}")

                scheduler.step(val_normal_loss)
                if val_normal_loss < best_val_normal_loss:
                    best_val_normal_loss = val_normal_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"[{ocean_name}] Early stopping at epoch {epoch + 1}")
                        stopped_epoch = epoch + 1
                        break

            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"[{ocean_name}] Best val-normal MSE: {best_val_normal_loss:.5f}")
            mlf_log_metric("best_val_normal_mse", best_val_normal_loss)
            mlf_log_param("stopped_epoch", stopped_epoch)

            # loss curve artifact
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(train_losses, label="train")
            ax.plot(val_normal_losses, label="val (normal-only)")
            ax.plot(val_all_losses, label="val (all, diagnostic)", alpha=0.5, linestyle="--")
            ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title(f"Loss curve — {ocean_name}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(loss_curve_path, dpi=150)
            plt.close(fig)

            # 5. Full reconstructions
            model.eval()
            with torch.no_grad():
                recon_train, z_train = model(torch.from_numpy(profiles_train).to(device))
                recon_val,   z_val   = model(torch.from_numpy(profiles_val).to(device))
                recon_test,  z_test  = model(torch.from_numpy(profiles_test).to(device))
                recon_train = recon_train.cpu().numpy()
                recon_val   = recon_val.cpu().numpy()
                recon_test  = recon_test.cpu().numpy()

            # 6. Evaluation vs baselines
            results = {}
            for split_name, recon, target, mask, y_true in [
                ("train", recon_train, target_train, mask_train, labels_train),
                ("val",   recon_val,   target_val,   mask_val,   labels_val),
                ("test",  recon_test,  target_test,  mask_test,  labels_test),
            ]:
                ae_rmse_t, ae_rmse_s, ae_summary = rmse_denorm_by_class(recon, target, mask, y_true, T_mean_L, T_std_L, S_mean_L, S_std_L)
                _, _, mean_baseline_summary = level_mean_baseline_rmse(target, mask, y_true, T_mean_L, T_std_L, S_mean_L, S_std_L)
                _, _, poly_baseline_summary = poly_baseline_rmse(target, mask, y_true, P_NORM_ROW, T_mean_L, T_std_L, S_mean_L, S_std_L)
                band_df = rmse_by_depth_band(recon, target, mask, y_true, DEPTH_BANDS, PRESSURE_GRID)
                var_expl_df = variance_explained_by_level(recon, target, mask, y_true, PRESSURE_GRID)

                results[split_name] = {
                    "ae_rmse_temp": ae_rmse_t, "ae_rmse_psal": ae_rmse_s, "ae_summary": ae_summary,
                    "level_mean_baseline_summary": mean_baseline_summary,
                    "poly_baseline_summary": poly_baseline_summary,
                    "rmse_by_depth_band": band_df,
                    "variance_explained_by_level": var_expl_df,
                }

                print(f"[{ocean_name}][{split_name}] normal profiles, RMSE denormalized")
                for var in ["TEMP", "PSAL"]:
                    ae_n = ae_summary[var]["normal"]["mean"]
                    mean_n = mean_baseline_summary[var]["normal"]["mean"]
                    poly_n = poly_baseline_summary[var]["normal"]["mean"]
                    print(f"    {var:4s} | AE={ae_n:.4f} | level-mean={mean_n:.4f} | poly-fit={poly_n:.4f}")
                    mlf_log_metric(f"{split_name}_{var.lower()}_rmse_ae_normal", ae_n)
                    mlf_log_metric(f"{split_name}_{var.lower()}_rmse_levelmean_normal", mean_n)
                    mlf_log_metric(f"{split_name}_{var.lower()}_rmse_poly_normal", poly_n)
                    ae_a = ae_summary[var]["anomaly"]["mean"]
                    if not np.isnan(ae_a):
                        mlf_log_metric(f"{split_name}_{var.lower()}_rmse_ae_anomaly", ae_a)

            results["train_losses"] = np.array(train_losses)
            results["val_normal_losses"] = np.array(val_normal_losses)
            results["val_all_losses_diagnostic"] = np.array(val_all_losses)
            results["best_val_normal_loss"] = best_val_normal_loss
            results["latent_dim"] = LATENT_DIM
            results["depth_bands"] = DEPTH_BANDS
            results["poly_degree"] = POLY_DEGREE
            results["ocean"] = ocean_name

            joblib.dump(results, metrics_path)
            print(f"[{ocean_name}] Saved metrics -> {metrics_path}")
            print(f"[{ocean_name}] Saved model   -> {best_model_path}")

            mlf_log_artifact(best_model_path)
            mlf_log_artifact(norm_stats_path)
            mlf_log_artifact(metrics_path)
            mlf_log_artifact(loss_curve_path)

        return {"ocean": ocean_name, "status": "ok", "best_val_normal_loss": best_val_normal_loss}

    except Exception as e:
        traceback.print_exc()
        return {"ocean": ocean_name, "status": "failed", "error": str(e)}


# ------------------------------------------------------------------
# Launch all oceans in parallel
# ------------------------------------------------------------------
def assign_device(idx):
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return "cpu"
    return f"cuda:{idx % n_gpus}"


def main():
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    tracking_uri = f"sqlite:///{MLFLOW_DIR}/mlflow.db"

    n_gpus = torch.cuda.device_count()
    n_oceans = len(OCEAN_CONFIGS)

    # Paralelizamos solo si hay >=1 GPU por océano (o al menos 2 GPUs libres).
    # Si hay 1 sola GPU (o CPU), corremos secuencial para evitar OOM / contención.
    run_parallel = n_gpus >= n_oceans

    print(f"GPUs detectadas: {n_gpus} | Océanos: {n_oceans} | "
          f"Modo: {'PARALLEL' if run_parallel else 'SECUENCIAL'}")

    if run_parallel:
        ctx = multiprocessing.get_context("spawn")
        futures = {}
        with ProcessPoolExecutor(max_workers=n_oceans, mp_context=ctx) as executor:
            for idx, (ocean_name, cfg) in enumerate(OCEAN_CONFIGS.items()):
                device_str = assign_device(idx)
                fut = executor.submit(
                    train_one_ocean,
                    ocean_name, cfg["preproc_dir"], cfg["output_dir"],
                    device_str, tracking_uri, EXPERIMENT_NAME,
                )
                futures[fut] = ocean_name

            for fut in as_completed(futures):
                ocean_name = futures[fut]
                result = fut.result()
                print(f"\n=== [{ocean_name}] finished: {result} ===")

    else:
        # Secuencial: mismo device para todos (o CPU si no hay GPU)
        device_str = "cuda:0" if n_gpus >= 1 else "cpu"
        for ocean_name, cfg in OCEAN_CONFIGS.items():
            print(f"\n--- Iniciando {ocean_name} en {device_str} ---")
            result = train_one_ocean(
                ocean_name, cfg["preproc_dir"], cfg["output_dir"],
                device_str, tracking_uri, EXPERIMENT_NAME,
            )
            print(f"\n=== [{ocean_name}] finished: {result} ===")

if __name__ == "__main__":
    main()