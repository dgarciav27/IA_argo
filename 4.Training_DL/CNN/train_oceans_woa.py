"""
CNN Multi-Ocean (Atlantic / Pacific / Indian) — WOA23 + Halving Search + K-Fold Confirmation
================================================================================

Design choices:
  1. Multi-océano independiente: datos, halving search, modelo final, output dir propios
  2. WOA23 climatology: cargada UNA VEZ, reutilizada para 3 océanos
  3. Halving Search: fast ranking de candidatos
  4. K-Fold Confirmation: solo sobre mejor config, 5-split sobre train+val, eval en test
  5. MLflow: un experimento, un run por océano, logs de params/metrics/artifacts
  6. Salida: loss curves (train/val), classification reports (train/val/test), predicciones
"""

import os
import gc
import json
import random
import traceback
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score, classification_report,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# PATHS / CONFIG
PREPROC_BASE = "/work/drgarcia/Dataset/DL_datasets"
YEARS_RANGE = "2018_2022"

OCEANS = {
    "atlantic": os.path.join(PREPROC_BASE, f"atlantic_ocean/{YEARS_RANGE}/split_time_ascA_dmodeD_masked_grid100"),
    "pacific": os.path.join(PREPROC_BASE, f"pacific_ocean/{YEARS_RANGE}/split_time_ascA_dmodeD_masked_grid100"),
    "indian": os.path.join(PREPROC_BASE, f"indian_ocean/{YEARS_RANGE}/split_time_ascA_dmodeD_masked_grid100"),
}

OUTPUT_DIR_BASE = "/work/drgarcia/Models_and_results/CNN_MultiOcean2/2018CNN_MultiOcean_WOA_vf"
WOA_DIR = r"/work/drgarcia/Dataset/WOA"
MLFLOW_DIR = "/home/drgarcia/Argo_ml_code/ML_flow"

MLFLOW_ENABLED = True
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# K-FOLD CONFIG (solo para mejor modelo)
RUN_KFOLD_CONFIRMATION = True
KFOLD_N_SPLITS = 5
KFOLD_MAX_EPOCHS = 60
KFOLD_PATIENCE = 15

# V2 BASE CONFIG (estable, no buscado)
V2_BASE = {
    "scheduler_metric": "f1_macro",
    "use_weighted_sampler": False,
    "use_pos_weight": True,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "context_hidden": 16,
    "seq_len": 100,
}

# Halving Search: pequeño y rápido, solo ranking de candidatos
HALVING_CONFIG = {
    "n_candidates": 6,
    "n_iterations": 2,
    "max_epochs_per_round": [25, 35],
    "patience_per_round": [6, 10],
    "param_distributions": {
        "label_smoothing": [0.02, 0.05],
        "dropout_rate": [0.15, 0.2],
        "kernel_size": [3, 5, 7],
        "conv1_filters": [32, 48],
        "conv2_filters": [64, 96],
        "conv3_filters": [128, 192],
        "pos_weight_boost": [1.0, 1.3, 1.6, 2.0],
    },
}

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# DATASET / MODEL
class OceanProfileDataset(Dataset):
    def __init__(self, X_profiles, context, y_labels):
        self.X = torch.from_numpy(X_profiles).float()
        self.context = torch.from_numpy(context).float()
        self.y = torch.from_numpy(y_labels).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.context[idx], self.y[idx]


class ProfileAnomalyCNN(nn.Module):
    def __init__(self, n_channels_in=10, seq_len=100, n_filters=(32, 64, 128),
                 kernel_size=5, dropout_rate=0.2, context_dim=5, context_hidden=16):
        super().__init__()
        assert seq_len % 4 == 0
        self.context_dim = context_dim

        self.conv1 = nn.Conv1d(n_channels_in, n_filters[0], kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(n_filters[0])
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(n_filters[1])
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(n_filters[2])

        self.gap = nn.AdaptiveAvgPool1d(1)

        if context_dim > 0:
            self.context_mlp = nn.Sequential(
                nn.Linear(context_dim, context_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
            )
            fc1_in = n_filters[2] + context_hidden
        else:
            self.context_mlp = None
            fc1_in = n_filters[2]

        self.fc1 = nn.Linear(fc1_in, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x, context=None):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)

        if self.context_mlp is not None and context is not None and context.shape[1] > 0:
            c = self.context_mlp(context)
            x = torch.cat([x, c], dim=1)

        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc_out(x)


# DATA LOADING + PREPROCESSING

def load_preprocessed_data(preproc_dir):
    print("  Cargando datos preprocesados...")
    train_temp = np.load(os.path.join(preproc_dir, "train_X_temp.npy"))
    train_psal = np.load(os.path.join(preproc_dir, "train_X_psal.npy"))
    train_mask = np.load(os.path.join(preproc_dir, "train_mask.npy"))
    train_meta = pd.read_parquet(os.path.join(preproc_dir, "train_meta.parquet"))

    val_temp = np.load(os.path.join(preproc_dir, "val_X_temp.npy"))
    val_psal = np.load(os.path.join(preproc_dir, "val_X_psal.npy"))
    val_mask = np.load(os.path.join(preproc_dir, "val_mask.npy"))
    val_meta = pd.read_parquet(os.path.join(preproc_dir, "val_meta.parquet"))

    test_temp = np.load(os.path.join(preproc_dir, "test_X_temp.npy"))
    test_psal = np.load(os.path.join(preproc_dir, "test_X_psal.npy"))
    test_mask = np.load(os.path.join(preproc_dir, "test_mask.npy"))
    test_meta = pd.read_parquet(os.path.join(preproc_dir, "test_meta.parquet"))

    pressure_grid = np.load(os.path.join(preproc_dir, "pressure_grid.npy"))

    print(f"    Train: {train_temp.shape[0]}  Anomalías: {train_meta['is_bad'].sum()}")
    print(f"    Val:   {val_temp.shape[0]}  Anomalías: {val_meta['is_bad'].sum()}")
    print(f"    Test:  {test_temp.shape[0]}  Anomalías: {test_meta['is_bad'].sum()}")

    return (train_temp, train_psal, train_mask, train_meta,
            val_temp, val_psal, val_mask, val_meta,
            test_temp, test_psal, test_mask, test_meta,
            pressure_grid)


def compute_gradient_channels(X, mask, P_grid):
    dP = np.gradient(P_grid).astype(np.float32)
    dP = np.where(dP == 0, 1e-3, dP)
    dX = np.gradient(X, axis=1) / dP[None, :]
    d2X = np.gradient(dX, axis=1) / dP[None, :]
    return (np.where(mask, dX, 0.0).astype(np.float32),
            np.where(mask, d2X, 0.0).astype(np.float32))


def compute_normalization_stats(X_temp, X_psal, mask, pressure_grid):
    T_mean_L = X_temp.mean(axis=0)
    T_std_L = X_temp.std(axis=0) + 1e-8
    S_mean_L = X_psal.mean(axis=0)
    S_std_L = X_psal.std(axis=0) + 1e-8

    dT, d2T = compute_gradient_channels(X_temp, mask, pressure_grid)
    dT_mu, dT_sd = float(dT[mask].mean()), float(dT[mask].std() + 1e-8)
    d2T_mu, d2T_sd = float(d2T[mask].mean()), float(d2T[mask].std() + 1e-8)

    dS, d2S = compute_gradient_channels(X_psal, mask, pressure_grid)
    dS_mu, dS_sd = float(dS[mask].mean()), float(dS[mask].std() + 1e-8)
    d2S_mu, d2S_sd = float(d2S[mask].mean()), float(d2S[mask].std() + 1e-8)

    return dict(T_mean_L=T_mean_L, T_std_L=T_std_L, S_mean_L=S_mean_L, S_std_L=S_std_L,
                dT_mu=dT_mu, dT_sd=dT_sd, d2T_mu=d2T_mu, d2T_sd=d2T_sd,
                dS_mu=dS_mu, dS_sd=dS_sd, d2S_mu=d2S_mu, d2S_sd=d2S_sd)


def build_channel_input(X_temp, X_psal, mask, P_grid, norm,
                         resid_T=None, resid_S=None,
                         resid_T_mu=0.0, resid_T_sd=1.0,
                         resid_S_mu=0.0, resid_S_sd=1.0):
    dT, d2T = compute_gradient_channels(X_temp, mask, P_grid)
    dS, d2S = compute_gradient_channels(X_psal, mask, P_grid)

    GRID_MAX_PRES = P_grid.max()
    P_norm_row = (P_grid / GRID_MAX_PRES).astype(np.float32)

    T_norm = np.where(mask, (X_temp - norm["T_mean_L"][None, :]) / norm["T_std_L"][None, :], 0.0).astype(np.float32)
    S_norm = np.where(mask, (X_psal - norm["S_mean_L"][None, :]) / norm["S_std_L"][None, :], 0.0).astype(np.float32)
    P_norm = np.broadcast_to(P_norm_row, T_norm.shape).astype(np.float32)
    M_chan = mask.astype(np.float32)
    dT_norm = np.where(mask, (dT - norm["dT_mu"]) / norm["dT_sd"], 0.0).astype(np.float32)
    d2T_norm = np.where(mask, (d2T - norm["d2T_mu"]) / norm["d2T_sd"], 0.0).astype(np.float32)
    dS_norm = np.where(mask, (dS - norm["dS_mu"]) / norm["dS_sd"], 0.0).astype(np.float32)
    d2S_norm = np.where(mask, (d2S - norm["d2S_mu"]) / norm["d2S_sd"], 0.0).astype(np.float32)

    channels = [T_norm, S_norm, P_norm, M_chan, dT_norm, d2T_norm, dS_norm, d2S_norm]

    if resid_T is not None:
        resid_T_norm = np.where(mask, (resid_T - resid_T_mu) / resid_T_sd, 0.0).astype(np.float32)
        resid_S_norm = np.where(mask, (resid_S - resid_S_mu) / resid_S_sd, 0.0).astype(np.float32)
        channels += [resid_T_norm, resid_S_norm]

    return np.stack(channels, axis=1).astype(np.float32)


# WOA23 CLIMATOLOGY (cargada UNA VEZ, compartida)

def compute_residual_channels_woa(X_temp, X_psal, mask, meta, pressure_grid, woa_ds):
    from scipy.interpolate import interp1d
    N, L = X_temp.shape
    lat = meta["LATITUDE"].values
    lon = meta["LONGITUDE"].values
    month = pd.to_datetime(meta["date"]).dt.month.values

    resid_T = np.zeros((N, L), dtype=np.float32)
    resid_S = np.zeros((N, L), dtype=np.float32)
    cache = {}
    for i in range(N):
        key = (round(lat[i] * 2) / 2, round(lon[i] * 2) / 2, int(month[i]))
        if key not in cache:
            try:
                point = woa_ds.sel(lat=lat[i], lon=lon[i], month=int(month[i]), method="nearest")
                depth = point["depth"].values
                t_an, s_an = point["t_an"].values, point["s_an"].values
                valid = ~np.isnan(t_an) & ~np.isnan(s_an)
                if valid.sum() < 3:
                    cache[key] = (np.full(L, np.nan, np.float32), np.full(L, np.nan, np.float32))
                else:
                    f_t = interp1d(depth[valid], t_an[valid], bounds_error=False, fill_value=np.nan)
                    f_s = interp1d(depth[valid], s_an[valid], bounds_error=False, fill_value=np.nan)
                    cache[key] = (f_t(pressure_grid).astype(np.float32), f_s(pressure_grid).astype(np.float32))
            except Exception:
                cache[key] = (np.full(L, np.nan, np.float32), np.full(L, np.nan, np.float32))
        clim_t, clim_s = cache[key]
        resid_T[i] = X_temp[i] - clim_t
        resid_S[i] = X_psal[i] - clim_s

    resid_T = np.where(mask, np.nan_to_num(resid_T, nan=0.0), 0.0).astype(np.float32)
    resid_S = np.where(mask, np.nan_to_num(resid_S, nan=0.0), 0.0).astype(np.float32)
    return resid_T, resid_S


def load_woa_climatology(woa_dir):
    import xarray as xr
    temp_list, psal_list = [], []
    for m in range(1, 13):
        f_t = os.path.join(woa_dir, f"woa23_decav_t{m:02d}_01.nc")
        f_s = os.path.join(woa_dir, f"woa23_decav_s{m:02d}_01.nc")
        ds_t = xr.open_dataset(f_t, decode_times=False)[["t_an"]].squeeze("time", drop=True).expand_dims(month=[m])
        ds_s = xr.open_dataset(f_s, decode_times=False)[["s_an"]].squeeze("time", drop=True).expand_dims(month=[m])
        temp_list.append(ds_t)
        psal_list.append(ds_s)
    return xr.merge([xr.concat(temp_list, dim="month"), xr.concat(psal_list, dim="month")]).load()


def build_context_features(meta):
    lat = meta["LATITUDE"].values.astype(np.float32)
    lon = meta["LONGITUDE"].values.astype(np.float32)
    month = pd.to_datetime(meta["date"]).dt.month.values.astype(np.float32)

    lat_norm = lat / 90.0
    lon_rad = np.deg2rad(lon)
    lon_sin, lon_cos = np.sin(lon_rad), np.cos(lon_rad)
    month_rad = 2 * np.pi * month / 12.0
    month_sin, month_cos = np.sin(month_rad), np.cos(month_rad)
    return np.stack([lat_norm, lon_sin, lon_cos, month_sin, month_cos], axis=1).astype(np.float32)


def prepare_woa_variant(raw_data, norm, woa_ds):
    """Builds X_train/val/test (10 channels) + ctx_train/val/test (5 features)"""
    (train_temp, train_psal, train_mask, train_meta,
     val_temp, val_psal, val_mask, val_meta,
     test_temp, test_psal, test_mask, test_meta,
     pressure_grid) = raw_data

    resid_train = compute_residual_channels_woa(train_temp, train_psal, train_mask, train_meta, pressure_grid, woa_ds)
    resid_val = compute_residual_channels_woa(val_temp, val_psal, val_mask, val_meta, pressure_grid, woa_ds)
    resid_test = compute_residual_channels_woa(test_temp, test_psal, test_mask, test_meta, pressure_grid, woa_ds)

    resid_T_mu = float(resid_train[0][train_mask].mean())
    resid_T_sd = float(resid_train[0][train_mask].std() + 1e-8)
    resid_S_mu = float(resid_train[1][train_mask].mean())
    resid_S_sd = float(resid_train[1][train_mask].std() + 1e-8)

    def make(X_temp, X_psal, mask, resid):
        return build_channel_input(X_temp, X_psal, mask, pressure_grid, norm,
                                    resid_T=resid[0], resid_S=resid[1],
                                    resid_T_mu=resid_T_mu, resid_T_sd=resid_T_sd,
                                    resid_S_mu=resid_S_mu, resid_S_sd=resid_S_sd)

    X_train = make(train_temp, train_psal, train_mask, resid_train)
    X_val = make(val_temp, val_psal, val_mask, resid_val)
    X_test = make(test_temp, test_psal, test_mask, resid_test)

    ctx_train = build_context_features(train_meta)
    ctx_val = build_context_features(val_meta)
    ctx_test = build_context_features(test_meta)

    return X_train, ctx_train, X_val, ctx_val, X_test, ctx_test, 10, 5


# ============================================================================
# TRAIN / EVAL HELPERS
# ============================================================================

def compute_class_weights(y_labels):
    n_normal = (y_labels == 0).sum()
    n_anomaly = (y_labels == 1).sum()
    return n_normal / n_anomaly if n_anomaly > 0 else 1.0


def make_train_loader(X, ctx, y, batch_size):
    return DataLoader(OceanProfileDataset(X, ctx, y), batch_size=batch_size, shuffle=True,
                       num_workers=4, pin_memory=True, persistent_workers=True)


def train_epoch(model, loader, optimizer, criterion, device, label_smoothing):
    model.train()
    total_loss = 0.0
    for batch_x, batch_ctx, batch_y in loader:
        batch_x, batch_ctx, batch_y = batch_x.to(device), batch_ctx.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x, batch_ctx).squeeze(-1)
        batch_y_smooth = batch_y.float() * (1 - label_smoothing) + 0.5 * label_smoothing
        loss = criterion(logits, batch_y_smooth)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_ctx, batch_y in loader:
            batch_x, batch_ctx, batch_y = batch_x.to(device), batch_ctx.to(device), batch_y.to(device)
            logits = model(batch_x, batch_ctx).squeeze(-1)
            loss = criterion(logits, batch_y.float())
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    pred_binary = (all_preds >= 0.5).astype(int)
    f1_macro = f1_score(all_labels, pred_binary, average='macro', zero_division=0)
    auc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), f1_macro, auc


def predict_on_split(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_ctx, batch_y in loader:
            batch_x, batch_ctx = batch_x.to(device), batch_ctx.to(device)
            logits = model(batch_x, batch_ctx).squeeze(-1)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(batch_y.numpy())
    return np.array(all_preds), np.array(all_labels)


def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.linspace(0, 1, 101)
    best_threshold, best_score = 0.5, 0
    for t in thresholds:
        pred = (y_pred_proba >= t).astype(int)
        score = f1_score(y_true, pred, average='macro', zero_division=0)
        if score > best_score:
            best_score, best_threshold = score, t
    return best_threshold


def compute_metrics_at_threshold(y_true, y_pred_proba, threshold):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "ap": average_precision_score(y_true, y_pred_proba),
        "threshold": threshold,
    }


# HALVING SEARCH (single split, rápido, solo ranking)

def train_and_score(config, X_train, ctx_train, y_train, X_val, ctx_val, y_val,
                     X_test, ctx_test, y_test, context_dim, device,
                     max_epochs, patience, plot_dir):
    class_weight = compute_class_weights(y_train)
    pos_weight_factor = config.get("pos_weight_boost", 1.0)

    train_loader = make_train_loader(X_train, ctx_train, y_train, V2_BASE["batch_size"])
    val_loader = DataLoader(OceanProfileDataset(X_val, ctx_val, y_val), batch_size=V2_BASE["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(OceanProfileDataset(X_test, ctx_test, y_test), batch_size=V2_BASE["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
  
    model = ProfileAnomalyCNN(
        n_channels_in=X_train.shape[1], seq_len=V2_BASE["seq_len"],
        n_filters=(config["conv1_filters"], config["conv2_filters"], config["conv3_filters"]),
        kernel_size=config["kernel_size"], dropout_rate=config["dropout_rate"],
        context_dim=context_dim, context_hidden=V2_BASE["context_hidden"],
    ).to(device)

    pos_weight = torch.tensor([class_weight * pos_weight_factor]).to(device) if V2_BASE["use_pos_weight"] else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=V2_BASE["learning_rate"], weight_decay=V2_BASE["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_f1_macro = -np.inf
    patience_counter = 0
    best_state = None
    ep_train_losses, ep_val_losses = [], []

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                  label_smoothing=config["label_smoothing"])
        val_loss, val_f1_macro, val_auc = validate_epoch(model, val_loader, criterion, device)
        ep_train_losses.append(train_loss)
        ep_val_losses.append(val_loss)

        scheduler.step(val_f1_macro)

        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)

    proba_val, labels_val = predict_on_split(model, val_loader, device)
    proba_test, labels_test = predict_on_split(model, test_loader, device)
    threshold = find_optimal_threshold(labels_val, proba_val)
    metrics = compute_metrics_at_threshold(labels_test, proba_test, threshold)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"config": config, **metrics}


def halving_search_woa(X_train, ctx_train, y_train, X_val, ctx_val, y_val,
                        X_test, ctx_test, y_test, context_dim, device, output_dir):
    print(f"\n{'='*90}\nHALVING SEARCH (fast, single-split) — woa_ctx\n{'='*90}")

    candidates = []
    for _ in range(HALVING_CONFIG["n_candidates"]):
        cfg = {}
        for param, values in HALVING_CONFIG["param_distributions"].items():
            cfg[param] = random.choice(values)
        candidates.append(cfg)

    all_results = []
    for it in range(HALVING_CONFIG["n_iterations"]):
        n_remaining = len(candidates)
        max_epochs = HALVING_CONFIG["max_epochs_per_round"][it]
        patience = HALVING_CONFIG["patience_per_round"][it]
        print(f"\n  Ronda {it+1}/{HALVING_CONFIG['n_iterations']} | {n_remaining} candidatos | "
              f"max_epochs={max_epochs} patience={patience}")

        scores = []
        for i, cfg in enumerate(candidates, 1):
            result = train_and_score(cfg, X_train, ctx_train, y_train, X_val, ctx_val, y_val,
                                      X_test, ctx_test, y_test, context_dim, device,
                                      max_epochs, patience, output_dir)
            scores.append(result)
            all_results.append(result)
            print(f"    [{i:2d}/{n_remaining}] recall={result['recall']:.4f} f1_macro={result['f1_macro']:.4f} "
                  f"f1={result['f1']:.4f} precision={result['precision']:.4f} pwb={cfg['pos_weight_boost']}")

        scores_sorted = sorted(scores, key=lambda x: x["f1_macro"], reverse=True)
        n_keep = max(1, n_remaining // 2)
        candidates = [s["config"] for s in scores_sorted[:n_keep]]

    best = max(all_results, key=lambda x: x["f1_macro"])
    print(f"\n  Mejor config woa_ctx: f1_macro={best['f1_macro']:.4f} recall={best['recall']:.4f}")
    for k, v in best["config"].items():
        print(f"    {k}: {v}")

    results_df = pd.DataFrame([
        {"recall": r["recall"], "f1_macro": r["f1_macro"], "f1": r["f1"], "precision": r["precision"],
         "auc_roc": r["auc_roc"], "ap": r["ap"], "threshold": r["threshold"], **r["config"]}
        for r in all_results
    ]).sort_values("f1_macro", ascending=False)
    results_df.to_csv(os.path.join(output_dir, "woa_ctx_halving_results.csv"), index=False)

    return best


# ============================================================================
# FINAL TRAINING (train+val -> test) — generoso, con loss curves completas
# ============================================================================

def train_final_model(best_config, X_train_full, ctx_train_full, y_train_full,
                       X_test, ctx_test, y_test, test_meta, context_dim, device, output_dir,
                       max_epochs=100, patience=20, mlflow_enabled=False):
    print("\n  Training final model (woa_ctx) with best config, in train+val completed...")

    class_weight = compute_class_weights(y_train_full)
    pos_weight_factor = best_config.get("pos_weight_boost", 1.0)

    n = len(y_train_full)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    train_loader = make_train_loader(X_train_full[tr_idx], ctx_train_full[tr_idx], y_train_full[tr_idx], V2_BASE["batch_size"])
    val_loader = DataLoader(OceanProfileDataset(X_train_full[val_idx], ctx_train_full[val_idx], y_train_full[val_idx]),
                             batch_size=V2_BASE["batch_size"], shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(OceanProfileDataset(X_test, ctx_test, y_test), batch_size=V2_BASE["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    model = ProfileAnomalyCNN(
        n_channels_in=X_train_full.shape[1], seq_len=V2_BASE["seq_len"],
        n_filters=(best_config["conv1_filters"], best_config["conv2_filters"], best_config["conv3_filters"]),
        kernel_size=best_config["kernel_size"], dropout_rate=best_config["dropout_rate"],
        context_dim=context_dim, context_hidden=V2_BASE["context_hidden"],
    ).to(device)

    pos_weight = torch.tensor([class_weight * pos_weight_factor]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=V2_BASE["learning_rate"], weight_decay=V2_BASE["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_f1_macro = -np.inf
    best_state = None
    patience_counter = 0
    best_epoch = 0
    ep_train_losses, ep_val_losses, ep_val_f1m, ep_val_auc = [], [], [], []

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                  label_smoothing=best_config["label_smoothing"])
        val_loss, val_f1_macro, val_auc = validate_epoch(model, val_loader, criterion, device)
        ep_train_losses.append(train_loss)
        ep_val_losses.append(val_loss)
        ep_val_f1m.append(val_f1_macro)
        ep_val_auc.append(val_auc)
        scheduler.step(val_f1_macro)

        if mlflow_enabled:
            mlflow.log_metrics({
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_f1_macro": float(val_f1_macro),
                "val_auc": float(val_auc),
            }, step=epoch)

        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:<4} train_loss={train_loss:.4f} val_f1m={val_f1_macro:.4f} val_auc={val_auc:.4f}")
        if patience_counter >= patience:
            print(f"    early stopping en epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(output_dir, "woa_ctx_final_model.pth"))

    # Loss curve: train vs val completa
    plt.figure(figsize=(12, 6))
    epochs_range = range(1, len(ep_train_losses) + 1)
    plt.plot(epochs_range, ep_train_losses, linestyle='--', linewidth=2, label='Train Loss')
    plt.plot(epochs_range, ep_val_losses, linestyle='-', linewidth=2, label='Val Loss')
    plt.axvline(best_epoch + 1, color='red', linestyle=':', alpha=0.7, label=f'Best (epoch {best_epoch+1})')
    plt.title('Final Model: Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "woa_ctx_final_loss.png"), dpi=130)
    plt.close()

    proba_train, labels_train = predict_on_split(model, train_loader, device)
    proba_val, labels_val = predict_on_split(model, val_loader, device)
    proba_test, labels_test = predict_on_split(model, test_loader, device)
    threshold = find_optimal_threshold(labels_val, proba_val)
    
    metrics_train = compute_metrics_at_threshold(labels_train, proba_train, threshold)
    metrics_val = compute_metrics_at_threshold(labels_val, proba_val, threshold)
    metrics_test = compute_metrics_at_threshold(labels_test, proba_test, threshold)

    pred_train = (proba_train >= threshold).astype(int)
    pred_val = (proba_val >= threshold).astype(int)
    pred_test = (proba_test >= threshold).astype(int)

    # Salida: Classification reports para train/val/test
    print(f"\n{'='*90}")
    print("CLASSIFICATION REPORT - TRAIN")
    print(f"{'='*90}")
    print(classification_report(labels_train, pred_train, target_names=["Normal", "Anomaly"], digits=4))

    print(f"\n{'='*90}")
    print("CLASSIFICATION REPORT - VAL")
    print(f"{'='*90}")
    print(classification_report(labels_val, pred_val, target_names=["Normal", "Anomaly"], digits=4))

    print(f"\n{'='*90}")
    print("CLASSIFICATION REPORT - TEST")
    print(f"{'='*90}")
    print(classification_report(labels_test, pred_test, target_names=["Normal", "Anomaly"], digits=4))

    print(f"\n[woa_ctx] TEST FINAL -> f1_macro={metrics_test['f1_macro']:.4f} "
          f"f1={metrics_test['f1']:.4f} recall={metrics_test['recall']:.4f} "
          f"precision={metrics_test['precision']:.4f} auc={metrics_test['auc_roc']:.4f}")

    # --- training_history.csv ---
    history_df = pd.DataFrame({
        "train_loss": ep_train_losses,
        "val_loss": ep_val_losses,
        "val_f1_macro": ep_val_f1m,
        "val_auc": ep_val_auc,
    })
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    if mlflow_enabled:
        mlflow.log_artifact(os.path.join(output_dir, "training_history.csv"))

    # --- cnn_results.joblib ---
    results_data = {
        "X_test": X_test,
        "y_test": labels_test,
        "proba_train": proba_train,
        "proba_val": proba_val,
        "proba_test": proba_test,
        "pred_test": pred_test,
        "optimal_threshold": threshold,
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "cnn_params": best_config,
        "best_epoch": best_epoch,
        "labels_train": labels_train,
        "labels_val": labels_val,
        "labels_test": labels_test,
    }
    results_path = os.path.join(output_dir, "cnn_results.joblib")
    joblib.dump(results_data, results_path)
    if mlflow_enabled:
        mlflow.log_artifact(results_path)

    # --- test_predictions.parquet ---
    test_meta_with_pred = test_meta.copy()
    test_meta_with_pred["proba"] = proba_test
    test_meta_with_pred["pred"] = pred_test
    test_meta_with_pred.to_parquet(os.path.join(output_dir, "test_predictions.parquet"), index=False)
    if mlflow_enabled:
        mlflow.log_artifact(os.path.join(output_dir, "test_predictions.parquet"))

    # --- config.json ---
    config_out = {
        "optimal_threshold": float(threshold),
        "best_epoch": int(best_epoch),
        **{k: (float(v) if isinstance(v, (int, np.integer, np.floating)) else v)
           for k, v in best_config.items()},
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_out, f, indent=2)
    if mlflow_enabled:
        mlflow.log_artifact(os.path.join(output_dir, "config.json"))

    return {
        "variant": "woa_ctx", "best_epoch": best_epoch, "threshold": threshold,
        "config": best_config, "metrics_test": metrics_test,
        "metrics_val": metrics_val, "metrics_train": metrics_train,
        "n_channels_in": X_train_full.shape[1], "context_dim": context_dim,
        "proba_train": proba_train, "labels_train": labels_train, "pred_train": pred_train,
        "proba_val": proba_val, "labels_val": labels_val, "pred_val": pred_val,
        "proba_test": proba_test, "labels_test": labels_test, "pred_test": pred_test,
    }


# K-FOLD CONFIRMATION (solo para best config, NO para cada trial)

def confirm_with_kfold(best_config, X_pool, ctx_pool, y_pool,
                        X_test, ctx_test, y_test, context_dim, device, output_dir,
                        n_splits=KFOLD_N_SPLITS, max_epochs=KFOLD_MAX_EPOCHS, patience=KFOLD_PATIENCE):
    print(f"\n{'='*90}\nK-FOLD CONFIRMATION (n_splits={n_splits}) — winning config only\n{'='*90}")
    for k, v in best_config.items():
        print(f"    {k}: {v}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_pool, y_pool), 1):
        print(f"\n  --- Fold {fold}/{n_splits} ---")
        X_tr, ctx_tr, y_tr = X_pool[tr_idx], ctx_pool[tr_idx], y_pool[tr_idx]
        X_v, ctx_v, y_v = X_pool[val_idx], ctx_pool[val_idx], y_pool[val_idx]

        result = train_and_score(best_config, X_tr, ctx_tr, y_tr, X_v, ctx_v, y_v,
                                  X_test, ctx_test, y_test, context_dim, device,
                                  max_epochs, patience, output_dir)
        fold_results.append(result)
        print(f"    Fold {fold} TEST -> recall={result['recall']:.4f} precision={result['precision']:.4f} "
              f"f1={result['f1']:.4f} f1_macro={result['f1_macro']:.4f} auc={result['auc_roc']:.4f}")

    metrics_keys = ["recall", "precision", "f1", "f1_macro", "auc_roc", "ap"]
    summary = {}
    for k in metrics_keys:
        vals = np.array([r[k] for r in fold_results])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}

    print(f"\n{'='*90}")
    print(f"K-FOLD CONFIRMATION SUMMARY (n_splits={n_splits})")
    print(f"{'='*90}")
    for k in metrics_keys:
        print(f"  {k:<12}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

    fold_df = pd.DataFrame([
        {"fold": i + 1, **{k: r[k] for k in metrics_keys}}
        for i, r in enumerate(fold_results)
    ])
    fold_df.to_csv(os.path.join(output_dir, "woa_ctx_kfold_confirmation.csv"), index=False)

    with open(os.path.join(output_dir, "woa_ctx_kfold_confirmation_summary.json"), "w") as f:
        json.dump({"n_splits": n_splits, "config": best_config, "summary": summary}, f, indent=2)

    return summary, fold_results


# PER-OCEAN PIPELINE

def run_for_ocean(ocean_name, preproc_dir, output_dir, woa_ds, device, mlflow_enabled):
    print("\n" + "#" * 100)
    print(f"# OCEAN: {ocean_name.upper()}")
    print(f"# preproc_dir: {preproc_dir}")
    print(f"# output_dir:  {output_dir}")
    print("#" * 100)

    os.makedirs(output_dir, exist_ok=True)
    variant_dir = os.path.join(output_dir, "woa_ctx")
    os.makedirs(variant_dir, exist_ok=True)

    run = None
    if mlflow_enabled:
        run = mlflow.start_run(run_name=f"CNN_{ocean_name}_{YEARS_RANGE}")
        mlflow.set_tag("ocean", ocean_name)
        mlflow.set_tag("years_range", YEARS_RANGE)
        mlflow.set_tag("variant", "woa_ctx")
        mlflow.log_param("preproc_dir", preproc_dir)
        mlflow.log_param("seed", SEED)
        mlflow.log_params({f"base__{k}": v for k, v in V2_BASE.items()})

    try:
        print("\n[1/4] Loading raw data...")
        raw_data = load_preprocessed_data(preproc_dir)
        (train_temp, train_psal, train_mask, train_meta,
         val_temp, val_psal, val_mask, val_meta,
         test_temp, test_psal, test_mask, test_meta,
         pressure_grid) = raw_data

        if mlflow_enabled:
            mlflow.log_params({
                "n_train": int(train_temp.shape[0]),
                "n_val": int(val_temp.shape[0]),
                "n_test": int(test_temp.shape[0]),
                "n_anomaly_train": int(train_meta["is_bad"].sum()),
                "n_anomaly_val": int(val_meta["is_bad"].sum()),
                "n_anomaly_test": int(test_meta["is_bad"].sum()),
            })

        print("\n[2/4] Calculating base normalization (8 canals) from train...")
        norm = compute_normalization_stats(train_temp, train_psal, train_mask, pressure_grid)

        y_train = train_meta["is_bad"].values.astype(np.int8)
        y_val = val_meta["is_bad"].values.astype(np.int8)
        y_test = test_meta["is_bad"].values.astype(np.int8)

        print("\n[3/4] Preparing WOA canals (10) + halving search + final train...")
        X_train, ctx_train, X_val, ctx_val, X_test, ctx_test, n_channels_in, context_dim = \
            prepare_woa_variant(raw_data, norm, woa_ds)
        print(f"  Canales de entrada: {n_channels_in}  |  context_dim: {context_dim}")

        best = halving_search_woa(X_train, ctx_train, y_train, X_val, ctx_val, y_val,
                                   X_test, ctx_test, y_test, context_dim, device, variant_dir)

        if mlflow_enabled:
            mlflow.log_params({f"search_best__{k}": v for k, v in best["config"].items()})
            halving_csv = os.path.join(variant_dir, "woa_ctx_halving_results.csv")
            if os.path.exists(halving_csv):
                mlflow.log_artifact(halving_csv, artifact_path="halving_search")

        X_pool = np.concatenate([X_train, X_val], axis=0)
        ctx_pool = np.concatenate([ctx_train, ctx_val], axis=0)
        y_pool = np.concatenate([y_train, y_val], axis=0)

        print("\n[4/4] Final training  + K-fold + complete output...")
        result = train_final_model(best["config"], X_pool, ctx_pool, y_pool,
                            X_test, ctx_test, y_test, test_meta, context_dim, device, output_dir,
                            mlflow_enabled=mlflow_enabled)

        # K-Fold confirmation solo para mejor modelo
        if RUN_KFOLD_CONFIRMATION:
            kfold_summary, _ = confirm_with_kfold(best["config"], X_pool, ctx_pool, y_pool,
                                                     X_test, ctx_test, y_test, context_dim,
                                                     device, variant_dir)
            if mlflow_enabled:
                for k, v in kfold_summary.items():
                    mlflow.log_metric(f"kfold_{k}_mean", v["mean"])
                    mlflow.log_metric(f"kfold_{k}_std", v["std"])

        # Guardar resultados finales
        final_result_path = os.path.join(output_dir, f"{ocean_name}_woa_ctx_final_result.json")
        with open(final_result_path, "w") as f:
            json.dump({
                "variant": "woa_ctx",
                "best_epoch": int(result["best_epoch"]),
                "threshold": float(result["threshold"]),
                "config": {k: (float(v) if isinstance(v, (int, np.integer, np.floating)) else v) 
                           for k, v in result["config"].items()},
                "metrics_train": {k: float(v) for k, v in result["metrics_train"].items()},
                "metrics_val": {k: float(v) for k, v in result["metrics_val"].items()},
                "metrics_test": {k: float(v) for k, v in result["metrics_test"].items()},
            }, f, indent=2)

        if mlflow_enabled:
            mlflow.log_metrics({f"test_{k}": float(v) for k, v in result["metrics_test"].items()})
            mlflow.log_metrics({f"val_{k}": float(v) for k, v in result["metrics_val"].items()})
            mlflow.log_metrics({f"train_{k}": float(v) for k, v in result["metrics_train"].items()})
            mlflow.log_param("best_epoch", result["best_epoch"])
            mlflow.log_param("threshold", result["threshold"])
            mlflow.log_artifact(final_result_path, artifact_path="final_result")

            loss_plot = os.path.join(variant_dir, "woa_ctx_final_loss.png")
            if os.path.exists(loss_plot):
                mlflow.log_artifact(loss_plot, artifact_path="plots")

            model_pth = os.path.join(variant_dir, "woa_ctx_final_model.pth")
            if os.path.exists(model_pth):
                mlflow.log_artifact(model_pth, artifact_path="model_state_dict")

        if mlflow_enabled:
            mlflow.set_tag("status", "OK")

        print(f"\n✓ [{ocean_name}] DONE. Results in: {output_dir}")
        return result

    except Exception as e:
        print(f"\n✗ [{ocean_name}] ERROR: {e}")
        traceback.print_exc()
        if mlflow_enabled:
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(e)[:250])
        raise

    finally:
        if mlflow_enabled and run is not None:
            mlflow.end_run()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# MAIN

def main():
    print("\n" + "=" * 100)
    print("CNN MULTI-OCEAN (Atlantic / Pacific / Indian) — WOA23, HALVING SEARCH, K-FOLD, MLflow")
    print("=" * 100)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    mlflow_enabled = MLFLOW_ENABLED and MLFLOW_AVAILABLE
    if MLFLOW_ENABLED and not MLFLOW_AVAILABLE:
        print("  [WARN] mlflow not installed.")

    if mlflow_enabled:
        os.makedirs(MLFLOW_DIR, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
        experiment_name = "CNN_Anomaly_MultiOcean_F1Macro"
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            mlflow.create_experiment(name=experiment_name, artifact_location=f"file://{MLFLOW_DIR}/mlruns")
        mlflow.set_experiment(experiment_name)
        print(f"  MLflow: {MLFLOW_DIR}/mlflow.db | experimento: {experiment_name}")

    print("\nLoading WOA23 climatology (once only, shared across 3 oceans)...")
    woa_ds = load_woa_climatology(WOA_DIR)

    all_results = {}
    for ocean_name, preproc_dir in OCEANS.items():
        ocean_output_dir = os.path.join(OUTPUT_DIR_BASE, YEARS_RANGE, ocean_name)
        try:
            result = run_for_ocean(ocean_name, preproc_dir, ocean_output_dir, woa_ds, DEVICE, mlflow_enabled)
            all_results[ocean_name] = result["metrics_test"]
        except Exception:
            print(f"  ✗ {ocean_name} fail, continue...")
            all_results[ocean_name] = None
            continue

    print("\n" + "=" * 100)
    print("RESUMEN FINAL — 3 OCÉANOS")
    print("=" * 100)
    for ocean_name, metrics in all_results.items():
        if metrics is None:
            print(f"  {ocean_name:<10}: FALLÓ")
        else:
            print(f"  {ocean_name:<10}: f1_macro={metrics['f1_macro']:.4f}  recall={metrics['recall']:.4f}  "
                  f"precision={metrics['precision']:.4f}  auc={metrics['auc_roc']:.4f}")

    summary_path = os.path.join(OUTPUT_DIR_BASE, "multiocean_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ DONE. Resumen en: {summary_path}")


if __name__ == "__main__":
    main()