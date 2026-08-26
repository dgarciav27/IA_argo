"""
Transformer-based Anomaly Classifier (Multi-Ocean with Successive Halving & K-Fold)
----------------------------------------------------------------------------------
- Iterates independently over specified Oceans (Atlantic, Pacific, Indian).
- Uses Successive Halving Random Search (with warm-started continuation) to
  quickly explore hyperparameter spaces.
- Evaluates the top hyperparameter candidate using 5-Fold Stratified Cross-Validation
  (run on the pooled train+val data, matching the CNN pipeline's methodology).
- Trains a final model with Early Stopping on full train/val splits.
- Computes train/val/test metrics (including f1_macro for comparability with the
  CNN pipeline), classification reports, saves outputs (including everything
  needed to reproduce plots later) and logs to a single shared MLflow experiment.
"""

import os
import json
import copy
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score,
                              precision_score, recall_score, accuracy_score,
                              average_precision_score, classification_report)
import mlflow
import mlflow.pytorch
from datetime import datetime
import joblib


# GLOBAL CONFIGURATION & PATHS
PREPROC_BASE = "/work/drgarcia/Dataset/DL_datasets"
YEARS_RANGE = "2018_2022"

OCEANS = {
    "atlantic": os.path.join(PREPROC_BASE, f"atlantic_ocean/{YEARS_RANGE}/split_time_ascA_dmodeD_masked_grid100"),
    "pacific":  os.path.join(PREPROC_BASE, f"pacific_ocean/{YEARS_RANGE}/split_time_ascA_dmodeD_masked_grid100"),
    "indian":   os.path.join(PREPROC_BASE, f"indian_ocean/{YEARS_RANGE}/split_time_ascA_dmodeD_masked_grid100"),
}

OUTPUT_DIR_BASE = "/work/drgarcia/Models_and_results/Transformer_MultiOceanvf/2018_2022"
MLFLOW_DIR = "/home/drgarcia/Argo_ml_code/ML_flow"

os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Successive Halving & K-Fold Settings.
# epochs_per_iter is INCREMENTAL (extra epochs trained on top of the
# previous stage), because candidates are warm-started between stages
# instead of being retrained from scratch.
KFOLD_N_SPLITS = 5
HALVING_CONFIG = {
    "n_candidates": 6,
    "n_iterations": 2,
    "epochs_per_iter": [5, 10],   # stage 0: 5 epochs; stage 1: +10 epochs (15 total)
    "top_k_per_iter": [3, 1],     # retain top 3 after stage 0, top 1 after stage 1
}

# Base hyperparameter template
BASE_TRANSFORMER_PARAMS = {
    "n_channels_in": 8,
    "seq_len": 100,
    "patience": 30,
    "max_epochs": 100,
}

# Single shared MLflow experiment across all oceans (one run per ocean),
# matching the CNN pipeline's approach so results are comparable side by side.
MLFLOW_EXPERIMENT_NAME = "Transformer_Anomaly_MultiOcean"

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# DATASET & MODEL DEFINITION
class OceanProfileDataset(Dataset):
    """Dataset for oceanographic profiles with 8 channels (T,S,P,Mask,dT,d2T,dS,d2S)."""

    def __init__(self, X_profiles, y_labels, metadata=None):
        self.X = torch.from_numpy(X_profiles).float()
        self.y = torch.from_numpy(y_labels).long()
        self.metadata = metadata

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding for the fixed pressure grid."""

    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        return x + self.pos_embedding


class ProfileAnomalyTransformer(nn.Module):
    """Transformer Encoder for binary classification of oceanographic anomalies."""

    def __init__(self, n_channels_in=8, seq_len=100, d_model=64, nhead=4,
                 num_layers=3, dim_feedforward=128, dropout=0.2, mask_channel_idx=3):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.mask_channel_idx = mask_channel_idx

        self.input_proj = nn.Linear(n_channels_in, d_model)
        self.pos_encoding = LearnedPositionalEncoding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 64)
        self.dropout1 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        mask_channel = x[:, self.mask_channel_idx, :]
        valid_counts = mask_channel.sum(dim=1)
        safe_counts = valid_counts.clamp(min=1.0)

        h = x.permute(0, 2, 1)
        h = self.input_proj(h)
        h = self.pos_encoding(h)

        key_padding_mask = (mask_channel == 0)
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)

        mask_f = mask_channel.unsqueeze(-1)
        pooled = (h * mask_f).sum(dim=1) / safe_counts.unsqueeze(-1)

        z = torch.relu(self.fc1(pooled))
        z = self.dropout1(z)
        out = self.fc_out(z)
        return out


# DATA PROCESSING UTILITIES
def load_preprocessed_data(preproc_dir):
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

    return (train_temp, train_psal, train_mask, train_meta,
            val_temp, val_psal, val_mask, val_meta,
            test_temp, test_psal, test_mask, test_meta,
            pressure_grid)


def compute_gradient_channels(X, mask, P_grid):
    dP = np.gradient(P_grid).astype(np.float32)
    dP = np.where(dP == 0, 1e-3, dP)
    dX = np.gradient(X, axis=1) / dP[None, :]
    d2X = np.gradient(dX, axis=1) / dP[None, :]
    dX = np.where(mask, dX, 0.0).astype(np.float32)
    d2X = np.where(mask, d2X, 0.0).astype(np.float32)
    return dX, d2X


def build_8channel_input(X_temp, X_psal, mask, P_grid, T_mean_L, T_std_L,
                          S_mean_L, S_std_L, dT_mu, dT_sd, d2T_mu, d2T_sd,
                          dS_mu, dS_sd, d2S_mu, d2S_sd):
    dT, d2T = compute_gradient_channels(X_temp, mask, P_grid)
    dS, d2S = compute_gradient_channels(X_psal, mask, P_grid)

    GRID_MAX_PRES = P_grid.max()
    P_norm_row = (P_grid / GRID_MAX_PRES).astype(np.float32)

    T_norm = np.where(mask, (X_temp - T_mean_L[None, :]) / T_std_L[None, :], 0.0).astype(np.float32)
    S_norm = np.where(mask, (X_psal - S_mean_L[None, :]) / S_std_L[None, :], 0.0).astype(np.float32)
    P_norm = np.broadcast_to(P_norm_row, T_norm.shape).astype(np.float32)
    M_chan = mask.astype(np.float32)
    dT_norm = np.where(mask, (dT - dT_mu) / dT_sd, 0.0).astype(np.float32)
    d2T_norm = np.where(mask, (d2T - d2T_mu) / d2T_sd, 0.0).astype(np.float32)
    dS_norm = np.where(mask, (dS - dS_mu) / dS_sd, 0.0).astype(np.float32)
    d2S_norm = np.where(mask, (d2S - d2S_mu) / d2S_sd, 0.0).astype(np.float32)

    profiles = np.stack([T_norm, S_norm, P_norm, M_chan, dT_norm, d2T_norm, dS_norm, d2S_norm], axis=1)
    return profiles.astype(np.float32)


# CLASS IMBALANCE HANDLING
def compute_class_weights(y_labels):
    """Weight for the positive (anomaly) class, inversely proportional to its frequency."""
    n_normal = (y_labels == 0).sum()
    n_anomaly = (y_labels == 1).sum()
    return n_normal / n_anomaly if n_anomaly > 0 else 1.0


# TRAINING & EVALUATION HELPER FUNCTIONS
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x).squeeze(-1)
        loss = criterion(logits, batch_y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
    return avg_loss, auc, acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate one epoch. Returns loss, AUC, accuracy, average precision, and
    F1-macro (needed to reproduce the same training-curve plots used for the CNN)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y.float())
            total_loss += loss.item()

            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    pred_binary = (all_preds >= 0.5).astype(int)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, pred_binary)
    ap = average_precision_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, pred_binary, average='macro', zero_division=0)
    return avg_loss, auc, acc, ap, f1_macro, all_preds, all_labels


def compute_metrics_anomaly(y_true, y_pred_proba, y_pred_binary, threshold=0.5):
    return {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "ap": average_precision_score(y_true, y_pred_proba),
        "threshold": threshold,
    }


def compute_metrics_macro(y_true, y_pred_proba, y_pred_binary, threshold=0.5):
    return {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision_macro": precision_score(y_true, y_pred_binary, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred_binary, average='macro', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred_binary, average='macro', zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "ap": average_precision_score(y_true, y_pred_proba),
        "threshold": threshold,
    }


def predict_on_split(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x).squeeze(-1)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
    return np.array(all_preds), np.array(all_labels)


# SEARCH & CROSS VALIDATION ROUTINES
def sample_random_params(rng):
    """Generate a valid random combination of hyperparameters."""
    d_model_choices = [32, 64, 128]
    d_model = int(rng.choice(d_model_choices))

    # nhead must evenly divide d_model
    possible_heads = [h for h in [2, 4, 8] if d_model % h == 0]
    nhead = int(rng.choice(possible_heads))

    num_layers = int(rng.choice([2, 3, 4]))
    dim_feedforward = int(rng.choice([64, 128, 256]))
    dropout = float(rng.choice([0.1, 0.2, 0.3]))
    learning_rate = float(rng.choice([5e-4, 1e-3, 2e-3]))
    weight_decay = float(rng.choice([1e-5, 1e-4]))
    batch_size = int(rng.choice([32, 64]))

    params = copy.deepcopy(BASE_TRANSFORMER_PARAMS)
    params.update({
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
    })
    return params


def _build_model_and_optimizer(params, device):
    model = ProfileAnomalyTransformer(
        n_channels_in=params["n_channels_in"],
        seq_len=params["seq_len"],
        d_model=params["d_model"],
        nhead=params["nhead"],
        num_layers=params["num_layers"],
        dim_feedforward=params["dim_feedforward"],
        dropout=params["dropout"],
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"],
                             weight_decay=params["weight_decay"])
    return model, optimizer


def train_candidate_for_epochs(state, train_dataset, val_dataset, n_epochs, device):
    """
    Trains a candidate (creating it fresh if `state` is None, otherwise
    resuming from its stored model/optimizer) for `n_epochs` additional
    epochs and returns the updated state plus the best validation AUC seen
    so far across ALL epochs trained (this stage and previous ones).
    """
    params = state["params"]
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

    if state["model"] is None:
        model, optimizer = _build_model_and_optimizer(params, device)
    else:
        model, optimizer = state["model"], state["optimizer"]

    # Class-imbalance-aware loss: weight the anomaly class inversely to its frequency.
    y_train_np = train_dataset.y.numpy()
    class_weight = compute_class_weights(y_train_np)
    pos_weight = torch.tensor([class_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = state.get("best_val_auc", -np.inf)

    for _ in range(n_epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        _, val_auc, _, _, _, _, _ = validate_epoch(model, val_loader, criterion, device)
        if val_auc > best_val_auc:
            best_val_auc = val_auc

    state["model"] = model
    state["optimizer"] = optimizer
    state["best_val_auc"] = best_val_auc
    state["epochs_trained"] = state.get("epochs_trained", 0) + n_epochs

    del train_loader, val_loader
    return state


def _free_state(state):
    if state.get("model") is not None:
        del state["model"]
    if state.get("optimizer") is not None:
        del state["optimizer"]
    state["model"] = None
    state["optimizer"] = None
    gc.collect()
    torch.cuda.empty_cache()


def run_successive_halving(train_dataset, val_dataset, device):
    """
    Runs Successive Halving Random Search. Surviving candidates are
    warm-started between stages (i.e. training continues from where it left
    off) instead of being retrained from scratch, so the epochs_per_iter
    values in HALVING_CONFIG are incremental. Returns the best hyperparameter
    set found and the full search history (for later plotting/inspection).
    """
    print("\n--- STARTING SUCCESSIVE HALVING RANDOM SEARCH ---")
    rng = np.random.default_rng(SEED)
    states = [
        {"params": sample_random_params(rng), "model": None, "optimizer": None,
         "best_val_auc": -np.inf, "epochs_trained": 0}
        for _ in range(HALVING_CONFIG["n_candidates"])
    ]

    search_history = []

    for stage in range(HALVING_CONFIG["n_iterations"]):
        extra_epochs = HALVING_CONFIG["epochs_per_iter"][stage]
        top_k = HALVING_CONFIG["top_k_per_iter"][stage]
        print(f"\n[Halving Stage {stage + 1}/{HALVING_CONFIG['n_iterations']}] "
              f"Training {len(states)} candidates for {extra_epochs} more epoch(s)...")

        for idx, state in enumerate(states):
            state = train_candidate_for_epochs(state, train_dataset, val_dataset, extra_epochs, device)
            params = state["params"]
            print(f"  Candidate {idx + 1} | Cumulative epochs: {state['epochs_trained']} | "
                  f"Val AUC: {state['best_val_auc']:.4f} | d_model={params['d_model']}, "
                  f"nhead={params['nhead']}, layers={params['num_layers']}, lr={params['learning_rate']}")
            search_history.append({
                "stage": stage,
                "candidate_idx": idx,
                "epochs_trained": state["epochs_trained"],
                "val_auc": float(state["best_val_auc"]),
                **{k: v for k, v in params.items() if k not in ("n_channels_in", "seq_len", "patience", "max_epochs")},
            })

        # Rank by validation AUC (descending) and keep the survivors; free GPU
        # memory for the candidates that get discarded.
        states.sort(key=lambda s: s["best_val_auc"], reverse=True)
        survivors, discarded = states[:top_k], states[top_k:]
        for s in discarded:
            _free_state(s)
        states = survivors

        print(f"-> Best AUC this stage: {states[0]['best_val_auc']:.4f}")

    best_state = states[0]
    best_params = best_state["params"]
    _free_state(best_state)

    print("--- BEST HYPERPARAMETER COMBINATION FOUND ---")
    print(json.dumps(best_params, indent=2))
    return best_params, search_history


def run_kfold_cv(best_params, X_pool, y_pool, device):
    """
    Runs K-Fold Cross-Validation on the best hyperparameter combination only.
    X_pool / y_pool should be the pooled train+val data (matching the CNN
    pipeline's methodology), giving the confirmation step more data to work with.
    """
    print(f"\n--- STARTING {KFOLD_N_SPLITS}-FOLD CROSS-VALIDATION ON THE BEST COMBINATION ---")
    skf = StratifiedKFold(n_splits=KFOLD_N_SPLITS, shuffle=True, random_state=SEED)
    fold_aucs, fold_accs, fold_aps, fold_f1_macros = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pool, y_pool)):
        fold_train_ds = OceanProfileDataset(X_pool[train_idx], y_pool[train_idx])
        fold_val_ds = OceanProfileDataset(X_pool[val_idx], y_pool[val_idx])

        train_loader = DataLoader(fold_train_ds, batch_size=best_params["batch_size"], shuffle=True)
        val_loader = DataLoader(fold_val_ds, batch_size=best_params["batch_size"], shuffle=False)

        model, optimizer = _build_model_and_optimizer(best_params, device)

        class_weight = compute_class_weights(y_pool[train_idx])
        pos_weight = torch.tensor([class_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc, best_acc, best_ap, best_f1_macro = -np.inf, 0.0, 0.0, 0.0

        # Quick 15-epoch training per fold
        for _ in range(15):
            train_epoch(model, train_loader, optimizer, criterion, device)
            _, val_auc, val_acc, val_ap, val_f1_macro, _, _ = validate_epoch(model, val_loader, criterion, device)
            if val_auc > best_auc:
                best_auc, best_acc, best_ap, best_f1_macro = val_auc, val_acc, val_ap, val_f1_macro

        fold_aucs.append(best_auc)
        fold_accs.append(best_acc)
        fold_aps.append(best_ap)
        fold_f1_macros.append(best_f1_macro)
        print(f" Fold {fold + 1}/{KFOLD_N_SPLITS} | Best Val AUC: {best_auc:.4f} | "
              f"Acc: {best_acc:.4f} | AP: {best_ap:.4f} | F1-macro: {best_f1_macro:.4f}")

        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    metrics_kfold = {
        "kfold_auc_mean": float(np.mean(fold_aucs)),
        "kfold_auc_std": float(np.std(fold_aucs)),
        "kfold_acc_mean": float(np.mean(fold_accs)),
        "kfold_ap_mean": float(np.mean(fold_aps)),
        "kfold_f1_macro_mean": float(np.mean(fold_f1_macros)),
        "kfold_f1_macro_std": float(np.std(fold_f1_macros)),
    }
    # Per-fold values are kept so downstream code can plot the CV spread
    # (box/violin plots, error bars, etc.) instead of only the aggregate.
    fold_details = {
        "fold_auc": [float(v) for v in fold_aucs],
        "fold_acc": [float(v) for v in fold_accs],
        "fold_ap": [float(v) for v in fold_aps],
        "fold_f1_macro": [float(v) for v in fold_f1_macros],
    }
    print(f"K-Fold Summary -> Mean AUC: {metrics_kfold['kfold_auc_mean']:.4f} "
          f"(+/- {metrics_kfold['kfold_auc_std']:.4f}) | "
          f"Mean F1-macro: {metrics_kfold['kfold_f1_macro_mean']:.4f}")
    return metrics_kfold, fold_details


# MAIN MULTI-OCEAN PIPELINE
def process_ocean(ocean_name, preproc_dir):
    print("\n" + "=" * 80)
    print(f" PROCESSING OCEAN: {ocean_name.upper()}")
    print("=" * 80)

    ocean_output_dir = os.path.join(OUTPUT_DIR_BASE, ocean_name)
    os.makedirs(ocean_output_dir, exist_ok=True)

    # 1. Loading data
    print("\n[1/6] Loading preprocessed data...")
    (train_temp, train_psal, train_mask, train_meta,
     val_temp, val_psal, val_mask, val_meta,
     test_temp, test_psal, test_mask, test_meta,
     pressure_grid) = load_preprocessed_data(preproc_dir)

    # Normalization stats computed from train only (avoids leakage)
    T_mean_L, T_std_L = train_temp.mean(axis=0), train_temp.std(axis=0) + 1e-8
    S_mean_L, S_std_L = train_psal.mean(axis=0), train_psal.std(axis=0) + 1e-8

    dT_train, d2T_train = compute_gradient_channels(train_temp, train_mask, pressure_grid)
    dT_mu, dT_sd = float(dT_train[train_mask].mean()), float(dT_train[train_mask].std() + 1e-8)
    d2T_mu, d2T_sd = float(d2T_train[train_mask].mean()), float(d2T_train[train_mask].std() + 1e-8)

    dS_train, d2S_train = compute_gradient_channels(train_psal, train_mask, pressure_grid)
    dS_mu, dS_sd = float(dS_train[train_mask].mean()), float(dS_train[train_mask].std() + 1e-8)
    d2S_mu, d2S_sd = float(d2S_train[train_mask].mean()), float(d2S_train[train_mask].std() + 1e-8)

    # 2. Build 8 channels
    print("\n[2/6] Building 8-channel tensors...")
    X_train = build_8channel_input(train_temp, train_psal, train_mask, pressure_grid,
                                    T_mean_L, T_std_L, S_mean_L, S_std_L,
                                    dT_mu, dT_sd, d2T_mu, d2T_sd, dS_mu, dS_sd, d2S_mu, d2S_sd)
    X_val = build_8channel_input(val_temp, val_psal, val_mask, pressure_grid,
                                  T_mean_L, T_std_L, S_mean_L, S_std_L,
                                  dT_mu, dT_sd, d2T_mu, d2T_sd, dS_mu, dS_sd, d2S_mu, d2S_sd)
    X_test = build_8channel_input(test_temp, test_psal, test_mask, pressure_grid,
                                   T_mean_L, T_std_L, S_mean_L, S_std_L,
                                   dT_mu, dT_sd, d2T_mu, d2T_sd, dS_mu, dS_sd, d2S_mu, d2S_sd)

    y_train = train_meta["is_bad"].values.astype(np.int8)
    y_val = val_meta["is_bad"].values.astype(np.int8)
    y_test = test_meta["is_bad"].values.astype(np.int8)

    train_dataset = OceanProfileDataset(X_train, y_train, train_meta)
    val_dataset = OceanProfileDataset(X_val, y_val, val_meta)
    test_dataset = OceanProfileDataset(X_test, y_test, test_meta)

    # 3. Successive Halving Random Search
    print("\n[3/6] Hyperparameter search (Successive Halving)...")
    best_params, search_history = run_successive_halving(train_dataset, val_dataset, DEVICE)

    # 4. K-Fold Cross-Validation on the best combination only, using the
    #    pooled train+val data (same methodology as the CNN pipeline).
    print("\n[4/6] Evaluating best combination with K-Fold CV (pooled train+val)...")
    X_pool = np.concatenate([X_train, X_val], axis=0)
    y_pool = np.concatenate([y_train, y_val], axis=0)
    metrics_kfold, kfold_fold_details = run_kfold_cv(best_params, X_pool, y_pool, DEVICE)

    # 5. MLflow Tracking Setup — single shared experiment, one run per ocean.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME, artifact_location=f"file://{MLFLOW_DIR}/mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 6. Final Model Training & Evaluation
    print("\n[5/6] Training final model with Early Stopping...")
    with mlflow.start_run(run_name=f"Transformer_{ocean_name}_{YEARS_RANGE}"):
        mlflow.set_tag("ocean", ocean_name)
        mlflow.set_tag("years_range", YEARS_RANGE)
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics_kfold)

        train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

        model, optimizer = _build_model_and_optimizer(best_params, DEVICE)

        class_weight = compute_class_weights(y_train)
        pos_weight = torch.tensor([class_weight]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_val_auc = -np.inf
        patience_counter = 0
        best_model_path = os.path.join(ocean_output_dir, "best_transformer_model.pth")

        history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": [],
                   "val_ap": [], "val_f1_macro": []}
        best_epoch = 0

        for epoch in range(best_params["max_epochs"]):
            tr_loss, tr_auc, _ = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            vl_loss, vl_auc, _, vl_ap, vl_f1_macro, _, _ = validate_epoch(model, val_loader, criterion, DEVICE)

            history["train_loss"].append(tr_loss)
            history["train_auc"].append(tr_auc)
            history["val_loss"].append(vl_loss)
            history["val_auc"].append(vl_auc)
            history["val_ap"].append(vl_ap)
            history["val_f1_macro"].append(vl_f1_macro)

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_auc": tr_auc,
                "val_loss": vl_loss, "val_auc": vl_auc, "val_ap": vl_ap,
                "val_f1_macro": vl_f1_macro,
            }, step=epoch)

            if vl_auc > best_val_auc:
                best_val_auc = vl_auc
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                best_epoch = epoch
            else:
                patience_counter += 1

            scheduler.step(vl_auc)
            if patience_counter >= best_params["patience"]:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Final evaluation on all splits
        print("\n[6/6] Evaluating on all splits and saving results...")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))

        proba_train, labels_train = predict_on_split(model, train_loader, DEVICE)
        proba_val, labels_val = predict_on_split(model, val_loader, DEVICE)
        proba_test, labels_test = predict_on_split(model, test_loader, DEVICE)

        # Threshold chosen on the validation set (Youden's J statistic), then
        # applied to all splits for consistent, non-leaky reporting.
        fpr, tpr, thresholds = roc_curve(labels_val, proba_val)
        optimal_threshold = float(thresholds[np.argmax(tpr - fpr)])

        pred_train = (proba_train >= optimal_threshold).astype(int)
        pred_val = (proba_val >= optimal_threshold).astype(int)
        pred_test = (proba_test >= optimal_threshold).astype(int)

        metrics_train = compute_metrics_anomaly(labels_train, proba_train, pred_train, optimal_threshold)
        metrics_val = compute_metrics_anomaly(labels_val, proba_val, pred_val, optimal_threshold)
        metrics_test = compute_metrics_anomaly(labels_test, proba_test, pred_test, optimal_threshold)
        metrics_test_macro = compute_metrics_macro(labels_test, proba_test, pred_test, optimal_threshold)

        # Merge macro metrics (notably f1_macro) into metrics_test so it matches
        # the shape expected by the plotting notebook (results['metrics_test']['f1_macro']).
        metrics_test["f1_macro"] = metrics_test_macro["f1_macro"]
        metrics_test["precision_macro"] = metrics_test_macro["precision_macro"]
        metrics_test["recall_macro"] = metrics_test_macro["recall_macro"]

        # Same merge for train/val, for consistency across splits.
        metrics_train_macro = compute_metrics_macro(labels_train, proba_train, pred_train, optimal_threshold)
        metrics_val_macro = compute_metrics_macro(labels_val, proba_val, pred_val, optimal_threshold)
        metrics_train["f1_macro"] = metrics_train_macro["f1_macro"]
        metrics_val["f1_macro"] = metrics_val_macro["f1_macro"]

        for split_name, m in (("train", metrics_train), ("val", metrics_val), ("test", metrics_test)):
            for k, v in m.items():
                mlflow.log_metric(f"{split_name}_{k}", v)
        for k, v in metrics_test_macro.items():
            mlflow.log_metric(f"test_macro_{k}", v)

        # Test-set ROC curve (useful for plotting alongside the val-derived threshold)
        fpr_test, tpr_test, thresholds_test = roc_curve(labels_test, proba_test)

        # --- Save artifacts & reports ---
        mlflow.pytorch.log_model(model, artifact_path="model")
        pd.DataFrame(history).to_csv(os.path.join(ocean_output_dir, "training_history.csv"), index=False)
        pd.DataFrame(search_history).to_csv(os.path.join(ocean_output_dir, "halving_search_history.csv"), index=False)

        # Bundle everything a plotting script would need: training curves,
        # hyperparameter-search history, per-fold CV metrics, ROC data,
        # per-split predictions/probabilities/labels, and final metrics.
        results_data = {
            # --- Experiment context ---
            "ocean": ocean_name,
            "years_range": YEARS_RANGE,
            "preproc_dir": preproc_dir,
            "timestamp": datetime.now().isoformat(),

            # --- Hyperparameter search ---
            "best_params": best_params,
            "halving_search_history": search_history,

            # --- Cross-validation ---
            "kfold_metrics": metrics_kfold,
            "kfold_fold_details": kfold_fold_details,

            # --- Training curves (final model) ---
            "training_history": history,
            "best_epoch": int(best_epoch),

            # --- Labels, predictions & probabilities per split ---
            "labels_train": labels_train,
            "labels_val": labels_val,
            "labels_test": labels_test,
            "proba_train": proba_train,
            "proba_val": proba_val,
            "proba_test": proba_test,
            "pred_train": pred_train,
            "pred_val": pred_val,
            "pred_test": pred_test,

            # --- Thresholding & ROC curves ---
            "optimal_threshold": optimal_threshold,
            "roc_val": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
            "roc_test": {"fpr": fpr_test.tolist(), "tpr": tpr_test.tolist(), "thresholds": thresholds_test.tolist()},

            # --- Metrics ---
            "metrics_train": metrics_train,
            "metrics_val": metrics_val,
            "metrics_test": metrics_test,
            "metrics_test_macro": metrics_test_macro,

            # --- Model artifact locations ---
            "model_state_dict_path": best_model_path,
        }
        joblib.dump(results_data, os.path.join(ocean_output_dir, "transformer_results.joblib"))

        test_meta_with_pred = test_meta.copy()
        test_meta_with_pred["proba"] = proba_test
        test_meta_with_pred["pred"] = pred_test
        test_meta_with_pred.to_parquet(os.path.join(ocean_output_dir, "test_predictions.parquet"), index=False)

        config = {
            "ocean": ocean_name,
            "preproc_dir": preproc_dir,
            "output_dir": ocean_output_dir,
            "device": str(DEVICE),
            "timestamp": datetime.now().isoformat(),
            "best_params": best_params,
            "best_epoch": int(best_epoch),
            "optimal_threshold": optimal_threshold,
            "metrics_train": {k: float(v) for k, v in metrics_train.items()},
            "metrics_val": {k: float(v) for k, v in metrics_val.items()},
            "metrics_test": {k: float(v) for k, v in metrics_test.items()},
            "kfold_metrics": metrics_kfold,
        }
        with open(os.path.join(ocean_output_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        mlflow.log_artifacts(ocean_output_dir)

        print(f"\nClassification report on Train ({ocean_name}):")
        print(classification_report(labels_train, pred_train, target_names=["Normal", "Anomaly"]))

        print(f"\nClassification report on Val ({ocean_name}):")
        print(classification_report(labels_val, pred_val, target_names=["Normal", "Anomaly"]))

        print(f"\nClassification report on Test ({ocean_name}):")
        print(classification_report(labels_test, pred_test, target_names=["Normal", "Anomaly"]))

        # Also persist all three reports to disk (as dict -> DataFrame -> CSV)
        for split_name, y_true, y_pred in (
            ("train", labels_train, pred_train),
            ("val", labels_val, pred_val),
            ("test", labels_test, pred_test),
        ):
            report_dict = classification_report(
                y_true, y_pred, target_names=["Normal", "Anomaly"], output_dict=True
            )
            pd.DataFrame(report_dict).transpose().to_csv(
                os.path.join(ocean_output_dir, f"classification_report_{split_name}.csv")
            )

        print(f"Results successfully saved to: {ocean_output_dir}")

        del model, optimizer, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()


def main():
    print(f"Starting Multi-Ocean Transformer Pipeline on device: {DEVICE}")
    for ocean_name, preproc_dir in OCEANS.items():
        if os.path.exists(preproc_dir):
            process_ocean(ocean_name, preproc_dir)
        else:
            print(f"WARNING: Directory not found for {ocean_name}: {preproc_dir}")


if __name__ == "__main__":
    main()