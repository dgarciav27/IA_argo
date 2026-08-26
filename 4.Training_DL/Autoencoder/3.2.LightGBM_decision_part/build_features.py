"""
build_features.py
=====================
Feature engineering (autoencoder recon-error + features extra), parametrizado
por OCEAN y YEARS_RANGE. Corre un loop sobre todas las combinaciones definidas
en CONFIG y guarda los parquet resultantes en un output_dir distinto por
combinación.

To run before train lightGBM_decision_part
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import BallTree
import xarray as xr
from scipy.interpolate import interp1d

# CONFIG

OCEANS = ["atlantic", "pacific", "indian"]
YEARS_RANGES = ["2018_2022"]          # agrega más strings si tienes varios rangos

BASE_PREPROC_DIR = "/work/drgarcia/Dataset/DL_datasets"
BASE_RECON_DIR   = "/work/drgarcia/Models_and_results/Autoencoder"
WOA_DIR          = r"/work/drgarcia/Dataset/WOA"

USE_LATENT_FEATURES = False
LATENT_DIM = 32
SEED = 42

DEPTH_BANDS_BASE = [("surface", 0, 200), ("mid", 200, 1000), ("deep", 1000, 100000)]
SPIKE_SIGMA  = 4.0
DEEP_BAND_LO = 1000.0
DRIFT_WINDOW = 5

NEIGHBOR_TIME_WINDOW_DAYS = 20
NEIGHBOR_SPATIAL_POOL     = 50
NEIGHBOR_K_FINAL          = 10

SEVERITY_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
WOA_TEMP_VAR = "t_an"
WOA_PSAL_VAR = "s_an"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_paths(ocean, years_range):
    """Centraliza TODOS los paths que dependen de ocean/years_range."""
    preproc_dir = os.path.join(
        BASE_PREPROC_DIR, f"{ocean}_ocean/{years_range}/split_time_ascA_dmodeD_masked_grid100"
    )
    recon_dir = os.path.join(BASE_RECON_DIR, f"reconstructor_{ocean}_{years_range}")
    output_dir = os.path.join(BASE_RECON_DIR, f"LightGBM_{ocean}_{years_range}")
    os.makedirs(output_dir, exist_ok=True)
    return {
        "preproc_dir": preproc_dir,
        "recon_dir": recon_dir,
        "output_dir": output_dir,
        "norm_stats_path": os.path.join(recon_dir, "norm_stats_per_level.npz"),
        "best_model_path": os.path.join(recon_dir, "best_reconstructor.pth"),
        "ts_reg_path": os.path.join(output_dir, "ts_reg.joblib"),
    }


# WOA climatology
def load_woa_climatology(woa_dir):
    temp_list, psal_list = [], []
    for m in range(1, 13):
        f_t = os.path.join(woa_dir, f"woa23_decav_t{m:02d}_01.nc")
        f_s = os.path.join(woa_dir, f"woa23_decav_s{m:02d}_01.nc")
        ds_t = xr.open_dataset(f_t, decode_times=False)[["t_an"]].squeeze("time", drop=True)
        ds_s = xr.open_dataset(f_s, decode_times=False)[["s_an"]].squeeze("time", drop=True)
        ds_t = ds_t.expand_dims(month=[m])
        ds_s = ds_s.expand_dims(month=[m])
        temp_list.append(ds_t)
        psal_list.append(ds_s)
    ds_temp = xr.concat(temp_list, dim="month")
    ds_psal = xr.concat(psal_list, dim="month")
    return xr.merge([ds_temp, ds_psal]).load()


# Loading / preprocessing helpers

def load_split_raw(preproc_dir, name):
    X_temp = np.load(os.path.join(preproc_dir, f"{name}_X_temp.npy"))
    X_psal = np.load(os.path.join(preproc_dir, f"{name}_X_psal.npy"))
    mask   = np.load(os.path.join(preproc_dir, f"{name}_mask.npy"))
    meta   = pd.read_parquet(os.path.join(preproc_dir, f"{name}_meta.parquet"))
    labels = meta["is_bad"].values.astype(np.int8)
    return X_temp, X_psal, mask, meta, labels


def compute_gradient_channels(X, mask, P_grid):
    dP = np.gradient(P_grid).astype(np.float32)
    dP = np.where(dP == 0, 1e-3, dP)
    dX = np.gradient(X, axis=1) / dP[None, :]
    d2X = np.gradient(dX, axis=1) / dP[None, :]
    dX = np.where(mask, dX, 0.0).astype(np.float32)
    d2X = np.where(mask, d2X, 0.0).astype(np.float32)
    return dX, d2X


def build_profile_tensor(X_temp, X_psal, mask, dT, d2T, dS, d2S, p_norm_row,
                          T_mean_L, T_std_L, S_mean_L, S_std_L,
                          dT_mu, dT_sd, d2T_mu, d2T_sd, dS_mu, dS_sd, d2S_mu, d2S_sd):
    T_norm = np.where(mask, (X_temp - T_mean_L[None, :]) / T_std_L[None, :], 0.0).astype(np.float32)
    S_norm = np.where(mask, (X_psal - S_mean_L[None, :]) / S_std_L[None, :], 0.0).astype(np.float32)
    P_norm = np.broadcast_to(p_norm_row, T_norm.shape).astype(np.float32)
    M_chan = mask.astype(np.float32)
    dT_norm  = np.where(mask, (dT  - dT_mu)  / dT_sd,  0.0).astype(np.float32)
    d2T_norm = np.where(mask, (d2T - d2T_mu) / d2T_sd, 0.0).astype(np.float32)
    dS_norm  = np.where(mask, (dS  - dS_mu)  / dS_sd,  0.0).astype(np.float32)
    d2S_norm = np.where(mask, (d2S - d2S_mu) / d2S_sd, 0.0).astype(np.float32)
    profiles = np.stack([T_norm, S_norm, P_norm, M_chan, dT_norm, d2T_norm, dS_norm, d2S_norm], axis=1)
    target = np.stack([T_norm, S_norm], axis=1)
    return profiles.astype(np.float32), target.astype(np.float32)


def denorm_temp(vals, T_mean_L, T_std_L):
    return vals * T_std_L + T_mean_L


def denorm_psal(vals, S_mean_L, S_std_L):
    return vals * S_std_L + S_mean_L


# Misma arquitectura que el reconstructor entrenado -- debe coincidir exacto.
class ProfileReconstructor(nn.Module):
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


# ============================================================================
# Feature helpers (idénticos a tu Parte 1, sin depender de globals de ocean)
# ============================================================================

def compute_profile_severity(meta):
    t = meta["PROFILE_TEMP_QC"].map(SEVERITY_ORDER)
    s = meta["PROFILE_PSAL_QC"].map(SEVERITY_ORDER)
    return np.where(t.notna() & s.notna(), np.maximum(t.values, s.values), np.nan)


def recon_error_channel_mean(recon, target, mask, channel):
    diff = (recon[:, channel, :] - target[:, channel, :]) ** 2
    mask_f = mask.astype(np.float32)
    diff = diff * mask_f
    valid = np.clip(mask_f.sum(axis=1), 1, None)
    return (diff.sum(axis=1) / valid).astype(np.float32)


def compute_deep_band_psal_bias(X_psal, mask, S_mean_L, pressure_grid, deep_lo=DEEP_BAND_LO):
    deep_level_mask = pressure_grid >= deep_lo
    bias = X_psal - S_mean_L[None, :]
    band_mask = mask & deep_level_mask[None, :]
    bias_masked = np.where(band_mask, bias, np.nan)
    with np.errstate(invalid="ignore"):
        result = np.nanmean(bias_masked, axis=1)
    return np.nan_to_num(result, nan=0.0).astype(np.float32)


def compute_float_drift_features(meta, deep_bias, window=DRIFT_WINDOW, long_window=10):
    df = meta[["PLATFORM_NUMBER", "DIRECTION", "date"]].copy()
    df["orig_idx"] = np.arange(len(df))
    df["deep_bias"] = deep_bias
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df = df.sort_values(["PLATFORM_NUMBER", "DIRECTION", "date"])

    slope = np.zeros(len(meta), dtype=np.float32)
    median_dev = np.zeros(len(meta), dtype=np.float32)
    slope_long = np.zeros(len(meta), dtype=np.float32)
    same_month_last_year_dev = np.zeros(len(meta), dtype=np.float32)

    for _, group in df.groupby(["PLATFORM_NUMBER", "DIRECTION"]):
        idxs = group["orig_idx"].values
        biases = group["deep_bias"].values
        months = group["month"].values
        years = group["year"].values

        for k in range(len(idxs)):
            cur_idx = idxs[k]

            hist = biases[max(0, k - window + 1): k + 1]
            if len(hist) >= 2:
                slope[cur_idx] = float(np.polyfit(np.arange(len(hist)), hist, 1)[0])
            median_dev[cur_idx] = float(biases[k] - np.median(biases[:k + 1]))

            hist_long = biases[max(0, k - long_window + 1): k + 1]
            if len(hist_long) >= 3:
                slope_long[cur_idx] = float(np.polyfit(np.arange(len(hist_long)), hist_long, 1)[0])

            target_year = years[k] - 1
            match = (months[:k] == months[k]) & (years[:k] == target_year)
            if match.any():
                same_month_last_year_dev[cur_idx] = float(biases[k] - np.mean(biases[:k][match]))

    return slope, median_dev, slope_long, same_month_last_year_dev


def compute_previous_cycle_deviation(meta, X_temp, X_psal, mask):
    df = meta[["PLATFORM_NUMBER", "CYCLE_NUMBER", "DIRECTION", "date"]].copy()
    df["orig_idx"] = np.arange(len(df))
    df = df.sort_values(["PLATFORM_NUMBER", "DIRECTION", "date"])

    dev_temp = np.full(len(meta), np.nan, dtype=np.float32)
    dev_psal = np.full(len(meta), np.nan, dtype=np.float32)

    for (plat, direction), group in df.groupby(["PLATFORM_NUMBER", "DIRECTION"]):
        idxs = group["orig_idx"].values
        for k in range(1, len(idxs)):
            cur_idx, prev_idx = idxs[k], idxs[k - 1]
            valid = mask[cur_idx] & mask[prev_idx]
            if valid.sum() < 5:
                continue
            dev_temp[cur_idx] = float(np.mean(np.abs(X_temp[cur_idx][valid] - X_temp[prev_idx][valid])))
            dev_psal[cur_idx] = float(np.mean(np.abs(X_psal[cur_idx][valid] - X_psal[prev_idx][valid])))

    return dev_temp, dev_psal


def simple_potential_density(temp, psal):
    T, S = temp, psal
    S_safe = np.clip(S, 0.0, None)
    rho = (999.842594 + 6.793952e-2 * T - 9.09529e-3 * T**2
           + 1.001685e-4 * T**3 - 1.120083e-6 * T**4 + 6.536332e-9 * T**5
           + (0.824493 - 4.0899e-3 * T + 7.6438e-5 * T**2
              - 8.2467e-7 * T**3 + 5.3875e-9 * T**4) * S
           + (-5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T**2) * S_safe**1.5
           + 4.8314e-4 * S**2)
    return rho - 1000.0


def compute_density_inversion_feature(X_temp, X_psal, mask):
    N = X_temp.shape[0]
    sigma0 = simple_potential_density(X_temp, X_psal)
    sigma0 = np.where(mask, sigma0, np.nan)

    inversion_frac = np.zeros(N, dtype=np.float32)
    inversion_max  = np.zeros(N, dtype=np.float32)

    for i in range(N):
        s = sigma0[i]
        valid_pairs = ~np.isnan(s[:-1]) & ~np.isnan(s[1:])
        if valid_pairs.sum() < 3:
            continue
        delta = s[1:][valid_pairs] - s[:-1][valid_pairs]
        n_inv = (delta < -0.01).sum()
        inversion_frac[i] = n_inv / valid_pairs.sum()
        inversion_max[i] = float(-delta.min()) if delta.min() < 0 else 0.0

    return inversion_frac, inversion_max


def compute_zscore_shape_features(T_norm, S_norm, mask):
    N = T_norm.shape[0]
    skew_T = np.zeros(N, dtype=np.float32); kurt_T = np.zeros(N, dtype=np.float32)
    skew_S = np.zeros(N, dtype=np.float32); kurt_S = np.zeros(N, dtype=np.float32)
    for i in range(N):
        m = mask[i]
        if m.sum() < 5:
            continue
        skew_T[i] = skew(T_norm[i][m]); kurt_T[i] = kurtosis(T_norm[i][m])
        skew_S[i] = skew(S_norm[i][m]); kurt_S[i] = kurtosis(S_norm[i][m])
    return skew_T, kurt_T, skew_S, kurt_S


def compute_local_spike_features(T_norm, S_norm, d2T_norm, d2S_norm, mask, spike_sigma=SPIKE_SIGMA):
    N = T_norm.shape[0]
    max_abs_zT = np.zeros(N, dtype=np.float32); max_abs_zS = np.zeros(N, dtype=np.float32)
    spike_rate_T = np.zeros(N, dtype=np.float32); spike_rate_S = np.zeros(N, dtype=np.float32)
    for i in range(N):
        m = mask[i]
        if m.sum() == 0:
            continue
        max_abs_zT[i] = float(np.max(np.abs(T_norm[i][m])))
        max_abs_zS[i] = float(np.max(np.abs(S_norm[i][m])))
        spike_rate_T[i] = float(np.mean(np.abs(d2T_norm[i][m]) > spike_sigma))
        spike_rate_S[i] = float(np.mean(np.abs(d2S_norm[i][m]) > spike_sigma))
    return max_abs_zT, max_abs_zS, spike_rate_T, spike_rate_S


def compute_stat_features(X, mask, prefix):
    N, L = X.shape
    means    = np.full(N, np.nan, dtype=np.float32)
    stds     = np.full(N, np.nan, dtype=np.float32)
    mins     = np.full(N, np.nan, dtype=np.float32)
    maxs     = np.full(N, np.nan, dtype=np.float32)
    mean_grad = np.full(N, np.nan, dtype=np.float32)
    max_grad  = np.full(N, np.nan, dtype=np.float32)
    grad_std  = np.full(N, np.nan, dtype=np.float32)

    for i in range(N):
        vals = X[i][mask[i]]
        if vals.size == 0:
            continue
        means[i] = vals.mean()
        stds[i]  = vals.std()
        mins[i]  = vals.min()
        maxs[i]  = vals.max()
        diffs = np.abs(np.diff(vals))
        if diffs.size:
            mean_grad[i] = diffs.mean()
            max_grad[i]  = diffs.max()
            grad_std[i]  = diffs.std()
        else:
            mean_grad[i] = 0.0
            max_grad[i]  = 0.0
            grad_std[i]  = 0.0

    ranges = maxs - mins
    return {
        f"{prefix}_mean":      np.nan_to_num(means, nan=0.0).astype(np.float32),
        f"{prefix}_std":       np.nan_to_num(stds, nan=0.0).astype(np.float32),
        f"{prefix}_min":       np.nan_to_num(mins, nan=0.0).astype(np.float32),
        f"{prefix}_max":       np.nan_to_num(maxs, nan=0.0).astype(np.float32),
        f"{prefix}_range":     np.nan_to_num(ranges, nan=0.0).astype(np.float32),
        f"{prefix}_mean_grad": np.nan_to_num(mean_grad, nan=0.0).astype(np.float32),
        f"{prefix}_max_grad":  np.nan_to_num(max_grad, nan=0.0).astype(np.float32),
        f"{prefix}_grad_std":  np.nan_to_num(grad_std, nan=0.0).astype(np.float32),
    }


def get_woa_profile(lat, lon, month, pressure_grid, woa_ds):
    try:
        point = woa_ds.sel(lat=lat, lon=lon, month=month, method="nearest")
        clim_depth = point["depth"].values
        clim_temp = point[WOA_TEMP_VAR].values
        clim_psal = point[WOA_PSAL_VAR].values
        valid = ~np.isnan(clim_temp) & ~np.isnan(clim_psal)
        if valid.sum() < 3:
            return (np.full_like(pressure_grid, np.nan, dtype=np.float32),
                    np.full_like(pressure_grid, np.nan, dtype=np.float32))
        f_temp = interp1d(clim_depth[valid], clim_temp[valid], bounds_error=False, fill_value=np.nan)
        f_psal = interp1d(clim_depth[valid], clim_psal[valid], bounds_error=False, fill_value=np.nan)
        return f_temp(pressure_grid).astype(np.float32), f_psal(pressure_grid).astype(np.float32)
    except Exception:
        return (np.full_like(pressure_grid, np.nan, dtype=np.float32),
                np.full_like(pressure_grid, np.nan, dtype=np.float32))


def compute_woa_residual_features(meta, X_temp, X_psal, mask, pressure_grid, woa_ds, depth_bands):
    N = len(meta)
    lat = meta["LATITUDE"].values
    lon = meta["LONGITUDE"].values
    month = pd.to_datetime(meta["date"]).dt.month.values

    lat_bin = np.round(lat * 2) / 2
    lon_bin = np.round(lon * 2) / 2
    cache = {}

    resid_temp = np.full((N, len(pressure_grid)), np.nan, dtype=np.float32)
    resid_psal = np.full((N, len(pressure_grid)), np.nan, dtype=np.float32)

    for i in range(N):
        key = (lat_bin[i], lon_bin[i], month[i])
        if key not in cache:
            cache[key] = get_woa_profile(lat[i], lon[i], month[i], pressure_grid, woa_ds)
        clim_t, clim_s = cache[key]
        resid_temp[i] = X_temp[i] - clim_t
        resid_psal[i] = X_psal[i] - clim_s

    feats = {}
    m = mask.astype(bool)

    def masked_mean(arr, mask2d):
        a = np.where(mask2d, arr, np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanmean(a, axis=1)

    def masked_absmax(arr, mask2d):
        a = np.where(mask2d, np.abs(arr), -np.inf)
        out = np.max(a, axis=1)
        out[np.isneginf(out)] = np.nan
        return out

    feats["woa_resid_temp_mean"] = masked_mean(resid_temp, m)
    feats["woa_resid_psal_mean"] = masked_mean(resid_psal, m)
    feats["woa_resid_temp_absmax"] = masked_absmax(resid_temp, m)
    feats["woa_resid_psal_absmax"] = masked_absmax(resid_psal, m)

    for name, lo, hi in depth_bands:
        band_level = (pressure_grid >= lo) & (pressure_grid < hi)
        band_mask = m & band_level[None, :]
        feats[f"woa_resid_temp_{name}"] = masked_mean(resid_temp, band_mask)
        feats[f"woa_resid_psal_{name}"] = masked_mean(resid_psal, band_mask)

    return {k: np.nan_to_num(v, nan=0.0).astype(np.float32) for k, v in feats.items()}


def fit_ts_relationship(X_temp, X_psal, mask, pressure_grid, idx_subset, deep_lo=DEEP_BAND_LO):
    deep_level_mask = pressure_grid >= deep_lo
    band_mask = mask[idx_subset] & deep_level_mask[None, :]
    T = X_temp[idx_subset][band_mask]
    S = X_psal[idx_subset][band_mask]
    P = np.broadcast_to(pressure_grid[None, :], mask[idx_subset].shape)[band_mask]
    P_norm = P / pressure_grid.max()
    feats = np.stack([T, T**2, P_norm, T * P_norm], axis=1)
    return LinearRegression().fit(feats, S)


def compute_ts_residual(X_temp, X_psal, mask, pressure_grid, reg, deep_lo=DEEP_BAND_LO):
    deep_level_mask = pressure_grid >= deep_lo
    band_mask = mask & deep_level_mask[None, :]
    N, L = X_temp.shape
    P_norm_row = (pressure_grid / pressure_grid.max()).astype(np.float32)
    P_norm = np.broadcast_to(P_norm_row[None, :], (N, L))

    T_flat = X_temp[band_mask]
    P_flat = P_norm[band_mask]
    S_flat = X_psal[band_mask]
    feats = np.stack([T_flat, T_flat**2, P_flat, T_flat * P_flat], axis=1)
    pred_flat = reg.predict(feats)
    resid_flat = (S_flat - pred_flat).astype(np.float32)

    residual_profile = np.full((N, L), np.nan, dtype=np.float32)
    residual_profile[band_mask] = resid_flat
    with np.errstate(invalid="ignore"):
        result = np.nanmean(residual_profile, axis=1)
    return np.nan_to_num(result, nan=0.0).astype(np.float32)


def compute_spatial_neighbor_features(meta, X_temp, X_psal, mask, pressure_grid,
                                       tree, ref_coords_rad, ref_X_temp, ref_X_psal,
                                       ref_mask, ref_dates, ref_platform,
                                       query_platform, query_dates,
                                       time_window_days=NEIGHBOR_TIME_WINDOW_DAYS,
                                       pool=NEIGHBOR_SPATIAL_POOL,
                                       k_final=NEIGHBOR_K_FINAL,
                                       exclude_self=False):
    N = X_temp.shape[0]
    coords_rad = np.radians(meta[["LATITUDE", "LONGITUDE"]].values.astype(np.float64))
    query_dates_d = pd.to_datetime(query_dates).values.astype("datetime64[D]")

    dist, idx = tree.query(coords_rad, k=min(pool, len(ref_dates)))

    resid_temp_mean = np.zeros(N, dtype=np.float32)
    resid_psal_mean = np.zeros(N, dtype=np.float32)
    resid_temp_max  = np.zeros(N, dtype=np.float32)
    resid_psal_max  = np.zeros(N, dtype=np.float32)
    n_neighbors_used = np.zeros(N, dtype=np.int32)

    for i in range(N):
        cand_idx = idx[i]
        day_diff = np.abs((ref_dates[cand_idx] - query_dates_d[i]).astype("timedelta64[D]").astype(int))
        time_ok = day_diff <= time_window_days
        plat_ok = ref_platform[cand_idx] != query_platform[i]
        valid = time_ok & plat_ok
        if exclude_self:
            valid = valid & (dist[i] > 1e-8)

        good_idx = cand_idx[valid][:k_final]
        n_neighbors_used[i] = len(good_idx)
        if len(good_idx) < 3:
            continue

        neigh_T = ref_X_temp[good_idx]
        neigh_S = ref_X_psal[good_idx]
        neigh_mask = ref_mask[good_idx]

        with np.errstate(invalid="ignore"):
            med_T = np.nanmedian(np.where(neigh_mask, neigh_T, np.nan), axis=0)
            med_S = np.nanmedian(np.where(neigh_mask, neigh_S, np.nan), axis=0)

        m = mask[i].astype(bool)
        both_valid = m & ~np.isnan(med_T) & ~np.isnan(med_S)
        if both_valid.sum() < 3:
            continue

        dT = X_temp[i][both_valid] - med_T[both_valid]
        dS = X_psal[i][both_valid] - med_S[both_valid]
        resid_temp_mean[i] = float(np.mean(dT))
        resid_psal_mean[i] = float(np.mean(dS))
        resid_temp_max[i]  = float(np.max(np.abs(dT)))
        resid_psal_max[i]  = float(np.max(np.abs(dS)))

    return {
        "neighbor_resid_temp_mean": resid_temp_mean,
        "neighbor_resid_psal_mean": resid_psal_mean,
        "neighbor_resid_temp_max":  resid_temp_max,
        "neighbor_resid_psal_max":  resid_psal_max,
        "neighbor_n_used": n_neighbors_used.astype(np.float32),
    }


# ============================================================================
# Build features para un (ocean, years_range)
# ============================================================================

def build_features_for_ocean(ocean, years_range, woa_ds):
    paths = get_paths(ocean, years_range)
    preproc_dir = paths["preproc_dir"]

    print(f"\n{'='*90}\nFEATURE ENGINEERING — ocean={ocean}  years_range={years_range}\n{'='*90}")

    with open(os.path.join(preproc_dir, "norm_stats.json")) as f:
        norm_stats_original = json.load(f)
    pressure_grid = np.load(os.path.join(preproc_dir, "pressure_grid.npy"))
    n_vert_levels = len(pressure_grid)
    grid_max_pres = norm_stats_original["PRES"]["grid_max"]
    p_norm_row = (pressure_grid / grid_max_pres).astype(np.float32)
    depth_bands = [(name, lo, min(hi, float(pressure_grid.max()) + 1.0)) for name, lo, hi in DEPTH_BANDS_BASE]

    norm_npz = np.load(paths["norm_stats_path"])
    T_mean_L, T_std_L = norm_npz["T_mean_L"], norm_npz["T_std_L"]
    S_mean_L, S_std_L = norm_npz["S_mean_L"], norm_npz["S_std_L"]
    dT_mu, dT_sd = float(norm_npz["dT_mu"]), float(norm_npz["dT_sd"])
    d2T_mu, d2T_sd = float(norm_npz["d2T_mu"]), float(norm_npz["d2T_sd"])
    dS_mu, dS_sd = float(norm_npz["dS_mu"]), float(norm_npz["dS_sd"])
    d2S_mu, d2S_sd = float(norm_npz["d2S_mu"]), float(norm_npz["d2S_sd"])

    if not os.path.exists(paths["best_model_path"]):
        raise FileNotFoundError(f"No existe el modelo reconstructor: {paths['best_model_path']}")

    model = ProfileReconstructor(
        n_channels_in=8, n_channels_out=2, seq_len=n_vert_levels, latent_dim=LATENT_DIM
    ).to(DEVICE)
    model.load_state_dict(torch.load(paths["best_model_path"], map_location=DEVICE))
    model.eval()
    print(f"  Reconstructor loaded on {DEVICE} for {ocean}/{years_range}")

    # --- ts_reg, fit una sola vez sobre train-normal de este ocean/years_range ---
    X_temp_train_raw, X_psal_train_raw, mask_train_raw, meta_train_raw, labels_train_raw = \
        load_split_raw(preproc_dir, "train")
    idx_normal_train = (labels_train_raw == 0)
    ts_reg = fit_ts_relationship(X_temp_train_raw, X_psal_train_raw, mask_train_raw,
                                  pressure_grid, idx_normal_train)
    joblib.dump(ts_reg, paths["ts_reg_path"])

    # --- spatial neighbor index sobre train-normal ---
    ref_meta = meta_train_raw[idx_normal_train].reset_index(drop=True)
    ref_X_temp = X_temp_train_raw[idx_normal_train]
    ref_X_psal = X_psal_train_raw[idx_normal_train]
    ref_mask = mask_train_raw[idx_normal_train]
    ref_dates = pd.to_datetime(ref_meta["date"]).values.astype("datetime64[D]")
    ref_platform = ref_meta["PLATFORM_NUMBER"].values
    ref_coords_rad = np.radians(ref_meta[["LATITUDE", "LONGITUDE"]].values.astype(np.float64))
    neighbor_tree = BallTree(ref_coords_rad, metric="haversine")
    print(f"  Neighbor index built on {len(ref_meta):,} normal train profiles")

    def _build_split(split_name):
        X_temp, X_psal, mask, meta, labels = load_split_raw(preproc_dir, split_name)
        dT, d2T = compute_gradient_channels(X_temp, mask, pressure_grid)
        dS, d2S = compute_gradient_channels(X_psal, mask, pressure_grid)
        profiles, target = build_profile_tensor(
            X_temp, X_psal, mask, dT, d2T, dS, d2S, p_norm_row,
            T_mean_L, T_std_L, S_mean_L, S_std_L,
            dT_mu, dT_sd, d2T_mu, d2T_sd, dS_mu, dS_sd, d2S_mu, d2S_sd,
        )
        with torch.no_grad():
            recon, z = model(torch.from_numpy(profiles).to(DEVICE))
            recon, z = recon.cpu().numpy(), z.cpu().numpy()

        temp_true_dn = np.apply_along_axis(lambda v: denorm_temp(v, T_mean_L, T_std_L), 1, target[:, 0, :])
        temp_pred_dn = np.apply_along_axis(lambda v: denorm_temp(v, T_mean_L, T_std_L), 1, recon[:, 0, :])
        psal_true_dn = np.apply_along_axis(lambda v: denorm_psal(v, S_mean_L, S_std_L), 1, target[:, 1, :])
        psal_pred_dn = np.apply_along_axis(lambda v: denorm_psal(v, S_mean_L, S_std_L), 1, recon[:, 1, :])

        err_t = (temp_true_dn - temp_pred_dn) ** 2
        err_s = (psal_true_dn - psal_pred_dn) ** 2
        m = mask.astype(bool)

        def masked_rmse(err2, mask2d):
            e = np.where(mask2d, err2, np.nan)
            with np.errstate(invalid="ignore"):
                return np.sqrt(np.nanmean(e, axis=1))

        def masked_peak(err2, mask2d):
            e = np.where(mask2d, err2, -np.inf)
            return np.sqrt(np.max(e, axis=1))

        feats = {}
        feats["rmse_temp_overall"] = masked_rmse(err_t, m)
        feats["rmse_psal_overall"] = masked_rmse(err_s, m)
        feats["peak_err_temp"] = masked_peak(err_t, m)
        feats["peak_err_psal"] = masked_peak(err_s, m)
        feats["coverage"] = m.mean(axis=1)

        for name, lo, hi in depth_bands:
            band_level = (pressure_grid >= lo) & (pressure_grid < hi)
            band_mask = m & band_level[None, :]
            feats[f"rmse_temp_{name}"] = masked_rmse(err_t, band_mask)
            feats[f"rmse_psal_{name}"] = masked_rmse(err_s, band_mask)

        feats["LATITUDE"] = meta["LATITUDE"].values.astype(np.float32)
        feats["LONGITUDE"] = meta["LONGITUDE"].values.astype(np.float32)
        feats["err_channel_temp"] = recon_error_channel_mean(recon, target, mask, channel=0)
        feats["err_channel_psal"] = recon_error_channel_mean(recon, target, mask, channel=1)

        deep_bias = compute_deep_band_psal_bias(X_psal, mask, S_mean_L, pressure_grid)
        feats["deep_psal_bias"] = deep_bias
        slope, median_dev, slope_long, same_month_dev = compute_float_drift_features(meta, deep_bias)
        feats["drift_slope"] = slope
        feats["drift_median_dev"] = median_dev
        feats["drift_slope_long"] = slope_long
        feats["drift_same_month_last_year_dev"] = same_month_dev

        dev_temp, dev_psal = compute_previous_cycle_deviation(meta, X_temp, X_psal, mask)
        feats["prevcycle_dev_temp"] = np.nan_to_num(dev_temp, nan=0.0)
        feats["prevcycle_dev_psal"] = np.nan_to_num(dev_psal, nan=0.0)

        inv_frac, inv_max = compute_density_inversion_feature(X_temp, X_psal, mask)
        feats["density_inv_frac"] = inv_frac
        feats["density_inv_max"] = inv_max

        skew_T, kurt_T, skew_S, kurt_S = compute_zscore_shape_features(
            target[:, 0, :], target[:, 1, :], mask
        )
        feats["skew_temp"], feats["kurt_temp"] = skew_T, kurt_T
        feats["skew_psal"], feats["kurt_psal"] = skew_S, kurt_S

        d2T_norm = np.where(mask, (d2T - d2T_mu) / d2T_sd, 0.0).astype(np.float32)
        d2S_norm = np.where(mask, (d2S - d2S_mu) / d2S_sd, 0.0).astype(np.float32)
        max_zT, max_zS, spike_T, spike_S = compute_local_spike_features(
            target[:, 0, :], target[:, 1, :], d2T_norm, d2S_norm, mask
        )
        feats["max_abs_z_temp"], feats["max_abs_z_psal"] = max_zT, max_zS
        feats["spike_rate_temp"], feats["spike_rate_psal"] = spike_T, spike_S

        feats["ts_residual"] = compute_ts_residual(X_temp, X_psal, mask, pressure_grid, ts_reg)

        neighbor_feats = compute_spatial_neighbor_features(
            meta, X_temp, X_psal, mask, pressure_grid,
            neighbor_tree, ref_coords_rad, ref_X_temp, ref_X_psal,
            ref_mask, ref_dates, ref_platform,
            query_platform=meta["PLATFORM_NUMBER"].values,
            query_dates=meta["date"].values,
            exclude_self=(split_name == "train"),
        )
        feats.update(neighbor_feats)

        woa_feats = compute_woa_residual_features(meta, X_temp, X_psal, mask, pressure_grid, woa_ds, depth_bands)
        feats.update(woa_feats)

        feats.update(compute_stat_features(X_temp, mask, "TEMP"))
        feats.update(compute_stat_features(X_psal, mask, "PSAL"))
        P_broadcast = np.broadcast_to(pressure_grid[None, :], mask.shape)
        feats.update(compute_stat_features(P_broadcast, mask, "PRES"))

        df_feat = pd.DataFrame(feats)
        for name, _, _ in depth_bands:
            df_feat[f"rmse_temp_{name}"] = df_feat[f"rmse_temp_{name}"].fillna(df_feat["rmse_temp_overall"])
            df_feat[f"rmse_psal_{name}"] = df_feat[f"rmse_psal_{name}"].fillna(df_feat["rmse_psal_overall"])

        if USE_LATENT_FEATURES:
            for i in range(LATENT_DIM):
                df_feat[f"z_{i:02d}"] = z[:, i]

        df_feat["mse_total"] = (feats["rmse_temp_overall"] ** 2 + feats["rmse_psal_overall"] ** 2) / 2.0
        df_feat["is_bad"] = labels
        df_feat["severity"] = compute_profile_severity(meta)
        df_feat["PLATFORM_NUMBER"] = meta["PLATFORM_NUMBER"].values
        df_feat["CYCLE_NUMBER"] = meta["CYCLE_NUMBER"].values
        return df_feat, meta

    df_train, _ = _build_split("train")
    df_val, _ = _build_split("val")
    df_test, meta_test = _build_split("test")

    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        df.to_parquet(os.path.join(paths["output_dir"], f"anomaly_features_{name}.parquet"), index=False)
        print(f"  {name}: {len(df):,} profiles, {df['is_bad'].mean():.2%} anomalous")

    extra_cols = ["PLATFORM_NUMBER", "CYCLE_NUMBER", "DIRECTION", "LATITUDE", "LONGITUDE",
                  "date", "TEMP_is_bad", "PSAL_is_bad", "PRES_is_bad"]
    extra_cols = [c for c in extra_cols if c in meta_test.columns]
    meta_test[extra_cols].to_parquet(os.path.join(paths["output_dir"], "test_meta_extra.parquet"), index=False)

    # liberar memoria de GPU antes de pasar al siguiente ocean/years_range
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return paths

# MAIN — loop sobre oceans x years_range
def main():
    print("Loading WOA23 climatology once (reused across oceans/years)...")
    woa_ds = load_woa_climatology(WOA_DIR)

    for years_range in YEARS_RANGES:
        for ocean in OCEANS:
            paths = get_paths(ocean, years_range)
            if not os.path.exists(paths["preproc_dir"]):
                print(f"\n WARNING: no existe {paths['preproc_dir']}, se salta {ocean}/{years_range}")
                continue
            try:
                build_features_for_ocean(ocean, years_range, woa_ds)
            except Exception as e:
                print(f"\n ERROR en {ocean}/{years_range}: {e}")
                continue

    print("\nListo. Los parquet de cada ocean/years_range quedaron en su output_dir "
          "(mismo patrón que usa 02_train_lgbm.py para leerlos).")


if __name__ == "__main__":
    main()