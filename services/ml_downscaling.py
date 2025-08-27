# services/ml_downscaling.py
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_pinball_loss,
)
from scipy.ndimage import uniform_filter, sobel, laplace, gaussian_filter

# Optional boosters
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:  # pragma: no cover
    def ssim(*args, **kwargs):
        return float("nan")

from .climate_ops import find_lat_lon

__all__ = [
    "align_and_prepare",
    "make_time_split_dataset",
    "train_downscaling",
    "hr_metrics_from_residuals",
    "quantile_coverage",
    "last_step_spatial_eval",
    "baseline_last_step_ssim",
]

# -----------------------------
# Feature engineering helpers
# -----------------------------
def _context_features_multiscale(lr3d: np.ndarray) -> Dict[str, np.ndarray]:
    """Multi-scale local features on LR↑. Returns dict with arrays (t,y,x)."""
    def _std_win(arr, k):
        m = uniform_filter(arr, size=(1, k, k), mode="nearest")
        m2 = uniform_filter(arr**2, size=(1, k, k), mode="nearest")
        return np.sqrt(np.maximum(m2 - m**2, 0.0))

    std3 = _std_win(lr3d, 3)
    std5 = _std_win(lr3d, 5)
    std9 = _std_win(lr3d, 9)

    gy = sobel(lr3d, axis=1, mode="nearest")
    gx = sobel(lr3d, axis=2, mode="nearest")
    grad = np.hypot(gx, gy)

    lap_ = laplace(lr3d, mode="nearest")

    g1 = gaussian_filter(lr3d, sigma=(0, 1.0, 1.0))
    g2 = gaussian_filter(lr3d, sigma=(0, 2.0, 2.0))
    dog = g1 - g2

    return {"std3": std3, "std5": std5, "std9": std9, "grad": grad, "lap": lap_, "dog": dog}


def _build_lag_stack(lr3d: np.ndarray, lags: Tuple[int, ...]) -> Dict[int, np.ndarray]:
    """Return dict {k: lagged LR↑ at t-k}, padded with NaNs for first k steps."""
    out = {}
    T, H, W = lr3d.shape
    for k in lags:
        arr = np.full_like(lr3d, np.nan, dtype=float)
        if k < T:
            arr[k:, :, :] = lr3d[:-k, :, :]
        out[k] = arr
    return out


def _flatten_arrays(
    lr3d: np.ndarray,
    hr3d: np.ndarray,
    month_vals: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    use_context: bool,
    lag_steps: Tuple[int, ...],
):
    """
    Vectorize features/targets consistently and return:
      X, y, lr_flat, weights, ids
    Includes automatic fallback if context features wipe out samples.
    """
    T, H, W = lr3d.shape
    ids_grid = np.arange(H * W, dtype=np.int32)
    id_v = np.tile(ids_grid, T)

    lr_v = lr3d.reshape(-1)
    hr_v = hr3d.reshape(-1)
    m_v = np.tile(month_vals, H * W)
    lat_v = np.repeat(lat2d.reshape(-1), T)
    lon_v = np.repeat(lon2d.reshape(-1), T)

    # Lag features
    lag_dict = _build_lag_stack(lr3d, lag_steps)
    lag_vs = [lag_dict[k].reshape(-1) for k in lag_steps]

    # Base finite mask (require all lags present)
    mask = (
        np.isfinite(lr_v) & np.isfinite(hr_v) & np.isfinite(m_v)
        & np.isfinite(lat_v) & np.isfinite(lon_v)
    )
    for v in lag_vs:
        mask &= np.isfinite(v)

    # Start with base subset
    lr_b, hr_b, m_b, lat_b, lon_b = lr_v[mask], hr_v[mask], m_v[mask], lat_v[mask], lon_v[mask]
    id_b = id_v[mask]
    lag_b_list = [v[mask] for v in lag_vs]

    if lr_b.size == 0:
        n_feats = 5 + len(lag_steps) + (6 if use_context else 0)
        return np.empty((0, n_feats)), np.empty((0,)), np.empty((0,)), None, np.empty((0,), dtype=np.int32)

    # Context features (optional)
    ctx_feats = []
    weights = None
    context_applied = False
    if use_context:
        ctx = _context_features_multiscale(lr3d)
        std3 = ctx["std3"].reshape(-1)[mask]
        std5 = ctx["std5"].reshape(-1)[mask]
        std9 = ctx["std9"].reshape(-1)[mask]
        grad = ctx["grad"].reshape(-1)[mask]
        lap_ = ctx["lap"].reshape(-1)[mask]
        dog = ctx["dog"].reshape(-1)[mask]

        ctx_mask = (
            np.isfinite(std3) & np.isfinite(std5) & np.isfinite(std9)
            & np.isfinite(grad) & np.isfinite(lap_) & np.isfinite(dog)
        )
        # If context kills all samples, fallback to base (no context)
        if np.any(ctx_mask):
            context_applied = True
            lr_b, hr_b, m_b, lat_b, lon_b = lr_b[ctx_mask], hr_b[ctx_mask], m_b[ctx_mask], lat_b[ctx_mask], lon_b[ctx_mask]
            id_b = id_b[ctx_mask]
            lag_b_list = [v[ctx_mask] for v in lag_b_list]
            std3, std5, std9, grad, lap_, dog = std3[ctx_mask], std5[ctx_mask], std9[ctx_mask], grad[ctx_mask], lap_[ctx_mask], dog[ctx_mask]
            ctx_feats = [std3, std5, std9, grad, lap_, dog]

            # Edge-weighted sampling weights (safe normalization)
            g = np.abs(grad) + 0.5 * std3 + 0.25 * std5
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            if g.size > 0 and np.isfinite(g).any():
                g_min, g_max = np.min(g), np.max(g)
                if g_max > g_min:
                    g = (g - g_min) / (g_max - g_min)
                    s = g.sum()
                    weights = g / s if s > 0 else np.ones_like(g) / g.size
                else:
                    weights = np.ones_like(g) / max(g.size, 1)

    # Build design matrix
    m_sin = np.sin(2 * np.pi * m_b / 12.0)
    m_cos = np.cos(2 * np.pi * m_b / 12.0)
    feats = [lr_b, lat_b, lon_b, m_sin, m_cos] + lag_b_list + ctx_feats
    n_feats = 5 + len(lag_steps) + (6 if (use_context and context_applied) else 0)

    X = np.column_stack(feats) if lr_b.size else np.empty((0, n_feats))
    y = hr_b - lr_b
    lr_flat = lr_b
    return X, y, lr_flat, weights, id_b


# -----------------------------
# Data alignment and splitting
# -----------------------------
def align_and_prepare(
    ds_lr: xr.Dataset, ds_hr: xr.Dataset, var_lr: str, var_hr: str
):
    lat_lr, lon_lr = find_lat_lon(ds_lr)
    lat_hr, lon_hr = find_lat_lon(ds_hr)
    da_lr = ds_lr[var_lr]
    da_hr = ds_hr[var_hr]
    t0 = max(pd.to_datetime(da_lr.time.values[0]), pd.to_datetime(da_hr.time.values[0]))
    t1 = min(pd.to_datetime(da_lr.time.values[-1]), pd.to_datetime(da_hr.time.values[-1]))
    da_lr = da_lr.sel(time=slice(t0, t1))
    da_hr = da_hr.sel(time=slice(t0, t1))
    da_lr_up = da_lr.interp({lat_lr: da_hr[lat_hr], lon_lr: da_hr[lon_hr]}, method="linear")
    month = xr.DataArray(da_hr["time"].dt.month, coords={"time": da_hr["time"]}, dims=["time"])
    lat2d = xr.broadcast(da_hr[lat_hr], da_hr[lon_hr])[0].values
    lon2d = xr.broadcast(da_hr[lat_hr], da_hr[lon_hr])[1].values
    return da_lr_up, da_hr, month, lat2d, lon2d, lat_hr, lon_hr


def make_time_split_dataset(
    da_lr_up: xr.DataArray,
    da_hr: xr.DataArray,
    month: xr.DataArray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    train_ratio: float,
    use_context: bool,
    max_samples: int,
    smooth_sigma: float = 0.0,
    sample_strategy: str = "edge",  # 'edge' or 'uniform'
    lag_steps: Tuple[int, ...] = (1, 2),
):
    """
    Build supervised dataset with a time-based split (avoids leakage).
    Auto-fallback: if context wipes samples, retry without context.
    Returns Xtr, ytr, Xte, yte, lr_te, ids_tr, ids_te, split info.
    """
    if da_hr.sizes.get("time", da_hr.shape[0]) < 3:  # need >= max(lag_steps)+1
        raise ValueError("Not enough time steps (need ≥3 with lags). Check LR/HR overlap.")

    # Optional smoothing of HR target (per-time gaussian)
    if smooth_sigma and smooth_sigma > 0:
        hr_vals = gaussian_filter(da_hr.values, sigma=(0, float(smooth_sigma), float(smooth_sigma)))
        da_hr = xr.DataArray(hr_vals, coords=da_hr.coords, dims=da_hr.dims, name=da_hr.name, attrs=da_hr.attrs)

    T = da_hr.shape[0]
    t_cut = max(2, min(T - 1, int(T * float(train_ratio))))  # leave room for lags

    def _build(use_ctx_flag: bool):
        Xtr_, ytr_, _, wtr_, ids_tr_ = _flatten_arrays(
            da_lr_up.isel(time=slice(0, t_cut)).values,
            da_hr.isel(time=slice(0, t_cut)).values,
            month.isel(time=slice(0, t_cut)).values,
            lat2d,
            lon2d,
            use_context=use_ctx_flag,
            lag_steps=lag_steps,
        )
        Xte_, yte_, lr_te_, _, ids_te_ = _flatten_arrays(
            da_lr_up.isel(time=slice(t_cut, None)).values,
            da_hr.isel(time=slice(t_cut, None)).values,
            month.isel(time=slice(t_cut, None)).values,
            lat2d,
            lon2d,
            use_context=use_ctx_flag,
            lag_steps=lag_steps,
        )
        return Xtr_, ytr_, Xte_, yte_, lr_te_, ids_tr_, ids_te_, wtr_

    Xtr, ytr, Xte, yte, lr_te, ids_tr, ids_te, wtr = _build(use_context)
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        Xtr, ytr, Xte, yte, lr_te, ids_tr, ids_te, wtr = _build(False)

    if Xtr.shape[0] == 0:
        raise ValueError("No valid training samples after masking. Try reducing smoothing or check data alignment.")
    if Xte.shape[0] == 0:
        raise ValueError("No valid test samples after masking. Increase holdout or check data/NaNs alignment.")

    # Subsample **only train**
    if len(ytr) > max_samples > 0:
        rng = np.random.RandomState(42)
        if wtr is not None and sample_strategy == "edge" and wtr.size == len(ytr):
            w = np.maximum(np.nan_to_num(wtr, nan=0.0), 1e-12)
            w = w / w.sum()
            idx = rng.choice(len(ytr), size=max_samples, replace=False, p=w)
        else:
            idx = rng.choice(len(ytr), size=max_samples, replace=False)
        Xtr, ytr, ids_tr = Xtr[idx], ytr[idx], ids_tr[idx]

    return Xtr, ytr, Xte, yte, lr_te, ids_tr, ids_te, {"t_cut": int(t_cut), "T": int(T)}


# -----------------------------
# A robust classical baseline: BCSD (Quantile Mapping)
# -----------------------------
class BCSDQuantileMapper:
    """
    Global quantile mapping: map LR↑ distribution to HR using training percentiles.
    Predict returns residual (mapped_hr - lr).
    """
    def __init__(self, q: np.ndarray | None = None):
        self.q = np.linspace(0, 100, 101) if q is None else q
        self.q_lr: Optional[np.ndarray] = None
        self.q_hr: Optional[np.ndarray] = None
        self.n_features_in_ = 5  # we only require [lr, lat, lon, m_sin, m_cos]
        self.alpha_ = 1.0
        self.use_model_ = True

    def fit(self, lr: np.ndarray, hr: np.ndarray):
        lr = lr[np.isfinite(lr)]
        hr = hr[np.isfinite(hr)]
        if lr.size == 0 or hr.size == 0:
            self.q_lr = np.array([0.0, 1.0])
            self.q_hr = np.array([0.0, 1.0])
            return self
        self.q_lr = np.percentile(lr, self.q)
        self.q_hr = np.percentile(hr, self.q)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        lr = X[:, 0]
        if self.q_lr is None or self.q_hr is None:
            return np.zeros_like(lr)
        mapped = np.interp(lr, self.q_lr, self.q_hr, left=self.q_hr[0], right=self.q_hr[-1])
        return mapped - lr  # residual


# -----------------------------
# α-shrinkage & per-cell safety gate
# -----------------------------
def _compute_alpha(y_true_res: np.ndarray, y_hat_res: np.ndarray) -> float:
    """Closed-form alpha minimizing ||y_true - alpha*y_hat||^2, clipped to [0,1]."""
    num = float(np.dot(y_hat_res, y_true_res))
    den = float(np.dot(y_hat_res, y_hat_res)) + 1e-12
    a = num / den
    return float(np.clip(a, 0.0, 1.0))


def _alpha_gate_per_cell(ids: np.ndarray, y_true: np.ndarray, y_hat: np.ndarray, min_n: int = 50):
    """Return alpha_map (len = n_cells) and use_model_map bools."""
    n_cells = int(ids.max()) + 1 if ids.size else 0
    alpha_map = np.ones(n_cells, dtype=float)
    use_map = np.zeros(n_cells, dtype=bool)

    # Precompute lists per id
    for cell in np.unique(ids):
        sel = ids == cell
        if sel.sum() < min_n:
            continue
        y_t = y_true[sel]
        y_h = y_hat[sel]
        a = _compute_alpha(y_t, y_h)
        rmse_m = float(np.sqrt(mean_squared_error(y_t, a * y_h)))
        rmse_b = float(np.sqrt(mean_squared_error(y_t, np.zeros_like(y_t))))
        alpha_map[cell] = a
        use_map[cell] = rmse_m < rmse_b
    return alpha_map, use_map


# -----------------------------
# Training & evaluation
# -----------------------------
def train_downscaling(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    ids_tr: np.ndarray,
    ids_te: np.ndarray,
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    model_type: str = "hgb",  # 'bcsd' | 'lgbm' | 'xgb' | 'hgb' | 'rf'
):
    model_type = model_type.lower()

    # ---- choose and fit model
    if model_type == "bcsd":
        lr_tr = Xtr[:, 0]
        hr_tr = lr_tr + ytr
        model = BCSDQuantileMapper().fit(lr_tr, hr_tr)
    elif model_type == "lgbm" and lgb is not None:
        model = lgb.LGBMRegressor(
            n_estimators=1500, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42
        )
        model.fit(Xtr, ytr)
    elif model_type == "xgb" and xgb is not None:
        model = xgb.XGBRegressor(
            n_estimators=1500, learning_rate=0.05, max_depth=8, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", random_state=42
        )
        model.fit(Xtr, ytr)
    elif model_type == "hgb":
        model = HistGradientBoostingRegressor(
            max_depth=8, learning_rate=0.08, max_iter=1500, early_stopping=True,
            n_iter_no_change=50, validation_fraction=0.1, l2_regularization=1e-3,
            min_samples_leaf=100, random_state=42,
        )
        model.fit(Xtr, ytr)
    else:  # rf
        model = RandomForestRegressor(
            n_estimators=400, n_jobs=-1, random_state=42, min_samples_leaf=50, max_depth=20
        )
        model.fit(Xtr, ytr)

    # ---- internal validation for global α and gate
    rng = np.random.RandomState(42)
    n_val = min(max(2000, int(0.1 * Xtr.shape[0])), Xtr.shape[0])
    val_idx = rng.choice(Xtr.shape[0], size=n_val, replace=False)
    Xv, yv, idv = Xtr[val_idx], ytr[val_idx], ids_tr[val_idx]
    yv_hat = model.predict(Xv)

    # bias-correct residuals (skip for BCSD)
    if not isinstance(model, BCSDQuantileMapper):
        yv_hat = yv_hat - np.mean(yv_hat) + np.mean(ytr)

    global_alpha = _compute_alpha(yv, yv_hat)
    rmse_val_model = float(np.sqrt(mean_squared_error(yv, global_alpha * yv_hat)))
    rmse_val_base = float(np.sqrt(mean_squared_error(yv, np.zeros_like(yv))))
    global_use = rmse_val_model < rmse_val_base

    # ---- per-cell α and gating (fallback to global if not enough samples)
    alpha_map, use_map = _alpha_gate_per_cell(idv, yv, yv_hat, min_n=50)
    # fill defaults for cells unseen in validation
    if alpha_map.size < (ids_tr.max() + 1):
        pad = (ids_tr.max() + 1) - alpha_map.size
        alpha_map = np.pad(alpha_map, (0, pad), constant_values=global_alpha)
        use_map = np.pad(use_map, (0, pad), constant_values=global_use)
    # persist
    setattr(model, "alpha_", float(global_alpha))
    setattr(model, "use_model_", bool(global_use))
    setattr(model, "alpha_map_", alpha_map.astype(float))
    setattr(model, "use_model_map_", use_map.astype(bool))

    # ---- test predictions with per-cell α & gate
    ypred = model.predict(Xte)
    if not isinstance(model, BCSDQuantileMapper):
        ypred = ypred - np.mean(ypred) + np.mean(ytr)
    a_vec = alpha_map[ids_te]
    u_vec = use_map[ids_te].astype(float)
    ypred_eff = a_vec * u_vec * ypred

    stats: Dict[str, float] = {
        "rmse": float(np.sqrt(mean_squared_error(yte, ypred_eff))),
        "mae": float(mean_absolute_error(yte, ypred_eff)),
        "r2": float(r2_score(yte, ypred_eff)),
        "alpha": float(global_alpha),
        "used_model": bool(global_use),
        "rmse_val_model_res": rmse_val_model,
        "rmse_val_baseline_res": rmse_val_base,
    }

    # Quantile models for uncertainty (portable)
    q_models: Dict[float, GradientBoostingRegressor] = {}
    q_metrics: Dict[float, float] = {}
    for q in quantiles:
        gbr = GradientBoostingRegressor(loss="quantile", alpha=float(q), random_state=42)
        gbr.fit(Xtr, ytr)
        q_pred = gbr.predict(Xte)
        q_models[q] = gbr
        q_metrics[q] = float(mean_pinball_loss(yte, q_pred, alpha=float(q)))
    stats["pinball"] = q_metrics
    return model, q_models, stats


def hr_metrics_from_residuals(lr_te: np.ndarray, yte: np.ndarray, ypred_eff: np.ndarray) -> Dict[str, float]:
    """Metrics on FINAL HR field and LR↑ baseline."""
    hr_true = lr_te + yte
    hr_pred = lr_te + ypred_eff
    rmse_hr = float(np.sqrt(mean_squared_error(hr_true, hr_pred)))
    mae_hr = float(mean_absolute_error(hr_true, hr_pred))
    r2_hr = float(r2_score(hr_true, hr_pred))

    rmse_hr_base = float(np.sqrt(mean_squared_error(hr_true, lr_te)))
    mae_hr_base = float(mean_absolute_error(hr_true, lr_te))
    r2_hr_base = float(r2_score(hr_true, lr_te))

    imp = (rmse_hr_base - rmse_hr) / (rmse_hr_base + 1e-9) * 100.0
    return {
        "rmse_hr": rmse_hr,
        "mae_hr": mae_hr,
        "r2_hr": r2_hr,
        "rmse_hr_baseline": rmse_hr_base,
        "mae_hr_baseline": mae_hr_base,
        "r2_hr_baseline": r2_hr_base,
        "rmse_improvement_pct": float(imp),
    }


def quantile_coverage(
    q_models: Dict[float, GradientBoostingRegressor],
    Xte: np.ndarray,
    yte: np.ndarray,
    lr_te: np.ndarray,
    low: float = 0.1,
    high: float = 0.9,
) -> Dict[str, Optional[float]]:
    ql = q_models.get(low)
    qh = q_models.get(high)
    cov_res: Optional[float] = None
    cov_hr: Optional[float] = None
    if ql is not None and qh is not None:
        ql_pred = ql.predict(Xte)
        qh_pred = qh.predict(Xte)
        cov_res = float(((ql_pred <= yte) & (yte <= qh_pred)).mean())
        cov_hr = float((((lr_te + ql_pred) <= (lr_te + yte)) & ((lr_te + yte) <= (lr_te + qh_pred))).mean())
    return {"coverage_residual_10_90": cov_res, "coverage_hr_10_90": cov_hr}


# -----------------------------
# Spatial eval (robust SSIM)
# -----------------------------
def _robust_ssim(true2d: np.ndarray, pred2d: np.ndarray) -> float:
    mask = np.isfinite(true2d) & np.isfinite(pred2d)
    if not np.any(mask):
        return float("nan")
    t = np.where(mask, true2d, np.nan)
    p = np.where(mask, pred2d, np.nan)
    try:
        pr1, pr99 = np.nanpercentile(t, [1, 99])
        rng = float(pr99 - pr1)
    except Exception:
        rng = float(np.nanmax(t) - np.nanmin(t))
    if not np.isfinite(rng) or rng == 0.0:
        rng = 1.0
    return float(ssim(np.nan_to_num(t), np.nan_to_num(p), data_range=rng))


def last_step_spatial_eval(
    model,
    da_lr_up: xr.DataArray,
    da_hr: xr.DataArray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    lat_name: str,
    lon_name: str,
):
    """
    Build features for last timestep; auto-match model's feature count.
    Apply per-cell α and gating to residuals.
    """
    t_sel = int(da_hr.shape[0] - 1)
    lr2d_t = da_lr_up.isel(time=t_sel).values
    # lags
    lags = []
    if t_sel - 1 >= 0:
        lags.append(da_lr_up.isel(time=t_sel - 1).values.reshape(-1))
    if t_sel - 2 >= 0:
        lags.append(da_lr_up.isel(time=t_sel - 2).values.reshape(-1))

    month_val = int(pd.to_datetime(da_hr.time.values[t_sel]).month)
    m_sin = np.sin(2 * np.pi * month_val / 12.0)
    m_cos = np.cos(2 * np.pi * month_val / 12.0)

    base = [
        lr2d_t.reshape(-1),
        lat2d.reshape(-1),
        lon2d.reshape(-1),
        np.full(lr2d_t.size, m_sin),
        np.full(lr2d_t.size, m_cos),
    ] + lags

    # Full (with context)
    lr3d = lr2d_t[None, ...]
    ctx = _context_features_multiscale(lr3d)
    ctx_list = [
        ctx["std3"].reshape(-1),
        ctx["std5"].reshape(-1),
        ctx["std9"].reshape(-1),
        ctx["grad"].reshape(-1),
        ctx["lap"].reshape(-1),
        ctx["dog"].reshape(-1),
    ]

    n_needed = getattr(model, "n_features_in_", None)
    full = base + ctx_list
    feats = full if (n_needed is None or n_needed >= len(full)) else full[: int(n_needed)]

    feat_mat = np.column_stack(feats) if lr2d_t.size else np.empty((0, len(feats)))
    res_pred = model.predict(feat_mat).reshape(lr2d_t.shape)

    # bias-correct for non-BCSD
    if not isinstance(model, BCSDQuantileMapper):
        res_pred = res_pred - np.mean(res_pred)

    # per-cell α & gate
    H, W = lr2d_t.shape
    ids_grid = np.arange(H * W, dtype=np.int32)
    alpha_map = getattr(model, "alpha_map_", np.ones(H * W, dtype=float))
    use_map = getattr(model, "use_model_map_", np.ones(H * W, dtype=bool))
    a_vec = alpha_map[ids_grid].reshape(H, W)
    u_mat = use_map[ids_grid].reshape(H, W).astype(float)
    pred2d = lr2d_t + (a_vec * u_mat * res_pred)

    true2d = da_hr.isel(time=t_sel).values
    ssim_val = _robust_ssim(true2d, pred2d)
    return pred2d, true2d, ssim_val


def baseline_last_step_ssim(da_lr_up: xr.DataArray, da_hr: xr.DataArray) -> float:
    """SSIM of LR↑ vs HR on the last timestep (baseline)."""
    t_sel = int(da_hr.shape[0] - 1)
    base2d = da_lr_up.isel(time=t_sel).values
    true2d = da_hr.isel(time=t_sel).values
    return _robust_ssim(true2d, base2d)
