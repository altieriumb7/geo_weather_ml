# tabs/models.py
import numpy as np
import streamlit as st

from services.data_io import open_xr, synthetic_temperature_ds
from services.ml_downscaling import (
    align_and_prepare,
    make_time_split_dataset,
    train_downscaling,
    last_step_spatial_eval,
    hr_metrics_from_residuals,
    quantile_coverage,
    baseline_last_step_ssim,
)
from services.viz import to_plotly_heatmap


def render_models_tab(cfg, state):
    st.subheader("🤖 Modelli AI — (2) Downscaling · (4) Anomaly/Regime · (6) Valutazione & Incertezza")
    subtab1, subtab2, subtab3 = st.tabs(["2️⃣ Downscaling", "4️⃣ Anomaly/Regime", "6️⃣ Valutazione & Incertezza"])

    # (2) Downscaling
    with subtab1:
        colA, colB = st.columns(2)
        with colA:
            up_lr = st.file_uploader("Low-res NetCDF (predictor, es. tas_lr)", type=["nc", "nc4"], key="dl_lr")
            var_lr = st.text_input("Variabile low-res", value="tas")
        with colB:
            up_hr = st.file_uploader("High-res NetCDF (target, es. tas_hr/E-OBS)", type=["nc", "nc4"], key="dl_hr")
            var_hr = st.text_input("Variabile high-res", value="tas")
        train_ratio = st.slider("Holdout temporale (frazione train)", 0.5, 0.95, 0.9, 0.05)
        max_samples = st.number_input("Max campioni train", 20000, 500000, 250000, step=10000)
        quantiles = st.multiselect("Quantili (GBR)", options=[0.1, 0.2, 0.5, 0.8, 0.9], default=[0.1, 0.5, 0.9])
        use_ctx = st.checkbox("Context multi-scala (std3/5/9, grad, lap, DoG)", value=True)
        model_choice = st.selectbox(
            "Modello",
            ["BCSD (Quantile Mapping)", "LightGBM", "XGBoost", "HistGradientBoosting (sklearn)", "RandomForest"],
            index=1,
        )
        smooth_sigma = st.slider("Smoothing target HR (σ px)", 0.0, 2.5, 1.0, 0.5)
        sample_strategy = st.selectbox("Campionamento train", ["edge", "uniform"], index=1)

        if st.button("Addestra modello di downscaling", type="primary"):
            with st.spinner("Preparo dati e addestro il modello..."):
                use_dask = cfg["use_dask"]
                if up_lr and up_hr:
                    ds_lr = open_xr(up_lr.getvalue(), {"time": 12} if use_dask else None)
                    ds_hr = open_xr(up_hr.getvalue(), {"time": 12} if use_dask else None)
                else:
                    ds_hr = synthetic_temperature_ds()
                    da_hr0 = ds_hr["t2m"].transpose("time", "latitude", "longitude")
                    da_lr0 = da_hr0.coarsen(latitude=4, longitude=4, boundary="trim").mean()
                    ds_lr = da_lr0.to_dataset(name="tas")
                    ds_hr = da_hr0.to_dataset(name="tas")
                    var_lr = var_hr = "tas"

                da_lr_up, da_hr, month, lat2d, lon2d, lat_name, lon_name = align_and_prepare(ds_lr, ds_hr, var_lr, var_hr)

                try:
                    Xtr, ytr, Xte, yte, lr_te, ids_tr, ids_te, split_info = make_time_split_dataset(
                        da_lr_up, da_hr, month, lat2d, lon2d,
                        train_ratio=float(train_ratio),
                        use_context=use_ctx,
                        max_samples=int(max_samples),
                        smooth_sigma=float(smooth_sigma),
                        sample_strategy=sample_strategy,
                        lag_steps=(1, 2),
                    )
                except ValueError as e:
                    st.error(str(e))
                    return

                model_map = {
                    "BCSD (Quantile Mapping)": "bcsd",
                    "LightGBM": "lgbm",
                    "XGBoost": "xgb",
                    "HistGradientBoosting (sklearn)": "hgb",
                    "RandomForest": "rf",
                }

                mdl, q_models, stats = train_downscaling(
                    Xtr, ytr, Xte, yte, ids_tr, ids_te,
                    quantiles=tuple(quantiles), model_type=model_map[model_choice]
                )

                # Effective residuals to evaluate HR metrics (use per-cell α & gating from model)
                ypred_te = mdl.predict(Xte)
                if mdl.__class__.__name__ != "BCSDQuantileMapper":
                    ypred_te = ypred_te - np.mean(ypred_te) + np.mean(ytr)
                a_map = getattr(mdl, "alpha_map_", np.ones(ids_te.max() + 1))
                u_map = getattr(mdl, "use_model_map_", np.ones(ids_te.max() + 1, dtype=bool))
                ypred_eff = a_map[ids_te] * u_map[ids_te].astype(float) * ypred_te

                # HR-level metrics & baseline comparison
                hrm = hr_metrics_from_residuals(lr_te, yte, ypred_eff)
                stats.update(hrm)

                # Quantile coverage (raw quantiles; independent of α)
                cov = quantile_coverage(q_models, Xte, yte, lr_te, low=0.1, high=0.9)
                stats.update(cov)

                # Persist
                state["down_model"] = mdl
                state["down_q_models"] = q_models
                state["down_stats"] = stats
                state["down_meta"] = {"lat": lat_name, "lon": lon_name, "var": var_hr, **split_info}

                # Spatial eval & baseline SSIM (internal α/gating)
                pred2d, true2d, ssim_val = last_step_spatial_eval(
                    mdl, da_lr_up, da_hr, lat2d, lon2d, lat_name, lon_name
                )
                base_ssim = baseline_last_step_ssim(da_lr_up, da_hr)
                state["down_stats"].update({"ssim_last_step": float(ssim_val), "ssim_baseline_last_step": float(base_ssim)})

                used = "sì" if stats.get("used_model") else "no (fallback LR↑)"
                st.success(
                    f"Downscaling addestrato. RES: RMSE={stats['rmse']:.3f}, MAE={stats['mae']:.3f}, R²={stats['r2']:.3f}, "
                    f"α={stats.get('alpha', 1.0):.2f}, usa_modello={used} | "
                    f"HR: RMSE={stats['rmse_hr']:.3f} (base {stats['rmse_hr_baseline']:.3f}, Δ={stats['rmse_improvement_pct']:.1f}%), "
                    f"MAE={stats['mae_hr']:.3f}, R²={stats['r2_hr']:.3f}, SSIM={ssim_val:.3f} (base {base_ssim:.3f})"
                )

                # Error map
                import xarray as xr
                diff = (pred2d - true2d)
                from services.viz import to_plotly_heatmap
                figd = to_plotly_heatmap(
                    xr.DataArray(diff, dims=(lat_name, lon_name), coords={lat_name: da_hr[lat_name], lon_name: da_hr[lon_name]}),
                    title="Errore (pred - true) — ultimo timestep",
                )
                st.plotly_chart(figd, use_container_width=True)

                if cov["coverage_hr_10_90"] is not None:
                    st.write({
                        "coverage_residual_10_90": round(cov["coverage_residual_10_90"], 3),
                        "coverage_hr_10_90": round(cov["coverage_hr_10_90"], 3),
                    })

    # (4) Anomaly/Regime)
    with subtab2:
        from services.ml_anomaly import detect_anomalies
        if "reg_ts_temp" not in state:
            st.info("Apri il tab Temperatura per generare l'anomalia regionale (reg_ts_temp).")
        else:
            reg_ts = state["reg_ts_temp"]
            anomalies, _ = detect_anomalies(reg_ts)
            st.metric("Anomalie rilevate", f"{len(anomalies)} eventi")

    # (6) Valutazione & Incertezza)
    with subtab3:
        stats = state.get("down_stats")
        if not stats:
            st.info("Addestra prima un modello di downscaling nel sub-tab 2.")
        else:
            core = {
                k: v for k, v in stats.items()
                if k in [
                    "rmse", "mae", "r2", "alpha", "used_model",
                    "rmse_val_model_res", "rmse_val_baseline_res",
                    "rmse_hr", "mae_hr", "r2_hr",
                    "rmse_hr_baseline", "rmse_improvement_pct",
                    "ssim_last_step", "ssim_baseline_last_step"
                ]
            }
            st.write({k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in core.items()})
            if stats.get("pinball"):
                st.write({f"pinball_{k}": round(v, 4) for k, v in stats["pinball"].items()})
            if stats.get("coverage_hr_10_90") is not None:
                st.write({
                    "coverage_residual_10_90": round(stats.get("coverage_residual_10_90", np.nan), 4),
                    "coverage_hr_10_90": round(stats.get("coverage_hr_10_90", np.nan), 4),
                })
# tabs/models.py
import numpy as np
import streamlit as st

from services.data_io import open_xr, synthetic_temperature_ds
from services.ml_downscaling import (
    align_and_prepare,
    make_time_split_dataset,
    train_downscaling,
    last_step_spatial_eval,
    hr_metrics_from_residuals,
    quantile_coverage,
    baseline_last_step_ssim,
)
from services.viz import to_plotly_heatmap


def render_models_tab(cfg, state):
    st.subheader("🤖 Modelli AI — (2) Downscaling · (4) Anomaly/Regime · (6) Valutazione & Incertezza")
    subtab1, subtab2, subtab3 = st.tabs(["2️⃣ Downscaling", "4️⃣ Anomaly/Regime", "6️⃣ Valutazione & Incertezza"])

    # (2) Downscaling
    with subtab1:
        colA, colB = st.columns(2)
        with colA:
            up_lr = st.file_uploader("Low-res NetCDF (predictor, es. tas_lr)", type=["nc", "nc4"], key="dl_lr")
            var_lr = st.text_input("Variabile low-res", value="tas")
        with colB:
            up_hr = st.file_uploader("High-res NetCDF (target, es. tas_hr/E-OBS)", type=["nc", "nc4"], key="dl_hr")
            var_hr = st.text_input("Variabile high-res", value="tas")
        train_ratio = st.slider("Holdout temporale (frazione train)", 0.5, 0.95, 0.9, 0.05)
        max_samples = st.number_input("Max campioni train", 20000, 500000, 250000, step=10000)
        quantiles = st.multiselect("Quantili (GBR)", options=[0.1, 0.2, 0.5, 0.8, 0.9], default=[0.1, 0.5, 0.9])
        use_ctx = st.checkbox("Context multi-scala (std3/5/9, grad, lap, DoG)", value=True)
        model_choice = st.selectbox(
            "Modello",
            ["BCSD (Quantile Mapping)", "LightGBM", "XGBoost", "HistGradientBoosting (sklearn)", "RandomForest"],
            index=1,
        )
        smooth_sigma = st.slider("Smoothing target HR (σ px)", 0.0, 2.5, 1.0, 0.5)
        sample_strategy = st.selectbox("Campionamento train", ["edge", "uniform"], index=1)

        if st.button("Addestra modello di downscaling", type="primary"):
            with st.spinner("Preparo dati e addestro il modello..."):
                use_dask = cfg["use_dask"]
                if up_lr and up_hr:
                    ds_lr = open_xr(up_lr.getvalue(), {"time": 12} if use_dask else None)
                    ds_hr = open_xr(up_hr.getvalue(), {"time": 12} if use_dask else None)
                else:
                    ds_hr = synthetic_temperature_ds()
                    da_hr0 = ds_hr["t2m"].transpose("time", "latitude", "longitude")
                    da_lr0 = da_hr0.coarsen(latitude=4, longitude=4, boundary="trim").mean()
                    ds_lr = da_lr0.to_dataset(name="tas")
                    ds_hr = da_hr0.to_dataset(name="tas")
                    var_lr = var_hr = "tas"

                da_lr_up, da_hr, month, lat2d, lon2d, lat_name, lon_name = align_and_prepare(ds_lr, ds_hr, var_lr, var_hr)

                try:
                    Xtr, ytr, Xte, yte, lr_te, ids_tr, ids_te, split_info = make_time_split_dataset(
                        da_lr_up, da_hr, month, lat2d, lon2d,
                        train_ratio=float(train_ratio),
                        use_context=use_ctx,
                        max_samples=int(max_samples),
                        smooth_sigma=float(smooth_sigma),
                        sample_strategy=sample_strategy,
                        lag_steps=(1, 2),
                    )
                except ValueError as e:
                    st.error(str(e))
                    return

                model_map = {
                    "BCSD (Quantile Mapping)": "bcsd",
                    "LightGBM": "lgbm",
                    "XGBoost": "xgb",
                    "HistGradientBoosting (sklearn)": "hgb",
                    "RandomForest": "rf",
                }

                mdl, q_models, stats = train_downscaling(
                    Xtr, ytr, Xte, yte, ids_tr, ids_te,
                    quantiles=tuple(quantiles), model_type=model_map[model_choice]
                )

                # Effective residuals to evaluate HR metrics (use per-cell α & gating from model)
                ypred_te = mdl.predict(Xte)
                if mdl.__class__.__name__ != "BCSDQuantileMapper":
                    ypred_te = ypred_te - np.mean(ypred_te) + np.mean(ytr)
                a_map = getattr(mdl, "alpha_map_", np.ones(ids_te.max() + 1))
                u_map = getattr(mdl, "use_model_map_", np.ones(ids_te.max() + 1, dtype=bool))
                ypred_eff = a_map[ids_te] * u_map[ids_te].astype(float) * ypred_te

                # HR-level metrics & baseline comparison
                hrm = hr_metrics_from_residuals(lr_te, yte, ypred_eff)
                stats.update(hrm)

                # Quantile coverage (raw quantiles; independent of α)
                cov = quantile_coverage(q_models, Xte, yte, lr_te, low=0.1, high=0.9)
                stats.update(cov)

                # Persist
                state["down_model"] = mdl
                state["down_q_models"] = q_models
                state["down_stats"] = stats
                state["down_meta"] = {"lat": lat_name, "lon": lon_name, "var": var_hr, **split_info}

                # Spatial eval & baseline SSIM (internal α/gating)
                pred2d, true2d, ssim_val = last_step_spatial_eval(
                    mdl, da_lr_up, da_hr, lat2d, lon2d, lat_name, lon_name
                )
                base_ssim = baseline_last_step_ssim(da_lr_up, da_hr)
                state["down_stats"].update({"ssim_last_step": float(ssim_val), "ssim_baseline_last_step": float(base_ssim)})

                used = "sì" if stats.get("used_model") else "no (fallback LR↑)"
                st.success(
                    f"Downscaling addestrato. RES: RMSE={stats['rmse']:.3f}, MAE={stats['mae']:.3f}, R²={stats['r2']:.3f}, "
                    f"α={stats.get('alpha', 1.0):.2f}, usa_modello={used} | "
                    f"HR: RMSE={stats['rmse_hr']:.3f} (base {stats['rmse_hr_baseline']:.3f}, Δ={stats['rmse_improvement_pct']:.1f}%), "
                    f"MAE={stats['mae_hr']:.3f}, R²={stats['r2_hr']:.3f}, SSIM={ssim_val:.3f} (base {base_ssim:.3f})"
                )

                # Error map
                import xarray as xr
                diff = (pred2d - true2d)
                from services.viz import to_plotly_heatmap
                figd = to_plotly_heatmap(
                    xr.DataArray(diff, dims=(lat_name, lon_name), coords={lat_name: da_hr[lat_name], lon_name: da_hr[lon_name]}),
                    title="Errore (pred - true) — ultimo timestep",
                )
                st.plotly_chart(figd, use_container_width=True)

                if cov["coverage_hr_10_90"] is not None:
                    st.write({
                        "coverage_residual_10_90": round(cov["coverage_residual_10_90"], 3),
                        "coverage_hr_10_90": round(cov["coverage_hr_10_90"], 3),
                    })

    # (4) Anomaly/Regime)
    with subtab2:
        from services.ml_anomaly import detect_anomalies
        if "reg_ts_temp" not in state:
            st.info("Apri il tab Temperatura per generare l'anomalia regionale (reg_ts_temp).")
        else:
            reg_ts = state["reg_ts_temp"]
            anomalies, _ = detect_anomalies(reg_ts)
            st.metric("Anomalie rilevate", f"{len(anomalies)} eventi")

    # (6) Valutazione & Incertezza)
    with subtab3:
        stats = state.get("down_stats")
        if not stats:
            st.info("Addestra prima un modello di downscaling nel sub-tab 2.")
        else:
            core = {
                k: v for k, v in stats.items()
                if k in [
                    "rmse", "mae", "r2", "alpha", "used_model",
                    "rmse_val_model_res", "rmse_val_baseline_res",
                    "rmse_hr", "mae_hr", "r2_hr",
                    "rmse_hr_baseline", "rmse_improvement_pct",
                    "ssim_last_step", "ssim_baseline_last_step"
                ]
            }
            st.write({k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in core.items()})
            if stats.get("pinball"):
                st.write({f"pinball_{k}": round(v, 4) for k, v in stats["pinball"].items()})
            if stats.get("coverage_hr_10_90") is not None:
                st.write({
                    "coverage_residual_10_90": round(stats.get("coverage_residual_10_90", np.nan), 4),
                    "coverage_hr_10_90": round(stats.get("coverage_hr_10_90", np.nan), 4),
                })
