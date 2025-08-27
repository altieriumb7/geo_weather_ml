import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.data_io import open_xr, synthetic_temperature_ds, synthetic_co2_ts
from services.climate_ops import kelvin_to_celsius, compute_anomaly, find_lat_lon, area_weights


def render_corr_tab(cfg, state):
    st.subheader("Relazione CO₂ ↔️ Temperatura (aggregata regione)")

    if "reg_ts_temp" in state:
        reg_ts = state["reg_ts_temp"]
    else:
        up_temp = cfg["up_temp"]; var_temp = cfg["var_temp"]
        use_dask = cfg["use_dask"]; chunk_time = cfg["chunk_time"]
        baseline = cfg["baseline"]; monthly = cfg["monthly"]
        if up_temp is not None:
            ds_temp = open_xr(up_temp.getvalue(), {"time": int(chunk_time)} if use_dask else None)
            varname = var_temp if var_temp in ds_temp else list(ds_temp.data_vars)[0]
            daT = kelvin_to_celsius(ds_temp[varname])
        else:
            daT = synthetic_temperature_ds()["t2m"]
        anom = compute_anomaly(daT, (baseline[0].strftime("%Y-%m-%d"), baseline[1].strftime("%Y-%m-%d")), monthly_climatology=monthly)
        lat_name, lon_name = find_lat_lon(anom.to_dataset(name="v"))
        w = area_weights(anom[lat_name])
        reg_ts = (anom.weighted(w).mean(dim=(lat_name, lon_name))).to_series()

    if cfg["up_co2"] is not None:
        from services.data_io import load_co2_csv
        co2 = load_co2_csv(cfg["up_co2"].getvalue(), cfg["co2_date_col"], cfg["co2_val_col"])
    else:
        co2 = synthetic_co2_ts()

    import pandas as pd
    df = pd.concat({"temp": reg_ts, "co2": co2}, axis=1).dropna()
    st.write(f"Periodo sovrapposto: {df.index.min():%Y-%m} → {df.index.max():%Y-%m} — N={len(df):,}")

    if len(df) < 6:
        st.warning("Periodo sovrapposto troppo breve per un'analisi significativa.")
        return

    corr = df.corr().loc["temp", "co2"]
    st.metric("Correlazione Pearson (temp anomalia vs CO₂)", f"{corr:.3f}")

    fig_sc = px.scatter(df, x="co2", y="temp", labels={"co2": co2.name or "CO₂ (ppm)", "temp": "Anomalia T (°C)"})
    xvals = df["co2"].to_numpy(dtype=float)
    yvals = df["temp"].to_numpy(dtype=float)
    mask_xy = np.isfinite(xvals) & np.isfinite(yvals)
    x_ = xvals[mask_xy]; y_ = yvals[mask_xy]
    if x_.size >= 2:
        A = np.vstack([x_, np.ones_like(x_)]).T
        beta, *_ = np.linalg.lstsq(A, y_, rcond=None)
        slope, intercept = float(beta[0]), float(beta[1])
        xs = np.linspace(x_.min(), x_.max(), 100)
        ys = slope * xs + intercept
        fig_sc.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Fit OLS"))
    st.plotly_chart(fig_sc, use_container_width=True)

    dfz = (df - df.mean()) / df.std()
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=dfz.index, y=dfz["temp"], mode="lines", name="Temp (z)"))
    fig_z.add_trace(go.Scatter(x=dfz.index, y=dfz["co2"], mode="lines", name="CO₂ (z)"))
    fig_z.update_layout(title="Serie normalizzate (z-score)")
    st.plotly_chart(fig_z, use_container_width=True)