import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.data_io import open_xr, synthetic_temperature_ds
from services.climate_ops import kelvin_to_celsius, compute_anomaly, find_lat_lon, area_weights, fit_trend
from services.geo_ops import europe_geometries, load_user_geodf, mask_from_geometries, regional_mean
from services.viz import to_plotly_heatmap


def render_temperature_tab(cfg, state):
    st.subheader("Anomalie di temperatura (Europa)")

    up_temp = cfg["up_temp"]; var_temp = cfg["var_temp"]
    use_dask = cfg["use_dask"]; chunk_time = cfg["chunk_time"]
    baseline = cfg["baseline"]; monthly = cfg["monthly"]
    up_shape = cfg["up_shape"]

    if up_temp is not None:
        chunks = {"time": int(chunk_time)} if use_dask else None
        ds_temp = open_xr(up_temp.getvalue(), chunks)
        if var_temp not in ds_temp:
            st.warning(f"Variabile '{var_temp}' non trovata. Variabili: {list(ds_temp.data_vars)}")
        varname = var_temp if var_temp in ds_temp else list(ds_temp.data_vars)[0]
        daT = ds_temp[varname]
    else:
        st.info("Nessun NetCDF caricato. Uso **dati sintetici** per la dimostrazione.")
        ds_temp = synthetic_temperature_ds()
        daT = ds_temp["t2m"]

    if "time" not in daT.dims:
        st.error("Il dataset temperatura deve avere la dimensione 'time'.")
        st.stop()

    daT = kelvin_to_celsius(daT)

    b0 = baseline[0].strftime("%Y-%m-%d"); b1 = baseline[1].strftime("%Y-%m-%d")
    try:
        anom = compute_anomaly(daT, (b0, b1), monthly_climatology=monthly)
    except Exception as e:
        st.error(f"Errore nel calcolo delle anomalie: {e}")
        st.stop()

    # regioni
    mask_bool = None
    region_source = "Paesi (Europa)"
    user_geodf = None
    if up_shape is not None:
        try:
            user_geodf = load_user_geodf(up_shape.getvalue(), up_shape.name)
            region_source = st.radio("Sorgente regioni", ["Paesi (Europa)", "Shape personalizzato"], horizontal=True, index=1)
        except Exception as e:
            st.warning(f"Impossibile leggere il file regioni caricato: {e}")
            user_geodf = None
            region_source = "Paesi (Europa)"
    else:
        st.caption("Suggerimento: carica un GeoJSON o uno Shapefile .zip nella sidebar per usare regioni personalizzate.")

    if region_source == "Paesi (Europa)" or user_geodf is None:
        gdf_eu = europe_geometries()
        countries = st.multiselect("Seleziona paesi europei", options=sorted(gdf_eu["name"].unique()), default=["Italy"])
        if len(countries) > 0:
            sel = gdf_eu[gdf_eu["name"].isin(countries)]
            try:
                mask_bool = mask_from_geometries(anom.to_dataset(name="v"), sel)
            except Exception as e:
                st.warning(f"Impossibile creare maschera regionale: {e}")
                mask_bool = None
    else:
        st.markdown("**Shape personalizzato caricato**")
        candidate_cols = [c for c in user_geodf.columns if c != "geometry" and user_geodf[c].dtype == object]
        name_col = st.selectbox("Campo nome (opzionale)", options=["(nessuno)"] + candidate_cols)
        if name_col != "(nessuno)":
            values = sorted(user_geodf[name_col].astype(str).unique().tolist())
            choose = st.multiselect("Seleziona features", options=values, default=values[: min(5, len(values))])
            sel = user_geodf[user_geodf[name_col].astype(str).isin(choose)] if choose else user_geodf
        else:
            sel = user_geodf
        try:
            mask_bool = mask_from_geometries(anom.to_dataset(name="v"), sel)
        except Exception as e:
            st.warning(f"Impossibile creare maschera dal tuo shape: {e}")
            mask_bool = None

    # tempo
    tmin = pd.to_datetime(anom.time.values[0]).to_pydatetime()
    tmax = pd.to_datetime(anom.time.values[-1]).to_pydatetime()
    sel_date = st.slider("Seleziona mese", min_value=tmin, max_value=tmax, value=tmax, format="%Y-%m")

    an2d = anom.sel(time=sel_date, method="nearest")
    state["anom_last"] = anom  # per altri tab

    fig_map = to_plotly_heatmap(an2d, title=f"Anomalia {an2d.name or 'T'} — {sel_date:%Y-%m}")
    st.plotly_chart(fig_map, use_container_width=True)

    if mask_bool is not None:
        reg_ts = regional_mean(anom, mask_bool).to_series()
    else:
        lat_name, lon_name = find_lat_lon(anom.to_dataset(name="v"))
        w = area_weights(anom[lat_name])
        reg_ts = (anom.weighted(w).mean(dim=(lat_name, lon_name))).to_series()

    state["reg_ts_temp"] = reg_ts

    st.markdown("### Serie temporale regionale (anomalia)")
    st.plotly_chart(px.line(reg_ts, labels={"index": "Tempo", "value": anom.attrs.get("units", "")}), use_container_width=True)

    yhat, slope_py = fit_trend(reg_ts)
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=reg_ts.index, y=reg_ts.values, mode="lines", name="Anomalia"))
    fig_ts.add_trace(go.Scatter(x=yhat.index, y=yhat.values, mode="lines", name=f"Trend (~{slope_py:.3f} {anom.attrs.get('units','')}/anno)"))
    fig_ts.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_ts, use_container_width=True)

    csv = reg_ts.to_csv(index=True, header=["anomaly"])
    st.download_button("⬇️ Scarica serie (CSV)", data=csv, file_name="temperature_anomaly_timeseries.csv", mime="text/csv")