import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.data_io import open_xr, synthetic_sea_level_ts
from services.climate_ops import find_lat_lon, area_weights, fit_trend


def render_sea_tab(cfg, state):
    st.subheader("Livello medio del mare")
    up_sea = cfg["up_sea"]; var_sea = cfg["var_sea"]
    use_dask = cfg["use_dask"]; chunk_time = cfg["chunk_time"]

    if up_sea is not None:
        ds_sea = open_xr(up_sea.getvalue(), {"time": int(chunk_time)} if use_dask else None)
        if var_sea not in ds_sea:
            st.warning(f"Variabile '{var_sea}' non trovata. Variabili: {list(ds_sea.data_vars)}")
        varname = var_sea if var_sea in ds_sea else list(ds_sea.data_vars)[0]
        daS = ds_sea[varname]
        if "time" not in daS.dims:
            st.error("Il dataset mare deve avere la dimensione 'time'.")
            st.stop()
        if set(["lat", "latitude"]).intersection(daS.dims) and set(["lon", "longitude"]).intersection(daS.dims):
            lat_name, lon_name = find_lat_lon(daS.to_dataset(name="v"))
            w = area_weights(daS[lat_name])
            ts = daS.weighted(w).mean(dim=(lat_name, lon_name)).to_series()
        else:
            ts = daS.to_series()
        ts.name = ts.name or "sea_level"
    else:
        st.info("Nessun NetCDF caricato. Uso **dati sintetici** (GMSL ~ mm).")
        ts = synthetic_sea_level_ts()

    st.plotly_chart(px.line(ts, labels={"index": "Tempo", "value": ts.name}), use_container_width=True)

    yhat, slope_py = fit_trend(ts)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines", name=ts.name))
    fig2.add_trace(go.Scatter(x=yhat.index, y=yhat.values, mode="lines", name=f"Trend (~{slope_py:.2f} unità/anno)"))
    st.plotly_chart(fig2, use_container_width=True)

    csv = ts.to_csv(index=True, header=["sea_level"])
    st.download_button("⬇️ Scarica serie (CSV)", data=csv, file_name="sea_level_timeseries.csv", mime="text/csv")