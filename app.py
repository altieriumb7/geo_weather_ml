import streamlit as st
from datetime import datetime

from tabs.temperature import render_temperature_tab
from tabs.sea import render_sea_tab
from tabs.co2 import render_co2_tab
from tabs.corr import render_corr_tab
from tabs.models import render_models_tab
from tabs.about import render_about_tab

st.set_page_config(page_title="Climate Data Explorer (EU)", page_icon="🌍", layout="wide")

# Sidebar — opzioni e upload
st.sidebar.title("⚙️ Opzioni")
st.sidebar.markdown("Se non carichi dati, puoi usare **dati sintetici**.")
use_dask = st.sidebar.toggle("Abilita Dask (chunked I/O)", value=False)
chunk_time = st.sidebar.number_input("Chunk time (passi)", min_value=1, value=12)

baseline = st.sidebar.date_input(
    "Baseline (inizio/fine)", value=(datetime(1991, 1, 1), datetime(2020, 12, 31))
)
monthly = st.sidebar.toggle("Climatologia mensile (destagionalizza)", value=True)

st.sidebar.markdown("---")

st.sidebar.subheader("📥 Dati — Temperatura (NetCDF)")
up_temp = st.sidebar.file_uploader("Carica NetCDF (tas/t2m)", type=["nc", "nc4", "grib", "grb"])
var_temp = st.sidebar.text_input("Nome variabile temperatura", value="t2m")

st.sidebar.subheader("🌊 Dati — Livello del mare (NetCDF)")
up_sea = st.sidebar.file_uploader("Carica NetCDF (zos/sea_level)", type=["nc", "nc4"])
var_sea = st.sidebar.text_input("Nome variabile mare", value="zos")

st.sidebar.subheader("🧪 Dati — CO₂ (CSV)")
up_co2 = st.sidebar.file_uploader("Carica CSV CO₂ (date,value)", type=["csv"])
co2_date_col = st.sidebar.text_input("Colonna data", value="date")
co2_val_col = st.sidebar.text_input("Colonna valore", value="value")

st.sidebar.subheader("🗺️ Regioni personalizzate (GeoJSON/Shapefile ZIP)")
up_shape = st.sidebar.file_uploader("Carica GeoJSON o Shapefile (.zip)", type=["geojson", "json", "zip"])

# Config condivisa
CFG = dict(
    use_dask=use_dask,
    chunk_time=int(chunk_time),
    baseline=baseline,
    monthly=monthly,
    up_temp=up_temp,
    var_temp=var_temp,
    up_sea=up_sea,
    var_sea=var_sea,
    up_co2=up_co2,
    co2_date_col=co2_date_col,
    co2_val_col=co2_val_col,
    up_shape=up_shape,
)

st.title("🌍 Climate Data Explorer — Europa")
st.caption("Analisi interattiva di anomalie di temperatura, livello del mare e CO₂. Streamlit · Xarray · Plotly · GeoPandas")

TAB_TEMP, TAB_SEA, TAB_CO2, TAB_CORR, TAB_AI, TAB_ABOUT = st.tabs([
    "🌡️ Temperatura",
    "🌊 Livello del mare",
    "🧪 CO₂",
    "🔗 Correlazioni",
    "🤖 Modelli",
    "ℹ️ Info",
])

with TAB_TEMP:
    render_temperature_tab(CFG, st.session_state)
with TAB_SEA:
    render_sea_tab(CFG, st.session_state)
with TAB_CO2:
    render_co2_tab(CFG, st.session_state)
with TAB_CORR:
    render_corr_tab(CFG, st.session_state)
with TAB_AI:
    render_models_tab(CFG, st.session_state)
with TAB_ABOUT:
    render_about_tab()