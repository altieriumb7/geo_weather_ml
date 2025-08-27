import io
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st

@st.cache_data(show_spinner=True)
def open_xr(file_bytes: bytes, chunks: Optional[dict]):
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tf:
        tf.write(file_bytes)
        path = tf.name
    if chunks:
        return xr.open_dataset(path, chunks=chunks)
    return xr.open_dataset(path)

@st.cache_data(show_spinner=True)
def load_co2_csv(file_bytes: bytes, date_col: str, value_col: str) -> pd.Series:
    df = pd.read_csv(io.BytesIO(file_bytes))
    dt = pd.to_datetime(df[date_col])
    ser = pd.Series(df[value_col].values, index=dt).sort_index()
    ser.name = "CO2"
    return ser

@st.cache_data(show_spinner=False)
def synthetic_temperature_ds() -> xr.Dataset:
    lats = np.arange(35.0, 71.0, 1.0)
    lons = np.arange(-10.0, 41.0, 1.0)
    time = pd.date_range("1980-01-01", "2024-12-01", freq="MS")
    month = time.month.to_numpy()
    seasonal = 10 + 15 * np.sin(2 * np.pi * (month - 1) / 12)
    trend = np.linspace(0.0, 1.2, len(time))
    noise = np.random.normal(scale=0.8, size=(len(lats), len(lons), len(time)))
    field = seasonal.reshape(1, 1, -1) + trend.reshape(1, 1, -1) + noise
    da = xr.DataArray(
        field,
        dims=("latitude", "longitude", "time"),
        coords={"latitude": lats, "longitude": lons, "time": time},
        name="t2m",
        attrs={"units": "°C"},
    ).transpose("time", "latitude", "longitude")
    return da.to_dataset()

@st.cache_data(show_spinner=False)
def synthetic_sea_level_ts() -> pd.Series:
    time = pd.date_range("1993-01-01", "2024-12-01", freq="MS")
    trend_mm = 3.5 * (np.arange(len(time)) / 12.0)
    seasonal = 2.0 * np.sin(2 * np.pi * (time.month - 1) / 12)
    noise = np.random.normal(scale=3.0, size=len(time))
    return pd.Series(trend_mm + seasonal + noise, index=time, name="GMSL (mm)")

@st.cache_data(show_spinner=False)
def synthetic_co2_ts() -> pd.Series:
    time = pd.date_range("1980-01-01", "2024-12-01", freq="MS")
    baseline = 340.0
    trend = np.linspace(0, 100, len(time))
    seasonal = 5.0 * np.sin(2 * np.pi * (time.month - 5) / 12)
    noise = np.random.normal(scale=0.6, size=len(time))
    return pd.Series(baseline + trend + seasonal + noise, index=time, name="CO2 (ppm)")