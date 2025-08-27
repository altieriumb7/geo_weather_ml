from typing import Tuple
import numpy as np
import pandas as pd
import xarray as xr


def find_lat_lon(ds: xr.Dataset) -> Tuple[str, str]:
    candidates = [("lat", "lon"), ("latitude", "longitude"), ("nav_lat", "nav_lon"), ("y", "x")]
    for la, lo in candidates:
        if la in ds.dims and lo in ds.dims:
            return la, lo
        if la in ds.coords and lo in ds.coords:
            return la, lo
    raise ValueError("Impossibile individuare le coordinate lat/lon nel dataset.")


def kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    units = da.attrs.get("units", "")
    if units.lower() in ["k", "kelvin"] or da.name in ["t2m", "tas"]:
        out = da - 273.15
        out.attrs.update({"units": "°C"})
        return out
    return da


def area_weights(lat: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def compute_anomaly(da: xr.DataArray, baseline: Tuple[str, str], monthly_climatology: bool = True) -> xr.DataArray:
    da = da.sortby("time")
    t0, t1 = pd.to_datetime(baseline[0]), pd.to_datetime(baseline[1])
    ref = da.sel(time=slice(t0, t1))
    if ref.time.size == 0:
        raise ValueError("La baseline selezionata non interseca il dataset.")
    if monthly_climatology:
        clim = ref.groupby("time.month").mean("time")
        anom = da.groupby("time.month") - clim
        anom.attrs["long_name"] = f"Anomalia {da.name} (climatologia mensile {t0.year}-{t1.year})"
    else:
        clim = ref.mean("time")
        anom = da - clim
        anom.attrs["long_name"] = f"Anomalia {da.name} (media {t0.year}-{t1.year})"
    if "units" in da.attrs:
        anom.attrs["units"] = da.attrs["units"]
    return anom


def fit_trend(ts: pd.Series):
    if ts.dropna().empty:
        return ts, float("nan")
    x = (ts.index - ts.index[0]).days.values.reshape(-1, 1)
    y = ts.values.reshape(-1, 1)
    A = np.hstack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = (A @ coeffs).flatten()
    slope_per_year = float(coeffs[1]) * 365.25
    return pd.Series(yhat, index=ts.index), slope_per_year