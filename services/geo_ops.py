import os
from typing import Optional
import geopandas as gpd
import regionmask
import xarray as xr

from .climate_ops import find_lat_lon, area_weights


def load_user_geodf(file_bytes: bytes, filename: str) -> gpd.GeoDataFrame:
    import tempfile
    suffix = ".geojson" if filename.lower().endswith((".geojson", ".json")) else ".zip"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(file_bytes)
        path = tf.name
    if suffix == ".zip":
        gdf = gpd.read_file(f"zip://{path}")
    else:
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(4326, inplace=True, allow_override=True)
    return gdf.to_crs(4326)


def europe_geometries() -> gpd.GeoDataFrame:
    import tempfile, urllib.request

    def _try_read(path: str) -> Optional[gpd.GeoDataFrame]:
        try:
            if path and path.lower().startswith("http"):
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
                    urllib.request.urlretrieve(path, tf.name)
                    return gpd.read_file(f"zip://{tf.name}")
            if path and path.lower().endswith(".zip"):
                return gpd.read_file(f"zip://{path}")
            if path and os.path.exists(path):
                return gpd.read_file(path)
        except Exception:
            return None
        return None

    candidates = [
        os.getenv("NE_ADMIN0_ZIP"),
        os.path.join("data", "ne_110m_admin_0_countries.zip"),
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip",
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
    ]
    gdf = None
    for c in candidates:
        gdf = _try_read(c)
        if gdf is not None and not gdf.empty:
            break
    if gdf is None or gdf.empty:
        raise RuntimeError("Impossibile caricare Natural Earth admin_0. Fornisci uno shape personalizzato.")

    name_col = None
    for c in ["ADMIN", "NAME_EN", "SOVEREIGNT", "name"]:
        if c in gdf.columns:
            name_col = c
            break
    if name_col != "name":
        gdf = gdf.rename(columns={name_col: "name"})
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    gdf = gdf.to_crs(4326)

    if "continent" in gdf.columns:
        gdf = gdf[(gdf["continent"].fillna("").str.contains("Europe")) | (gdf["name"].isin(["Turkey", "Russia"]))]
    else:
        gdf = gdf[gdf.geometry.centroid.y.between(34, 72) & gdf.geometry.centroid.x.between(-11, 41)]
    return gdf[["name", "geometry"]]


def mask_from_geometries(ds: xr.Dataset, geodf: gpd.GeoDataFrame) -> xr.DataArray:
    lat_name, lon_name = find_lat_lon(ds)
    try:
        regions = regionmask.Regions.from_geopandas(geodf, names=None)
    except AttributeError:
        regions = regionmask.from_geopandas(geodf, names=None)
    lon = ds[lon_name]; lat = ds[lat_name]
    mask = regions.mask(lon, lat)
    return mask.notnull()


def regional_mean(da: xr.DataArray, mask_bool: xr.DataArray) -> xr.DataArray:
    lat_name, lon_name = find_lat_lon(da.to_dataset(name="v"))
    w = area_weights(da[lat_name])
    w2 = w.broadcast_like(da)
    masked = da.where(mask_bool)
    num = (masked * w2).sum(dim=(lat_name, lon_name))
    den = (w2.where(mask_bool)).sum(dim=(lat_name, lon_name))
    return num / den