import plotly.graph_objects as go
from .climate_ops import find_lat_lon
import xarray as xr


def to_plotly_heatmap(da2d: xr.DataArray, title: str = "") -> go.Figure:
    lat_name, lon_name = find_lat_lon(da2d.to_dataset(name="v"))
    z = da2d.values
    x = da2d[lon_name].values
    y = da2d[lat_name].values
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorbar=dict(title=da2d.attrs.get("units", ""))))
    fig.update_layout(title=title, xaxis_title=lon_name, yaxis_title=lat_name, margin=dict(l=0, r=0, t=40, b=0), height=520)
    return fig