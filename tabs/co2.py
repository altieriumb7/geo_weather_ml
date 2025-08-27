import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.data_io import load_co2_csv, synthetic_co2_ts
from services.climate_ops import fit_trend


def render_co2_tab(cfg, state):
    st.subheader("Concentrazione di CO₂")
    up_co2 = cfg["up_co2"]; dcol = cfg["co2_date_col"]; vcol = cfg["co2_val_col"]

    if up_co2 is not None:
        try:
            co2 = load_co2_csv(up_co2.getvalue(), dcol, vcol)
        except Exception as e:
            st.error(f"Errore nel CSV CO₂: {e}")
            st.stop()
    else:
        st.info("Nessun CSV caricato. Uso **dati sintetici** (ppm).")
        co2 = synthetic_co2_ts()

    st.plotly_chart(px.line(co2, labels={"index": "Tempo", "value": co2.name}), use_container_width=True)

    yhat, slope_py = fit_trend(co2)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=co2.index, y=co2.values, mode="lines", name=co2.name))
    fig2.add_trace(go.Scatter(x=yhat.index, y=yhat.values, mode="lines", name=f"Trend (~{slope_py:.2f} ppm/anno)"))
    st.plotly_chart(fig2, use_container_width=True)

    csv = co2.to_csv(index=True, header=["co2_ppm"])
    st.download_button("⬇️ Scarica serie (CSV)", data=csv, file_name="co2_timeseries.csv", mime="text/csv")