import streamlit as st

def render_about_tab():
    st.markdown(
        """
        ### Guida rapida
        1. **Carica i tuoi NetCDF** per temperatura e mare, e un **CSV** per la CO₂; in alternativa abilita i **dati sintetici** lasciando vuoti gli upload.
        2. Scegli la **baseline** e il tipo di **climatologia** per calcolare le anomalie.
        3. Seleziona uno o più **paesi europei** per l'aggregazione regionale (media area-pesata), oppure lascia vuoto per una media sull'area completa del dataset.
        4. Esporta le serie come **CSV** dai rispettivi tab.

        #### Estensioni possibili
        - Indici estremi (ETCCDI) calcolati on-the-fly.
        - Explorer ensemble CMIP6/EURO-CORDEX con incertezza (p10–p90).
        - What-if (storylines locali) e downscaling DL (UNet leggero).
        """
    )