import streamlit as st
from frontend.utils.helpers import parse_text_input
from frontend.utils.api import FastAPIClient

def render_predict_page(client: FastAPIClient, model_type: str):
    """
    Page pour prédire avec le modèle.
    Args:
        client: instance de FastAPIClient
    """
    st.title("Prédictions")

    x_input = st.text_area("Entrez vos X (ligne par ligne, séparés par des virgules)")
    if st.button("Prédire"):
        x = parse_text_input(x_input)
        predictions = client.predict(x, model_type=model_type)
        st.success(f"Résultats : {predictions}")
