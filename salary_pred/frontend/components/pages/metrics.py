import streamlit as st
from frontend.utils.api import FastAPIClient

def render_metrics_page(client: FastAPIClient):
    """
    Page affichant les métriques du modèle.
    Args:
        client: instance de FastAPIClient
    """
    st.title("Métriques du modèle")

    if st.button("Voir dernières métriques"):
        metrics = client.get_metrics()
        st.write(metrics)

    if st.button("Voir l'historique complet"):
        history = client.get_metrics_history()
        st.write(history)
