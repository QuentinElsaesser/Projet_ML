import streamlit as st
import numpy as np

from frontend.utils.helpers import parse_text_input
from frontend.utils.api import FastAPIClient

def render_train_page(client: FastAPIClient):
    """
    Page pour entraîner le modèle.
    Args:
        client: instance de FastAPIClient
    """
    st.title("Entraînement du modèle")

    method = st.radio("Méthode :", ["Données manuelles", "Fichier CSV"])

    if method == "Données manuelles":
        x_input = st.text_area("Entrez vos X (ligne par ligne, séparés par des virgules)")
        y_input = st.text_area("Entrez vos Y (séparés par des virgules)")
        if st.button("Lancer l'entraînement"):
            x = parse_text_input(x_input)
            y = parse_text_input(y_input).ravel()
            metrics = client.train_model(x, y)
            st.success(f"Entraînement terminé ! Metrics : {metrics}")

    elif method == "Fichier CSV":
        uploaded_file = st.file_uploader("Sélectionnez un fichier CSV", type="csv")
        if uploaded_file and st.button("Lancer l'entraînement"):
            metrics = client.train_model_file(uploaded_file)
            st.success(f"Entraînement terminé ! Metrics : {metrics}")
