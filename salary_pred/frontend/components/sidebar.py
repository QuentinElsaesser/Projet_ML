import streamlit as st
from typing import Dict, Callable

from frontend.utils.api import FastAPIClient

def render_sidebar(pages: Dict[str, Callable], client: FastAPIClient):
    """
    Affiche la sidebar et retourne la page sélectionnée.

    Args:
        pages: dictionnaire {nom_page: fonction_affichage}

    Returns:
        nom_page sélectionnée
    """
    st.sidebar.title("Navigation")
    page_name = st.sidebar.radio("Aller à :", list(pages.keys()))
    try:
        health = client.health()
        st.sidebar.success("Backend OK")
        st.sidebar.info(f"Status : {health['status']}")
        if 'model_type' in health:
            st.sidebar.info(f"Modele : {health['model_type']}")
        else:
            st.sidebar.info(f"Aucun modele chargé")
    except Exception:
        st.sidebar.error("Backend DOWN")
    return page_name
