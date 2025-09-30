import streamlit as st

from frontend.utils.api import FastAPIClient
from frontend.components.sidebar import render_sidebar
from frontend.components.header import render_header
from frontend.components.pages import home, train, predict, metrics, image_classification

from backend.utils.config import config

def main():
    st.set_page_config(page_title=config.front.title, layout="wide")
    
    # Header
    render_header(title=config.front.title, subtitle="Prédiction et entraînement de modèle")
    
    # Instanciation unique du client
    client = FastAPIClient(port=config.api.port, host=config.api.dockerhost)

    # Pages avec passage du client
    PAGES = {
        "Accueil": lambda *_: home.render_home_page(),
        # "Entraînement": lambda client, model_type: train.render_train_page(client, model_type),
        # "Prédiction": lambda client, model_type: predict.render_predict_page(client, model_type),
        # "Métriques": lambda client, model_type: metrics.render_metrics_page(client),
        "Image classification": lambda client, model_type: image_classification.render_image_classification_page(client, model_type)
    }

    page_name, model_type = render_sidebar(PAGES, client)
    PAGES[page_name](client, model_type)

if __name__ == "__main__":
    main()
