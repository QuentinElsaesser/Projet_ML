import streamlit as st

from frontend.utils.api import FastAPIClient
from frontend.components.sidebar import render_sidebar
from frontend.components.pages import home, train, predict, metrics
from frontend.components.header import render_header
# from components.charts import plot_metrics, plot_metrics_history

from backend.utils.config import config

def main():
    st.set_page_config(page_title=config.front.title, layout="wide")
    
    # Header
    render_header(title=config.front.title, subtitle="Prédiction et entraînement de modèle")
    
    # Instanciation unique du client
    client = FastAPIClient(port=config.api.port, host=config.api.dockerhost)

    # Pages avec passage du client
    PAGES = {
        "Accueil": lambda: home.render_home_page(),
        "Entraînement": lambda: train.render_train_page(client),
        "Prédiction": lambda: predict.render_predict_page(client),
        "Métriques": lambda: metrics.render_metrics_page(client),
    }
    
    page_name = render_sidebar(PAGES, client)
    page_func = PAGES[page_name]
    # Affichage de la page selectionnée
    page_func()
    
    # Exemple de graphique
    # metrics = client.get_metrics()
    # history = client.get_metrics_history()
    # plot_metrics(metrics)
    # plot_metrics_history(history)

if __name__ == "__main__":
    main()
