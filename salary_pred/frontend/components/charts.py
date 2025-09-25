import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

sns.set_style("whitegrid")

def plot_metrics(metrics: Dict[str, float], title: str = "Métriques du modèle"):
    """
    Affiche un graphique des métriques sous forme de bar chart.
    
    Args:
        metrics (Dict[str, float]): dictionnaire {"mae": ..., "mse": ..., ...}
        title (str): titre du graphique
    """
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    fig, ax = plt.subplots()
    sns.barplot(x="Metric", y="Value", data=df, palette="Blues_d", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def plot_metrics_history(history: List[Dict[str, Any]], title: str = "Historique des métriques"):
    """
    Affiche l'historique des métriques pour chaque entraînement.
    
    Args:
        history (List[Dict[str, Any]]): liste de dictionnaires metrics
        title (str): titre du graphique
    """
    if not history:
        st.warning("Aucun historique disponible")
        return
    
    df = pd.DataFrame(history)
    df.index.name = "Entraînement"
    fig, ax = plt.subplots()
    df.plot(kind="bar", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)
