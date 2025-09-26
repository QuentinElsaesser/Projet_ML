import streamlit as st

def render_header(title: str = "ML App", subtitle: str = ""):
    """
    Affiche le header de l'application.
    Args:
        title (str): titre principal
        subtitle (str): sous-titre optionnel
    """
    st.markdown(
        f"""
        <div style="padding: 10px; background-color: #8a8db5; border-radius: 5px; text-align: center;">
            <h1 style="margin-bottom: 5px;">{title}</h1>
            <p style="margin-top: 0px; color: #555;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")  # ligne de séparation
