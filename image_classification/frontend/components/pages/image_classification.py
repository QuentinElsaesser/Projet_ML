import streamlit as st

from torchvision import datasets

from frontend.utils.api import FastAPIClient

def render_image_classification_page(client: FastAPIClient, model_type: str = "image_classifier"):
    st.title("Classification d'images (Chest X-ray Pneumonia)")

    st.subheader("Entraîner depuis un dossier")
    root_dir = st.text_input("Chemin dataset (ex: data/raw)", value="data/raw/")
    backbone = st.selectbox("Backbone", ["resnet18", "resnet50"])
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=3)
    batch_size = st.selectbox("Batch size", [8, 16, 32, 64], index=1)
    mode = st.selectbox("Mode", ["balanced", "augmented", "normal"])
    if st.button("Lancer l'entraînement"):
        with st.spinner("Initialisation de l'entraînement..."):
            init_resp = client.train_image_init(
                root_dir=root_dir,
                epochs=epochs,
                backbone=backbone,
                batch_size=batch_size,
                mode=mode
            )
            st.success("Initialisation terminée")
        st.write("### Progression")
        progress = st.progress(0)
        log_area = st.empty()
        
        metrics_history = []
        for epoch in range(epochs):
            with st.spinner(f"Epoch {epoch+1}/{epochs}..."):
                step_metrics = client.train_image_step()
                metrics_history.append(step_metrics)
                progress.progress((epoch + 1) / epochs)
                log_area.json(step_metrics)
        st.success("✅ Entraînement terminé")
        st.write("### Résumé des métriques")
        st.json(metrics_history)

    st.markdown("---")
    st.subheader("Prédire une image")
    ###
    train_dataset = datasets.ImageFolder("data/raw/train")
    st.write(f"Class : {train_dataset.class_to_idx}")
    ###
    uploaded = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg"])
    pred_backbone = st.selectbox("Backbone (inference)", ["resnet18", "resnet50"], index=0)
    if uploaded and st.button("Prédire l'image"):
        with st.spinner("Prédiction en cours..."):
            resp = client.predict_image(uploaded, backbone=pred_backbone)
            st.write("Résultat:", resp)
            if "predictions" in resp:
                mapping = {0: "NORMAL", 1: "PNEUMONIA"}
                st.write("Classe prédite:", mapping.get(resp["predictions"][0], resp["predictions"][0]))
