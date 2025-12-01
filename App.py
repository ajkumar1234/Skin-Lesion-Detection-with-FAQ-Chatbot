import streamlit as st
from utils import preprocess_image
from model import get_model, predict
import torch

st.title("🩺 Skin Lesion Detection")

uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess
    image_tensor = preprocess_image(uploaded_file)

    # Load model (replace with your trained weights if available)
    model = get_model(num_classes=7)
    # Example: model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))

    # Predict
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    label_idx = predict(model, image_tensor, device=device)

    labels = ["Melanoma", "Nevus", "Basal Cell Carcinoma", "Actinic Keratosis", 
              "Benign Keratosis", "Dermatofibroma", "Vascular Lesion"]
    st.write(f"### Predicted Label: {labels[label_idx]}")
