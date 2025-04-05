import streamlit as st
from PIL import Image
from utils import load_model, predict_emotion

st.set_page_config(page_title="Happy/Sad Detector", layout="centered")

st.title("😊 Happy or 😢 Sad?")
st.write("Upload a photo and I'll guess your emotion!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner("Analyzing..."):
        model = load_model()
        label, prob = predict_emotion(model, image)

    st.success(f"Prediction: **{label}** (confidence: {prob:.2f})")
