import streamlit as st
from PIL import Image
from utils import load_model, predict_emotion

# Set page layout and title
st.set_page_config(
    page_title="Emotion Detector",
    layout="centered",
    page_icon="🧠"
)

st.title("🧠 Emotion Detector")
st.write("Upload a photo of a face, and I'll try to guess the emotion!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing emotion..."):
        model = load_model()
        label, prob = predict_emotion(model, image)

    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {prob:.2f}")
