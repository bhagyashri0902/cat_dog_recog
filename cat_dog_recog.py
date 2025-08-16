import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2

# Load the saved model
@st.cache_resource  # cache so model loads only once
def load_catdog_model():
    model = load_model("cd_model.h5")
    return model

model = load_catdog_model()

# Define class names
class_names = ["Cat", "Dog"]

# ---------------- UI Layout ----------------
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱🐶", layout="centered")

st.markdown(
    """
    <div style="text-align: center;">
        <h1> Cat vs Dog Recognition App</h1>
        <p style="font-size:18px;">Upload an image and let the AI predict whether it's a cat or a dog.</p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([2, 1])  # image left, prediction right

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("🔍 Predict"):
            # Preprocess image
            img_resized = cv2.resize(img_array, (224, 224))
            img_resized = img_resized / 255.0
            img_resized = img_resized.reshape(1, 224, 224, 3)

            # Predict
            yp = model.predict_on_batch(img_resized).argmax()
            label = "Cat 🐱" if yp == 0 else "Dog 🐶"

            st.markdown(
                f"""
                <div style="padding:20px; background-color:#f0f2f6; border-radius:10px; text-align:center;">
                    <h3>✅ Prediction:</h3>
                    <h2 style="color:green;">{label}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
