import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
import os

# -----------------------------
# 🔧 CONFIGURATION
# -----------------------------
MODEL_PATH = "models/model.keras"

# -----------------------------
# 🔑 GEMINI API
# -----------------------------
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    except:
        gemini_model = None
else:
    gemini_model = None

# -----------------------------
# 🧠 LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# 🎨 UI
# -----------------------------
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection System")

# -----------------------------
# 📸 UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze"):

            with st.spinner("Analyzing..."):

                # -----------------------------
                # 🔥 AUTO-FIX SHAPE (KEY PART)
                # -----------------------------
                input_shape = model.input_shape
                height, width, channels = input_shape[1], input_shape[2], input_shape[3]

                # Fix color channels
                if channels == 3:
                    image = image.convert("RGB")
                else:
                    image = image.convert("L")

                # Resize correctly
                img = image.resize((width, height))

                # Convert to array
                img_array = np.array(img)

                # Handle grayscale
                if channels == 1:
                    img_array = np.expand_dims(img_array, axis=-1)

                # Normalize
                img_array = img_array / 255.0

                # Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)

                # -----------------------------
                # 🔍 DEBUG (optional)
                # -----------------------------
                # st.write("Input shape:", img_array.shape)
                # st.write("Model expects:", model.input_shape)

                # -----------------------------
                # 🤖 PREDICT
                # -----------------------------
                prediction = model.predict(img_array)
                predicted_class = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                st.success(f"Prediction Class Index: {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}")

                # -----------------------------
                # 🌱 GEMINI RESPONSE
                # -----------------------------
                if gemini_model:
                    try:
                        query = f"Plant disease detected: class {predicted_class}. Give treatment and prevention steps."
                        response = gemini_model.generate_content(query)

                        st.subheader("🌱 Treatment Advice")
                        st.write(response.text)

                    except Exception as e:
                        st.error(f"Gemini Error: {e}")
                else:
                    st.warning("Chatbot not available (API key missing)")

    except Exception as e:
        st.error(f"Error: {e}")