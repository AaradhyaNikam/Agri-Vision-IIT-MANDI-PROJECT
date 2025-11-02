import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
import os

# -----------------------------
# 🔧 CONFIGURATION
# -----------------------------
MODEL_PATH = "models/model.keras"   # your trained CNN model file

# ✅ Load Gemini API Key from Streamlit Secrets
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ No Gemini API key found. Please set GOOGLE_API_KEY in Streamlit Secrets.")
else:
    genai.configure(api_key=api_key)

# -----------------------------
# 🧠 LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -----------------------------
# 🗂️ DEFINE CLASS LABELS
# -----------------------------
class_mapping = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy",
    28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight",
    30: "Tomato___Late_blight",
    37: "Tomato___healthy"
    # Add your full mapping here as per your model
}

# -----------------------------
# 🤖 CONFIGURE GEMINI MODEL
# -----------------------------
try:
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")  # ✅ Updated model name
except Exception as e:
    st.warning(f"⚠️ Gemini Model not initialized: {e}")
    gemini_model = None

# -----------------------------
# 🧩 STREAMLIT UI
# -----------------------------
st.title("🌿 Plant Disease Detection & Treatment Chatbot")
st.write("Upload a plant leaf image to detect disease and get instant treatment advice.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing..."):
            img = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = model.predict(img_array)
            predicted_class = int(np.argmax(prediction))
            disease_name = class_mapping.get(predicted_class, "Unknown disease")

            st.success(f"🩺 Detected: **{disease_name}**")

            if gemini_model:
                try:
                    query = f"My plant has {disease_name}. Suggest treatment steps, prevention methods, and organic solutions."
                    response = gemini_model.generate_content(query)
                    st.subheader("🌱 Treatment Advice")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"⚠️ Gemini API Error: {e}")
            else:
                st.warning("Gemini model not configured. Please add your API key.")

# -----------------------------
# 🗨️ CHATBOT SECTION
# -----------------------------
st.markdown("---")
st.subheader("💬 Chat with the Plant Expert")

user_query = st.text_input("Ask your plant-related question:")

if st.button("Send"):
    if user_query.strip() == "":
        st.warning("Please type a question first.")
    elif not gemini_model:
        st.error("❌ Gemini model not configured. Add your API key.")
    else:
        try:
            response = gemini_model.generate_content(user_query)
            st.write(response.text)
        except Exception as e:
            st.error(f"⚠️ Gemini API Error: {e}")
