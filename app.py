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
class_mapping = { 0: "Apple___Apple_scab", 
                 1: "Apple___Black_rot", 
                 2: "Apple___Cedar_apple_rust", 
                 3: "Apple___healthy", 
                 4: "Blueberry___healthy", 
                 5: "Cherry_(including_sour)___Powdery_mildew", 
                 6: "Cherry_(including_sour)___healthy", 
                 7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 
                 8: "Corn_(maize)___Common_rust_", 
                 9: "Corn_(maize)___Northern_Leaf_Blight", 
                 10: "Corn_(maize)___healthy", 
                 11: "Grape___Black_rot", 
                 12: "Grape___Esca_(Black_Measles)", 
                 13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 
                 14: "Grape___healthy", 
                 15: "Orange___Haunglongbing_(Citrus_greening)", 
                 16: "Peach___Bacterial_spot", 
                 17: "Peach___healthy", 
                 18: "Pepper,_bell___Bacterial_spot", 
                 19: "Pepper,_bell___healthy", 
                 20: "Potato___Early_blight", 
                 21: "Potato___Late_blight", 
                 22: "Potato___healthy", 
                 23: "Raspberry___healthy", 
                 24: "Soybean___healthy", 
                 25: "Squash___Powdery_mildew", 
                 26: "Strawberry___Leaf_scorch", 
                 27: "Strawberry___healthy", 
                 28: "Tomato___Bacterial_spot", 
                 29: "Tomato___Early_blight", 
                 30: "Tomato___Late_blight", 
                 31: "Tomato___Leaf_Mold", 
                 32: "Tomato___Septoria_leaf_spot", 
                 33: "Tomato___Spider_mites Two-spotted_spider_mite", 
                 34: "Tomato___Target_Spot", 
                 35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus", 
                 36: "Tomato___Tomato_mosaic_virus", 
                 37: "Tomato___healthy" }

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
