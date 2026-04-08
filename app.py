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
# 🔑 LOAD GEMINI API KEY
# -----------------------------
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.warning("⚠️ Gemini API key not found. Chatbot will be disabled.")
    gemini_model = None
else:
    genai.configure(api_key=api_key)
    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        st.warning(f"⚠️ Gemini initialization failed: {e}")
        gemini_model = None

# -----------------------------
# 🧠 LOAD MODEL (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# 🗂️ CLASS LABELS
# -----------------------------
class_mapping = {
    0: "Apple___Apple_scab", 1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust", 3: "Apple___healthy",
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
    37: "Tomato___healthy"
}

# -----------------------------
# 🎨 UI CONFIG
# -----------------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")

st.title("🌿 AI Plant Disease Detection & Treatment System")
st.write("Upload a plant leaf image to detect disease and get treatment advice.")

# -----------------------------
# 📸 IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # ✅ FIX: Ensure RGB
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing image... Please wait ⏳"):

                # -----------------------------
                # 🧠 PREPROCESS IMAGE
                # -----------------------------
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # -----------------------------
                # 🔍 DEBUG (optional)
                # -----------------------------
                # st.write("Input shape:", img_array.shape)
                # st.write("Model expects:", model.input_shape)

                # -----------------------------
                # 🤖 PREDICTION
                # -----------------------------
                prediction = model.predict(img_array)
                predicted_class = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                disease_name = class_mapping.get(predicted_class, "Unknown disease")

                st.success(f"🩺 Detected: **{disease_name}**")
                st.info(f"📊 Confidence: {confidence:.2f}")

                # -----------------------------
                # 🌱 GEMINI RESPONSE
                # -----------------------------
                if gemini_model:
                    try:
                        query = f"My plant has {disease_name}. Suggest treatment steps, prevention methods, and organic solutions."
                        response = gemini_model.generate_content(query)

                        st.subheader("🌱 Treatment Advice")
                        st.write(response.text)

                    except Exception as e:
                        st.error(f"⚠️ Gemini API Error: {e}")
                else:
                    st.warning("⚠️ Chatbot unavailable (API key missing).")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

# -----------------------------
# 💬 CHATBOT SECTION
# -----------------------------
st.markdown("---")
st.subheader("💬 Chat with Plant Expert")

user_query = st.text_input("Ask your plant-related question:")

if st.button("Send"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    elif not gemini_model:
        st.error("Chatbot unavailable. Add API key.")
    else:
        try:
            response = gemini_model.generate_content(user_query)
            st.write(response.text)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")