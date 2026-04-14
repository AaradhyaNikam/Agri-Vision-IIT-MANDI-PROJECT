import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# -----------------------------
# 🔧 CONFIGURATION
# -----------------------------
MODEL_PATH = "models/model.keras"

# -----------------------------
# 🔑 GEMINI API
# -----------------------------
api_key = None
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    try:
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
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
# 🗂️ DEFINE CLASS LABELS
# -----------------------------
class_mapping = {
    0: "Apple___Apple_scab",
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
    37: "Tomato___healthy"
}

# -----------------------------
# 🎨 UI
# -----------------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌿 AI Plant Disease Detection System")

# -----------------------------
# 📸 UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyze Image"):

            with st.spinner("Analyzing image... ⏳"):

                # -----------------------------
                # 🔥 AUTO INPUT SHAPE HANDLING
                # -----------------------------
                input_shape = model.input_shape

                # Handle models like (None, 224,224,3)
                if len(input_shape) == 4:
                    height = input_shape[1]
                    width = input_shape[2]
                    channels = input_shape[3]
                else:
                    st.error("❌ Unsupported model input shape")
                    st.stop()

                # -----------------------------
                # 🎯 FIX IMAGE CHANNELS
                # -----------------------------
                if channels == 3:
                    image = image.convert("RGB")
                elif channels == 1:
                    image = image.convert("L")
                else:
                    st.error("❌ Invalid channel size in model")
                    st.stop()

                # -----------------------------
                # 🖼️ RESIZE IMAGE CORRECTLY
                # -----------------------------
                img = image.resize((width, height))

                # Convert to numpy
                img_array = np.array(img)

                # Handle grayscale case
                if channels == 1:
                    img_array = np.expand_dims(img_array, axis=-1)

                # Normalize safely
                img_array = img_array.astype("float32") / 255.0

                # Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)

                # -----------------------------
                # 🔍 DEBUG INFO (REMOVE LATER)
                # -----------------------------
                st.write("Input Shape:", img_array.shape)
                st.write("Model Expected:", model.input_shape)

                # -----------------------------
                # 🤖 PREDICTION
                # -----------------------------
                prediction = model.predict(img_array)
                predicted_class = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
                disease_name = class_mapping.get(predicted_class, f"Unknown (Class {predicted_class})")

                st.success(f"🩺 Detected: **{disease_name}**")
                st.info(f"📊 Confidence: **{confidence:.2%}**")

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
                        st.error(f"Gemini Error: {e}")
                else:
                    st.warning("⚠️ Chatbot unavailable (API key missing)")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

# -----------------------------
# 💬 CHATBOT SECTION
# -----------------------------
st.markdown("---")
st.subheader("💬 Ask Plant Expert")

user_query = st.text_input("Ask anything about plant diseases:")

if st.button("Send"):
    if not user_query.strip():
        st.warning("Enter a question first.")
    elif not gemini_model:
        st.error("Chatbot unavailable.")
    else:
        try:
            response = gemini_model.generate_content(user_query)
            st.write(response.text)
        except Exception as e:
            st.error(f"Error: {e}")