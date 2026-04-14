import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# -----------------------------
# 🔧 CONFIGURATION

# -----------------------------
MODEL_PATH = "models/model.keras"   # your trained CNN model file

# Set up authentication using service account
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gen-lang-client-0327394450-bb75c29f5afb.json"

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
    "0": "Apple___Apple_scab",
    "1": "Apple___Black_rot",
    "2": "Apple___Cedar_apple_rust",
    "3": "Apple___healthy",
    "4": "Blueberry___healthy",
    "5": "Cherry_(including_sour)___Powdery_mildew",
    "6": "Cherry_(including_sour)___healthy",
    "7": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "8": "Corn_(maize)___Common_rust_",
    "9": "Corn_(maize)___Northern_Leaf_Blight",
    "10": "Corn_(maize)___healthy",
    "11": "Grape___Black_rot",
    "12": "Grape___Esca_(Black_Measles)",
    "13": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "14": "Grape___healthy",
    "15": "Orange___Haunglongbing_(Citrus_greening)",
    "16": "Peach___Bacterial_spot",
    "17": "Peach___healthy",
    "18": "Pepper,_bell___Bacterial_spot",
    "19": "Pepper,_bell___healthy",
    "20": "Potato___Early_blight",
    "21": "Potato___Late_blight",
    "22": "Potato___healthy",
    "23": "Raspberry___healthy",
    "24": "Soybean___healthy",
    "25": "Squash___Powdery_mildew",
    "26": "Strawberry___Leaf_scorch",
    "27": "Strawberry___healthy",
    "28": "Tomato___Bacterial_spot",
    "29": "Tomato___Early_blight",
    "30": "Tomato___Late_blight",
    "31": "Tomato___Leaf_Mold",
    "32": "Tomato___Septoria_leaf_spot",
    "33": "Tomato___Spider_mites Two-spotted_spider_mite",
    "34": "Tomato___Target_Spot",
    "35": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "36": "Tomato___Tomato_mosaic_virus",
    "37": "Tomato___healthy"
}

# Convert keys to integers to match np.argmax output
class_mapping = {int(k): v for k, v in class_mapping.items()}

# -----------------------------
# 🤖 CONFIGURE GEMINI MODEL
# -----------------------------
# Configure with default credentials from GOOGLE_APPLICATION_CREDENTIALS
genai.configure()
# Initialize the model with a supported model that accepts generate_content
gemini_model = genai.GenerativeModel('models/gemini-flash-latest')

# -----------------------------
# 🧩 STREAMLIT UI
# -----------------------------
st.title("🌿 Plant Disease Detection & Treatment Chatbot")
st.write("Upload a leaf image and get treatment advice instantly!")

uploaded_file = st.file_uploader("📸 Upload an image of the plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing the image..."):
            # Preprocess image
            img = image.resize((224, 224))  # adjust to your model input size
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Predict
            prediction = model.predict(img_array)
            predicted_class = int(np.argmax(prediction))
            disease_name = class_mapping.get(predicted_class, "Unknown disease")

            st.success(f"🩺 Detected: **{disease_name}**")

            # Ask Gemini for treatment advice
            try:
                query = f"My plant has {disease_name}. Suggest treatment steps, prevention methods, and organic solutions."
                response = gemini_model.generate_content(query)
                st.subheader("🌱 Treatment Advice")
                st.write(response.text)
            except Exception as e:
                st.error(f"⚠️ Gemini API Error: {e}")

# -----------------------------
# 🗨️ Chat Interface
# -----------------------------
st.markdown("---")
st.subheader("💬 Ask More Questions About Plant Care")

user_query = st.text_input("Type your question here...")

if st.button("Send to Chatbot"):
    if user_query.strip() != "":
        try:
            response = gemini_model.generate_content(user_query)
            st.write(response.text)
        except Exception as e:
            st.error(f"⚠️ Gemini API Error: {e}")
    else:
        st.warning("Please type a question before sending.")