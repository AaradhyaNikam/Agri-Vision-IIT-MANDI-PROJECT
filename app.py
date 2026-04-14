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

                st.success(f"Prediction Class Index: {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}")

                # -----------------------------
                # 🌱 GEMINI RESPONSE
                # -----------------------------
                if gemini_model:
                    try:
                        query = f"Plant disease detected (class {predicted_class}). Suggest treatment and prevention."
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