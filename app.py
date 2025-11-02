import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import io

# ---------------------------------------------
# 🎯 PAGE CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection using Gemini AI",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection using Gemini AI")
st.write("Upload a leaf image to identify possible diseases and get treatment suggestions.")

# ---------------------------------------------
# 🔒 GEMINI API CONFIGURATION
# ---------------------------------------------
api_key = st.secrets.get("GOOGLE_API_KEY")  # ✅ Streamlit Secrets variable

if not api_key:
    st.error("❌ No Gemini API key found. Please set GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=api_key)

# ---------------------------------------------
# ⚙️ IMAGE UPLOAD
# ---------------------------------------------
uploaded_file = st.file_uploader("📸 Upload a clear image of the leaf", type=["jpg", "jpeg", "png"])

# ---------------------------------------------
# 🧠 ANALYSIS FUNCTION
# ---------------------------------------------
def analyze_image(image):
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = (
        "You are an expert plant pathologist. Identify the disease from this leaf image "
        "and provide: 1) The disease name, 2) Short description, "
        "3) Treatment methods using organic and chemical approaches, "
        "4) Prevention tips. Respond in simple, clear points."
    )
    
    result = model.generate_content([prompt, image])
    return result.text

# ---------------------------------------------
# 🚀 PREDICTION
# ---------------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
    
    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing the image with Gemini..."):
            try:
                result = analyze_image(image)
                st.success("✅ Analysis complete!")
                st.markdown(result)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# ---------------------------------------------
# 🪴 FOOTER
# ---------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Google Gemini and Streamlit")