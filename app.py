import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Import your custom XAI modules from Week 2
from xai_analysis import (
    load_and_preprocess_image,
    get_nested_gradcam_heatmap,
    overlay_gradcam,
    get_lime_explanation
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Jaaniv Krushi - Intelligent Crop Diagnostics", 
    page_icon="🌿", 
    layout="wide"
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """Loads the Keras model. Cached to ensure fast page reloads."""
    return tf.keras.models.load_model('models/model.keras')

def get_class_names():
    """Dynamically fetches class names from local dataset or uses standard PlantVillage fallback."""
    dataset_dir = 'plantvillage dataset/color/'
    if os.path.exists(dataset_dir):
        # Scan the directory to get alphabetical folder names
        classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        if len(classes) > 0:
            return classes
            
    # Standard PlantVillage 38 classes fallback if folder isn't available
    return [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]

# ---------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------
st.sidebar.title("🌿 Jaaniv Krushi")
st.sidebar.subheader("Intelligent Crop Diagnostics")

st.sidebar.markdown("---")
st.sidebar.markdown("### About Project")
st.sidebar.info(
    "**Jaaniv Krushi** leverages a cutting-edge **Hybrid CNN-ViT** architecture to accurately diagnose crop diseases. "
    "We integrate **Explainable AI (Grad-CAM + LIME)** to ensure our diagnostic models are transparent, trustworthy, and actionable for farmers."
)
st.sidebar.markdown("---")

# Image Uploader Widget
uploaded_file = st.sidebar.file_uploader(
    "Upload a Leaf Scan", 
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------
st.title("Agri-Vision: Crop Disease Analysis")

if uploaded_file is not None:
    # 1. Cleanly handle temporary image saving so xai_analysis path functions work
    temp_img_path = "temp_uploaded_image.jpg"
    with open(temp_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Load Model & Preprocess
    with st.spinner("Loading AI Model and Preprocessing Image..."):
        model = load_model()
        class_names = get_class_names()
        
        preprocessed_img, raw_img = load_and_preprocess_image(temp_img_path)
        
        # 3. Model Inference
        preds = model.predict(preprocessed_img)
        pred_idx = np.argmax(preds[0])
        confidence = np.max(preds[0]) * 100
        predicted_disease = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
        
        # Format the display name (e.g. 'Apple___Apple_scab' -> 'Apple - Apple scab')
        display_disease_name = predicted_disease.replace('___', ' - ').replace('_', ' ')

    # 4. Header Card - Display Prediction & Confidence
    st.markdown("### Diagnosis Result")
    st.markdown(f"<h2 style='color: #2e7b32;'>{display_disease_name}</h2>", unsafe_allow_html=True)
    st.progress(int(confidence), text=f"Diagnostic Confidence Score: {confidence:.2f}%")
    
    st.markdown("---")
    
    # 5. Tabbed Interface
    tab1, tab2, tab3 = st.tabs([
        "📷 Visual Inspection & Diagnosis", 
        "🔍 Explainable AI Diagnostics (XAI)", 
        "💊 Treatment & Prevention"
    ])
    
    # TAB 1: Visual Inspection
    with tab1:
        st.markdown("#### Original Leaf Scan")
        st.image(Image.open(temp_img_path), caption="Uploaded Image", use_column_width=True)
        
        st.markdown("#### Top Class Probabilities")
        # Get top 3 predictions
        top_indices = np.argsort(preds[0])[-3:][::-1]
        for idx in top_indices:
            label = class_names[idx].replace('___', ' - ').replace('_', ' ') if idx < len(class_names) else f"Class {idx}"
            prob = preds[0][idx] * 100
            st.write(f"- **{label}**: {prob:.2f}%")
            
    # TAB 2: Explainable AI
    with tab2:
        st.markdown("#### Model Transparency & Feature Localization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Grad-CAM: Global Heatmap**")
            with st.spinner("Computing spatial gradients for Grad-CAM..."):
                heatmap = get_nested_gradcam_heatmap(
                    model=model, 
                    img_array=preprocessed_img, 
                    inner_model_name='mobilenetv2_1.00_224', 
                    pred_index=pred_idx
                )
                gradcam_img = overlay_gradcam(raw_img, heatmap)
                st.image(gradcam_img.astype('uint8'), use_column_width=True)
        
        with col2:
            st.markdown("**LIME: Localized Superpixels**")
            with st.spinner("Perturbing superpixels for LIME Explanations (this takes a moment)..."):
                lime_img = get_lime_explanation(
                    model=model, 
                    raw_img=raw_img, 
                    top_labels=1, 
                    num_samples=500, # Using 500 to keep UI responsive
                    num_features=5
                )
                st.image(lime_img, use_column_width=True)
                
        # Brief technical caption
        st.info(
            "**Technical Insight:** Explainable AI (XAI) mapping validates our model's diagnostic transparency. "
            "The **Grad-CAM Heatmap** highlights the broad regions of the leaf that strongly influenced the model globally. "
            "The **LIME Superpixels** define the exact textual anomalies and localized boundary spots driving the top prediction, "
            "proving the model is identifying actual pathological symptoms rather than background artifacts."
        )
        
    # TAB 3: Treatment Guidance
    with tab3:
        st.markdown(f"#### Disease Management Guidance for {display_disease_name.split('-')[0].strip()}")
        
        with st.expander("🌱 Organic Solutions"):
            st.write("""
            * **Neem Oil Extract:** Spray a 1% solution of Neem oil for broad-spectrum fungal and pest control.
            * **Compost Teas:** Enhance leaf microbiome resistance by applying aerated compost teas.
            * **Pruning:** Immediately prune and safely burn/destroy infected leaves to limit spore spread.
            """)
            
        with st.expander("🧪 Chemical Treatments"):
            st.write("""
            * **Copper-based Fungicides:** Effective against bacterial spots and early blights. Apply strictly according to label intervals.
            * **Systemic Fungicides:** If the infection is widespread, systemic intervention may be required to protect new growth.
            * *Note: Always consult your local agricultural extension for approved chemical regulations.*
            """)
            
        with st.expander("🛡️ Prevention Steps"):
            st.write("""
            * **Crop Rotation:** Do not plant crops from the same family in the same soil consecutively.
            * **Water Management:** Avoid overhead watering which encourages fungal spores. Use drip irrigation.
            * **Tool Sanitation:** Sterilize pruning shears with 70% isopropyl alcohol between cuts.
            """)

    # Clean up the temporary file to prevent clutter
    try:
        os.remove(temp_img_path)
    except Exception:
        pass
        
else:
    # Default landing screen instruction
    st.info("👈 Please upload a leaf image from the sidebar to begin the diagnostic analysis.")