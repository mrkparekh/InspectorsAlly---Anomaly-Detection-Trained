import streamlit as st # type: ignore
import numpy as np
from PIL import Image # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# --- Page Setup ---python
st.set_page_config(
    page_title="Wood Inspector - AI Anomaly Detection",
    page_icon="🌲",
    layout="centered"
)

# --- Sidebar Info ---
with st.sidebar:
    st.title("🌲 Wood Inspector")
    # 
    # st.image("assets/logo.jpg", use_column_width=True)  # Optional logo image
    st.markdown("""
    **Anomaly Detection for Wood Products**  
    This app uses a deep learning model trained via [Teachable Machine](https://teachablemachine.withgoogle.com/) to detect defects in wood surfaces.

    **Instructions:**
    - Upload or capture an image
    - The AI will analyze and classify it as **Good** or **Defective**
    """)
    st.markdown("---")
    st.caption("Developed for Week-13 Assignment")

# --- Load Model (Cached) ---
@st.cache_resource
def load_tm_model():
    return load_model("keras_model.h5",compile=False)  # Ensure this file is in your local directory

model = load_tm_model()
class_names = ["Good", "Bad"]  # Adjust based on your training

# --- Prediction Function ---
def predict_image(img_pil):
    img = img_pil.resize((224, 224)).convert("RGB")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    if predicted_class == "Good":
        return f"✅ **Good Product Detected** with {confidence:.2f}% confidence."
    else:
        return f"⚠️ **Defect Detected** with {confidence:.2f}% confidence."



# --- Main UI ---
st.title("🔍 Wood Inspector")
st.caption("AI-Powered Defect Detection in Wood Products")

st.markdown("Upload or capture an image of a wood product. The AI will inspect it for defects.")

# --- Input Method ---
input_method = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"], horizontal=True)

image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload a wood product image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

elif input_method == "Use Camera":
    captured_image = st.camera_input("Capture Image")
    if captured_image:
        image = Image.open(captured_image)
        st.image(image, caption="Captured Image", use_container_width=True)

# --- Prediction Button ---
if st.button("🔎 Analyze Image", type="primary"):
    if image:
        with st.spinner("Analyzing image..."):
            result = predict_image(image)
            st.success(result)
    else:
        st.warning("Please upload or capture an image first.")
