import streamlit as st
import numpy as np
from PIL import Image
import io
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Optional: Gemini API
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
    GEMINI_ENABLED = True
except Exception as e:
    GEMINI_ENABLED = False
    gemini_model = None

# Load Keras model
@st.cache_resource
def load_skin_model():
    return load_model("facial_skin_model.keras")

model = load_skin_model()
class_names = ['Acne', 'Blackheads', 'Clear', 'Dry', 'Oily', 'Scars', 'Spots', 'Whiteheads']

# Get Gemini advice
def get_advice(condition):
    if not GEMINI_ENABLED:
        return "⚠ Gemini not available or API key not configured."

    prompt = f"""
You are a medical assistant. Explain the skin condition: "{condition}" in both English and Tamil.

🔹 English  
💡 Description:  
🔥 Causes:  
🌿 Remedies:  
📊 Severity:  
👨‍⚕ Doctor Advice:

🔹 தமிழ்  
💡 விளக்கம்:  
🔥 காரணங்கள்:  
🌿 வழிகள்:  
📊 நிலைமை:  
👨‍⚕ மருத்துவர் ஆலோசனை:
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"

# Streamlit UI
st.set_page_config(page_title="MediScan AI", layout="centered")
st.title("🧑‍⚕️ MediScan AI – Skin Analyzer")
st.write("Upload a facial image to get the skin condition and remedies.")

uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✅ Predicted Condition: **{predicted_class}**")

    with st.spinner("🧠 Getting advice from Gemini..."):
        advice = get_advice(predicted_class)
        st.subheader("📋 Health Advice")
        st.markdown(advice)
