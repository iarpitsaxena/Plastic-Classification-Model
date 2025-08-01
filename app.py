
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Minimal, formal Streamlit page config
st.set_page_config(
    page_title="Plastic Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# Load the trained model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# Class labels (modify based on your dataset)
class_labels = ["Organic", "Recyclable"]

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image




# Minimal sidebar with improved color
st.sidebar.markdown(
    """
    <h2 style='color:#009688; font-weight:600; margin-bottom:0.5em;'>Plastic Waste Classifier</h2>
    <p style='color:#37474f; font-size:1.05em;'>Upload a plastic waste image to classify it as <b>Organic</b> or <b>Recyclable</b>.</p>
    """,
    unsafe_allow_html=True
)



# Main UI with modern color palette
st.markdown("""
<h2 style='text-align:center; color:#009688; font-weight:700; margin-bottom:0.5em;'>Plastic Waste Classification</h2>
<p style='text-align:center; color:#37474f; font-size:1.13em;'>Upload an image to classify it as <b style='color:#ff9800;'>Organic</b> or <b style='color:#1976d2;'>Recyclable</b>.</p>
""", unsafe_allow_html=True)


# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=False,
    help="Upload a clear image of plastic waste"
)



if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #e0f7fa 0%, #fffde7 100%); border-radius:10px; padding:22px 0; margin-top:22px; text-align:center; border:1.5px solid #b2dfdb;'>
        <span style='font-size:20px; color:#009688; font-weight:600;'><b>Prediction Result</b></span><br>
        <span style='font-size:17px; color:#37474f;'><b>Class:</b> <span style='color:#1976d2;'>{predicted_class}</span></span><br>
        <span style='font-size:16px; color:#37474f;'><b>Confidence:</b> <span style='color:#ff9800;'>{confidence:.2f}%</span></span>
    </div>
    """, unsafe_allow_html=True)



st.markdown("""
<hr style='margin-top:2em; margin-bottom:1em; border: none; border-top: 1.5px solid #b2dfdb;'>
<div style='text-align:center; color:#009688; font-size:0.97em;'>
    Powered by <a href='https://streamlit.io/' target='_blank' style='color:#1976d2;'>Streamlit</a> & <a href='https://www.tensorflow.org/' target='_blank' style='color:#ff9800;'>TensorFlow</a>
</div>
""", unsafe_allow_html=True)
