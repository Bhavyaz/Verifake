import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to load the Xception model
@st.cache(allow_output_mutation=True)
def load_xception_model():
    model_path = "final_model.h5"
    model = tf.keras.models.load_model(model_path)
    return model

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128,128))  # Resize image to 299x299 for Xception
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Load the Xception model
model = load_xception_model()

# Streamlit app
st.title("Deepfake Detection App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(preprocessed_image)

    # Display prediction
    st.subheader("Prediction")
    if prediction >= 0.5:
        st.write("This image is predicted to be a real image.")
    else:
        st.write("This image is predicted to be a fake image.")
