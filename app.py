
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

st.set_page_config(
    page_title="Brain MRI Image Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("Brain MRI Image Classification")
st.write(
    "An educational machine-learning project for classifying "
    "MRI images into four dataset categories."
)

st.warning(
    "This application is for educational purposes only. "
    "It is not a medical diagnostic tool. Model predictions "
    "must not be used to make medical decisions. Consult a "
    "qualified medical professional for interpretation."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mobilenetv2_final.keras",
        custom_objects={"preprocess_input": preprocess_input}
    )

with open("class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded MRI image",
            use_container_width=True
        )

        resized = image.resize((224, 224))
        array = np.asarray(resized, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

        probabilities = model.predict(array, verbose=0)[0]
        best_index = int(np.argmax(probabilities))

        st.subheader("Model output")
        st.write("Predicted dataset category:", class_names[best_index])
        st.write(
            "Model score:",
            f"{probabilities[best_index] * 100:.2f}%"
        )

        st.caption(
            "This score reflects the model's output, not the "
            "probability that a person has a disease."
        )

        st.subheader("Scores by category")

        for name, score in zip(class_names, probabilities):
            st.write(f"{name}: {score * 100:.2f}%")

    except Exception as error:
        st.error("Could not process this image.")
        st.exception(error)
