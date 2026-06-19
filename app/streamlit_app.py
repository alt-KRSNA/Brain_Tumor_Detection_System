import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    layout="centered"
)

# FastAPI URL
API_URL = "http://127.0.0.1:8000/api/predict"

# Title
st.title("Brain Tumor Classification")

st.write("Upload an MRI image for tumor classification.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        with st.spinner("Predicting..."):

            try:

                # Reset file pointer
                uploaded_file.seek(0)

                # Read image bytes
                image_bytes = uploaded_file.read()

                # Debug
                st.write(f"File Size : {len(image_bytes)} bytes")

                files = {
                    "file": (
                        uploaded_file.name,
                        image_bytes,
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files
                )

                if response.status_code == 200:

                    result = response.json()

                    st.subheader("Prediction")

                    st.write(
                        f"Prediction : {result['prediction']}"
                    )

                    st.write(
                        f"Confidence : {result['confidence']} %"
                    )

                    st.subheader("Class Probabilities")

                    probs = result["class_probabilities"]

                    labels = list(probs.keys())
                    values = list(probs.values())

                    fig, ax = plt.subplots(figsize=(6,4))

                    ax.bar(labels, values)

                    ax.set_xlabel("Class")
                    ax.set_ylabel("Probability (%)")
                    ax.set_ylim(0,100)

                    st.pyplot(fig)

                else:

                    st.error("Prediction Failed")

                    st.code(response.text)

            except Exception as e:

                st.error(str(e))