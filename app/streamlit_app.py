
import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor AI Assistant",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# API CONFIGURATION
# --------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8000/api"

PREDICT_API_URL = f"{API_BASE_URL}/predict"
ASK_API_URL = f"{API_BASE_URL}/ask"


# --------------------------------------------------
# APPLICATION TITLE
# --------------------------------------------------

st.title("Brain Tumor Detection System")

st.write(
    "An AI-powered system for brain MRI classification "
    "and educational medical question answering."
)

st.divider()


# --------------------------------------------------
# APPLICATION TABS
# --------------------------------------------------

mri_tab, assistant_tab = st.tabs(
    ["MRI Classification", "Medical AI Assistant"]
)


# ==================================================
# TAB 1: MRI CLASSIFICATION
# ==================================================

with mri_tab:

    st.header("Brain MRI Classification")

    st.write(
        "Upload a brain MRI image to classify its category."
    )

    uploaded_file = st.file_uploader(
        "Choose an MRI Image",
        type=["jpg", "jpeg", "png"],
        key="mri_uploader"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded MRI Image",
            use_container_width=True
        )

        if st.button("Predict", key="predict_button"):

            with st.spinner("Analyzing MRI image..."):

                try:

                    uploaded_file.seek(0)

                    image_bytes = uploaded_file.read()

                    files = {
                        "file": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type
                        )
                    }

                    response = requests.post(
                        PREDICT_API_URL,
                        files=files,
                        timeout=120
                    )

                    if response.status_code == 200:

                        result = response.json()

                        st.success("Prediction completed!")

                        st.subheader("Prediction")

                        st.write(
                            f"**Class:** {result['prediction']}"
                        )

                        st.write(
                            f"**Model Confidence:** "
                            f"{result['confidence']}%"
                        )

                        st.subheader("Class Probabilities")

                        probs = result["class_probabilities"]

                        labels = list(probs.keys())
                        values = list(probs.values())

                        fig, ax = plt.subplots(figsize=(7, 4))

                        ax.bar(labels, values)

                        ax.set_xlabel("Tumor Class")
                        ax.set_ylabel("Probability (%)")
                        ax.set_ylim(0, 100)

                        ax.set_title("Classification Results")

                        plt.xticks(rotation=15)

                        st.pyplot(fig)

                        plt.close(fig)

                        st.caption(
                            "Educational and research use only. "
                            "This model is not a medical diagnostic tool."
                        )

                    else:

                        st.error("Prediction failed.")

                        st.code(response.text)

                except requests.exceptions.RequestException as error:

                    st.error(
                        f"Unable to connect to the API: {error}"
                    )

                except Exception as error:

                    st.error(f"An error occurred: {error}")


# ==================================================
# TAB 2: MEDICAL AI ASSISTANT
# ==================================================

with assistant_tab:

    st.header("Medical AI Assistant")

    st.write(
        "Ask educational questions about brain tumors, "
        "MRI scans, symptoms, and related topics."
    )

    st.info(
        "This assistant uses Retrieval-Augmented Generation (RAG) "
        "to generate answers from the medical PDF knowledge base."
    )

    st.caption(
        "Educational purposes only. It does not provide "
        "medical diagnosis or treatment recommendations."
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous conversation
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            # Display sources for assistant responses
            if message["role"] == "assistant":

                sources = message.get("sources", [])

                if sources:

                    with st.expander("View Medical Sources"):

                        for source in sources:

                            st.write(
                                f"- {source['filename']} "
                                f"(Page {source['page']})"
                            )

    # Chat input
    question = st.chat_input(
        "Ask a brain tumor-related question..."
    )

    if question:

        # Display user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate AI response
        with st.chat_message("assistant"):

            with st.spinner("Searching medical knowledge base..."):

                try:

                    response = requests.post(
                        ASK_API_URL,
                        json={"question": question},
                        timeout=180
                    )

                    if response.status_code == 200:

                        result = response.json()

                        answer = result["answer"]
                        sources = result.get("sources", [])

                        st.markdown(answer)

                        # Display source references
                        if sources:

                            with st.expander("View Medical Sources"):

                                for source in sources:

                                    st.write(
                                        f"- {source['filename']} "
                                        f"(Page {source['page']})"
                                    )

                        # Save assistant response
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            }
                        )

                    else:

                        error_message = (
                            "Unable to generate an answer. "
                            "Please try again."
                        )

                        st.error(error_message)
                        st.code(response.text)

                except requests.exceptions.RequestException as error:

                    st.error(
                        f"Unable to connect to the AI service: {error}"
                    )

                except Exception as error:

                    st.error(f"An error occurred: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Brain Tumor Detection System | "
    "Deep Learning + Generative AI + RAG"
)
