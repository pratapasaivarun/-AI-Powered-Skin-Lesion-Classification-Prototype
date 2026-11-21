"""Streamlit interface for the skin lesion classification project."""

import os
import pickle
import sys

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    CLASS_INFO,
    FINAL_MODEL_PATH,
    IMAGE_DIR_PART1,
    IMAGE_DIR_PART2,
    IMAGE_SIZE,
    LABEL_ENCODER_PATH,
    RISK_COLORS,
)
from src.image_utils import get_model_image_size, preprocess_rgb_image
from src.triage import build_triage, check_image_quality


st.set_page_config(
    page_title="DermAssist | Skin Lesion Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stAppViewContainer"] { background: #f5f7fb; color: #23364a; }
    [data-testid="stSidebar"] { background: #eef2f8; color: #23364a; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] { color: #34495e !important; }
    h1, h2, h3, h4 { color: #17324b !important; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #263d52 !important; }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        color: #f8fafc;
        background: linear-gradient(120deg, #10243a 0%, #174e5b 100%);
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: #fff !important; margin: 0.3rem 0 0.6rem; }
    .hero p { color: #d7e5ec !important; font-size: 1.05rem; max-width: 760px; }
    .eyebrow {
        color: #8de0c1;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #e6ebf2;
        border-radius: 14px;
        padding: 1rem;
    }
    .note {
        background: #fff;
        border: 1px solid #e6ebf2;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model_and_encoder(model_path, encoder_path, model_modified, encoder_modified):
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    with open(encoder_path, "rb") as encoder_file:
        encoder = pickle.load(encoder_file)
    return model, encoder


def main():
    model_ready = os.path.isfile(FINAL_MODEL_PATH) and os.path.isfile(LABEL_ENCODER_PATH)
    dataset_csv = os.path.join(PROJECT_ROOT, "dataset", "hmnist_28_28_RGB.csv")
    metadata_csv = os.path.join(PROJECT_ROOT, "dataset", "HAM10000_metadata.csv")
    image_data_ready = (
        os.path.isfile(metadata_csv)
        and os.path.isdir(IMAGE_DIR_PART1)
        and os.path.isdir(IMAGE_DIR_PART2)
    )

    st.sidebar.markdown("## 🔬 DermAssist")
    st.sidebar.caption("Skin lesion classification · research prototype")
    st.sidebar.divider()
    if model_ready:
        st.sidebar.success("Model artifacts found")
    else:
        st.sidebar.info("Model not trained yet")
    st.sidebar.markdown("### Supported classes")
    for code, info in CLASS_INFO.items():
        marker = RISK_COLORS.get(info["risk"], "⚪")
        st.sidebar.markdown(f"{marker} **{code}** · {info['name']}")
    st.sidebar.divider()
    st.sidebar.caption(
        "For educational research only. This tool cannot diagnose skin conditions."
    )

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Explainable AI · HAM10000</div>
            <h1>Skin lesion analysis, made interpretable.</h1>
            <p>
                Explore a deep-learning workflow for classifying dermoscopic images,
                visualising model attention, and reviewing the limitations of
                image-based predictions.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, analyze_tab, about_tab = st.tabs(
        ["Project overview", "Image analysis", "About & limitations"]
    )

    with overview_tab:
        st.subheader("Project status")
        status_text = "Ready for analysis" if model_ready else "Waiting for a trained model"
        st.write(
            "The interface is running. It will enable predictions after compatible "
            "model and label-encoder artifacts are available."
        )
        status_columns = st.columns(3)
        status_columns[0].metric("Inference model", "Available" if model_ready else "Not trained")
        status_columns[1].metric(
            "Original image set", "Available" if image_data_ready else "Not set up"
        )
        status_columns[2].metric("Bundled 28×28 CSV", "Available" if os.path.isfile(dataset_csv) else "Not found")

        if model_ready:
            st.success(f"{status_text}. Open **Image analysis** to continue.")
        else:
            st.info(
                "The app is intentionally running in preview mode; it does not train "
                "a model or download data automatically."
            )
            st.markdown("#### Choose a training route when you're ready")
            st.markdown(
                "**Original HAM10000 images (224×224):** install dependencies, configure "
                "Kaggle credentials, then run `python main.py --mode download` followed "
                "by `python main.py --mode train --model hybrid`.\n\n"
                "**Bundled HMNIST CSV (28×28):** install dependencies and run "
                "`python run_full_pipeline.py`. This separate script trains a model and "
                "writes artifacts compatible with this app."
            )
            st.warning(
                "Both routes start model training when you run their commands. "
                "No training or data download is started by this app."
            )
            if os.path.isfile(dataset_csv) and not os.path.isfile(metadata_csv):
                st.caption(
                    "The bundled HMNIST CSV is a different 28×28 dataset; it is not "
                    "the image/metadata input used by the 224×224 pipeline."
                )

        st.markdown("#### What this project demonstrates")
        feature_columns = st.columns(3)
        feature_columns[0].markdown(
            "**Classification**  \n"
            "Seven lesion classes with predicted probabilities."
        )
        feature_columns[1].markdown(
            "**Explainability**  \n"
            "Grad-CAM overlays highlight image regions used by the network."
        )
        feature_columns[2].markdown(
            "**Reproducible workflow**  \n"
            "Separate data, model, evaluation, and inference modules."
        )

    with analyze_tab:
        st.subheader("Analyze a dermoscopic image")
        st.caption(
            "Upload a JPG or PNG. Images are processed in memory and are not saved by this app."
        )
        if not model_ready:
            st.warning(
                "Analysis is unavailable until both the trained model and label encoder "
                "are present in the `models` directory. No model is included or trained automatically."
            )
        else:
            model_modified = os.path.getmtime(FINAL_MODEL_PATH)
            encoder_modified = os.path.getmtime(LABEL_ENCODER_PATH)
            try:
                model, encoder = load_model_and_encoder(
                    FINAL_MODEL_PATH,
                    LABEL_ENCODER_PATH,
                    model_modified,
                    encoder_modified,
                )
            except (ImportError, OSError, ValueError) as error:
                st.error(f"Could not load the model artifacts: {error}")
                return

            uploaded = st.file_uploader(
                "Choose a lesion image",
                type=["jpg", "jpeg", "png"],
                help="Use a clear dermoscopic image. Maximum file size: 15 MB.",
            )
            if uploaded is not None:
                if uploaded.size > 15 * 1024 * 1024:
                    st.error("This file is larger than 15 MB. Choose a smaller image.")
                    return
                try:
                    image = Image.open(uploaded).convert("RGB")
                except (UnidentifiedImageError, OSError, ValueError) as error:
                    st.error(f"Unable to read this image: {error}")
                    return

                image_array = np.asarray(image)
                quality_warnings = check_image_quality(image_array)
                for quality_warning in quality_warnings:
                    st.info(f"Photo check: {quality_warning}")
                input_array = preprocess_rgb_image(
                    image_array, get_model_image_size(model, IMAGE_SIZE)
                )
                with st.spinner("Analyzing image..."):
                    predictions = np.asarray(model.predict(input_array, verbose=0))[0]

                if predictions.ndim != 1 or len(predictions) != len(encoder.classes_):
                    st.error(
                        "The model output does not match the label encoder. "
                        "Check that both artifacts were created by the same training run."
                    )
                    return
                if not np.all(np.isfinite(predictions)):
                    st.error("The model returned invalid prediction values.")
                    return

                predicted_index = int(np.argmax(predictions))
                predicted_class = encoder.classes_[predicted_index]
                confidence = float(predictions[predicted_index])
                info = CLASS_INFO.get(predicted_class, {})
                risk = info.get("risk", "UNKNOWN")
                marker = RISK_COLORS.get(risk, "⚪")
                triage = build_triage(predicted_class, confidence)

                image_column, explanation_column = st.columns(2)
                with image_column:
                    st.markdown("#### Uploaded image")
                    st.image(image_array, use_container_width=True)
                with explanation_column:
                    st.markdown("#### Model attention")
                    try:
                        from src.gradcam import (
                            find_last_conv_layer,
                            make_gradcam_heatmap,
                            overlay_gradcam,
                        )

                        last_conv_layer = find_last_conv_layer(model)
                        heatmap = make_gradcam_heatmap(
                            input_array, model, last_conv_layer
                        )
                        st.image(
                            overlay_gradcam(image_array, heatmap),
                            caption="Grad-CAM visualization (not a clinical explanation)",
                            use_container_width=True,
                        )
                    except (ValueError, TypeError, KeyError, IndexError) as error:
                        st.warning(f"Grad-CAM could not be generated: {error}")

                st.divider()
                st.markdown("#### Prediction summary")
                result_columns = st.columns(3)
                result_columns[0].metric(
                    "Predicted class", info.get("name", predicted_class)
                )
                result_columns[1].metric("Model confidence", f"{confidence:.1%}")
                result_columns[2].metric("Project risk category", f"{marker} {risk}")
                st.markdown(f"**Next-step guidance: {triage['label']}**")
                for recommendation in triage["recommendations"]:
                    st.write(f"- {recommendation}")
                st.caption(
                    f"{info.get('description', 'No class description available.')}"
                )
                st.bar_chart(
                    {
                        encoder.classes_[index]: float(probability)
                        for index, probability in enumerate(predictions)
                    },
                    y_label="Model score",
                )
                st.warning(
                    "This is an experimental model output, not a diagnosis or medical "
                    "recommendation. Model confidence is not a measure of clinical certainty. "
                    "Consult a qualified healthcare professional about health concerns."
                )

    with about_tab:
        st.subheader("A research and portfolio project")
        st.markdown(
            "This project explores a seven-class image-classification workflow using "
            "HAM10000, with custom CNN and MobileNetV2-based architectures. Grad-CAM is "
            "included as a visualization of model activation; it does not establish "
            "causality or clinical validity."
        )
        st.markdown("#### Important limitations")
        st.markdown(
            "- The project is not validated for diagnosis, screening, or treatment decisions.\n"
            "- Performance can vary with image quality, device, skin tone, and dataset bias.\n"
            "- A high score is not a guarantee, and a low score does not rule out disease.\n"
            "- Predictions should not replace assessment by a qualified clinician."
        )
        st.caption(
            "Educational use only · No uploaded image is intentionally persisted by the app."
        )


if __name__ == "__main__":
    main()
