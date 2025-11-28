# app.py (UPDATED - robust Grad-CAM + safe Lottie + animations + nicer UI)
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import requests
from typing import Optional
from streamlit_lottie import st_lottie

# ---------------------------
# Page config (MUST be first Streamlit command)
# ---------------------------
st.set_page_config(page_title="Bird vs Drone AI Classifier",
                   page_icon="🚁",
                   layout="wide")

# ---------------------------
# Safe Lottie loader
# ---------------------------
def load_lottie_url(url: str, timeout: int = 5) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

# public Lottie urls (examples)
drone_lottie_url = "https://assets6.lottiefiles.com/packages/lf20_u4yrau.json"
bird_lottie_url = "https://assets1.lottiefiles.com/packages/lf20_q5pk6p1k.json"
drone_anim = load_lottie_url(drone_lottie_url)
bird_anim = load_lottie_url(bird_lottie_url)

# ---------------------------
# Load model (cached)
# ---------------------------
@st.cache_resource
def load_model():
    # adjust path if needed
    model = tf.keras.models.load_model("saved_models/best_model.h5")
    return model

model = load_model()

# ---------------------------
# Preprocessing helper
# ---------------------------
def preprocess(img_pil: Image.Image, target_size=(224, 224)):
    img = img_pil.convert("RGB").resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------
# Utility: find backbone and a last conv layer robustly
# ---------------------------
def find_backbone_and_last_conv(model):
    """
    Returns tuple (backbone_layer, last_conv_layer_name)
    backbone_layer is a tf.keras.Model inside the full model (e.g., efficientnetb0)
    last_conv_layer_name is the name of a Conv2D/DepthwiseConv2D layer inside that backbone.
    If not found returns (None, None)
    """
    # prefer a layer with 'efficient' in name
    backbone = None
    for lyr in model.layers:
        if isinstance(lyr, tf.keras.Model) and 'efficient' in lyr.name:
            backbone = lyr
            break

    # fallback: first nested model
    if backbone is None:
        for lyr in model.layers:
            if isinstance(lyr, tf.keras.Model):
                backbone = lyr
                break

    # find last conv inside backbone
    if backbone is not None:
        from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
        for lyr in reversed(backbone.layers):
            if isinstance(lyr, (Conv2D, DepthwiseConv2D)):
                return backbone, lyr.name

    # fallback: search top-level model layers
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
    for lyr in reversed(model.layers):
        if isinstance(lyr, (Conv2D, DepthwiseConv2D)):
            return None, lyr.name

    return None, None

# ---------------------------
# Grad-CAM (robust)
# ---------------------------
def generate_gradcam(input_array, model):
    """
    Returns heatmap (H, W) numpy array with values in [0,1],
    or None if Grad-CAM cannot be computed.
    """
    try:
        backbone, conv_name = find_backbone_and_last_conv(model)
        if conv_name is None:
            return None

        # get the target tensor from the full model graph
        target_tensor = None
        if backbone is not None:
            try:
                outer_backbone = model.get_layer(backbone.name)  # this should exist
                target_tensor = outer_backbone.get_layer(conv_name).output
            except Exception:
                # fallback: try get_layer on model with conv_name
                try:
                    target_tensor = model.get_layer(conv_name).output
                except Exception:
                    target_tensor = None
        else:
            try:
                target_tensor = model.get_layer(conv_name).output
            except Exception:
                target_tensor = None

        if target_tensor is None:
            return None

        # build model that maps input -> conv outputs + predictions
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[target_tensor, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_array)
            loss = predictions[:, 0]  # binary output logits/probs

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        convs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, convs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        denom = np.max(heatmap) if np.max(heatmap) != 0 else 1e-9
        heatmap /= denom
        heatmap = cv2.resize(heatmap.numpy(), (224, 224))
        return heatmap
    except Exception:
        # do not crash the app — return None and show fallback
        return None

# ---------------------------
# Overlay heatmap on PIL image
# ---------------------------
def apply_heatmap_on_image(pil_img: Image.Image, heatmap):
    img = np.array(pil_img.convert("RGB").resize((224, 224)))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb

# ---------------------------
# Probability bar plot
# ---------------------------
def plot_probability(pred_score):
    labels = ["Bird", "Drone"]
    values = [1 - pred_score, pred_score]
    fig, ax = plt.subplots(figsize=(4, 1.2))
    bars = ax.bar(labels, values, color=['#2b8cbe', '#f03b20'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}", ha='center', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

# ---------------------------
# UI layout + sidebar
# ---------------------------
st.markdown("<style> .stApp { background-color: #0e0f11; color: #fff; } </style>", unsafe_allow_html=True)

st.sidebar.title("Navigation")
mode = st.sidebar.radio("Input mode", ["Upload Image", "Camera (web)"])
st.sidebar.markdown("**Model:** EfficientNetB0 (fine-tuned)")
st.sidebar.markdown("**Classes:** Bird, Drone")

with st.sidebar.expander("Animations", expanded=True):
    st.markdown("### Live demos")
    # use lottie only if loaded; else show animated gifs
    if drone_anim is not None:
        st_lottie(drone_anim, height=140)
    else:
        st.image("https://i.imgur.com/7bKae8x.gif", width=140)

    if bird_anim is not None:
        st_lottie(bird_anim, height=140)
    else:
        st.image("https://i.imgur.com/0fV6F2x.gif", width=140)

st.title("🦅 Bird vs 🚁 Drone — Advanced Classifier")
st.write("Upload an image or use your camera. Grad-CAM interpretability shown when available.")

# ---------------------------
# Upload mode
# ---------------------------
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload an image (jpg, png)", type=['jpg', 'jpeg', 'png'])
    if uploaded:
        try:
            img = Image.open(uploaded)
        except Exception as e:
            st.error("Could not open image. Try another file.")
            st.stop()

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Input image", use_column_width=True)

        arr = preprocess(img)
        preds = model.predict(arr)
        pred = float(preds[0][0])
        label = "Drone" if pred > 0.5 else "Bird"
        confidence = pred if label == "Drone" else 1 - pred

        # Grad-CAM
        heatmap = generate_gradcam(arr, model)
        if heatmap is not None:
            overlay = apply_heatmap_on_image(img, heatmap)
            with col2:
                st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
        else:
            with col2:
                st.info("Grad-CAM not available for this model architecture.")
                st.image(img, caption="No Grad-CAM available", use_column_width=True)

        # Results
        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"### Confidence: **{confidence:.4f}**")
        if label == "Drone":
            st.error("🚨 ALERT: Drone detected!")
        else:
            st.success("🕊️ Bird detected.")
        plot_probability(pred)

# ---------------------------
# Camera mode
# ---------------------------
elif mode == "Camera (web)":
    cam_img = st.camera_input("Capture from camera")
    if cam_img is not None:
        try:
            img = Image.open(cam_img)
        except Exception:
            st.error("Could not read camera image.")
            st.stop()

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Captured image", use_column_width=True)

        arr = preprocess(img)
        preds = model.predict(arr)
        pred = float(preds[0][0])
        label = "Drone" if pred > 0.5 else "Bird"
        confidence = pred if label == "Drone" else 1 - pred

        heatmap = generate_gradcam(arr, model)
        if heatmap is not None:
            overlay = apply_heatmap_on_image(img, heatmap)
            with col2:
                st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
        else:
            with col2:
                st.info("Grad-CAM not available for this model architecture.")
                st.image(img, caption="No Grad-CAM available", use_column_width=True)

        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"### Confidence: **{confidence:.4f}**")
        if label == "Drone":
            st.error("🚨 ALERT: Drone detected!")
        else:
            st.success("🕊️ Bird detected.")
        plot_probability(pred)
