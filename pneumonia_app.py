import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import tempfile
import os
import zipfile

# --------------------------
# Config
# --------------------------
MODEL_PATH = r"C:\Users\harik\Downloads\archive\pneumonia_mobilenetv2.h5"
IMG_SIZE = (224, 224)

# --------------------------
# Load Model
# --------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --------------------------
# Grad-CAM function
# --------------------------
def grad_cam(model, img_array, layer_name=None):
    if layer_name is None:
        for layer in reversed(model.layers):
            try:
                if len(layer.output.shape) == 4:
                    layer_name = layer.name
                    break
            except:
                continue

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    grads = grads / (tf.reduce_max(tf.abs(grads)) + 1e-10)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1).numpy()

    cam = np.maximum(cam[0], 0)
    cam = cam / (cam.max() + 1e-10)
    cam = cv2.resize(cam, IMG_SIZE)
    return cam

# --------------------------
# Helper: Predict from image
# --------------------------
def predict_image(image_rgb):
    image_resized = cv2.resize(image_rgb, IMG_SIZE)
    image_array = np.expand_dims(image_resized, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    prob = model.predict(image_array)[0][0]
    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    cam = grad_cam(model, image_array)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image_resized, 0.6, heatmap, 0.4, 0)
    return label, prob, superimposed_img

# --------------------------
# Helper: Read and validate uploaded image
# --------------------------
def read_image_safe(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.warning(f"⚠️ Skipping invalid or unsupported file: {uploaded_file.name}")
        return None

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --------------------------
# Streamlit UI
# --------------------------
st.title("🫁 Pneumonia Detection with Grad-CAM")
st.write("Upload chest X-rays for detection. Supports **single image** or **batch upload with CSV + ZIP export**.")

mode = st.radio("Choose Mode", ["Single Image", "Batch Folder"])

if mode == "Single Image":
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if uploaded_file.name.startswith("._") or uploaded_file.name.startswith("."):
            st.warning("⚠️ Hidden/system files are ignored.")
        else:
            image_rgb = read_image_safe(uploaded_file)
            if image_rgb is not None:
                label, prob, heatmap_img = predict_image(image_rgb)

                st.subheader(f"Prediction: **{label}**")
                st.write(f"Confidence: `{prob*100:.2f}%`")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_rgb, caption="Original X-ray", use_container_width=True)
                with col2:
                    st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)

elif mode == "Batch Folder":
    uploaded_files = st.file_uploader("Upload multiple X-ray images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        results = []
        csv_data = []

        with tempfile.TemporaryDirectory() as tmpdirname:
            heatmap_dir = os.path.join(tmpdirname, "heatmaps")
            os.makedirs(heatmap_dir, exist_ok=True)

            for uploaded_file in uploaded_files:
                if uploaded_file.name.startswith("._") or uploaded_file.name.startswith("."):
                    continue  # Skip hidden/system files

                image_rgb = read_image_safe(uploaded_file)
                if image_rgb is None:
                    continue  # Skip invalid images

                label, prob, heatmap_img = predict_image(image_rgb)
                results.append((uploaded_file.name, label, prob, heatmap_img, image_rgb))
                csv_data.append({"File Name": uploaded_file.name, "Prediction": label, "Confidence (%)": f"{prob*100:.2f}"})

                heatmap_path = os.path.join(heatmap_dir, f"{os.path.splitext(uploaded_file.name)[0]}_heatmap.jpg")
                cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))

            for name, label, prob, heatmap_img, orig_img in results:
                st.write(f"**{name}** → Prediction: **{label}** (Confidence: `{prob*100:.2f}%`)")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(orig_img, caption="Original X-ray", use_container_width=True)
                with col2:
                    st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
                st.markdown("---")

            df = pd.DataFrame(csv_data)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="pneumonia_predictions.csv",
                mime="text/csv"
            )

            zip_path = os.path.join(tmpdirname, "heatmaps.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in os.listdir(heatmap_dir):
                    zipf.write(os.path.join(heatmap_dir, file), file)

            with open(zip_path, "rb") as f:
                st.download_button(
                    label="📦 Download All Heatmaps (ZIP)",
                    data=f,
                    file_name="pneumonia_heatmaps.zip",
                    mime="application/zip"
                )
