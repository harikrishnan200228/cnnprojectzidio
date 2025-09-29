import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
            if len(layer.output_shape) == 4:
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
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
# Streamlit UI
# --------------------------
st.title("🫁 Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to detect **PNEUMONIA** or **NORMAL**.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, IMG_SIZE)
    image_array = np.expand_dims(image_resized, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    # Predict
    prob = model.predict(image_array)[0][0]
    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: `{prob:.2f}`")

    # Grad-CAM
    cam = grad_cam(model, image_array)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image_resized, 0.6, heatmap, 0.4, 0)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Original X-ray", use_column_width=True)
    with col2:
        st.image(superimposed_img, caption="Grad-CAM Heatmap", use_column_width=True)
