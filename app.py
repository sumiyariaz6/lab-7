import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# Load model correctly (TF Hub way)
# -----------------------------
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    return model

model = load_model()

# -----------------------------
# COCO Labels
# -----------------------------
LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    44: 'bottle', 47: 'cup', 64: 'keyboard', 65: 'cell phone',
    72: 'tv'
}

# -----------------------------
# Detection function
# -----------------------------
def detect(image):
    img = np.array(image)

    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]

    outputs = model(input_tensor)

    boxes = outputs["detection_boxes"][0].numpy()
    classes = outputs["detection_classes"][0].numpy().astype(int)
    scores = outputs["detection_scores"][0].numpy()

    h, w, _ = img.shape

    for i in range(len(scores)):
        if scores[i] > 0.5:
            cls = classes[i]
            label = LABELS.get(cls, str(cls))

            y1, x1, y2, x2 = boxes[i]
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {scores[i]:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    return img

# -----------------------------
# UI
# -----------------------------
st.title("🔍 AI Object Detection (Free TensorFlow Model)")
st.write("Upload image and detect objects (NO API KEY).")

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file)
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Detect Objects"):
        result = detect(image)
        st.image(result, caption="Detected Objects", use_container_width=True)
