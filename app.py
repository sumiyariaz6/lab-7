import streamlit as st
import tensorflow as tf
import numpy as np

st.title("TensorFlow Sentiment Demo App")

st.write("Enter text and get AI prediction")

# Simple fixed TensorFlow model (NO training on cloud)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

text = st.text_area("Enter text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        # simple numeric feature (safe for demo)
        feature = np.array([[len(text)]], dtype=np.float32)

        prediction = model(feature, training=False).numpy()[0][0]

        if prediction > 0.5:
            st.success(f"Positive 😊 Score: {prediction:.2f}")
        else:
            st.error(f"Negative 😞 Score: {prediction:.2f}")
