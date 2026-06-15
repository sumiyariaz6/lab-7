import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="TensorFlow Demo App")

st.title("AI Text Sentiment Demo")

st.write("Enter a sentence:")

text = st.text_area("Text")

# Simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Dummy feature based on text length
        feature = np.array([[len(text)]], dtype=np.float32)

        prediction = model(feature, training=False).numpy()[0][0]

        st.success(f"Prediction Score: {prediction:.4f}")

        if prediction > 0.5:
            st.write("Positive")
        else:
            st.write("Negative")
