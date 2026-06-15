import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

st.set_page_config(page_title="AI Sentiment Analyzer")

st.title("AI Sentiment Analysis App")
st.write("Enter a movie review and predict whether it is Positive or Negative.")

VOCAB_SIZE = 10000
MAX_LEN = 200

@st.cache_resource
def train_model():

    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=VOCAB_SIZE
    )

    x_train = pad_sequences(
        x_train,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    model = Sequential([
        Embedding(VOCAB_SIZE, 32, input_length=MAX_LEN),
        LSTM(32),
        Dense(24, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        epochs=2,
        batch_size=128,
        verbose=0
    )

    word_index = imdb.get_word_index()

    return model, word_index


model, word_index = train_model()

review = st.text_area("Enter Review")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter some text.")
    else:

        words = review.lower().split()

        encoded = []

        for word in words:
            encoded.append(word_index.get(word, 2) + 3)

        padded = pad_sequences(
            [encoded],
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(padded, verbose=0)[0][0]

        st.subheader("Result")

        if prediction >= 0.5:
            st.success(
                f"Positive Review ({prediction:.2%})"
            )
        else:
            st.error(
                f"Negative Review ({(1-prediction):.2%})"
            )
