import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.title("AI Sentiment Analysis App (Free Deploy)")

# Sample training data
texts = [
    "I love this product",
    "This is amazing",
    "Very good experience",
    "I hate this",
    "Very bad service",
    "Worst experience"
]

labels = [1, 1, 1, 0, 0, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

user_input = st.text_area("Enter text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")
