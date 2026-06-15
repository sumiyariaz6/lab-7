import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.title("AI Sentiment Analysis App")

# training data (mini AI model)
texts = [
    "I love this",
    "This is amazing",
    "Very good",
    "I hate this",
    "Very bad",
    "Worst experience"
]

labels = [1, 1, 1, 0, 0, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

user_input = st.text_area("Enter text")

if st.button("Predict"):
    if user_input.strip():
        x = vectorizer.transform([user_input])
        pred = model.predict(x)[0]

        if pred == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")
    else:
        st.warning("Please enter text")
