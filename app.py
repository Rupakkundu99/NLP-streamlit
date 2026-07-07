import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the word index and model
word_index = tf.keras.datasets.imdb.get_word_index()
# FIXED: changed load_all to load_model
model = tf.keras.models.load_model('sentiment_model.keras')

# Helper function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    # Offset by 3 for IMDb dataset specific tokens
    tokens = [word_index.get(word, 2) + 3 for word in words]
    padded = pad_sequences([tokens], maxlen=250, padding='post', truncating='post')
    return padded

st.title("IMDb Sentiment Predictor")
st.write("Enter a movie review below to see if it's Positive or Negative!")

user_input = st.text_area("Review:", "This movie was fantastic!")

if st.button("Predict"):
    processed_input = preprocess_text(user_input)
    prediction = model.predict(processed_input)[0][0]

    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.metric("Sentiment", sentiment)
    st.write(f"Confidence: {confidence:.2%}")
