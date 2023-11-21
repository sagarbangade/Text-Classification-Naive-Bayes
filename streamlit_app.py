import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import joblib

# Load the model and vectorizer
nb = joblib.load('nb_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit UI
st.title("Active or Passive Voice Classifier")
user_input = st.text_input("Enter a sentence: ")

if user_input:
    # Vectorize the user input
    user_input_dtm = vectorizer.transform([user_input])

    # Make a prediction
    user_input_pred = nb.predict(user_input_dtm)

    st.write(f'The sentence is in {user_input_pred[0]} voice.')
