import streamlit as st
import spacy
import numpy as np
from joblib import load
import subprocess
import sys

# Function to download and load the SpaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading 'en_core_web_sm' model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Function to preprocess text
def preprocess(text):
    if not text.strip():
        return ""  # Handle empty input safely

    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]
    return " ".join(filtered_tokens)

# Load the trained model
@st.cache_resource
def load_model():
    return load("random_forest_model.joblib")

clf = load_model()

# Title of the app
st.title("Spam Detection with Streamlit")

# Add a text input field
user_input = st.text_input("Enter some text:")

if user_input:
    processed_text = preprocess(user_input)
    
    if processed_text:
        st.subheader("Preprocessed Text:")
        st.write(processed_text)

        doc = nlp(processed_text)
        
        # Check if the vector exists (some words may not have embeddings)
        if doc.vector_norm > 0:
            prediction = clf.predict(doc.vector.reshape(1, -1))
            labels = ["ham", "spam"]
            st.subheader("Prediction:")
            st.write(f"📢 This message is likely: **{labels[int(prediction[0])]}**")
        else:
            st.warning("⚠️ Unable to generate a vector for this text. Try using more meaningful words.")
    else:
        st.warning("⚠️ No valid words detected. Please enter a different text.")
else:
    st.write("Please enter some text to analyze.")
