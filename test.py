import streamlit as st
import spacy
from joblib import load
import numpy as np

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)


# Title of the app
st.title("Simple Streamlit Input Example")

# Add a text input field
user_input = st.text_input("Enter some text:")

propcced_text = preprocess(user_input)

# Display the input
if user_input:
    st.write(propcced_text)

else:
    st.write("Please enter some text.")

resp = ["ham", "spam"]
if len(propcced_text) != 0:
    doc = nlp(propcced_text)
    # X_test_2d =  np.stack(doc.vector)
    # print(X_test_2d)

    loaded_clf = load("random_forest_model.joblib")
    result = loaded_clf.predict(doc.vector.reshape(1, -1))
    st.write(resp[int(result[0])])
