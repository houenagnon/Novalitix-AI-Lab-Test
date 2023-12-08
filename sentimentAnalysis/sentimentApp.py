import streamlit as st
import pickle
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

st.title("Analyse de Sentiments pour les Critiques de Films")

# Charger le modèle et le vectorizer
with open("final_model.pkl", "rb") as file:
    final_model = pickle.load(file)

with open("cv.pkl", "rb") as file:
    cv = pickle.load(file)

# Fonction de prétraitement
def preprocess_input(input_text):
    input_text = input_text.lower()
    input_text = re.sub("[;:!\'?,\"()\[\]]", "", input_text)
    input_text = re.sub("(<br\s*/><br\s*/>)|(\-)|(\/)|[.]", " ", input_text)
    input_text = re.sub("[0-9]", "", input_text)
    return input_text

# Fonction pour préparer la chaîne
def prepare_string(input_text):
    input_text = preprocess_input(input_text)
    english_stopwords = set(stopwords.words('english'))
    filtre_stopen = lambda text: [token for token in text if token.lower() not in english_stopwords]
    input_text = ' '.join(filtre_stopen(word_tokenize(input_text)))
    stemmer = EnglishStemmer()
    input_text = stemmer.stem(input_text)
    return [input_text]

# Zone de texte pour l'entrée de l'utilisateur
user_input = st.text_area("Entrez votre critique ici", "")

# Bouton pour effectuer la prédiction
if st.button("Prédire le sentiment"):
    input_data = prepare_string(user_input)
    input_vectorized = cv.transform(input_data)
    prediction = final_model.predict(input_vectorized)
    st.write(f"Sentiment prédit : {'Positif' if prediction[0] == 1 else 'Négatif'}")
