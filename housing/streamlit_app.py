import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def main():
    st.header("Prédiction de la classe d'appartenance d'un Iris 2")
    run_the_app()

def run_the_app():
    # Read the dataset
    @st.cache
    def load_data():
        return load_iris()
    
    iris_dataset =  load_data()
    
    # Data splitting into train and test 
    X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)

    st.write("L'exactitude de ce modèle est de :", round(accuracy, 2), "%")
    
    long_sepal = st.number_input('Longueur Sépal', value=0.)
    larg_sepal = st.number_input('Largeur Sépal', value=0.)
    long_petal = st.number_input('Longueur Pétal', value=0.)
    larg_petal = st.number_input('Largeur Pétal', value=0.)
  
    valeurs = np.array([long_sepal, larg_sepal, long_petal, larg_petal]).reshape(1,-1)
    prediction = knn.predict(valeurs)

    # Résultat
    prédition  = iris_dataset['target_names'][prediction]

    if st.button('Prédire'):
        st.subheader('Résultat de la prédiction')
        for pred in prédition:
            st.success(pred)

if __name__ == "__main__":
    main()
'''from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline("translate_fr_en", lang = "xx") 
pipeline.annotate("Your sentence to translate!")
'''

def translation1(text):
    
    translator = pipeline('translation', 'facebook/m2m100_418M', src_lang='en', tgt_lang="de")
    outputs = translator(text, clean_up_tokenization_spaces = True, max_length = 100000)
    return(outputs[0]["translation_text"])