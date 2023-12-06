import streamlit as st
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 
import pandas as pd


st.title("Problème 1 : Prédiction du Prix des Maisons")

#Charegement du model dans la variable model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

#Chargement du données de test
with open("data.pkl", "rb") as file:
    data = pickle.load(file)



X_test = data[0]
y_test = data[1]


#Vérification
y_test_predict = model.predict(X_test)


rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
# score R carré du modèle
r2 = r2_score(y_test, y_test_predict)
print("La performance du Modèle pour le set de Test")
print("--------------------------------------------")
print("l'erreur RMSE est {}".format(rmse))
print('le score R2 score est {}'.format(r2))
    
#Création des champs LSTAT et  RM
Lstat = st.number_input('LSTAT', value=0.)
rm = st.number_input('RM', value=0.)

#Chargement du données de test
with open("scale.pkl", "rb") as file:
    scale= pickle.load(file)

if(st.button("Prédir")):
    x_input = np.array([Lstat, rm]).reshape(1, -1)
    x_input = scale.transform(x_input)
    result = abs(float(model.predict(x_input)))
    print(result)
    #st.success(result)
    st.write("Le prix de cette maison est : ", result, " dollars")



  