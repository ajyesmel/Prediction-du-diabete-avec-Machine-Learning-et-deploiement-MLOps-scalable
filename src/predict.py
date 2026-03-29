import numpy as np 
import pandas as pd
import joblib 

#Charger le modèle et le scaler 
model = joblib.load("models/model.pkl")
scaler = joblib.load("scalers/scaler.pkl")

# Fonction de prédiction
def predict(data): 
    # Convertir les données d'entrée en dataframe
    data = pd.DataFrame(data, index=[0])
    # Normaliser les données d'entrée 
    data_scaled = scaler.transform(data)
    # Prédire le résultat
    prediction = model.predict(data_scaled)
    return prediction[0]

    