#api/main.py
from fastapi import FastAPI
import joblib
import numpy as np 
import pandas as pd 

app = FastAPI(title="API DE PREDICTION DU DIABETE")

# Charger le modèle et le scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("scalers/scaler.pkl")

@app.post("/predict")
def predict(data: dict):
    # Convertir les données d'entrée en dataframe
    data = list(data.values())
    # Normaliser les données d'entrée 
    data_scaled = scaler.transform([data])
    # Prédire le résultat
    prediction = model.predict([data_scaled])
    return {"prediction": int(prediction[0])}   

@app.post("/explication")
def explication(data: dict):
    
    data = list(data.values())
    data_scaled = scaler.transform([data])
    
    explainer = shap.Explainer(model)
    shap_values = explainer(scaled)
    
    return {"shap_values": shap_values.values.tolist()}
    
