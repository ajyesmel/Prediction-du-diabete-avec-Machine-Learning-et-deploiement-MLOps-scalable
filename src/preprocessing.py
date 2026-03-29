
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib 


def chargement_des_donnees (path):
    data = pd.read_csv(path)
    return data

# Remplacer les valeurs manquantes par NaN
def remplacer_les_donnees_manquantes(data): 
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[cols] = data[cols].replace(0, np.NaN)

# Remplacer les valeurs manquantes par la mediane de chaque colonne
    data.fillna(data.median(), inplace=True)
    return data 


# Separation des variables eexplicatives et les variables cibles
def selection_des_variables(data):
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    return X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# Normaliser les données 
def normalisation_des_donnees(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test) 
    joblib.dump(scaler, "scalers/scaler.pkl")
    
    return X_train_scaled, X_test_scaled