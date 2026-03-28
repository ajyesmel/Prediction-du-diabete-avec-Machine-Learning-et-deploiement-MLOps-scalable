
import numpy as np
import pandas as pd

# Remplacer les valeurs manquantes par NaN
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols] = data[cols].replace(0, np.NaN)

# Remplacer les valeurs manquantes par la mediane de chaque colonne
data.fillna(data.median(), inplace=True)

# Separation des variables eexplicatives et les variables cibles
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Diviser des données en ensembes d'entrainements et de tests
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Normaliser les données 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)