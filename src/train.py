import numpy as np 
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocessing import chargement_des_donnees, remplacer_les_donnees_manquantes, selection_des_variables, normalisation_des_donnees

def entrainer():
    """
    Pipeline complet d'entraînement
    """
    
    # 1. Chargement
    df = chargement_des_donnees("data/diabetes.csv")
    
    # 2. Nettoyage
    df = remplacer_les_donnees_manquantes(df)
    
    # 3. Split
    X_train, X_test, y_train, y_test = selection_des_variables(df)
    
    # 4. Scaling
    X_train, X_test = normalisation_des_donnees(X_train, X_test)
    
    # 5. Modèle + GridSearch
    params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5]
    }
    
    grid = GridSearchCV(RandomForestClassifier(), params, cv=3)
    
    # 6. Tracking MLflow
    with mlflow.start_run():
        
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        
        # Prédictions
        y_pred = best_model.predict(X_test)
        
        # Log métriques
        mlflow.log_params(grid.best_params_)
        
        # Sauvegarde modèle
        joblib.dump(best_model, "models/model.pkl")
        
        print("Modèle entraîné et sauvegardé ✅")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    entrainer()
