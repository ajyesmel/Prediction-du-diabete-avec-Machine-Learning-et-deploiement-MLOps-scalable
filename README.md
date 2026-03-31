# 🧠 Prédiction du diabète avec Machine Learning & MLOps

## 📌 Description du projet

Ce projet vise à développer un système intelligent capable de **prédire
le risque de diabète chez un patient** à partir de données médicales.

Il s'inscrit dans une démarche de **Data Science appliquée à la santé**,
avec une architecture **MLOps scalable et industrialisable**.

------------------------------------------------------------------------

## 🎯 Objectifs

-   Détecter précocement les patients à risque
-   Aider à la prise de décision médicale
-   Réduire les coûts hospitaliers liés aux complications
-   Mettre en place un pipeline ML reproductible

------------------------------------------------------------------------

## 🏥 Contexte métier

Le diabète est une maladie chronique qui peut entraîner :

-   AVC
-   Insuffisance rénale
-   Amputations

👉 Une détection précoce permet de sauver des vies.

------------------------------------------------------------------------

## 📊 Dataset

-   Source : UCI Machine Learning Repository
-   Nombre d'observations : 768
-   Variables principales :
    -   Glucose
    -   BMI
    -   Age
    -   Insulin
    -   BloodPressure

⚠️ Limites : - Dataset de petite taille - Données bruitées (valeurs 0) -
Population spécifique

------------------------------------------------------------------------

## ⚙️ Architecture du projet

    project/
    │
    ├── data/
    ├── notebooks/
    ├── src/
    │   ├── preprocessing.py
    │   ├── train.py
    │   ├── predict.py
    │
    ├── models/
    ├── api/
    ├── docker/
    ├── mlruns/
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🧪 Pipeline Machine Learning

### 1. Préprocessing

-   Nettoyage des données
-   Gestion des valeurs manquantes
-   Standardisation

### 2. Modélisation

-   Random Forest (optimisé via GridSearch)
-   Comparaison avec d'autres modèles

### 3. Évaluation

-   Accuracy
-   Precision
-   Recall (prioritaire en santé)
-   F1-score

------------------------------------------------------------------------

## 🔍 Interprétabilité

Utilisation de SHAP pour : - Identifier les variables importantes -
Expliquer les prédictions - Apporter de la transparence au modèle

------------------------------------------------------------------------

## ⚙️ MLOps

-   Tracking des expériences avec MLflow
-   Versioning des modèles
-   Pipeline reproductible

------------------------------------------------------------------------

## 🚀 Installation

``` bash
conda create -n diabetes_ml python=3.10
conda activate diabetes_ml
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🏋️ Entraînement du modèle

``` bash
python src/train.py
```

------------------------------------------------------------------------

## 🌐 Lancer l'API

``` bash
python -m uvicorn api.main:app --reload
```

------------------------------------------------------------------------

## 📡 Endpoints API

### 🔹 Prédiction

POST /predict

``` json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 85,
  "BMI": 30,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 45
}
```

------------------------------------------------------------------------

## 🐳 Docker

``` bash
docker build -t diabetes-api .
docker run -p 8000:8000 diabetes-api
```

------------------------------------------------------------------------

## ☁️ Déploiement

-   AWS SageMaker
-   Azure ML
-   Docker / Kubernetes

------------------------------------------------------------------------

## 📈 Monitoring

-   Suivi des performances du modèle
-   Détection du data drift
-   Alertes en cas de dégradation

------------------------------------------------------------------------

## ⚠️ Limites

-   Dataset limité
-   Non validé cliniquement
-   Ne remplace pas un diagnostic médical

------------------------------------------------------------------------

## 👩‍💻 Auteur

Esmel Amary Jean-Yves\
Data Scientist \| Machine Learning \| IA Médicale

------------------------------------------------------------------------

## ⭐ Si ce projet t'a aidé

N'hésite pas à laisser une ⭐ sur GitHub !
