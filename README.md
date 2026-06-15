# 🩺 Diabetes Risk Prediction — ML & MLOps Pipeline

> **Système intelligent de détection précoce du diabète, construit avec une architecture MLOps industrialisable.**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=flat)](https://shap.readthedocs.io/)

---

## 📋 Table des matières

- [Contexte métier](#-contexte-métier)
- [Stack technologique](#-stack-technologique)
- [Dataset](#-dataset)
- [Architecture du projet](#-architecture-du-projet)
- [Pipeline ML](#-pipeline-machine-learning)
- [MLOps](#-mlops)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [API Reference](#-api-reference)
- [Déploiement](#-déploiement)
- [Limites & avertissements](#-limites--avertissements)
- [Auteur](#-auteur)

---

## 🏥 Contexte métier

Le diabète est une maladie chronique mondiale dont les complications non détectées à temps peuvent mener à des AVC, une insuffisance rénale ou des amputations. **Une détection précoce permet de sauver des vies et de réduire significativement les coûts hospitaliers.**

Ce projet fournit un modèle de prédiction du risque diabétique à partir de données médicales, conçu pour **assister** (et non remplacer) la prise de décision clinique.

**Objectifs :**
- Détecter précocement les patients à risque
- Aider à la prise de décision médicale
- Proposer un pipeline ML reproductible et scalable
- Garantir la transparence des prédictions via l'explicabilité

---

## 🛠 Stack technologique

| Catégorie | Technologie | Usage |
|---|---|---|
| **Langage** | Python 3.10 | Développement principal |
| **ML / Modélisation** | scikit-learn, Random Forest | Entraînement & évaluation |
| **Explicabilité** | SHAP | Interprétation des prédictions |
| **Tracking ML** | MLflow | Suivi des expériences & versioning |
| **API** | FastAPI + Uvicorn | Service de prédiction REST |
| **Conteneurisation** | Docker | Packaging & déploiement |
| **Orchestration** | Kubernetes | Scalabilité en production |
| **Cloud** | AWS SageMaker / Azure ML | Déploiement managé |
| **Monitoring** | MLflow + Evidently | Suivi des performances & data drift |
| **Environnement** | Conda | Gestion des dépendances |

---

## 📊 Dataset

**Source :** [UCI Machine Learning Repository — Pima Indians Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes)

| Propriété | Valeur |
|---|---|
| Observations | 768 |
| Variables | 8 features + 1 cible |
| Cible | `Outcome` (0 = Non-diabétique, 1 = Diabétique) |

**Features principales :**

| Feature | Description |
|---|---|
| `Glucose` | Concentration plasmatique en glucose |
| `BMI` | Indice de masse corporelle |
| `Age` | Âge du patient |
| `Insulin` | Insuline sérique (2h) |
| `BloodPressure` | Pression artérielle diastolique |
| `Pregnancies` | Nombre de grossesses |
| `SkinThickness` | Épaisseur du pli cutané |
| `DiabetesPedigreeFunction` | Antécédents familiaux |

> ⚠️ **Limites du dataset :** taille réduite, valeurs physiologiquement impossibles encodées à `0`, population spécifique (femmes Pima d'au moins 21 ans). Un préprocessing rigoureux est appliqué pour atténuer ces biais.

---

## 🗂 Architecture du projet

```
diabetes-prediction/
│
├── data/                     # Données brutes et prétraitées
├── notebooks/                # Exploration & analyse (EDA)
│
├── src/
│   ├── preprocessing.py      # Nettoyage, imputation, normalisation
│   ├── train.py              # Entraînement & optimisation hyperparamètres
│   └── predict.py            # Inférence
│
├── models/                   # Modèles sérialisés (.pkl)
├── api/
│   └── main.py               # API FastAPI
│
├── docker/
│   └── Dockerfile
│
├── mlruns/                   # Artefacts MLflow (expériences & modèles)
├── requirements.txt
└── README.md
```

---

## 🧪 Pipeline Machine Learning

### 1. Préprocessing

- Remplacement des valeurs `0` incohérentes par `NaN`
- Imputation par la médiane (features numériques)
- Standardisation via `StandardScaler`
- Séparation `train / test` stratifiée

### 2. Modélisation

Le modèle principal est un **Random Forest Classifier**, optimisé via `GridSearchCV` avec validation croisée k-fold.

Des modèles de comparaison sont également entraînés : Logistic Regression, SVM, Gradient Boosting.

### 3. Évaluation

| Métrique | Description |
|---|---|
| **Recall** ⭐ | Prioritaire en santé — minimiser les faux négatifs |
| **Precision** | Réduire les faux positifs |
| **F1-Score** | Équilibre précision / recall |
| **Accuracy** | Vue globale |
| **AUC-ROC** | Performance discriminative |

### 4. Explicabilité avec SHAP

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

SHAP permet d'identifier les features les plus influentes par prédiction et d'apporter une transparence indispensable dans un contexte médical.

---

## ⚙️ MLOps

### Tracking avec MLflow

Chaque expérience est automatiquement loguée :

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({"recall": recall, "f1": f1})
    mlflow.sklearn.log_model(model, "random_forest")
```

Accès à l'interface MLflow :

```bash
mlflow ui
# → http://localhost:5000
```

**Fonctionnalités MLOps couvertes :**
- Tracking des hyperparamètres et métriques
- Versioning et registre de modèles
- Pipeline reproductible end-to-end
- Monitoring des performances & détection du data drift

---

## 🚀 Installation

**Prérequis :** Python 3.10, Conda, Docker (optionnel)

```bash
# 1. Cloner le dépôt
git clone https://github.com/username/diabetes-prediction.git
cd diabetes-prediction

# 2. Créer et activer l'environnement Conda
conda create -n diabetes_ml python=3.10
conda activate diabetes_ml

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🧑‍💻 Utilisation

### Entraîner le modèle

```bash
python src/train.py
```

### Générer une prédiction

```bash
python src/predict.py
```

### Lancer l'API localement

```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

### Docker

```bash
# Build
docker build -t diabetes-api .

# Run
docker run -p 8000:8000 diabetes-api
```

---

## 📡 API Reference

### `POST /predict`

Retourne la probabilité de diabète pour un patient.

**Body (JSON) :**

```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 85,
  "BMI": 30.0,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 45
}
```

**Réponse :**

```json
{
  "prediction": 1,
  "probability": 0.73,
  "risk_level": "High"
}
```

Documentation interactive disponible sur `/docs` (Swagger UI) et `/redoc`.

---

## ☁️ Déploiement

| Plateforme | Description |
|---|---|
| **AWS SageMaker** | Entraînement & inférence managés |
| **Azure ML** | Pipeline cloud Microsoft |
| **Docker + Kubernetes** | Déploiement containerisé scalable |

---

## ⚠️ Limites & avertissements

> **Ce système est un outil d'aide à la décision et ne constitue en aucun cas un diagnostic médical.**

- Dataset de petite taille (768 observations)
- Population d'entraînement spécifique (non généralisable universellement)
- Non validé cliniquement
- Toute décision médicale doit impliquer un professionnel de santé qualifié

---

## 👨‍💻 Auteur

**Esmel Amary Jean-Yves**
Data Scientist | Machine Learning | IA Médicale

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com)

---

*Si ce projet t'a été utile, n'hésite pas à lui laisser une ⭐ sur GitHub — ça aide à le faire connaître !*
