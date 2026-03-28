import numpy as np 
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_reprot, confusion_matrix, accuracy_score

from preprocessing import chargement_des_donnees, remplacer_les_donnees_manquantes, selection_des_variables, normalisation_des_donnees
