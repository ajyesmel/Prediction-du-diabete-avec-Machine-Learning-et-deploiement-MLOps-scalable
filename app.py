import streamlit as st
import joblib
import numpy as np 
import pandas as pd

# Configurer de a page
st.set_page_config(
    page_title="Prédiction du diabète avec l'IA",
    page_icon="🩺",
    layout="centered"
)

# Charger le modèle et le scaler
@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("scalers/scaler.pkl")
    return model, scaler
model, scaler = load_model()

# Titre de l'application 
st.title("Prédiction du diabète avec l'IA")
st.markdown("Entrez vos données pour avoir une prédiction de facon instantanée")


with st.form("prediction_diabetes"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glycémie (mg/dL)", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Pression artérielle (mm Hg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Épaisseur peau (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insuline (μU/mL)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("IMC (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        dpf = st.number_input("Antécédents familiaux (DPF)", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        age = st.number_input("Âge (années)", min_value=1, max_value=120, value=30)
    
    submitted = st.form_submit_button("🔍 Prédire", use_container_width=True)

if submitted:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    input_scaled = scaler.transform(input_data)  # retire si pas de scaler
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    
    st.divider()
    
    if prediction == 1:
        st.error(f"⚠️ **Risque élevé de diabète détecté**")
        st.metric("Probabilité de diabète", f"{proba[1]*100:.1f}%")
    else:
        st.success(f"✅ **Faible risque de diabète**")
        st.metric("Probabilité de non-diabète", f"{proba[0]*100:.1f}%")
    
    st.info("⚕️ *Cette prédiction est un outil d'aide, pas un diagnostic médical. Consultez un médecin.*")
        
        
