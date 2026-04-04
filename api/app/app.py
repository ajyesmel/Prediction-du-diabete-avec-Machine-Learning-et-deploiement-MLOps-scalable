# ============================================================
# IMPORTS
# os : permet de lire les variables d'environnement (clé API)
# streamlit : framework pour créer l'interface web
# joblib : pour charger le modèle ML sauvegardé (.pkl)
# numpy : pour créer le tableau de données à envoyer au modèle
# pandas : pour créer le tableau de l'historique
# anthropic : SDK officiel pour appeler l'API Claude
# ============================================================
import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import anthropic

# ============================================================
# CONFIGURATION DE LA PAGE
# set_page_config doit toujours être la 1ère commande Streamlit
# page_title : titre dans l'onglet du navigateur
# page_icon : emoji affiché dans l'onglet
# layout="wide" : utilise toute la largeur de l'écran
# ============================================================
st.set_page_config(
    page_title="Prédiction du Diabète — IA",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# CLÉ API ANTHROPIC
# os.getenv lit la variable d'environnement ANTHROPIC_API_KEY
# que tu configures sur Render (jamais dans le code directement)
# ============================================================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ============================================================
# CHARGEMENT DU MODÈLE ML
# @st.cache_resource : charge le modèle UNE SEULE FOIS
# et le garde en mémoire pour toutes les requêtes suivantes.
# Sans ça, le modèle se rechargerait à chaque interaction,
# ce qui serait très lent.
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")    # Le modèle RandomForest entraîné
    scaler = joblib.load("scalers/scaler.pkl") # Le scaler pour normaliser les données
    return model, scaler

model, scaler = load_model()

# ============================================================
# INITIALISATION DE L'HISTORIQUE
# st.session_state est comme une mémoire locale à la session
# de l'utilisateur. Si "historique" n'existe pas encore,
# on crée une liste vide pour la stocker.
# Sans session_state, tout serait perdu à chaque interaction.
# ============================================================
if "historique" not in st.session_state:
    st.session_state.historique = []

# ============================================================
# FONCTION D'INTERPRÉTATION PAR VARIABLE
# Prend chaque valeur et retourne un commentaire coloré.
# Les seuils sont basés sur les normes médicales internationales.
# Retourne une liste de strings formatés en Markdown.
# ============================================================
def interpreter_variables(glucose, bmi, age, blood_pressure, insulin, pregnancies, skin_thickness, dpf):
    interpretations = []

    # --- Glycémie ---
    # Norme : < 100 mg/dL à jeun
    if glucose < 70:
        interpretations.append("🟡 **Glycémie** : Trop basse — hypoglycémie possible")
    elif glucose <= 99:
        interpretations.append("🟢 **Glycémie** : Normale (< 100 mg/dL)")
    elif glucose <= 125:
        interpretations.append("🟠 **Glycémie** : Zone prédiabète (100–125 mg/dL)")
    else:
        interpretations.append("🔴 **Glycémie** : Élevée (> 125 mg/dL) — risque de diabète")

    # --- IMC ---
    # Norme OMS : 18.5 à 24.9 kg/m²
    if bmi < 18.5:
        interpretations.append("🟡 **IMC** : Insuffisance pondérale (< 18.5)")
    elif bmi <= 24.9:
        interpretations.append("🟢 **IMC** : Poids normal (18.5–24.9)")
    elif bmi <= 29.9:
        interpretations.append("🟠 **IMC** : Surpoids (25–29.9) — vigilance recommandée")
    else:
        interpretations.append("🔴 **IMC** : Obésité (> 30) — facteur de risque majeur")

    # --- Pression artérielle ---
    # Valeur diastolique — norme : 60 à 80 mm Hg
    if blood_pressure < 60:
        interpretations.append("🟡 **Pression artérielle** : Trop basse (hypotension)")
    elif blood_pressure <= 80:
        interpretations.append("🟢 **Pression artérielle** : Normale (60–80 mm Hg)")
    else:
        interpretations.append("🔴 **Pression artérielle** : Élevée — risque d'hypertension")

    # --- Insuline ---
    # 0 signifie souvent une donnée manquante dans ce dataset
    if insulin == 0:
        interpretations.append("⚪ **Insuline** : Donnée manquante ou non mesurée")
    elif insulin <= 166:
        interpretations.append("🟢 **Insuline** : Dans la plage normale (≤ 166 μU/mL)")
    else:
        interpretations.append("🔴 **Insuline** : Élevée — possible résistance à l'insuline")

    # --- Âge ---
    # Le risque de diabète de type 2 augmente avec l'âge
    if age < 30:
        interpretations.append("🟢 **Âge** : Jeune (< 30 ans) — risque naturellement plus faible")
    elif age <= 45:
        interpretations.append("🟠 **Âge** : Tranche intermédiaire (30–45 ans)")
    else:
        interpretations.append("🔴 **Âge** : Âge avancé (> 45 ans) — facteur de risque accru")

    # --- Antécédents familiaux (DPF) ---
    # Diabetes Pedigree Function : mesure la prédisposition génétique
    if dpf > 1.0:
        interpretations.append("🔴 **Antécédents familiaux (DPF)** : Forte prédisposition génétique")
    elif dpf > 0.5:
        interpretations.append("🟠 **Antécédents familiaux (DPF)** : Prédisposition modérée")
    else:
        interpretations.append("🟢 **Antécédents familiaux (DPF)** : Faible prédisposition génétique")

    return interpretations


# ============================================================
# FONCTION D'EXPLICATION PAR CLAUDE (LLM)
# Cette fonction :
# 1. Crée un client Anthropic avec ta clé API
# 2. Construit un prompt médical avec toutes les données patient
# 3. Envoie le prompt à Claude via l'API
# 4. Retourne le texte généré par Claude
#
# Le prompt est en français et demande à Claude d'agir comme
# un assistant médical — il analyse les données et génère
# une explication personnalisée, compréhensible, bienveillante.
# ============================================================
def generer_explication_llm(
    glucose, bmi, age, blood_pressure, insulin,
    pregnancies, skin_thickness, dpf,
    prediction, probabilite
):
    # Création du client Anthropic avec la clé API
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Traduction du résultat numérique en texte lisible
    resultat_texte = "RISQUE ÉLEVÉ DE DIABÈTE" if prediction == 1 else "FAIBLE RISQUE DE DIABÈTE"

    # Construction du prompt envoyé à Claude
    # Les f-strings insèrent les vraies valeurs du patient
    prompt = f"""Tu es un assistant médical spécialisé en diabétologie et en médecine préventive.
Un modèle de Machine Learning (Random Forest) vient d'analyser les données d'un patient.

Voici les données complètes du patient :
- Nombre de grossesses : {pregnancies}
- Glycémie : {glucose} mg/dL
- Pression artérielle (diastolique) : {blood_pressure} mm Hg
- Épaisseur du pli cutané tricipital : {skin_thickness} mm
- Insuline sérique : {insulin} μU/mL
- IMC (Indice de Masse Corporelle) : {bmi} kg/m²
- Antécédents familiaux — Diabetes Pedigree Function : {dpf}
- Âge : {age} ans

Résultat du modèle ML : {resultat_texte}
Probabilité estimée de diabète : {probabilite:.1f}%

Ta mission est de générer une analyse structurée en 4 parties :

1. **Synthèse du résultat** (2-3 phrases) : explique simplement ce que signifie ce résultat pour ce patient.

2. **Facteurs les plus préoccupants** : identifie les 2 ou 3 variables qui contribuent le plus au risque, avec une explication claire pour chacune.

3. **Recommandations personnalisées** : propose 3 actions concrètes et réalistes adaptées à ce profil spécifique (alimentation, activité physique, suivi médical).

4. **Message de conclusion** : une phrase encourageante et bienveillante.

Règles importantes :
- Utilise un langage accessible, pas de jargon médical excessif
- Sois précis mais humain
- Termine toujours par ce rappel : "⚕️ Cette analyse est un outil d'aide à la décision et ne remplace pas une consultation médicale."
- Réponds uniquement en français
"""

    # Appel à l'API Claude
    # model : version de Claude à utiliser
    # max_tokens : longueur maximale de la réponse (environ 450 mots)
    # messages : liste des messages — ici un seul message utilisateur
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=800,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extraction du texte depuis la réponse de l'API
    # message.content est une liste de blocs — on prend le premier bloc texte
    return message.content[0].text


# ============================================================
# INTERFACE PRINCIPALE — EN-TÊTE
# st.title : titre principal H1
# st.markdown : texte formaté en Markdown
# st.divider : ligne horizontale de séparation
# ============================================================
st.title("🩺 Prédiction du Diabète avec l'IA")
st.markdown(
    "Entrez les données médicales du patient pour obtenir "
    "une prédiction instantanée accompagnée d'une analyse par intelligence artificielle."
)
st.divider()

# ============================================================
# FORMULAIRE DE SAISIE
# st.form : regroupe tous les champs — la prédiction ne se
# déclenche que quand l'utilisateur clique sur "Prédire".
# Sans st.form, Streamlit recalcule à chaque modification
# d'un champ, ce qui serait lent et coûteux (appels API).
#
# st.columns(2) : crée 2 colonnes côte à côte
# st.number_input : champ numérique avec min, max et valeur par défaut
# ============================================================
with st.form("formulaire_prediction"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "🤰 Nombre de grossesses",
            min_value=0, max_value=20, value=1,
            help="Nombre total de grossesses"
        )
        glucose = st.number_input(
            "🍬 Glycémie (mg/dL)",
            min_value=0, max_value=300, value=120,
            help="Concentration de glucose plasmatique à 2h lors d'un test oral de tolérance au glucose"
        )
        blood_pressure = st.number_input(
            "💓 Pression artérielle diastolique (mm Hg)",
            min_value=0, max_value=200, value=70,
            help="Pression artérielle diastolique"
        )
        skin_thickness = st.number_input(
            "📏 Épaisseur du pli cutané (mm)",
            min_value=0, max_value=100, value=20,
            help="Épaisseur du pli cutané tricipital"
        )

    with col2:
        insulin = st.number_input(
            "💉 Insuline sérique (μU/mL)",
            min_value=0, max_value=900, value=80,
            help="Insuline sérique à 2 heures"
        )
        bmi = st.number_input(
            "⚖️ IMC (kg/m²)",
            min_value=0.0, max_value=70.0, value=25.0, format="%.1f",
            help="Indice de Masse Corporelle = poids / taille²"
        )
        dpf = st.number_input(
            "🧬 Antécédents familiaux (DPF)",
            min_value=0.0, max_value=3.0, value=0.5, format="%.3f",
            help="Diabetes Pedigree Function — mesure la prédisposition génétique au diabète"
        )
        age = st.number_input(
            "🎂 Âge (années)",
            min_value=1, max_value=120, value=30
        )

    # Bouton de soumission — use_container_width : prend toute la largeur
    submitted = st.form_submit_button("🔍 Lancer la prédiction", use_container_width=True)


# ============================================================
# TRAITEMENT APRÈS SOUMISSION DU FORMULAIRE
# Ce bloc ne s'exécute que si l'utilisateur a cliqué sur
# "Lancer la prédiction" (submitted == True)
# ============================================================
if submitted:

    # ── Préparation des données ────────────────────────────
    # np.array crée un tableau numpy avec les 8 valeurs
    # Le double crochet [[...]] crée une matrice 1 ligne x 8 colonnes
    # car scikit-learn attend une matrice, pas un vecteur simple
    input_data = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]])

    # Le scaler normalise les données sur la même échelle
    # que les données d'entraînement (très important pour la précision)
    input_scaled = scaler.transform(input_data)

    # Prédiction : 0 = pas de diabète, 1 = diabète
    prediction = model.predict(input_scaled)[0]

    # predict_proba retourne [proba_classe_0, proba_classe_1]
    # proba[0] = probabilité de ne PAS avoir le diabète
    # proba[1] = probabilité d'AVOIR le diabète
    proba = model.predict_proba(input_scaled)[0]

    st.divider()

    # ── Affichage du résultat ML ───────────────────────────
    # On utilise 3 colonnes pour un affichage équilibré
    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        if prediction == 1:
            st.error("⚠️ **Risque élevé de diabète**")
        else:
            st.success("✅ **Faible risque de diabète**")

    with col_res2:
        # st.metric affiche une métrique avec un titre et une valeur
        st.metric("Probabilité de diabète", f"{proba[1]*100:.1f}%")

    with col_res3:
        st.metric("Probabilité sans diabète", f"{proba[0]*100:.1f}%")

    st.divider()

    # ── Interprétation par variable ────────────────────────
    # On appelle notre fonction et on affiche chaque ligne
    st.subheader("📊 Analyse de vos indicateurs")
    interpretations = interpreter_variables(
        glucose, bmi, age, blood_pressure,
        insulin, pregnancies, skin_thickness, dpf
    )
    for msg in interpretations:
        st.markdown(f"- {msg}")

    st.divider()

    # ── Explication par Claude ─────────────────────────────
    # st.spinner affiche un indicateur de chargement
    # pendant l'appel à l'API Claude (qui prend 3-5 secondes)
    st.subheader("🤖 Analyse approfondie par Intelligence Artificielle")

    if not ANTHROPIC_API_KEY:
        # Si la clé API n'est pas configurée, on affiche un avertissement
        st.warning("⚠️ Clé API Anthropic non configurée. Ajoutez ANTHROPIC_API_KEY dans les variables d'environnement Render.")
    else:
        with st.spinner("Claude analyse votre profil médical..."):
            try:
                explication = generer_explication_llm(
                    glucose, bmi, age, blood_pressure, insulin,
                    pregnancies, skin_thickness, dpf,
                    prediction, proba[1] * 100
                )
                # st.markdown pour afficher le texte formaté (gras, listes, etc.)
                st.markdown(explication)
            except Exception as e:
                # En cas d'erreur API (quota, réseau...), on affiche le message
                st.error(f"Erreur lors de l'appel à Claude : {str(e)}")

    st.divider()

    # ── Ajout à l'historique ───────────────────────────────
    # On ajoute un dictionnaire (une ligne) à la liste historique
    # pd.Timestamp.now() : date et heure actuelles
    st.session_state.historique.append({
        "Date/Heure": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "Glycémie": glucose,
        "IMC": bmi,
        "Âge": age,
        "Pression": blood_pressure,
        "Insuline": insulin,
        "Grossesses": pregnancies,
        "DPF": dpf,
        "Résultat": "⚠️ Risque élevé" if prediction == 1 else "✅ Faible risque",
        "Prob. diabète": f"{proba[1]*100:.1f}%"
    })


# ============================================================
# SECTION HISTORIQUE
# Affichée en dehors du bloc "if submitted" pour être
# toujours visible, même après une nouvelle prédiction.
# ============================================================
if st.session_state.historique:
    st.divider()
    st.subheader("📋 Historique des prédictions de la session")

    # Conversion de la liste de dicts en DataFrame pandas
    df_historique = pd.DataFrame(st.session_state.historique)

    # st.dataframe : tableau interactif (triable, scrollable)
    # use_container_width : prend toute la largeur disponible
    st.dataframe(df_historique, use_container_width=True)

    # Bouton pour télécharger l'historique en CSV
    # .encode("utf-8") : nécessaire pour les accents français
    csv = df_historique.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Télécharger l'historique (CSV)",
        data=csv,
        file_name="historique_predictions_diabete.csv",
        mime="text/csv"
    )

    # Bouton pour vider l'historique de la session
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.historique = []
        # st.rerun() : force Streamlit à recharger la page
        st.rerun()