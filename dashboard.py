# dashboard.py

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os


# Point de départ = dossier courant où se trouve le script Python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Chemins relatifs
DATA_PATH = os.path.join(BASE_DIR,"data", "rh_data.csv")
MODEL_TURNOVER_PATH = os.path.join(BASE_DIR, "models", "model_turnover.pkl")
MODEL_ABS_PATH = os.path.join(BASE_DIR, "models", "model_absenteisme.pkl")
SYNTH_PATH = os.path.join(BASE_DIR,"synthese_rh.txt")

# Chargement du fichier
df = pd.read_csv(DATA_PATH, parse_dates=["Date embauche", "Date absence", "Date départ"])
df["Statut_actuel"] = df["Date départ"].apply(lambda x: "Actif" if pd.isna(x) or x == "Aucune" else "Sorti")
df["Anciennete_mois"] = (pd.Timestamp(datetime.now()) - df["Date embauche"]) / pd.Timedelta(days=30)

# Indicateurs clés
nb_actifs = (df["Statut_actuel"] == "Actif").sum()
taux_turnover = (df["Statut_actuel"] == "Sorti").mean() * 100
taux_absenteisme = (df["Date absence"] != "Aucune").mean() * 100

# Layout
st.set_page_config(page_title="Dashboard RH IA", layout="wide")
st.title("📊 Dashboard RH avec Intelligence Artificielle")

col1, col2, col3 = st.columns(3)
col1.metric("👥 Effectif actuel", nb_actifs)
col2.metric("🔁 Taux de Turnover", f"{taux_turnover:.2f}%")
col3.metric("🚫 Taux d'Absentéisme", f"{taux_absenteisme:.2f}%")

st.markdown("---")

# Prédictions IA
st.subheader("🔮 Prédiction personnalisée")

with st.form("form_predict"):
    anciennete = st.slider("Ancienneté (mois)", 0, 200, 12)
    ug = st.selectbox("Unité de Gestion", df["UG"].unique())
    ghu = st.selectbox("GHU", df["GHU"].unique())
    contrat = st.selectbox("Type de Contrat", df["Contrat"].unique())
    statut = st.selectbox("Statut (Temps plein/partiel)", df["Statut"].unique())
    submit = st.form_submit_button("Prédire")

if submit:
    from sklearn.preprocessing import LabelEncoder

    input_df = pd.DataFrame([{
        "Anciennete_mois": anciennete,
        "UG": ug,
        "GHU": ghu,
        "Contrat": contrat,
        "Statut": statut
    }])

    for col in ["UG", "GHU", "Contrat", "Statut"]:
        le = LabelEncoder()
        le.fit(df[col])
        input_df[col] = le.transform(input_df[col])

    # Chargement des modèles avec sécurité
    if not os.path.exists(MODEL_TURNOVER_PATH) or not os.path.exists(MODEL_ABS_PATH):
        st.error("❌ Un ou plusieurs fichiers modèles sont manquants.")
    else:
        model_turnover = joblib.load(MODEL_TURNOVER_PATH)
        model_abs = joblib.load(MODEL_ABS_PATH)

        # Prédiction avec gestion des cas à une seule classe
        probas_turnover = model_turnover.predict_proba(input_df)[0]
        proba_turnover = probas_turnover[1] if len(probas_turnover) > 1 else 0.0

        probas_abs = model_abs.predict_proba(input_df)[0]
        proba_abs = probas_abs[1] if len(probas_abs) > 1 else 0.0

        # Affichage
        st.success(f"🧠 Probabilité de départ : **{proba_turnover * 100:.1f}%**")
        st.info(f"🧠 Probabilité d'absentéisme : **{proba_abs * 100:.1f}%**")

        if len(probas_turnover) == 1:
            st.warning("⚠️ Le modèle Turnover a été entraîné avec une seule classe (prédiction limitée).")
        if len(probas_abs) == 1:
            st.warning("⚠️ Le modèle Absentéisme a été entraîné avec une seule classe (prédiction limitée).")

    
# Synthèse GPT
if os.path.exists(SYNTH_PATH):
    with open(SYNTH_PATH, "r", encoding="utf-8") as f:
        synthese = f.read()
else:
    synthese = "Synthèse non encore générée."
# Affichage de la synthèse
st.subheader("📝 Synthèse GPT (dernière génération)")
st.text_area("Synthèse mensuelle RH", synthese, height=250)
