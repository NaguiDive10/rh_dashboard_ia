# train_models.py

import pandas as pd
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Répertoire racine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fichiers
DATA_PATH = os.path.join(BASE_DIR, "rh_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_TURNOVER_PATH = os.path.join(MODEL_DIR, "model_turnover.pkl")
MODEL_ABS_PATH = os.path.join(MODEL_DIR, "model_absenteisme.pkl")

# Création du dossier modèles si nécessaire
os.makedirs(MODEL_DIR, exist_ok=True)

# Chargement des données
df = pd.read_csv(DATA_PATH, parse_dates=["Date embauche", "Date absence", "Date départ"])
df["Statut_actuel"] = df["Date départ"].apply(lambda x: "Actif" if pd.isna(x) or x == "Aucune" else "Sorti")
df["Anciennete_mois"] = (pd.Timestamp(datetime.now()) - df["Date embauche"]) / pd.Timedelta(days=30)

# Préparation
features = ["Anciennete_mois", "UG", "GHU", "Contrat", "Statut"]
df_model = df.copy()

# Encodage
for col in ["UG", "GHU", "Contrat", "Statut"]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])

# 🎯 Modèle Turnover
df_model["Target_Turnover"] = df_model["Statut_actuel"].apply(lambda x: 1 if x == "Sorti" else 0)
X = df_model[features]
y_turnover = df_model["Target_Turnover"]
X_train, X_test, y_train, y_test = train_test_split(X, y_turnover, test_size=0.2, random_state=42)

model_turnover = RandomForestClassifier(n_estimators=100, random_state=42)
model_turnover.fit(X_train, y_train)
joblib.dump(model_turnover, MODEL_TURNOVER_PATH)
print(f"✅ Modèle Turnover enregistré : {MODEL_TURNOVER_PATH}")

# 🎯 Modèle Absentéisme
df_model["Target_Abs"] = df["Date absence"].apply(lambda x: 0 if x == "Aucune" else 1)
y_abs = df_model["Target_Abs"]
X_train_abs, X_test_abs, y_train_abs, y_test_abs = train_test_split(X, y_abs, test_size=0.2, random_state=42)

model_abs = RandomForestClassifier(n_estimators=100, random_state=42)
model_abs.fit(X_train_abs, y_train_abs)
joblib.dump(model_abs, MODEL_ABS_PATH)
print(f"✅ Modèle Absentéisme enregistré : {MODEL_ABS_PATH}")
