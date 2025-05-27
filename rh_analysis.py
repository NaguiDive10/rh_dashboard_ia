# rh_dashboard_ia - Indicateurs RH + Anomalies + Prédiction IA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Chargement du jeu de données
df = pd.read_csv("data/rh_data.csv", parse_dates=["Date embauche", "Date absence", "Date départ"])

# Remplissage des valeurs manquantes
missing_cols = ["Date absence", "Motif absence", "Date départ"]
for col in missing_cols:
    df[col] = df[col].fillna("Aucune")

# Colonnes enrichies
now = pd.Timestamp(datetime.now())
df["Anciennete_mois"] = (now - df["Date embauche"]) / pd.Timedelta(days=30)
df["Statut_actuel"] = df["Date départ"].apply(lambda x: "Actif" if x == "Aucune" else "Sorti")
df["Annee_entree"] = df["Date embauche"].dt.year

# Indicateurs RH
nb_actifs = (df["Statut_actuel"] == "Actif").sum()
taux_turnover = (df["Statut_actuel"] == "Sorti").mean() * 100
taux_absenteisme = (df["Date absence"] != "Aucune").mean() * 100
print(f"Effectif actuel : {nb_actifs}")
print(f"Taux de turnover : {taux_turnover:.2f}%")
print(f"Taux d'absentéisme : {taux_absenteisme:.2f}%")

# --- DÉTECTION D'ANOMALIES ---
df_abs = df[df["Date absence"] != "Aucune"].copy()
df_abs["Mois_absence"] = pd.to_datetime(df_abs["Date absence"]).dt.to_period("M").astype(str)
df_depart = df[df["Date départ"] != "Aucune"].copy()
df_depart["Mois_depart"] = pd.to_datetime(df_depart["Date départ"]).dt.to_period("M").astype(str)
abs_monthly = df_abs.groupby("Mois_absence").size().reset_index(name="Nb_absences")
depart_monthly = df_depart.groupby("Mois_depart").size().reset_index(name="Nb_depart")
stats_mensuelles = pd.merge(abs_monthly, depart_monthly, left_on="Mois_absence", right_on="Mois_depart", how="outer")
stats_mensuelles.fillna(0, inplace=True)
stats_mensuelles["Mois"] = stats_mensuelles["Mois_absence"].combine_first(stats_mensuelles["Mois_depart"])
stats_mensuelles = stats_mensuelles[["Mois", "Nb_absences", "Nb_depart"]].sort_values("Mois")
iso = IsolationForest(contamination=0.1, random_state=42)
anom_features = stats_mensuelles[["Nb_absences", "Nb_depart"]]
stats_mensuelles["Anomalie"] = iso.fit_predict(anom_features)

# Visualisation anomalies
plt.figure(figsize=(10, 5))
plt.plot(stats_mensuelles["Mois"], stats_mensuelles["Nb_absences"], label="Absences")
plt.plot(stats_mensuelles["Mois"], stats_mensuelles["Nb_depart"], label="Départs")
anomalies = stats_mensuelles[stats_mensuelles["Anomalie"] == -1]
plt.scatter(anomalies["Mois"], anomalies["Nb_absences"], color="red", label="Anomalie détectée", zorder=5)
plt.xticks(rotation=45)
plt.title("Anomalies dans les absences et départs")
plt.legend()
plt.tight_layout()
plt.show()

# Sauvegarde des anomalies
stats_mensuelles.to_csv("data/stats_mensuelles_anomalies.csv", index=False)

# --- PRÉDICTION DE DÉPART (Turnover) ---

# Préparation des features
features = ["Anciennete_mois", "UG", "GHU", "Contrat", "Statut"]
df_model = df.copy()

# Encodage des variables catégorielles
for col in ["UG", "GHU", "Contrat", "Statut"]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])

# Cible : 1 si "Sorti", 0 sinon
df_model["Target"] = df_model["Statut_actuel"].apply(lambda x: 1 if x == "Sorti" else 0)

X = df_model[features]
y = df_model["Target"]

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
print("\nÉvaluation du modèle de prédiction de départs :")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Sauvegarde du modèle
joblib.dump(model, "models/model_turnover.pkl")
print("\nModèle sauvegardé : models/model_turnover.pkl")