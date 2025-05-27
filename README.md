# RH Dashboard IA

Un projet d’analyse RH intégrant des indicateurs clés, la détection d’anomalies, la prédiction d’absentéisme et de turnover, ainsi qu’une synthèse automatique générée via GPT.

## Objectifs

- Analyser les données RH (effectifs, absences, départs)
- Identifier les anomalies mensuelles (pics d’absences, turnover inhabituel)
- Prédire la probabilité de départ ou d’absentéisme d’un agent
- Générer une synthèse professionnelle des résultats avec GPT

## Fonctionnalités

- Interface interactive via Streamlit
- Indicateurs RH dynamiques (effectif, taux de turnover, absentéisme)
- Détection d’anomalies par Isolation Forest
- Modèles IA avec RandomForest pour les prédictions
- Génération d'une synthèse textuelle automatique
- Séparation claire entre traitement, modèles et interface

## Arborescence du projet

```
rh_dashboard_ia/
│
├── data/
│   ├── rh_data.csv                # Données sources
│   └── synthese_rh.txt            # Résumé généré par GPT
│
├── models/
│   ├── model_turnover.pkl
│   └── model_absenteisme.pkl
│
├── notebooks/                     # Espace pour explorations Jupyter
│
├── dashboard.py                   # Interface Streamlit
├── rh_analysis.py                 # Analyse des données et IA
├── train_models.py                # Génération des modèles
├── requirements.txt               # Dépendances Python
└── README.md                      # Ce fichier
```

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/ton-utilisateur/rh_dashboard_ia.git
cd rh_dashboard_ia
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Générer les modèles (si absents) :

```bash
python train_models.py
```
Le script crée automatiquement le dossier models/ et enregistre deux fichiers : models/model_turnover.pkl & models/model_absenteisme.pkl

4. Lancer l'application Streamlit :
```bash
streamlit run dashboard.py
```
ou 
```bash
python -m streamlit run dashboard.py
```

## Modèles IA

Les modèles sont générés à l'aide de `train_models.py` :
- Modèle de prédiction de turnover
- Modèle de prédiction d’absentéisme

Les modèles sont stockés dans le dossier `models/`.

## Dépendances clés

- pandas, matplotlib, seaborn
- scikit-learn
- joblib
- streamlit
- openai (optionnel, pour la synthèse GPT)

## Licence

Projet libre, distribué à des fins pédagogiques. Licence à définir selon votre politique (MIT, Apache, etc.).
