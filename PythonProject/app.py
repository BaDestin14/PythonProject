# 2. Application Streamlit (`app.py`)

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Charger le modèle et les encodeurs
model = joblib.load('churn_model.joblib')
df_encoded = joblib.load('df_encoded.joblib')

# Configuration de l'application
st.title('🔮 Prédiction de Churn Expresso')
st.subheader("Prédisez la probabilité de désabonnement d'un client")

# Définition des caractéristiques
features = {
    'region': ['DAKAR', 'THIES', 'SAINT-LOUIS', 'AUTRE'],
    'tenure': (0, 2000),
    'montant': (0, 50000),
    'frequence_rech': (0, 100),
    'revenue': (0, 50000),
    'arpu': (0, 10000),
    'frequence': (0, 100),
    'data_volume': (0, 1000),
    'on_net': (0, 1000),
    'orange': (0, 500),
    'tigo': (0, 500),
    'zone1': (0, 500),
    'zone2': (0, 500),
    'mrg': ['Oui', 'Non'],
    'regularity': (0, 100),
    'top_pack': ['Pack1', 'Pack2', 'Pack3', 'Pack4']
}

# Formulaire de saisie
inputs = {}
cols = st.columns(3)

for i, (feature, values) in enumerate(features.items()):
    with cols[i % 3]:
        if isinstance(values, tuple):  # Valeur numérique
            inputs[feature] = st.slider(
                label=feature.capitalize(),
                min_value=values[0],
                max_value=values[1],
                value=(values[0] + values[1]) // 2
            )
        else:  # Valeur catégorielle
            inputs[feature] = st.selectbox(
                label=feature.capitalize(),
                options=values
            )

# Bouton de prédiction
if st.button('Prédire le Churn'):
    # Création du DataFrame
    input_df = pd.DataFrame([inputs])

    # Encodage des variables catégorielles
    for col, encoder in df_encoded.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Prédiction
    proba = model.predict_proba(input_df)[0][1]
    churn_prob = round(proba * 100, 2)

    # Affichage des résultats
    st.subheader(f"Résultat : {'⚠️ Client à risque' if churn_prob > 50 else '✅ Client fidèle'}")
    st.progress(churn_prob / 100)
    st.metric(label="Probabilité de désabonnement", value=f"{churn_prob}%")
