from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
import pandas as pd
import logging
import os
from google.cloud import storage
import tempfile
from dotenv import load_dotenv
import shap
import numpy as np
import streamlit as st

# load_dotenv('.env')

# Configurer la journalisation
logging.basicConfig(level=logging.INFO)


# Charger le modèle sauvegardé
model_path = "./mlflow_model_"
model = mlflow.sklearn.load_model(model_path)

try:
    model = mlflow.sklearn.load_model(model_path)
    logging.info("Modèle chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle: {e}")
    raise HTTPException(status_code=500, detail="Erreur lors du chargement du modèle")

model_ = model.named_steps['classifier']

# Charger les données des clients
data_path = 'https://raw.githubusercontent.com/imanitou/projet_8/main/app_train_with_feature_selection_subset.csv'
try:
    #clients_df = pd.read_csv(data_path)
    clients_df = pd.read_csv("app_train_with_feature_selection_subset.csv")
    logging.info("Données des clients chargées avec succès.")
    logging.info(f"En-tête du DataFrame des clients :\n{clients_df.head()}")
except Exception as e:
    logging.error(f"Erreur lors du chargement des données des clients : {e}")
    raise HTTPException(status_code=500, detail="Erreur lors du chargement des données des clients")

# Fonction pour obtenir les informations du client
def get_client_info(client_id, clients_df):
    return clients_df[clients_df['SK_ID_CURR'] == client_id]

# Fonction de prédiction
def predict(client_id: int, model, clients_df):
    try:
        # Rechercher le client par ID
        logging.info(f"Recherche du client ID {client_id}")
        client_data = clients_df[clients_df['SK_ID_CURR'] == client_id]
        if client_data.empty:
            logging.warning(f"Client ID {client_id} non trouvé.")
            return None

        # On extrait les features du client
        client_features = client_data.drop('SK_ID_CURR', axis=1).values

        # Obtenir les prédictions et les valeurs SHAP
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(clients_df.drop('SK_ID_CURR', axis=1).values, 10))
        # Calcul des valeurs SHAP pour le client
        shap_values = explainer.shap_values(client_features)

        # Prédiction
        prediction = model.predict(client_features)
        # Probabilité associée
        prediction_proba = model.predict_proba(client_features)
        # Probabilité de la classe positive (1)
        score = prediction_proba[:, 1]

        logging.info(f"Prédiction pour le client ID {client_id} : {prediction[0]}, Score : {score[0]}")

        return {
            "prediction": prediction.tolist(),
            "score": score.tolist(),
            "features": client_data.drop('SK_ID_CURR', axis=1).columns.tolist(),
            "shap_values": shap_values if isinstance(shap_values, np.ndarray) else [s.tolist() for s in shap_values]
        }

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return None

# Charger le modèle et les données
model = load_mlflow_model()
clients_df = load_client_data()

# Interface Streamlit
st.title("Prédiction de crédit")

client_id = st.number_input("Entrez l'ID du client", min_value=1, step=1)

if st.button("Prédire"):
    if model is not None and clients_df is not None:
        client_info = get_client_info(client_id, clients_df)
        if client_info.empty:
            st.warning(f"Client ID {client_id} non trouvé.")
        else:
            # Afficher les informations du client
            st.write("Informations concernant le client :")
            formatted_info = client_info.applymap(lambda x: format_number(x) if isinstance(x, (int, float)) else x)
            st.dataframe(formatted_info)
            
            # Faire la prédiction
            result = predict(client_id, model, clients_df)
            
            if result:
                prediction = result['prediction'][0]
                score = result['score'][0]
                shap_values = result['shap_values']
                features = result['features']

                # Afficher les résultats
                if score < 0.18:
                    st.write("**Prédiction : BON CLIENT ! Le client devrait rembourser son crédit.**")
                else:
                    st.write("**Prédiction : ATTENTION ! Le client risque de ne pas rembourser son crédit.**")
                st.write(f"Probabilité de faire défaut : {score:.2f}")
                
                # Graphique SHAP
                st.subheader("Importance locale des caractéristiques (SHAP values)")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values[1], client_info.drop('SK_ID_CURR', axis=1), plot_type="bar", show=False)
                st.pyplot(fig)
                
                # Afficher les valeurs SHAP dans un tableau
                st.subheader("Valeurs SHAP détaillées")
                shap_df = pd.DataFrame({
                    'Feature': features,
                    'SHAP value': shap_values[1][0]
                }).sort_values('SHAP value', key=abs, ascending=False)
                st.table(shap_df)
            else:
                st.error("Impossible de faire une prédiction pour ce client.")
    else:
        st.error("Impossible de faire une prédiction. Vérifiez que le modèle et les données sont correctement chargés.")