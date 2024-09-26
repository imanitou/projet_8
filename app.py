import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import shap
import matplotlib.pyplot as plt
import os

# Fonction pour charger le modèle MLflow
@st.cache_resource
def load_mlflow_model():
    try:
        # Assurez-vous que le chemin est correct
        model_path = "mlflow_model_"
        model = mlflow.pyfunc.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle MLflow: {e}")
        return None

# Fonction pour charger les données des clients
@st.cache_data
def load_client_data():
    try:
        clients_df = pd.read_csv('app_train_with_feature_selection_subset.csv')
        return clients_df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données des clients : {e}")
        return None

# Charger le modèle et les données
model = load_mlflow_model()
clients_df = load_client_data()

# Créer un explainer SHAP
@st.cache_resource
def create_explainer(model, data):
    # Nous devons adapter cette partie en fonction du type de modèle sous-jacent
    # Pour cet exemple, nous supposons qu'il s'agit d'un modèle arborescent
    return shap.TreeExplainer(model.unwrap_python_model())

if model is not None and clients_df is not None:
    explainer = create_explainer(model, clients_df.drop(['SK_ID_CURR'], axis=1))

# Interface Streamlit
st.title("Prédiction de crédit")

client_id = st.number_input("Entrez l'ID du client", min_value=1, step=1)

if st.button("Prédire"):
    if model is not None and clients_df is not None:
        client_data = clients_df[clients_df['SK_ID_CURR'] == client_id]
        if client_data.empty:
            st.warning(f"Client ID {client_id} non trouvé.")
        else:
            client_features = client_data.drop('SK_ID_CURR', axis=1)
            
            # Prédiction
            prediction = model.predict(client_features)
            
            # Pour la probabilité, nous devons vérifier si le modèle a une méthode predict_proba
            if hasattr(model.unwrap_python_model(), 'predict_proba'):
                prediction_proba = model.unwrap_python_model().predict_proba(client_features)
                score = prediction_proba[:, 1][0]
            else:
                score = prediction[0]  # Utiliser la prédiction comme score si predict_proba n'est pas disponible
            
            # SHAP values
            shap_values = explainer.shap_values(client_features)
            
            # Afficher les résultats
            st.write(f"Prédiction pour le client ID {client_id} : {'Crédit refusé' if prediction[0] == 1 else 'Crédit accordé'}")
            st.write(f"Score de probabilité : {score:.2f}")
            
            # Graphique SHAP
            st.subheader("Importance locale des caractéristiques (SHAP values)")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, client_features, plot_type="bar", show=False)
            st.pyplot(fig)
            
            # Afficher les valeurs SHAP dans un tableau
            st.subheader("Valeurs SHAP détaillées")
            shap_df = pd.DataFrame({
                'Feature': client_features.columns,
                'SHAP value': shap_values[0]
            }).sort_values('SHAP value', key=abs, ascending=False)
            st.table(shap_df)
    else:
        st.error("Impossible de faire une prédiction. Vérifiez que le modèle et les données sont correctement chargés.")