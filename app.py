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
from babel.numbers import format_decimal
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import plotly.graph_objects as go
import plotly.express as px
import time



# Configurer la journalisation
logging.basicConfig(level=logging.INFO)

# Fonction pour formater les nombres en utilisant un séparateur de millier français
def format_number(number):
    try:
        return format_decimal(number, locale='fr_FR')
    except:
        return number
    
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
data_path = './app_train_with_feature_selection_subset.csv'
try:
    #clients_df = pd.read_csv(data_path)
    clients_df = pd.read_csv("app_train_with_feature_selection_subset.csv")
    logging.info("Données des clients chargées avec succès.")
    logging.info(f"En-tête du DataFrame des clients :\n{clients_df.head()}")
except Exception as e:
    logging.error(f"Erreur lors du chargement des données des clients : {e}")
    raise HTTPException(status_code=500, detail="Erreur lors du chargement des données des clients")





def predict(client_id: int):
    try:
        # Utiliser les variables globales `model` et `clients_df`
        global model, clients_df

        # Rechercher le client par ID
        logging.info(f"Recherche du client ID {client_id}")
        client_data = clients_df[clients_df['SK_ID_CURR'] == client_id]
        if client_data.empty:
            logging.warning(f"Client ID {client_id} non trouvé.")
            raise HTTPException(status_code=404, detail="Client non trouvé")
        
        # On extrait les features du client
        client_features = client_data.values

        # Obtenir les prédictions et les valeurs SHAP
        explainer = shap.KernelExplainer(model_.predict_proba, shap.sample(clients_df.values, 10))  # Choisir l'explainer adapté à ton modèle
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
            "features": client_data.columns.tolist(),
            "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else [s.tolist() for s in shap_values]

        }

    
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Entrée pour l'ID du client
client_id = st.number_input("Entrez le SK_ID_CURR du client :", min_value=int(clients_df['SK_ID_CURR'].min()), max_value=int(clients_df['SK_ID_CURR'].max()))

# Fonction pour obtenir les informations d'un client
def get_client_info(client_id):
    client_info = clients_df[clients_df['SK_ID_CURR'] == client_id]
    return client_info

# Afficher les informations du client
if client_id:
    client_info = get_client_info(client_id)
    if not client_info.empty:
        st.write("Informations concernant le client :")
        # formatted_info = client_info.applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
        formatted_info = client_info.applymap(lambda x: format_number(x) if isinstance(x, (int, float)) else x)
        st.dataframe(formatted_info)
        
        # Envoyer la requête à l'API pour obtenir la prédiction
        result = predict(client_id)
        if result:
                prediction = result['prediction'][0]
                score = result['score'][0]
                shap_values = result['shap_values']
                features = result['features']

                # Convertir shap_values en numpy array
                shap_values = np.array(result['shap_values'])

                # Afficher les résultats
                if score < 0.18:
                    st.write("**Prédiction : BON CLIENT ! Le client devrait rembourser son crédit.**")
                else:
                    st.write("**Prédiction : ATTENTION ! Le client risque de ne pas rembourser son crédit.**")
                st.write(f"Probabilité de faire défaut : {score:.2f}")

                # Créer une jauge avec Plotly
                final_score = score
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=0, # Valeur initiale à 0
                    title={'text': "<b>Probabilité de faire défaut</b>",
                        'font': {'size' : 20, 'color' : 'black'}},
                    gauge={
                        'axis': {
                            'range': [0, 1],
                            'tickwidth' : 2,  
                            'tickcolor': 'black'},
                        'bar': {'color': "linen"},
                        'steps': [
                            {'range': [0, 0.18], 'color': "forestgreen"},
                            {'range': [0.18, 1], 'color': "darkred"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.18
                        }
                    },
                    # Customisation du score affiché
                    number={
                        'font': {'size' : 30, 'color': 'black'} 
                    }
                ))
                # On affiche la jauge initiale dans Streamlit
                gauge_plot = st.plotly_chart(fig)

                # Simulation du chargement progressif de la jauge
                for i in range(1, int(final_score * 100) + 1):
                    time.sleep(0.03)  # Pause pour simuler le chargement
                    fig.update_traces(value=i/100)  # Mise à jour de la valeur de la jauge
                    gauge_plot.plotly_chart(fig)  # Rafraîchissement de la jauge dans Streamlit
            
# Importance locale des caractéristiques 

                # Titre
                st.markdown("""<p style='text-align: center; font-size: 24px; text-decoration: underline;'>
                I. Importance locale des caractéristiques
                </p>
                <div style='height: 30px;'></div>  <!-- Ajouter un espace de 30 pixels -->
                """, 
                unsafe_allow_html=True)

                # Assurer que les dimensions correspondent
                client_features = client_info.values

                # Ajuster les dimensions des shap_values
                if shap_values.shape[2] == 2:  # Cas pour un modèle binaire
                    shap_values = shap_values[:, :, 1]  # Sélectionner les valeurs pour la classe positive (1)
                
                # Vérifier si les dimensions correspondent
                if client_features.shape[1] != shap_values.shape[1]:
                    st.error(f"Les dimensions des shap_values ({shap_values.shape[1]}) ne correspondent pas à celles des données client ({client_features.shape[1]})")
                else:
                    # Extraire les valeurs SHAP locales pour le client
                    local_shap_values = shap_values[0]  # Prendre les valeurs SHAP pour le seul client

                    # Calculer l'importance des caractéristiques
                    top_indices = np.argsort(local_shap_values)[-10:]  # Indices des 10 plus importantes caractéristiques
                    top_indices = top_indices[np.argsort(local_shap_values[top_indices])[::-1]]  # Tri décroissant des indices
                    top_features = np.array(features)[top_indices]
                    top_shap_values = local_shap_values[top_indices]

                    top_shap_values_df = pd.DataFrame({
                        'Caractéristique': top_features,
                        'Valeur SHAP': top_shap_values
                        })
                    
                    # Dataframe
                    st.dataframe(top_shap_values_df, height=250)

                    # Ajout d'un espace
                    st.markdown(""" <div style='height: 30px;'></div>  <!-- Ajouter un espace de 30 pixels -->
                    """, 
                    unsafe_allow_html=True)
                    
                    # Trier les valeurs dans l'ordre décroissant
                    top_shap_values_df = top_shap_values_df.sort_values(by='Valeur SHAP', ascending=True)
                

                    # Créer un graphique en barres avec Plotly
                    fig = px.bar(
                        top_shap_values_df,
                        x='Valeur SHAP',
                        y='Caractéristique',
                        orientation='h',  # Pour un graphique horizontal
                        #title="Importance des caractéristiques pour la prédiction",
                        text=top_shap_values_df['Valeur SHAP'].map('{:.3f}'.format),  # Ajouter des labels pour chaque barre
                        color='Valeur SHAP',  # Gradient de couleur basé sur l'importance
                        color_continuous_scale='brwnyl',  # Choisir une palette de couleurs
                    )

                    # Personnalisation du graphique
                    fig.update_layout(
                        title={
                            'text': "Top 10 des caractéristiques qui ont le plus influencé<br>la prédiction du modèle pour le client sélectionné",
                            'y': 0.95,  # Position verticale
                            'x': 0.5,  # Centrer le titre
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': dict(size=18, color='black')
                        },
                        xaxis_title="Valeur SHAP",
                        xaxis=dict(
                            title_font=dict(size=16, color='black'),  # Couleur du texte du titre de l'axe des x
                            tickfont=dict(size=14, color='black')  # Couleur du texte des ticks de l'axe des x
                        ),
                        yaxis_title=None,
                        yaxis=dict(
                            title_font=dict(size=14, color='black'),  # Couleur du texte du titre de l'axe des y
                            tickfont=dict(size=14, color='black')  # Couleur du texte des ticks de l'axe des y
                        ),
                        font=dict(size=12, color='black'),
                        plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
                        paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,  # Cacher la légende
                        coloraxis_colorbar=dict(
                            title='Valeur SHAP',  # Titre de la barre de couleur
                            title_font=dict(size=14, color='black'),  # Couleur du titre de la barre de couleur
                            tickfont=dict(size=12, color='black')  # Couleur du texte des ticks de la barre de couleur
                        ),
                    )


                    # Ajouter une ligne horizontale pour l'axe des abscisses
                    fig.update_xaxes(
                        # zeroline=True,  # Afficher la ligne de l'axe des abscisses
                        # zerolinecolor='white',  # Couleur de la ligne de l'axe des abscisses
                        # zerolinewidth=2,  # Épaisseur de la ligne de l'axe des abscisses
                        showline=True,  # Afficher la ligne de l'axe
                        linecolor='black',  # Couleur de la ligne de l'axe
                        linewidth=0.5  # Épaisseur de la ligne de l'axe
                    )

                    # Ajustez la taille de la figure
                    fig.update_layout(
                        width=1400,  # Largeur en pixels
                        height=600,  # Hauteur en pixels
                    )

                    # Afficher le graphique avec Streamlit
                    st.plotly_chart(fig)

            

