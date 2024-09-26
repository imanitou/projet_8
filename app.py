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

# load_dotenv('.env')

app = FastAPI()

# Configurer la journalisation
logging.basicConfig(level=logging.INFO)

# # Lire le contenu JSON des credentials depuis la variable d'environnement
# key_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')

# if key_json is None:
#     raise ValueError("La variable d'environnement GOOGLE_APPLICATION_CREDENTIALS_JSON n'est pas définie.")

# # Créer un fichier temporaire pour les credentials JSON
# with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as temp_file:
#     temp_file.write(key_json)
#     temp_file_path = temp_file.name

# # Définir la variable d'environnement pour le chemin du fichier temporaire
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path

# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     """Télécharge un blob depuis le bucket."""
#     try:
#         storage_client = storage.Client() 
#     except:
#         storage_client = storage.Client.from_service_account_json(".env.json")
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)
#     logging.info(f"Blob {source_blob_name} téléchargé vers {destination_file_name}.")

# # Configurer les chemins
# bucket_name = 'bucket_mlflow_model'
# model_blob_name = 'mlflow_model_/'
# model_local_path = 'C:/Users/guill/Imane/P7/mlflow_model_/'

# # Assurez-vous que le chemin local existe
# os.makedirs(model_local_path, exist_ok=True)

# # Télécharger chaque fichier du répertoire du modèle
# files_to_download = ['conda.yaml', 'MLmodel', 'model.pkl', 'python_env.yaml', 'requirements.txt']  # Ajoutez d'autres fichiers si nécessaire

# for file_name in files_to_download:
#     download_blob(bucket_name, model_blob_name + file_name, os.path.join(model_local_path, file_name))

# # Charger le modèle sauvegardé
# model_path = os.path.abspath(model_local_path)

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

@app.get("/")
def read_root():
    return {"message": "Bienvenue à l'API du modèle MLFlow"}



@app.get("/predict/{client_id}")
def predict(client_id: int):
    try:
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

#     finally:
#         # Nettoyer le fichier temporaire après utilisation
#         if os.path.exists(temp_file_path):
#             os.remove(temp_file_path)

# # Ce bloc permet de démarrer l'application en mode standalone (lorsque le script est directement exécuté)
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))  # Utilisation de la variable PORT définie par Render
#     uvicorn.run(app, host="0.0.0.0", port=port)
    
# Dans le terminal lancer : uvicorn api:app --reload

# Test pour faire une requête GET à l'API avec un ID de client existant : http://127.0.0.1:8000/predict/100006