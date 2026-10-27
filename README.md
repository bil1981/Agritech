# Projet MLOps - Agritech Answers

Ce dépôt conserve un pipeline historique Home Credit et comprend désormais un prototype Agritech de prédiction de rendement agricole. L'application Agritech s'appuie sur un modèle scikit-learn, une API FastAPI et une interface Streamlit. Les principales composantes sont :

- les scripts et notebooks d'entraînement dans `notebooks/` ;
- le modèle de rendement et ses métadonnées dans `models/` ;
- le script `src/build_agritech_database.py`, qui construit la base `agritech.db` ;
- l'API FastAPI `src/api.py`, qui expose la prédiction et la recommandation ;
- l'interface Streamlit `app.py`, qui utilise la base Agritech et l'API ;
- une structure dédiée pour les tests et les workflows CI/CD.

## 1. Structure du projet

```text
P1_Code_Python/
├── src/
│   ├── api.py              # API FastAPI
│   ├── build_agritech_database.py # Construction de agritech.db
├── models/
│   └── best_yield_model.joblib
│   └── best_model_metadata.json
├── notebooks/
│   └── step1_eda_fusion_acp.ipynb
│   └── step2_train_yield_model.ipynb
├── data/                    # Données initiales CSV et fichiers de résultats
├── tests/                  # Tests automatisés API et tests unitaires/API (Pytest/Pandera)
├── .github/workflows/      # Workflows GitHub Actions, CI/CD yaml
├── monitoring/      		# rapports de dérive (Drif) Evidently AI
├── logs/            		# le fichier de sauvegarde de données de production (production_data.csv)
├── docs/            		# documentation mkdocs
├── main.py             	# code principal lançant l'API et l'interface Streamlit
├── app.py              	# Interface/Dashboard Streamlit
├── CHANGELOG.md			# fichier Markdown qui liste chronologiquement tous les changements apportés au fil de ses version
├── Dockerfile				# liste d'instructions pour créer automatiquement une image
├── .gitignore			 	# liste des fichiers/dossiers ignorés dans le contrôle de version
├── requirements.txt		# liste l'ensemble des bibliothèques et packages externes nécessaires
└── README.md

```

## 2. Prérequis

- Windows avec Python 3.10 ou version supérieure ;
- les données agritech dans `data/raw` ;
- le modèle `models/` ;
- PowerShell ou un terminal VS Code.

Les chemins utilisés par les scripts sont relatifs à la racine du projet. Toutes les commandes suivantes doivent donc être exécutées depuis `P1_Code_Python`.

## 3. Installation de l'environnement

Créer un environnement virtuel :

```powershell
python -m venv .venv
```

Activer l'environnement dans PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Les dépendances principales sont `pandas`, `numpy`, `lightgbm`, `mlflow`, `scikit-learn`, `streamlit`, `plotly`, `fastapi`, `uvicorn`, `pydantic`, `pytest` et `httpx`.

### Notebook d'entraînement Agritech et suivi MLflow

Le notebook `notebooks/step2_train_yield_model.ipynb` entraîne et compare les modèles de rendement. Il charge le jeu de données `data/processed/crop_yield_unified.parquet` et utilise MLflow pour enregistrer les expériences, métriques et modèles.

Après l'installation des dépendances ci-dessus, sélectionnez le kernel Python de cet environnement dans VS Code et démarrez le serveur MLflow depuis la racine du projet :

```powershell
mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Laissez ce terminal ouvert, puis exécutez les cellules du notebook dans l'ordre. L'interface de suivi est disponible sur `http://127.0.0.1:5000`. Si MLflow n'est pas reconnu par le notebook, vérifiez que son kernel utilise bien l'environnement où `requirements.txt` a été installé, puis redémarrez le kernel.

## 4. Préparation des données et du modèle

Le script de modélisation situé dans `notebooks/lightgbm_script.py` lit les fichiers suivants :

- `Data/raw/Agriculture_Crop_Yield/*.csv` ;
- `Data/raw/Crop_Yield_Prediction_Dataset/*.csv`.

Le pipeline effectue notamment le one-hot encoding, les agrégations par `SK_ID_CURR`, le feature engineering et l'entraînement LightGBM avec validation croisée. Le modèle final doit être disponible sous :

```text
Artefacts attendus :
- models/best_yield_model.joblib
- models/best_model_metadata.json
```

Les fichiers de prédictions et d'importance des variables utilisés par la base sont notamment :

```fichier parquet de resultats Forest random
data/processed/crop_yield_unified.parquet
```

## 5. Construction de la base Agritech

Depuis la racine du projet, créer `agritech.db` :

```powershell
python .\src\build_agritech_database.py
```

Le script charge les données consolidées, le modèle et ses métadonnées. La base comprend notamment le référentiel des cultures (`crop_reference`), l'historique des rendements (`crop_yield_history`), les indicateurs du modèle (`model_metadata`) et les classements d'exemple. Le fichier est créé à la racine sous `agritech.db`.

## 6. Lancer l'API de rendement

Depuis la racine du projet :

```powershell
python -m uvicorn src.api:app --reload
```

L'API est disponible à l'adresse `http://127.0.0.1:8000` et sa documentation interactive sur `http://127.0.0.1:8000/docs`.

- `GET /health` vérifie que l'API répond.
- `POST /predict` estime le rendement d'une culture fournie.
- `POST /recommend` classe toutes les cultures pour les conditions fournies.

Exemple de requête `/predict` :

```json
{
  "features": {
    "country": "France",
    "year": 2013,
    "crop": "Maize",
    "rainfall_mm": 800,
    "avg_temp_c": 12,
    "pesticides_tonnes": 5000
  }
}
```

Exemple de requête `/recommend` : même objet `features`, sans la propriété `crop`.

## 7. Lancer l'application Streamlit

Générer d'abord la base, puis lancer l'API et, dans un second terminal, l'interface :

```powershell
python .\src\build_agritech_database.py
python -m uvicorn src.api:app --reload
```

```powershell
streamlit run app.py
```

L'interface utilise `agritech.db` pour proposer les pays et cultures, afficher les indicateurs du modèle et enrichir les prédictions avec les références agronomiques. Elle permet d'estimer un rendement ou d'obtenir le classement des cultures via l'API. Pour configurer les chemins, les variables d'environnement `AGRITECH_DB_PATH` et `API_URL` peuvent être définies.

## 8. Tests et validations

Lancer les tests ciblés de l'API et de Streamlit :

```powershell
python -m pytest tests\test_api.py tests\test_prediction_logic.py tests\test_streamlit.py
```

## Assistant RAG Mistral

Le sous-projet `P10_DSML` contient un assistant RAG basé sur FAISS et Mistral ainsi qu'un script d'évaluation RAGAS. Depuis la racine du dépôt, entrez dans le dossier du sous-projet, puis créez l'environnement et installez ses dépendances :

```powershell
Set-Location .\P10_DSML
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Créez un fichier `.env` dans `P10_DSML` et configurez votre clé API sans la publier :

```text
MISTRAL_API_KEY=votre_clé_api_mistral
```

Vérifiez que `vector_db/faiss_index.idx` et `vector_db/document_chunks.pkl` existent; sinon, lancez l'indexation avec `uv run python indexer.py`.

Lancer l'évaluation :

```powershell
uv run python evaluate_ragas.py
```

Le script calcule `faithfulness`, `answer_relevancy`, `context_precision` et `context_recall`, puis enregistre les résultats dans `ragas_evaluation_results.csv`. Il envoie des requêtes à l'API Mistral, susceptibles d'être facturées et soumises à des quotas. La concurrence RAGAS est limitée à un worker. En cas de réponse HTTP `429`, vérifiez les limites de débit et les crédits du compte avant de relancer.

### arcitecture stramlit et Fastapi

[ Utilisateur ]
         │
         ▼ (Saisit les données sur l'écran)
┌──────────────────────────────────────────────────────────┐
│              Streamlit (app_2.py) - Front                │
└──────────────────────────┬───────────────────────────────┘
                           │
                           │ Requête HTTP POST (JSON)
                           │ ex: http://localhost:8000/predict
                           ▼
┌──────────────────────────────────────────────────────────┐
│              FastAPI (main.py) - Back                    │
│  - Reçoit les données                                    │
│  - Exécute le modèle Forest Random chargé au démarrage       │
│  - Renvoie probabilité + décision                        │
└──────────────────────────┬───────────────────────────────┘
                           │
                           │ Réponse HTTP (JSON)
                           │ ex: {"probability": 0.15}
                           ▼
┌──────────────────────────────────────────────────────────┐
│              Streamlit (app_2.py) - Front                │
│  - Affiche la metric et le message de succès/erreur      │
└──────────────────────────────────────────────────────────┘
