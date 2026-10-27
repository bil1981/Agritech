# -*- coding: utf-8 -*-
"""
Etape 2 - Moteur de prediction du rendement agricole
Agritech Answers

Prerequis :
- Placer `dataset_consolidated.csv` (issu de l'etape 1) dans le meme dossier.
- Installer les dependances :
    pip install pandas numpy scikit-learn mlflow matplotlib shap

Lancement :
    mlflow ui --backend-store-uri ./mlruns
    python train_yield_model.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = os.getenv("DATA_PATH", "dataset_consolidated.csv")
TARGET = os.getenv("TARGET_COLUMN", "yield")  # adapter si necessaire
EXPERIMENT_NAME = "agritech_yield_prediction"

# ------------------------------------------------------------------
# 1. Chargement et preparation
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Dimensions :", df.shape)
print(df.head())

if TARGET not in df.columns:
    candidates = [
        c for c in df.columns if "yield" in c.lower() or "rendement" in c.lower()
    ]
    raise ValueError(
        f"Colonne cible '{TARGET}' absente. Colonnes candidates : {candidates}"
    )

# Suppression des doublons stricts et des lignes sans cible
df = df.drop_duplicates()
df = df.dropna(subset=[TARGET])

# Colonnes a exclure : identifiants, fuites potentielles, colonnes redondantes
EXCLUDE_COLS = [
    c
    for c in df.columns
    if c.lower() in {"id", "index", "unnamed: 0", "area", "country", "region"}
    or c.lower().startswith("pca")
]
# ATTENTION : garder 'Area'/'Country' si elles constituent une feature metier utile.
# Adapter cette liste apres lecture du dataset consolide.

feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS + [TARGET]]
X = df[feature_cols].copy()
y = df[TARGET].astype(float)

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print("Cible :", TARGET)
print("Variables numeriques :", numeric_cols)
print("Variables categorielles :", categorical_cols)

# Split stratifie approximatif par culture si disponible, sinon aleatoire
stratify = None
if "crop" in [c.lower() for c in X.columns]:
    crop_col = [c for c in X.columns if c.lower() == "crop"][0]
    stratify = X[crop_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
)

# ------------------------------------------------------------------
# 2. Preprocessing reproductible
# ------------------------------------------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)


# ------------------------------------------------------------------
# 3. Fonctions d'evaluation
# ------------------------------------------------------------------
def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    )
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_pct": mape}


def log_run(model_name, pipeline, params, X_train, X_test, y_train, y_test, tags=None):
    with mlflow.start_run(run_name=model_name, tags=tags or {}):
        mlflow.log_params(
            {
                "model": model_name,
                "random_state": RANDOM_STATE,
                **params,
            }
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = evaluate(y_test, preds)
        mlflow.log_metrics(metrics)

        # Importance par permutation sur le jeu de test
        try:
            perm = permutation_importance(
                pipeline,
                X_test,
                y_test,
                n_repeats=5,
                random_state=RANDOM_STATE,
                scoring="neg_root_mean_squared_error",
            )
            importance = pd.Series(
                perm.importances_mean, index=X_test.columns
            ).sort_values(ascending=False)
            plt.figure(figsize=(8, 5))
            importance.head(15).plot(kind="barh")
            plt.gca().invert_yaxis()
            plt.title(f"Importance des variables - {model_name}")
            plt.tight_layout()
            plt.savefig(f"mlflow_importance_{model_name}.png", dpi=150)
            plt.close()
            mlflow.log_artifact(f"mlflow_importance_{model_name}.png")
        except Exception as e:
            print("Importance non calculee :", e)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        print(model_name, metrics)
        return metrics, pipeline


# ------------------------------------------------------------------
# 4. Comparaison de modeles
# ------------------------------------------------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)

models = {
    "ridge": Ridge(random_state=RANDOM_STATE),
    "random_forest": RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}

results = {}
fitted = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    metrics, pipe = log_run(
        name,
        pipe,
        {"hyperparameters": "baseline"},
        X_train,
        X_test,
        y_train,
        y_test,
        tags={"stage": "baseline_comparison"},
    )
    results[name] = metrics
    fitted[name] = pipe

# ------------------------------------------------------------------
# 5. Optimisation du meilleur modele
# ------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["r2"])
print("Meilleur modele baseline :", best_name)

param_grids = {
    "ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    "random_forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 12, 20],
        "model__min_samples_leaf": [1, 3, 5],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": [0.03, 0.06, 0.1],
        "model__max_iter": [200, 400],
        "model__max_leaf_nodes": [15, 31],
        "model__l2_regularization": [0.0, 1.0],
    },
}

grid = GridSearchCV(
    fitted[best_name],
    param_grids[best_name],
    scoring="neg_root_mean_squared_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1,
    verbose=1,
)

with mlflow.start_run(
    run_name=f"{best_name}_optimized", tags={"stage": "hyperparameter_optimization"}
):
    grid.fit(X_train, y_train)
    best_params = {k: v for k, v in grid.best_params_.items()}
    mlflow.log_params({"model": best_name, "random_state": RANDOM_STATE, **best_params})
    preds = grid.predict(X_test)
    metrics = evaluate(y_test, preds)
    mlflow.log_metrics(metrics)
    mlflow.log_metric("cv_best_rmse", -grid.best_score_)
    mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")
    joblib.dump(grid.best_estimator_, "best_yield_model.joblib")
    with open("best_model_metadata.json", "w") as f:
        json.dump(
            {"best_model": best_name, "params": best_params, "test_metrics": metrics},
            f,
            indent=2,
        )
    mlflow.log_artifact("best_model_metadata.json")
    print("Modele optimise :", best_name, metrics)

# ------------------------------------------------------------------
# 6. Interpretation metier
# ------------------------------------------------------------------
# Le classement des importances identifie les leviers agronomiques :
# temperature, precipitations, pesticides, fertilisation, irrigation, type de culture.
# A integrer au rapport : expliquer chaque variable en langage simple et formuler
# des recommandations par culture (ex. culture recommandee selon conditions).
