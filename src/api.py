# -*- coding: utf-8 -*-
"""
API FastAPI - Agritech Answers

Endpoints :
- GET  /health : verification de disponibilite
- GET  /crops  : liste des cultures disponibles
- POST /predict : prediction du rendement pour une culture donnee
- POST /recommend : classement des cultures pour une parcelle

La base SQLite agritech.db est utilisee pour :
- enregistrer les conditions de parcelle ;
- stocker les predictions ;
- conserver le classement des cultures recommandees.

Artefacts attendus :
- models/best_yield_model.joblib
- models/best_model_metadata.json
- agritech.db
"""

import json
import math
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIRECTORIES = (
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "notebooks" / "models",
)
DATABASE_PATH = PROJECT_ROOT / "agritech.db"

TARGET = "yield_t_ha"
MODEL_VERSION = "v1"

REQUIRED_PARCEL_FEATURES = (
    "country",
    "year",
    "rainfall_mm",
    "avg_temp_c",
    "pesticides_tonnes",
)

CATEGORICAL_FEATURES = {"country", "crop"}


# ----------------------------------------------------------------------
# Modeles Pydantic
# ----------------------------------------------------------------------


class YieldRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country: str = Field(description="Pays de la parcelle.")
    year: int = Field(ge=1900, le=2100, description="Annee de culture.")
    rainfall_mm: float = Field(
        ge=0, allow_inf_nan=False, description="Pluviometrie annuelle en mm."
    )
    avg_temp_c: float = Field(
        allow_inf_nan=False, description="Temperature moyenne en degres Celsius."
    )
    pesticides_tonnes: float = Field(
        ge=0, allow_inf_nan=False, description="Pesticides utilises en tonnes."
    )
    crop: str | None = Field(
        default=None,
        description="Culture a evaluer. Obligatoire pour /predict, interdit pour /recommend.",
    )

    @model_validator(mode="before")
    @classmethod
    def unwrap_features(cls, value: Any) -> Any:
        if isinstance(value, dict) and "features" in value:
            if len(value) != 1:
                raise ValueError(
                    "Fournissez soit les champs de parcelle à la racine, "
                    "soit un objet 'features', mais pas les deux."
                )
            features = value["features"]
            if not isinstance(features, dict):
                raise ValueError("'features' doit être un objet JSON.")
            return features
        return value

    @model_validator(mode="after")
    def normalize_text_fields(self) -> "YieldRequest":
        self.country = self.country.strip()
        if not self.country:
            raise ValueError("'country' ne peut pas être vide.")
        if self.crop is not None:
            self.crop = self.crop.strip()
        return self


class YieldPredictionResponse(BaseModel):
    parcel_id: int
    crop: str
    estimated_yield_t_ha: float
    model_version: str


class CropRecommendation(BaseModel):
    rank: int
    crop: str
    estimated_yield_t_ha: float
    recommended: bool


class RecommendationResponse(BaseModel):
    parcel_id: int
    recommendations: list[CropRecommendation]


class CropListResponse(BaseModel):
    crops: list[str]


# ----------------------------------------------------------------------
# Chargement des artefacts
# ----------------------------------------------------------------------


def _load_artifacts() -> tuple[Any, dict[str, Any]]:
    for artifact_directory in ARTIFACT_DIRECTORIES:
        metadata_path = artifact_directory / "best_model_metadata.json"
        model_path = artifact_directory / "best_yield_model.joblib"

        if metadata_path.is_file() and model_path.is_file():
            with metadata_path.open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

            expected_version = metadata.get("sklearn_version")
            current_version = sklearn.__version__

            if expected_version and expected_version != current_version:
                raise RuntimeError(
                    "Version scikit-learn incompatible. "
                    f"Modele entraîne avec {expected_version}, "
                    f"environnement actuel : {current_version}."
                )

            model = joblib.load(model_path)
            return model, metadata

    expected_paths = [
        str(directory / filename)
        for directory in ARTIFACT_DIRECTORIES
        for filename in ("best_model_metadata.json", "best_yield_model.joblib")
    ]
    raise RuntimeError(
        "Artefacts du modele Agritech introuvables. "
        f"Attendu dans : {', '.join(str(d) for d in ARTIFACT_DIRECTORIES)}. "
        f"Fichiers recherches : {', '.join(expected_paths)}"
    )


def _get_crop_options(model: Any) -> list[str]:
    try:
        preprocessor = model.named_steps["preprocessor"]
        categorical_columns = next(
            columns for name, _, columns in preprocessor.transformers_ if name == "cat"
        )
        crop_index = categorical_columns.index("crop")
        categorical_transformer = preprocessor.named_transformers_["cat"]
        one_hot_encoder = categorical_transformer.named_steps["onehot"]
        return [str(crop) for crop in one_hot_encoder.categories_[crop_index]]
    except (AttributeError, KeyError, StopIteration, ValueError, IndexError) as error:
        raise RuntimeError(
            "Impossible de determiner les cultures a partir du pipeline entraîne."
        ) from error


# ----------------------------------------------------------------------
# Acces SQLite
# ----------------------------------------------------------------------


def _get_connection() -> sqlite3.Connection:
    if not DATABASE_PATH.is_file():
        raise RuntimeError(
            f"Base SQLite introuvable : {DATABASE_PATH}. "
            "Executez d'abord build_agritech_database.py."
        )
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _insert_parcel(conn: sqlite3.Connection, payload: YieldRequest) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO parcel_conditions (
            country, year, rainfall_mm, avg_temp_c,
            pesticides_tonnes, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            payload.country.strip(),
            payload.year,
            payload.rainfall_mm,
            payload.avg_temp_c,
            payload.pesticides_tonnes,
            created_at,
        ),
    )
    return int(cursor.lastrowid)


def _get_crop_reference(conn: sqlite3.Connection, crop: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT crop, crop_irrigation_rate, crop_fertilizer_rate,
               crop_mean_days_to_harvest, crop_mean_rainfall_local_mm,
               crop_mean_temperature_local_c, has_agronomic_proxy
        FROM crop_reference
        WHERE crop = ?
        """,
        (crop,),
    ).fetchone()

    if row is None:
        return {
            "crop": crop,
            "crop_irrigation_rate": None,
            "crop_fertilizer_rate": None,
            "crop_mean_days_to_harvest": None,
            "crop_mean_rainfall_local_mm": None,
            "crop_mean_temperature_local_c": None,
            "has_agronomic_proxy": 0,
        }

    return dict(row)


def _build_model_input(
    payload: YieldRequest,
    feature_names: list[str],
    crop: str,
    crop_reference: dict[str, Any],
) -> pd.DataFrame:
    row: dict[str, Any] = {}

    for feature in feature_names:
        if feature == "country":
            row[feature] = payload.country.strip()
        elif feature == "year":
            row[feature] = payload.year
        elif feature == "rainfall_mm":
            row[feature] = payload.rainfall_mm
        elif feature == "avg_temp_c":
            row[feature] = payload.avg_temp_c
        elif feature == "pesticides_tonnes":
            row[feature] = payload.pesticides_tonnes
        elif feature == "crop":
            row[feature] = crop
        else:
            row[feature] = crop_reference.get(feature)

    return pd.DataFrame([row], columns=feature_names)


def _predict_crop(
    model: Any,
    payload: YieldRequest,
    feature_names: list[str],
    crop: str,
    conn: sqlite3.Connection,
) -> float:
    crop_reference = _get_crop_reference(conn, crop)
    model_input = _build_model_input(payload, feature_names, crop, crop_reference)

    try:
        estimate = float(model.predict(model_input)[0])
    except (TypeError, ValueError, KeyError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Impossible de calculer le rendement pour {crop!r} : {error}",
        ) from error

    if not math.isfinite(estimate):
        raise HTTPException(
            status_code=500,
            detail=f"Le modele a retourne une estimation non finie pour {crop!r}.",
        )

    return estimate


# ----------------------------------------------------------------------
# Cycle de vie de l'application
# ----------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, metadata = _load_artifacts()

    feature_names = metadata.get("features")
    if not isinstance(feature_names, list) or not feature_names:
        raise RuntimeError("Les metadonnees du modele ne contiennent pas 'features'.")

    if metadata.get("target") != TARGET:
        raise RuntimeError(
            f"Cible incompatible dans les metadonnees : "
            f"{metadata.get('target')!r}, attendu {TARGET!r}."
        )

    if (
        not hasattr(model, "feature_names_in_")
        or list(model.feature_names_in_) != feature_names
    ):
        raise RuntimeError(
            "Les variables du modele ne correspondent pas a celles des metadonnees."
        )

    app.state.model = model
    app.state.feature_names = feature_names
    app.state.crop_options = _get_crop_options(model)
    app.state.model_version = metadata.get("model_version", MODEL_VERSION)

    yield


app = FastAPI(
    title="Agritech Yield API",
    description=(
        "Prediction du rendement agricole et recommandation de la culture "
        "la plus performante pour une parcelle donnee."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/crops", response_model=CropListResponse)
def crops(request: Request) -> CropListResponse:
    return CropListResponse(crops=request.app.state.crop_options)


@app.post("/predict", response_model=YieldPredictionResponse)
def predict(payload: YieldRequest, request: Request) -> YieldPredictionResponse:
    if not payload.crop:
        raise HTTPException(
            status_code=422,
            detail="La variable 'crop' est obligatoire pour /predict.",
        )

    crop = payload.crop.strip()
    if crop not in request.app.state.crop_options:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Culture inconnue : {crop!r}. Cultures disponibles : "
                f"{', '.join(request.app.state.crop_options)}"
            ),
        )

    conn = _get_connection()
    try:
        model = request.app.state.model
        feature_names = request.app.state.feature_names
        estimate = _predict_crop(model, payload, feature_names, crop, conn)
        parcel_id = _insert_parcel(conn, payload)
        predicted_at = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """
            INSERT INTO yield_predictions (
                parcel_id, crop, predicted_yield_t_ha,
                model_version, predicted_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (parcel_id, crop, estimate, request.app.state.model_version, predicted_at),
        )
        conn.commit()
    finally:
        conn.close()

    return YieldPredictionResponse(
        parcel_id=parcel_id,
        crop=crop,
        estimated_yield_t_ha=estimate,
        model_version=request.app.state.model_version,
    )


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: YieldRequest, request: Request) -> RecommendationResponse:
    if payload.crop:
        raise HTTPException(
            status_code=422,
            detail=(
                "Ne fournissez pas 'crop' pour /recommend : "
                "l'API evalue automatiquement toutes les cultures."
            ),
        )

    model = request.app.state.model
    feature_names = request.app.state.feature_names
    created_at = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()
    try:
        parcel_id = _insert_parcel(conn, payload)

        recommendations: list[CropRecommendation] = []
        for crop in request.app.state.crop_options:
            estimate = _predict_crop(model, payload, feature_names, crop, conn)

            conn.execute(
                """
                INSERT INTO yield_predictions (
                    parcel_id, crop, predicted_yield_t_ha,
                    model_version, predicted_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    parcel_id,
                    crop,
                    estimate,
                    request.app.state.model_version,
                    created_at,
                ),
            )

            recommendations.append(
                CropRecommendation(
                    rank=0,
                    crop=crop,
                    estimated_yield_t_ha=estimate,
                    recommended=False,
                )
            )

        recommendations.sort(
            key=lambda recommendation: recommendation.estimated_yield_t_ha,
            reverse=True,
        )

        for rank, recommendation in enumerate(recommendations, start=1):
            recommendation.rank = rank
            recommendation.recommended = rank == 1

            conn.execute(
                """
                INSERT INTO crop_recommendations (
                    parcel_id, crop, predicted_yield_t_ha,
                    rank, recommended, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    parcel_id,
                    recommendation.crop,
                    recommendation.estimated_yield_t_ha,
                    rank,
                    1 if recommendation.recommended else 0,
                    created_at,
                ),
            )

        conn.commit()
    finally:
        conn.close()

    return RecommendationResponse(
        parcel_id=parcel_id,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
