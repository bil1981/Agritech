# -*- coding: utf-8 -*-
"""
Creation de la base SQLite Agritech Answers.

Sources attendues :
- data/processed/crop_yield_unified.csv : dataset consolide (etape 1)
- models/best_yield_model.joblib        : modele entraîne (etape 2)
- models/best_model_metadata.json       : metadonnees du modele
- feature_importance.csv                : importance des variables (optionnel)

Tables creees :
- parcel_conditions : conditions de parcelle saisies par l'agriculteur
- crop_yield_history : historique consolide pays-annee-culture
- crop_reference : referentiel des cultures et proxys agronomiques
- yield_predictions : predictions generees par le modele
- crop_recommendations : classement des cultures recommandees
- model_metadata : informations sur le modele en production
- feature_importance : importance des variables
"""

import os
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import joblib

UNIFIED_CSV = "data/processed/crop_yield_unified.csv"
MODEL_PATH = "models/best_yield_model.joblib"
METADATA_PATH = "models/best_model_metadata.json"
FEATURE_IMPORTANCE_CSV = "feature_importance.csv"
DB_PATH = "agritech.db"


def create_schema(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS parcel_conditions (
            parcel_id INTEGER PRIMARY KEY AUTOINCREMENT,
            country TEXT NOT NULL,
            year INTEGER NOT NULL,
            rainfall_mm REAL,
            avg_temp_c REAL,
            pesticides_tonnes REAL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS crop_yield_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            country TEXT NOT NULL,
            year INTEGER NOT NULL,
            crop TEXT NOT NULL,
            yield_t_ha REAL,
            rainfall_mm REAL,
            avg_temp_c REAL,
            pesticides_tonnes REAL,
            crop_irrigation_rate REAL,
            crop_fertilizer_rate REAL,
            crop_mean_days_to_harvest REAL,
            crop_mean_rainfall_local_mm REAL,
            crop_mean_temperature_local_c REAL,
            has_agronomic_proxy INTEGER,
            UNIQUE(country, year, crop)
        );

        CREATE TABLE IF NOT EXISTS crop_reference (
            crop_id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop TEXT UNIQUE NOT NULL,
            crop_irrigation_rate REAL,
            crop_fertilizer_rate REAL,
            crop_mean_days_to_harvest REAL,
            crop_mean_rainfall_local_mm REAL,
            crop_mean_temperature_local_c REAL,
            has_agronomic_proxy INTEGER
        );

        CREATE TABLE IF NOT EXISTS yield_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            parcel_id INTEGER NOT NULL,
            crop TEXT NOT NULL,
            predicted_yield_t_ha REAL NOT NULL,
            model_version TEXT,
            predicted_at TEXT NOT NULL,
            FOREIGN KEY (parcel_id) REFERENCES parcel_conditions(parcel_id)
        );

        CREATE TABLE IF NOT EXISTS crop_recommendations (
            recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            parcel_id INTEGER NOT NULL,
            crop TEXT NOT NULL,
            predicted_yield_t_ha REAL NOT NULL,
            rank INTEGER NOT NULL,
            recommended INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (parcel_id) REFERENCES parcel_conditions(parcel_id)
        );

        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            model_version TEXT,
            trained_at TEXT,
            target TEXT,
            rmse REAL,
            mae REAL,
            r2 REAL,
            mape_pct REAL,
            parameters_json TEXT,
            features_json TEXT
        );

        CREATE TABLE IF NOT EXISTS feature_importance (
            feature TEXT PRIMARY KEY,
            importance REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_history_country_year_crop
            ON crop_yield_history(country, year, crop);

        CREATE INDEX IF NOT EXISTS idx_predictions_parcel
            ON yield_predictions(parcel_id);

        CREATE INDEX IF NOT EXISTS idx_recommendations_parcel
            ON crop_recommendations(parcel_id);
        """
    )


def load_unified_data(conn):
    if not os.path.exists(UNIFIED_CSV):
        raise FileNotFoundError(f"Fichier introuvable : {UNIFIED_CSV}")

    df = pd.read_csv(UNIFIED_CSV)
    print(f"Dataset consolide charge : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

    df.to_sql(
        "crop_yield_history",
        conn,
        if_exists="replace",
        index=False,
        dtype={"has_agronomic_proxy": "INTEGER"},
    )

    # Referentiel culture : une ligne par culture, avec les proxys agronomiques
    proxy_cols = [
        "crop",
        "crop_irrigation_rate",
        "crop_fertilizer_rate",
        "crop_mean_days_to_harvest",
        "crop_mean_rainfall_local_mm",
        "crop_mean_temperature_local_c",
        "has_agronomic_proxy",
    ]

    proxy_cols = [c for c in proxy_cols if c in df.columns]
    crop_ref = (
        df[proxy_cols]
        .drop_duplicates(subset=["crop"])
        .sort_values("crop")
        .reset_index(drop=True)
    )

    crop_ref.to_sql("crop_reference", conn, if_exists="replace", index=False)
    print(f"Referentiel cultures enregistre : {len(crop_ref)} cultures.")


def load_model_metadata(conn):
    if not os.path.exists(METADATA_PATH):
        print("Metadonnees du modele introuvables, table model_metadata laissee vide.")
        return

    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    metrics = metadata.get("final_metrics", {})
    conn.execute(
        """
        INSERT INTO model_metadata (
            model_name, model_version, trained_at, target,
            rmse, mae, r2, mape_pct, parameters_json, features_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metadata.get("best_model"),
            "v1",
            datetime.now(timezone.utc).isoformat(),
            metadata.get("target"),
            metrics.get("RMSE"),
            metrics.get("MAE"),
            metrics.get("R2"),
            metrics.get("MAPE_%"),
            json.dumps(metadata.get("best_params", {}), ensure_ascii=False),
            json.dumps(metadata.get("features", []), ensure_ascii=False),
        ),
    )
    print("Metadonnees du modele enregistrees.")


def load_feature_importance(conn):
    if not os.path.exists(FEATURE_IMPORTANCE_CSV):
        print(
            "feature_importance.csv introuvable, table feature_importance laissee vide."
        )
        return

    fi = pd.read_csv(FEATURE_IMPORTANCE_CSV)

    # Formats possibles : feature/importance, ou feature/importances_mean
    if "importances_mean" in fi.columns and "importance" not in fi.columns:
        fi = fi.rename(columns={"importances_mean": "importance"})

    required = {"feature", "importance"}
    if not required.issubset(fi.columns):
        print("Colonnes attendues absentes dans feature_importance.csv.")
        return

    fi = fi[["feature", "importance"]].drop_duplicates("feature")
    fi.to_sql("feature_importance", conn, if_exists="replace", index=False)
    print(f"Importance des variables enregistree : {len(fi)} variables.")


def insert_parcel_and_recommendations(
    conn,
    model,
    feature_cols,
    country,
    year,
    rainfall_mm,
    avg_temp_c,
    pesticides_tonnes,
):
    created_at = datetime.now(timezone.utc).isoformat()

    cursor = conn.execute(
        """
        INSERT INTO parcel_conditions (
            country, year, rainfall_mm, avg_temp_c,
            pesticides_tonnes, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (country, year, rainfall_mm, avg_temp_c, pesticides_tonnes, created_at),
    )
    parcel_id = cursor.lastrowid

    crops = pd.read_sql_query(
        "SELECT crop FROM crop_reference ORDER BY crop",
        conn,
    )["crop"].tolist()

    history = pd.read_sql_query(
        "SELECT * FROM crop_yield_history",
        conn,
    )

    rows = []
    for crop in crops:
        crop_history = history.loc[history["crop"] == crop]

        def first(column):
            if column in crop_history.columns and not crop_history.empty:
                return crop_history[column].iloc[0]
            return None

        rows.append(
            {
                "country": country,
                "year": year,
                "crop": crop,
                "rainfall_mm": rainfall_mm,
                "avg_temp_c": avg_temp_c,
                "pesticides_tonnes": pesticides_tonnes,
                "crop_irrigation_rate": first("crop_irrigation_rate"),
                "crop_fertilizer_rate": first("crop_fertilizer_rate"),
                "crop_mean_days_to_harvest": first("crop_mean_days_to_harvest"),
                "crop_mean_rainfall_local_mm": first("crop_mean_rainfall_local_mm"),
                "crop_mean_temperature_local_c": first("crop_mean_temperature_local_c"),
                "has_agronomic_proxy": int(first("has_agronomic_proxy") or 0),
            }
        )

    input_df = pd.DataFrame(rows)
    input_df["predicted_yield_t_ha"] = model.predict(input_df[feature_cols])
    input_df = input_df.sort_values(
        "predicted_yield_t_ha", ascending=False
    ).reset_index(drop=True)

    for rank, row in input_df.iterrows():
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
                row["crop"],
                float(row["predicted_yield_t_ha"]),
                "v1",
                created_at,
            ),
        )

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
                row["crop"],
                float(row["predicted_yield_t_ha"]),
                rank + 1,
                1 if rank == 0 else 0,
                created_at,
            ),
        )

    print(f"Parcelle {parcel_id} enregistree avec {len(input_df)} predictions.")
    return parcel_id, input_df[["crop", "predicted_yield_t_ha"]]


def build_database():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modele introuvable : {MODEL_PATH}. Executez d'abord train_yield_model.ipynb."
        )

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["features"]

    conn = sqlite3.connect(DB_PATH)
    try:
        create_schema(conn)
        load_unified_data(conn)
        load_model_metadata(conn)
        load_feature_importance(conn)
        conn.commit()

        # Exemple : a remplacer par les valeurs saisies dans Streamlit
        parcel_id, ranking = insert_parcel_and_recommendations(
            conn=conn,
            model=model,
            feature_cols=feature_cols,
            country="France",
            year=2013,
            rainfall_mm=800,
            avg_temp_c=12,
            pesticides_tonnes=5000,
        )
        conn.commit()

        print("\nClassement des cultures recommandees :")
        print(ranking.to_string(index=False))

    finally:
        conn.close()

    print(f"\nBase SQLite creee : {DB_PATH}")


if __name__ == "__main__":
    build_database()
