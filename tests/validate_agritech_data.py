# -*- coding: utf-8 -*-
"""
Tests unitaires de validation des donnees Agritech Answers.

Objectif :
- Verifier que le dataset consolide respecte le schema Pandera.
- Verifier que les requetes de parcelle sont validees correctement.
- Verifier que les valeurs invalides sont rejetees.
- Tester la validation independamment de l'API HTTP.

Lancement :
    pytest tests/test_validation_agritech.py -v
"""

import pandas as pd
import pytest
import pandera.pandas as pa

from validation.schemas import (
    validate_agritech_data,
    validate_parcel_input,
)


# ----------------------------------------------------------------------
# Jeux de donnees de reference
# ----------------------------------------------------------------------


@pytest.fixture
def valid_unified_df():
    return pd.DataFrame(
        {
            "country": ["France"],
            "year": [2013],
            "crop": ["Wheat"],
            "yield_t_ha": [6.5],
            "rainfall_mm": [800.0],
            "avg_temp_c": [12.0],
            "pesticides_tonnes": [5000.0],
            "crop_irrigation_rate": [0.5],
            "crop_fertilizer_rate": [0.5],
            "crop_mean_days_to_harvest": [120.0],
            "crop_mean_rainfall_local_mm": [550.0],
            "crop_mean_temperature_local_c": [27.5],
            "has_agronomic_proxy": [1],
        }
    )


@pytest.fixture
def valid_parcel():
    return {
        "country": "France",
        "year": 2013,
        "rainfall_mm": 800.0,
        "avg_temp_c": 12.0,
        "pesticides_tonnes": 5000.0,
    }


# ----------------------------------------------------------------------
# Validation du dataset consolide
# ----------------------------------------------------------------------


def test_valid_unified_dataset(valid_unified_df):
    validated_df = validate_agritech_data(valid_unified_df)

    assert len(validated_df) == 1
    assert validated_df.loc[0, "country"] == "France"
    assert validated_df.loc[0, "crop"] == "Wheat"
    assert validated_df.loc[0, "yield_t_ha"] == 6.5


def test_missing_required_column(valid_unified_df):
    df = valid_unified_df.drop(columns=["yield_t_ha"])

    with pytest.raises(ValueError, match="Colonnes obligatoires manquantes"):
        validate_agritech_data(df)


def test_negative_yield_is_rejected(valid_unified_df):
    df = valid_unified_df.copy()
    df.loc[0, "yield_t_ha"] = -1.0

    with pytest.raises(pa.errors.SchemaError):
        validate_agritech_data(df)


def test_negative_rainfall_is_rejected(valid_unified_df):
    df = valid_unified_df.copy()
    df.loc[0, "rainfall_mm"] = -100.0

    with pytest.raises(pa.errors.SchemaError):
        validate_agritech_data(df)


def test_invalid_temperature_is_rejected(valid_unified_df):
    df = valid_unified_df.copy()
    df.loc[0, "avg_temp_c"] = 80.0

    with pytest.raises(pa.errors.SchemaError):
        validate_agritech_data(df)


def test_invalid_year_is_rejected(valid_unified_df):
    df = valid_unified_df.copy()
    df.loc[0, "year"] = 1800

    with pytest.raises(pa.errors.SchemaError):
        validate_agritech_data(df)


def test_invalid_irrigation_rate_is_rejected(valid_unified_df):
    df = valid_unified_df.copy()
    df.loc[0, "crop_irrigation_rate"] = 1.5

    with pytest.raises(pa.errors.SchemaError):
        validate_agritech_data(df)


def test_invalid_fertilizer_rate_is_rejected(valid_unified_df):
    df = valid_unified_df.copy()
    df.loc[0, "crop_fertilizer_rate"] = -0.1

    with pytest.raises(pa.errors.SchemaError):
        validate_agritech_data(df)


def test_invalid_has_agronomic_proxy_is_rejected(valid_unified_df):
    df = valid_unified_df.copy()
    df.loc[0, "has_agronomic_proxy"] = 2

    with pytest.raises(pa.errors.SchemaError):
        validate_agritech_data(df)


def test_nullable_agronomic_proxies_are_accepted(valid_unified_df):
    df = valid_unified_df.copy()
    df.loc[0, "crop_irrigation_rate"] = None
    df.loc[0, "crop_fertilizer_rate"] = None
    df.loc[0, "crop_mean_days_to_harvest"] = None

    validated_df = validate_agritech_data(df)

    assert len(validated_df) == 1
    assert pd.isna(validated_df.loc[0, "crop_irrigation_rate"])


# ----------------------------------------------------------------------
# Validation d'une requete de parcelle
# ----------------------------------------------------------------------


def test_valid_parcel_input(valid_parcel):
    validated = validate_parcel_input(valid_parcel)

    assert validated["country"] == "France"
    assert validated["year"] == 2013
    assert validated["rainfall_mm"] == 800.0
    assert validated["avg_temp_c"] == 12.0
    assert validated["pesticides_tonnes"] == 5000.0


def test_parcel_negative_rainfall_is_rejected(valid_parcel):
    data = valid_parcel.copy()
    data["rainfall_mm"] = -10.0

    with pytest.raises(pa.errors.SchemaError):
        validate_parcel_input(data)


def test_parcel_negative_pesticides_are_rejected(valid_parcel):
    data = valid_parcel.copy()
    data["pesticides_tonnes"] = -1.0

    with pytest.raises(pa.errors.SchemaError):
        validate_parcel_input(data)


def test_parcel_invalid_year_is_rejected(valid_parcel):
    data = valid_parcel.copy()
    data["year"] = 2200

    with pytest.raises(pa.errors.SchemaError):
        validate_parcel_input(data)


def test_parcel_invalid_temperature_is_rejected(valid_parcel):
    data = valid_parcel.copy()
    data["avg_temp_c"] = 100.0

    with pytest.raises(pa.errors.SchemaError):
        validate_parcel_input(data)


def test_parcel_missing_required_field_is_rejected(valid_parcel):
    data = valid_parcel.copy()
    data.pop("rainfall_mm")

    with pytest.raises(pa.errors.SchemaError):
        validate_parcel_input(data)


def test_parcel_unknown_field_is_rejected(valid_parcel):
    data = valid_parcel.copy()
    data["unknown_column"] = "test"

    with pytest.raises(pa.errors.SchemaError):
        validate_parcel_input(data)


"""
pytest pour les tests de fonctions, a la demande
Pandera/pydantic pour la validation des données, ca tourne tt le temps. il apparait dans les logs de l'API, et si ca ne passe pas, ca bloque la requete.
logging : creation de logger, lors de lancement du code
"""
