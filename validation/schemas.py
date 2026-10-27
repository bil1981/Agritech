# -*- coding: utf-8 -*-
"""
Validation Pandera du dataset consolide Agritech Answers.

Le schema controle les colonnes attendues, leurs types, les valeurs nulles
et les regles metier avant l'entrainement ou l'inference.
"""

from pandera.pandas import Column, DataFrameSchema, Check

# Colonnes obligatoires pour la modelisation
REQUIRED_COLUMNS = [
    "country",
    "year",
    "crop",
    "yield_t_ha",
    "rainfall_mm",
    "avg_temp_c",
    "pesticides_tonnes",
]

# Proxys agronomiques : peuvent etre absents pour certaines cultures
OPTIONAL_AGRONOMIC_COLUMNS = [
    "crop_irrigation_rate",
    "crop_fertilizer_rate",
    "crop_mean_days_to_harvest",
    "crop_mean_rainfall_local_mm",
    "crop_mean_temperature_local_c",
    "has_agronomic_proxy",
]

agritech_schema = DataFrameSchema(
    {
        # Cles analytiques
        "country": Column(
            str, nullable=False, coerce=True, checks=Check.str_length(min_value=1)
        ),
        "year": Column(
            int, nullable=False, coerce=True, checks=Check.in_range(1900, 2100)
        ),
        "crop": Column(
            str, nullable=False, coerce=True, checks=Check.str_length(min_value=1)
        ),
        # Cible
        "yield_t_ha": Column(
            float,
            nullable=False,
            coerce=True,
            checks=Check.ge(0),
        ),
        # Conditions climatiques et intrants
        "rainfall_mm": Column(
            float,
            nullable=True,
            required=False,
            coerce=True,
            checks=Check.ge(0),
        ),
        "avg_temp_c": Column(
            float,
            nullable=True,
            required=False,
            coerce=True,
            checks=Check.in_range(-50, 60),
        ),
        "pesticides_tonnes": Column(
            float,
            nullable=True,
            required=False,
            coerce=True,
            checks=Check.ge(0),
        ),
        # Proxys agronomiques
        "crop_irrigation_rate": Column(
            float,
            nullable=True,
            required=False,
            coerce=True,
            checks=Check.in_range(0, 1),
        ),
        "crop_fertilizer_rate": Column(
            float,
            nullable=True,
            required=False,
            coerce=True,
            checks=Check.in_range(0, 1),
        ),
        "crop_mean_days_to_harvest": Column(
            float,
            nullable=True,
            required=False,
            coerce=True,
            checks=Check.gt(0),
        ),
        "crop_mean_rainfall_local_mm": Column(
            float,
            nullable=True,
            required=False,
            coerce=True,
            checks=Check.ge(0),
        ),
        "crop_mean_temperature_local_c": Column(
            float,
            nullable=True,
            required=False,
            coerce=True,
            checks=Check.in_range(-50, 60),
        ),
        "has_agronomic_proxy": Column(
            int,
            nullable=True,
            required=False,
            coerce=True,
            checks=Check.isin([0, 1]),
        ),
    },
    strict=False,
)


def validate_agritech_data(df):
    """
    Valide le dataset consolide Agritech avec Pandera.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset issu de l'etape 1.

    Returns
    -------
    pandas.DataFrame
        Dataset valide et coerce.

    Raises
    ------
    pandera.errors.SchemaError
        Si une colonne obligatoire, un type ou une regle metier est invalide.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Colonnes obligatoires manquantes : " + ", ".join(missing))

    return agritech_schema.validate(df)


def validate_parcel_input(data):
    """
    Valide une requete de prediction ou de recommandation.

    Parameters
    ----------
    data : dict
        Conditions de parcelle : country, year, rainfall_mm,
        avg_temp_c et pesticides_tonnes.

    Returns
    -------
    dict
        Donnees validees.
    """
    parcel_schema = DataFrameSchema(
        {
            "country": Column(
                str, nullable=False, coerce=True, checks=Check.str_length(min_value=1)
            ),
            "year": Column(
                int, nullable=False, coerce=True, checks=Check.in_range(1900, 2100)
            ),
            "rainfall_mm": Column(
                float, nullable=False, coerce=True, checks=Check.ge(0)
            ),
            "avg_temp_c": Column(
                float, nullable=False, coerce=True, checks=Check.in_range(-50, 60)
            ),
            "pesticides_tonnes": Column(
                float, nullable=False, coerce=True, checks=Check.ge(0)
            ),
        },
        strict=True,
    )

    validated = parcel_schema.validate(pd.DataFrame([data]))
    return validated.iloc[0].to_dict()


if __name__ == "__main__":
    import pandas as pd

    sample = pd.DataFrame(
        [
            {
                "country": "France",
                "year": 2013,
                "crop": "Wheat",
                "yield_t_ha": 6.5,
                "rainfall_mm": 800.0,
                "avg_temp_c": 12.0,
                "pesticides_tonnes": 5000.0,
                "crop_irrigation_rate": 0.5,
                "crop_fertilizer_rate": 0.5,
                "crop_mean_days_to_harvest": 120.0,
                "crop_mean_rainfall_local_mm": 550.0,
                "crop_mean_temperature_local_c": 27.5,
                "has_agronomic_proxy": 1,
            }
        ]
    )

    validated = validate_agritech_data(sample)
    print("Dataset valide :", validated.shape)

    parcel = validate_parcel_input(
        {
            "country": "France",
            "year": 2013,
            "rainfall_mm": 800,
            "avg_temp_c": 12,
            "pesticides_tonnes": 5000,
        }
    )
    print("Requete parcelle valide :", parcel)
