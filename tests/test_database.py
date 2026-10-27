import sqlite3
from pathlib import Path

import pytest


DATABASE_PATH = Path(__file__).resolve().parents[1] / "agritech.db"

REQUIRED_TABLE_COLUMNS = {
    "parcel_conditions": {
        "parcel_id",
        "country",
        "year",
        "rainfall_mm",
        "avg_temp_c",
        "pesticides_tonnes",
        "created_at",
    },
    "crop_yield_history": {
        "country",
        "year",
        "crop",
        "yield_t_ha",
    },
    "crop_reference": {"crop"},
    "yield_predictions": {
        "prediction_id",
        "parcel_id",
        "crop",
        "predicted_yield_t_ha",
        "model_version",
        "predicted_at",
    },
    "crop_recommendations": {
        "recommendation_id",
        "parcel_id",
        "crop",
        "predicted_yield_t_ha",
        "rank",
        "recommended",
        "created_at",
    },
    "model_metadata": {
        "model_name",
        "model_version",
        "target",
        "rmse",
        "mae",
        "r2",
    },
    "feature_importance": {"feature", "importance"},
}


@pytest.fixture(scope="module")
def db_connection():
    if not DATABASE_PATH.is_file():
        pytest.fail(f"Base Agritech introuvable : {DATABASE_PATH}")

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
    finally:
        conn.close()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        is not None
    )


def get_table_columns(conn: sqlite3.Connection, name: str) -> set[str]:
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{name}")')}


@pytest.mark.parametrize("table_name", REQUIRED_TABLE_COLUMNS)
def test_agritech_tables_exist(db_connection: sqlite3.Connection, table_name: str):
    assert table_exists(db_connection, table_name), (
        f"La table {table_name!r} est absente de {DATABASE_PATH.name}"
    )


@pytest.mark.parametrize(
    ("table_name", "required_columns"),
    REQUIRED_TABLE_COLUMNS.items(),
)
def test_agritech_tables_have_required_columns(
    db_connection: sqlite3.Connection,
    table_name: str,
    required_columns: set[str],
):
    actual_columns = get_table_columns(db_connection, table_name)
    assert required_columns <= actual_columns, (
        f"Colonnes manquantes dans {table_name!r} : "
        f"{sorted(required_columns - actual_columns)}"
    )


def display_database_info() -> None:
    if not DATABASE_PATH.is_file():
        print(f"Base Agritech introuvable : {DATABASE_PATH}")
        return

    with sqlite3.connect(DATABASE_PATH) as conn:
        table_names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        print("\n===== TABLES AGRITECH =====")
        for table_name in table_names:
            print(f"\nTABLE : {table_name}")
            for row in conn.execute(f'PRAGMA table_info("{table_name}")'):
                print(f"  {row[1]} ({row[2]})")


if __name__ == "__main__":
    display_database_info()
