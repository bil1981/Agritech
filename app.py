import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = Path(
    os.getenv("AGRITECH_DB_PATH", str(PROJECT_ROOT / "agritech.db"))
).resolve()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
AGRONOMIC_FEATURES = (
    "crop_irrigation_rate",
    "crop_fertilizer_rate",
    "crop_mean_days_to_harvest",
    "crop_mean_rainfall_local_mm",
    "crop_mean_temperature_local_c",
    "has_agronomic_proxy",
)
MODEL_FEATURES = {
    "country",
    "year",
    "crop",
    "rainfall_mm",
    "avg_temp_c",
    "pesticides_tonnes",
    *AGRONOMIC_FEATURES,
}

st.set_page_config(
    page_title="Agritech Answers | Rendement agricole",
    page_icon="🌱",
    layout="wide",
)


@st.cache_data(ttl=300)
def load_reference_data(database_path: str) -> dict[str, Any]:
    with sqlite3.connect(database_path) as connection:
        countries = pd.read_sql_query(
            """
            SELECT DISTINCT country
            FROM crop_yield_history
            WHERE country IS NOT NULL AND country <> ''
            ORDER BY country
            """,
            connection,
        )["country"].tolist()
        crops = pd.read_sql_query(
            "SELECT * FROM crop_reference ORDER BY crop",
            connection,
        )
        year_range = pd.read_sql_query(
            "SELECT MIN(year) AS min_year, MAX(year) AS max_year FROM crop_yield_history",
            connection,
        ).iloc[0]
        metadata = pd.read_sql_query(
            "SELECT * FROM model_metadata ORDER BY id DESC LIMIT 1",
            connection,
        )
        feature_importance = pd.read_sql_query(
            "SELECT feature, importance FROM feature_importance ORDER BY importance DESC",
            connection,
        )

    if crops.empty:
        raise ValueError("La table crop_reference ne contient aucune culture.")
    if not countries:
        raise ValueError("La table crop_yield_history ne contient aucun pays.")
    if pd.isna(year_range["min_year"]) or pd.isna(year_range["max_year"]):
        raise ValueError("La table crop_yield_history ne contient aucune année.")

    return {
        "countries": countries,
        "crops": crops,
        "min_year": int(year_range["min_year"]),
        "max_year": int(year_range["max_year"]),
        "metadata": metadata.iloc[0].to_dict() if not metadata.empty else None,
        "feature_importance": feature_importance,
    }


def build_conditions(
    country: str,
    year: int,
    rainfall_mm: float,
    avg_temp_c: float,
    pesticides_tonnes: float,
) -> dict[str, Any]:
    return {
        "country": country,
        "year": year,
        "rainfall_mm": rainfall_mm,
        "avg_temp_c": avg_temp_c,
        "pesticides_tonnes": pesticides_tonnes,
    }


def request_api(endpoint: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}",
            json=payload,
            timeout=30,
        )
    except requests.exceptions.RequestException as error:
        st.error(
            f"API Agritech inaccessible ({API_URL}) : {error}. "
            "Démarrez-la avec `python -m uvicorn src.api:app --reload`."
        )
        return None

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        st.error(
            f"Réponse invalide de l'API (HTTP {response.status_code}) : "
            f"{response.text[:500]}"
        )
        return None

    if not response.ok:
        detail = result.get("detail", result) if isinstance(result, dict) else result
        st.error(f"Erreur API (HTTP {response.status_code}) : {detail}")
        return None
    if not isinstance(result, dict):
        st.error("L'API a retourné une réponse au format inattendu.")
        return None
    return result


try:
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            f"Base Agritech introuvable : {DATABASE_PATH}. "
            "Créez-la depuis la racine du projet avec "
            "`python .\\src\\build_agritech_database.py`."
        )
    reference_data = load_reference_data(str(DATABASE_PATH))
except (sqlite3.Error, OSError, ValueError) as error:
    st.error(f"Impossible de charger la base Agritech : {error}")
    st.stop()

st.title("🌱 Agritech Answers")
st.caption(
    "Estimez le rendement d'une culture ou comparez les cultures adaptées "
    "aux conditions de votre parcelle."
)

with st.sidebar:
    st.header("Conditions de la parcelle")
    st.caption("Ces conditions sont partagées par la prédiction et la recommandation.")

    selected_country = st.selectbox(
        "Pays",
        reference_data["countries"],
        index=(
            reference_data["countries"].index("France")
            if "France" in reference_data["countries"]
            else 0
        ),
    )
    selected_year = st.number_input(
        "Année",
        min_value=reference_data["min_year"],
        max_value=reference_data["max_year"],
        value=min(2013, reference_data["max_year"]),
        step=1,
        help=(
            "La plage disponible provient de l'historique de la base "
            f"({reference_data['min_year']}–{reference_data['max_year']})."
        ),
    )
    rainfall_mm = st.number_input(
        "Précipitations (mm)",
        min_value=0.0,
        value=800.0,
        step=25.0,
    )
    avg_temp_c = st.number_input(
        "Température moyenne (°C)",
        min_value=-20.0,
        max_value=60.0,
        value=15.0,
        step=0.5,
    )
    pesticides_tonnes = st.number_input(
        "Pesticides utilisés (tonnes)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
    )
    st.divider()
    st.caption(f"Base utilisée : `{DATABASE_PATH.name}`")
    st.caption(f"API : `{API_URL}`")

parcel_conditions = build_conditions(
    selected_country,
    int(selected_year),
    rainfall_mm,
    avg_temp_c,
    pesticides_tonnes,
)

prediction_tab, recommendation_tab, model_tab = st.tabs(
    ["Prédire un rendement", "Recommander une culture", "Données & modèle"]
)

with prediction_tab:
    st.subheader("Estimation pour une culture")
    st.write(
        "Choisissez une culture et obtenez son rendement estimé en tonnes par hectare."
    )

    crop_reference = reference_data["crops"]
    crop_names = crop_reference["crop"].dropna().astype(str).tolist()
    selected_crop = st.selectbox("Culture", crop_names, key="prediction_crop")

    if st.button("Estimer le rendement", type="primary", key="predict_yield"):
        crop_row = crop_reference.loc[crop_reference["crop"] == selected_crop].iloc[0]
        features = dict(parcel_conditions)
        features["crop"] = selected_crop

        with st.spinner("Calcul du rendement..."):
            result = request_api("predict", {"features": features})
        if result is not None:
            estimate = result.get("estimated_yield_t_ha")
            if not isinstance(estimate, (int, float)):
                st.error("La réponse de l'API ne contient pas de rendement valide.")
            else:
                left, right = st.columns([1, 2])
                with left:
                    st.metric("Rendement estimé", f"{estimate:.2f} t/ha")
                    if crop_row.get("has_agronomic_proxy", 0):
                        st.caption(
                            "Les informations agronomiques de la culture "
                            "proviennent du référentiel SQLite."
                        )
                    else:
                        st.info(
                            "Le référentiel ne contient pas de proxy agronomique "
                            "pour cette culture ; le modèle complète les valeurs "
                            "manquantes avec son prétraitement."
                        )
                with right:
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Condition": [
                                    "Pays",
                                    "Année",
                                    "Précipitations",
                                    "Température moyenne",
                                    "Pesticides utilisés",
                                ],
                                "Valeur": [
                                    selected_country,
                                    str(int(selected_year)),
                                    f"{rainfall_mm:g} mm",
                                    f"{avg_temp_c:g} °C",
                                    f"{pesticides_tonnes:g} tonnes",
                                ],
                            }
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

with recommendation_tab:
    st.subheader("Classement des cultures")
    st.write(
        "Le modèle simule le rendement de toutes les cultures du référentiel "
        "pour les conditions sélectionnées."
    )

    if st.button(
        "Comparer les cultures",
        type="primary",
        key="recommend_crops",
    ):
        with st.spinner("Comparaison des cultures..."):
            result = request_api("recommend", {"features": parcel_conditions})

        if result is not None:
            recommendations = result.get("recommendations")
            if not isinstance(recommendations, list) or not recommendations:
                st.error("L'API n'a retourné aucune recommandation.")
            else:
                ranking = pd.DataFrame(recommendations)
                required_columns = {
                    "rank",
                    "crop",
                    "estimated_yield_t_ha",
                }
                if not required_columns.issubset(ranking.columns):
                    st.error("Le format du classement retourné par l'API est invalide.")
                else:
                    ranking = ranking.sort_values("rank")
                    top_crop = ranking.iloc[0]
                    st.success(
                        f"Culture la mieux classée : **{top_crop['crop']}** "
                        f"— {top_crop['estimated_yield_t_ha']:.2f} t/ha estimées."
                    )
                    chart_data = ranking.sort_values(
                        "estimated_yield_t_ha",
                        ascending=True,
                    )
                    figure = px.bar(
                        chart_data,
                        x="estimated_yield_t_ha",
                        y="crop",
                        orientation="h",
                        labels={
                            "estimated_yield_t_ha": "Rendement estimé (t/ha)",
                            "crop": "Culture",
                        },
                        text_auto=".2f",
                        color="estimated_yield_t_ha",
                        color_continuous_scale="Greens",
                    )
                    figure.update_layout(
                        coloraxis_showscale=False,
                        yaxis={
                            "categoryorder": "array",
                            "categoryarray": chart_data["crop"],
                        },
                        margin={"l": 10, "r": 10, "t": 20, "b": 10},
                    )
                    st.plotly_chart(figure, use_container_width=True)
                    st.dataframe(
                        ranking.rename(
                            columns={
                                "rank": "Rang",
                                "crop": "Culture",
                                "estimated_yield_t_ha": "Rendement estimé (t/ha)",
                            }
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

with model_tab:
    st.subheader("Référentiel Agritech")
    st.write(
        f"La base `{DATABASE_PATH.name}` fournit les cultures, les pays, "
        "les plages historiques et les indicateurs du modèle."
    )

    metadata = reference_data["metadata"]
    if metadata is None:
        st.warning("Aucune métadonnée de modèle n'est enregistrée dans la base.")
    else:
        st.markdown(
            f"**Modèle :** {metadata.get('model_name') or 'Non renseigné'}  \n"
            f"**Version :** {metadata.get('model_version') or 'Non renseignée'}  \n"
            f"**Cible :** {metadata.get('target') or 'Non renseignée'}"
        )
        metric_columns = st.columns(4)
        for column, label, key, suffix in zip(
            metric_columns,
            ("RMSE", "MAE", "R²", "MAPE"),
            ("rmse", "mae", "r2", "mape_pct"),
            (" t/ha", " t/ha", "", " %"),
        ):
            value = metadata.get(key)
            column.metric(label, f"{value:.3f}{suffix}" if value is not None else "N/D")

    importance = reference_data["feature_importance"]
    relevant_importance = importance.loc[
        importance["feature"].isin(MODEL_FEATURES)
    ].head(15)
    if not relevant_importance.empty:
        st.subheader("Variables importantes du modèle")
        figure = px.bar(
            relevant_importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importance", "feature": "Variable"},
        )
        st.plotly_chart(figure, use_container_width=True)
    elif not importance.empty:
        st.info(
            "La table d'importance présente dans la base ne correspond pas "
            "aux variables du modèle de rendement ; elle n'est pas affichée."
        )

    st.subheader("Cultures disponibles")
    st.dataframe(
        reference_data["crops"].drop(columns=["crop_id"], errors="ignore"),
        hide_index=True,
        use_container_width=True,
    )
