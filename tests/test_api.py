from src.api import app


PARCEL_FEATURES = {
    "country": "France",
    "year": 2020,
    "rainfall_mm": 800,
    "avg_temp_c": 15,
    "pesticides_tonnes": 1000,
}


def test_app_exists():
    assert app is not None
    assert app.title == "Agritech Yield API"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_estimated_yield(client):
    response = client.post(
        "/predict",
        json={"features": {**PARCEL_FEATURES, "crop": "Maize"}},
    )
    assert response.status_code == 200
    assert set(response.json()) == {
        "parcel_id",
        "crop",
        "estimated_yield_t_ha",
        "model_version",
    }
    assert response.json()["crop"] == "Maize"
    assert isinstance(response.json()["estimated_yield_t_ha"], float)
    assert response.json()["parcel_id"] > 0


def test_predict_accepts_flat_request(client):
    response = client.post(
        "/predict",
        json={**PARCEL_FEATURES, "crop": "Maize"},
    )
    assert response.status_code == 200


def test_predict_requires_crop(client):
    response = client.post("/predict", json={"features": PARCEL_FEATURES})
    assert response.status_code == 422
    assert "crop" in response.json()["detail"]


def test_predict_rejects_crop_not_present_in_training_data(client):
    response = client.post(
        "/predict",
        json={"features": {**PARCEL_FEATURES, "crop": "Unknown crop"}},
    )
    assert response.status_code == 422
    assert "Cultures disponibles" in response.json()["detail"]


def test_predict_requires_parcel_conditions(client):
    response = client.post("/predict", json={"features": {"crop": "Maize"}})
    assert response.status_code == 422
    assert any(error["loc"][-1] == "country" for error in response.json()["detail"])


def test_predict_rejects_non_numeric_measurement(client):
    response = client.post(
        "/predict",
        json={
            "features": {
                **PARCEL_FEATURES,
                "crop": "Maize",
                "rainfall_mm": "pluie",
            }
        },
    )
    assert response.status_code == 422
    assert any(error["loc"][-1] == "rainfall_mm" for error in response.json()["detail"])


def test_recommend_ranks_all_crops_by_estimated_yield(client):
    response = client.post("/recommend", json={"features": PARCEL_FEATURES})
    assert response.status_code == 200
    recommendations = response.json()["recommendations"]
    assert len(recommendations) > 1
    assert [item["rank"] for item in recommendations] == list(
        range(1, len(recommendations) + 1)
    )
    yields = [item["estimated_yield_t_ha"] for item in recommendations]
    assert yields == sorted(yields, reverse=True)
    assert all(item["crop"] for item in recommendations)
    assert sum(item["recommended"] for item in recommendations) == 1
    assert response.json()["parcel_id"] > 0


def test_recommend_rejects_crop_input(client):
    response = client.post(
        "/recommend",
        json={"features": {**PARCEL_FEATURES, "crop": "Maize"}},
    )
    assert response.status_code == 422
    assert "Ne fournissez pas 'crop'" in response.json()["detail"]
