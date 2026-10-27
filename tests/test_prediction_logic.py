import pytest


PARCEL_FEATURES = {
    "country": "France",
    "year": 2020,
    "rainfall_mm": 800,
    "avg_temp_c": 15,
    "pesticides_tonnes": 1000,
}


@pytest.mark.parametrize("crop", ["Maize", "Wheat", "Rice, Paddy"])
def test_prediction_is_finite_for_supported_crops(client, crop):
    response = client.post(
        "/predict",
        json={"features": {**PARCEL_FEATURES, "crop": crop}},
    )
    assert response.status_code == 200
    assert response.json()["estimated_yield_t_ha"] >= 0


def test_recommendations_are_descending(client):
    response = client.post("/recommend", json={"features": PARCEL_FEATURES})
    assert response.status_code == 200
    estimates = [
        item["estimated_yield_t_ha"] for item in response.json()["recommendations"]
    ]
    assert estimates == sorted(estimates, reverse=True)
