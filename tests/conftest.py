import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
import src.api as api_module


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    test_database = tmp_path / "agritech.db"
    shutil.copy2(api_module.DATABASE_PATH, test_database)
    monkeypatch.setattr(api_module, "DATABASE_PATH", test_database)

    with TestClient(api_module.app) as test_client:
        yield test_client


#
# Cla permet à tous les tests de faire simplement :
#
# def test_predict(client):
#  response = client.post(...)

# s #ans recréer le TestClient à chaque fois.
