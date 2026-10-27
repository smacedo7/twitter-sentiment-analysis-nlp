import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app, get_model_path  # noqa: E402


@pytest.fixture
def client(model_path):
    app.dependency_overrides[get_model_path] = lambda: model_path
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def client_without_model(tmp_path):
    app.dependency_overrides[get_model_path] = lambda: tmp_path / "missing.pkl"
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_with_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_without_model(client_without_model):
    response = client_without_model.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_predict_positive_and_negative(client):
    positive = client.post("/predict", json={"text": "I love this song, what a wonderful day!"})
    negative = client.post("/predict", json={"text": "I hate this, the worst service ever"})

    assert positive.status_code == 200
    assert positive.json()["label"] == "Positive"
    assert 0.5 <= positive.json()["confidence"] <= 1.0
    assert negative.json()["label"] == "Negative"


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "a" * 1001}, {"text": 123}])
def test_predict_rejects_invalid_payloads(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_text_without_analyzable_words(client):
    response = client.post("/predict", json={"text": "@user http://t.co/abc !!!"})

    assert response.status_code == 422
    assert "no analyzable words" in response.json()["detail"]


def test_predict_returns_503_when_model_is_missing(client_without_model):
    response = client_without_model.post("/predict", json={"text": "I love this"})

    assert response.status_code == 503
    assert "python -m src.train" in response.json()["detail"]


def test_predict_batch(client):
    response = client.post("/predict/batch", json={"texts": ["I love this", "@user http://t.co/abc", "I hate this"]})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [p["label"] for p in predictions] == ["Positive", None, "Negative"]
    assert predictions[1]["confidence"] is None


@pytest.mark.parametrize("payload", [{"texts": []}, {"texts": ["ok"] * 101}, {"texts": "not a list"}])
def test_predict_batch_rejects_invalid_payloads(client, payload):
    assert client.post("/predict/batch", json=payload).status_code == 422


def test_predict_batch_returns_503_when_model_is_missing(client_without_model):
    response = client_without_model.post("/predict/batch", json={"texts": ["I love this"]})

    assert response.status_code == 503
