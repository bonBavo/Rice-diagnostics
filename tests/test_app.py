"""Tests for the Rice Leaf Doctor Flask application."""
import io
import pytest
from PIL import Image

from app import create_app
@pytest.fixture()
def app(tmp_path):
    """Create the app with a temporary database and upload folder."""
    app = create_app({
        "TESTING": True,
        "DB_PATH": str(tmp_path / "test.db"),
        "UPLOAD_DIR": tmp_path / "uploads",
    })
    (tmp_path / "uploads").mkdir(exist_ok=True)
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


def _make_jpeg():
    """Create a small valid JPEG in memory."""
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color="green").save(buf, "JPEG")
    buf.seek(0)
    return buf


# ------------------------------------------------------------------
# 1. App starts
# ------------------------------------------------------------------
def test_app_creates(app):
    assert app is not None


# ------------------------------------------------------------------
# 2. /health works
# ------------------------------------------------------------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] in ("ok", "model_not_loaded")
    if client.application.config["PREDICTOR"] is not None:
        assert data["classes"] == client.application.config["PREDICTOR"].class_names


# ------------------------------------------------------------------
# 3. /api/v1/predict rejects no file
# ------------------------------------------------------------------
def test_predict_no_file(client):
    r = client.post("/api/v1/predict")
    assert r.status_code == 400
    data = r.get_json()
    assert data["success"] is False
    assert "No image" in data["error"]


# ------------------------------------------------------------------
# 4. Unsupported file type rejected
# ------------------------------------------------------------------
def test_predict_bad_file(client):
    data = {"image": (io.BytesIO(b"not an image"), "test.txt")}
    r = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json()["success"] is False


# ------------------------------------------------------------------
# 5. Valid image reaches predictor and returns prediction
# ------------------------------------------------------------------
def test_predict_valid_image(client):
    """A valid JPEG should return a prediction (requires trained model)."""
    buf = _make_jpeg()
    data = {"image": (buf, "leaf.jpg")}
    r = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")
    if client.application.config["PREDICTOR"] is not None:
        assert r.status_code == 200
        body = r.get_json()
        assert body["success"] is True
        assert body["status"] in ("confident", "uncertain")
        if body["status"] == "confident":
            assert body["prediction"] in client.application.config["PREDICTOR"].class_names
        else:
            assert body["prediction"] == "unsure"
        assert 0.0 <= body["confidence"] <= 1.0
    else:
        assert r.status_code == 503


# ------------------------------------------------------------------
# 6. Prediction response has expected fields
# ------------------------------------------------------------------
def test_predict_response_fields(client):
    buf = _make_jpeg()
    data = {"image": (buf, "leaf.jpg")}
    r = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")
    body = r.get_json()
    assert "success" in body
    if body["success"]:
        assert isinstance(body["confidence"], float)
        assert body["status"] in ("confident", "uncertain")
        if body["status"] == "confident":
            assert body["prediction"] in client.application.config["PREDICTOR"].class_names
        else:
            assert body["prediction"] == "unsure"
    else:
        assert r.status_code == 503


def test_uncertain_image_returns_no_disease_guess(client, monkeypatch):
    class UncertainPredictor:
        def predict(self, _path, top_k, threshold):
            assert top_k == 3
            assert threshold == client.application.config["CONFIDENCE_THRESHOLD"]
            return {
                "prediction": "blast",
                "confidence": 0.42,
                "status": "uncertain",
                "top_k": [{"label": "blast", "confidence": 0.42}],
            }

    monkeypatch.setitem(client.application.config, "PREDICTOR", UncertainPredictor())
    response = client.post(
        "/api/v1/predict",
        data={"image": (_make_jpeg(), "non_leaf.jpg")},
        content_type="multipart/form-data",
    )
    body = response.get_json()

    assert response.status_code == 200
    assert body["success"] is True
    assert body["prediction"] == "unsure"
    assert body["status"] == "uncertain"
    assert "may not be a rice plant or leaf" in body["message"]


# ------------------------------------------------------------------
# 7. Index page loads
# ------------------------------------------------------------------
def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Rice Leaf Doctor" in r.data


# ------------------------------------------------------------------
# 8. History page loads
# ------------------------------------------------------------------
def test_history(client):
    r = client.get("/history")
    assert r.status_code == 200
