from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SPAM_EXAMPLE = (
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. "
    "Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
)
HAM_EXAMPLE = (
    "Go until jurong point, crazy.. Available only in bugis n great world "
    "la e buffet... Cine there got amore wat..."
)


class TestPredictRoute:
    def test_spam_message(self):
        response = client.post("/predict", json={"message": SPAM_EXAMPLE})
        assert response.status_code == 200

        body = response.json()
        assert body["prediction"] == "spam"
        assert 0 <= body["confidence"] <= 100

    def test_ham_message(self):
        response = client.post("/predict", json={"message": HAM_EXAMPLE})
        assert response.status_code == 200

        body = response.json()
        assert body["prediction"] == "ham"
        assert 0 <= body["confidence"] <= 100

    def test_predict_rejects_missing_field(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_rejects_empty_message(self):
        response = client.post("/predict", json={"message": ""})
        assert response.status_code == 422


class TestModelInfoRoute:
    def test_model_info_returns_expected_shape(self):
        response = client.get("/model-info")
        assert response.status_code == 200

        body = response.json()
        assert "algorithm" in body
        assert "accuracy" in body
        assert 0 <= body["accuracy"] <= 1


class TestHealthRoute:
    def test_health_when_model_loaded(self):
        response = client.get("/health")
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["model_algorithm"] is not None
