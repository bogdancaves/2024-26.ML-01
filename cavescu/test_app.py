import pytest
from cavescu.app import app as flask_app


@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client


def test_hello(client):
    response = client.post(
        "/infer",
        json={
            "data": {
                "Date": "2021/01/04",
                "Hour": "09:00",
                "high": 1.22864,
                "low": 1.22619,
                "Volume": 7418,
                "ATR": 0.001945978492279958,
                "Range": 0.0017499999999999183,
                "Range_vs_ATR": "Below",
                "Currency": "EUR",
                "Impact": "L",
                "Event": "Spanish Manufacturing PMI",
                "Actual": 51.0,
                "Forecast": 52.6,
                "Previous": 49.8,
                "Actual_vs_Forecast": "worse",
            }
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["result"]["value"] in ['Long', 'Short']
