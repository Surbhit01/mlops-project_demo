# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Health check ───────────────────────────────────────────────────
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ── Predict local — valid input ────────────────────────────────────
def test_predict_local_valid_input():
    payload = {
        "Age": 35,
        "BusinessTravel": 1,
        "DailyRate": 800,
        "Department": 1,
        "DistanceFromHome": 10,
        "Education": 3,
        "EducationField": 0,
        "EnvironmentSatisfaction": 2,
        "Gender": 1,
        "HourlyRate": 60,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": 1,
        "JobSatisfaction": 2,
        "MaritalStatus": 1,
        "MonthlyIncome": 5000,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 3,
        "OverTime": 1,
        "PercentSalaryHike": 13,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 2,
        "YearsAtCompany": 3,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2
    }
    response = client.post("/predict/local", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "attrition_risk" in result
    assert "probability" in result
    assert "model_source" in result
    assert result["attrition_risk"] in ["High", "Low"]
    assert 0.0 <= result["probability"] <= 1.0


# ── Predict local — invalid input triggers 422 ─────────────────────
def test_predict_local_invalid_input():
    payload = {"Age": "not a number", "MonthlyIncome": 5000}
    response = client.post("/predict/local", json=payload)
    assert response.status_code == 422


# ── Predict local — missing fields triggers 422 ────────────────────
def test_predict_local_missing_fields():
    payload = {"Age": 35}
    response = client.post("/predict/local", json=payload)
    assert response.status_code == 422