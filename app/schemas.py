from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):
    """Input schema — one row of employee features."""

    Age:                      int   = Field(..., ge=18,  le=65)
    BusinessTravel:           int   = Field(..., ge=0,   le=2)
    DailyRate:                int   = Field(..., gt=0)
    Department:               int   = Field(..., ge=0,   le=2)
    DistanceFromHome:         int   = Field(..., ge=0)
    Education:                int   = Field(..., ge=1,   le=5)
    EducationField:           int   = Field(..., ge=0,   le=5)
    EnvironmentSatisfaction:  int   = Field(..., ge=1,   le=4)
    Gender:                   int   = Field(..., ge=0,   le=1)
    HourlyRate:               int   = Field(..., gt=0)
    JobInvolvement:           int   = Field(..., ge=1,   le=4)
    JobLevel:                 int   = Field(..., ge=1,   le=5)
    JobRole:                  int   = Field(..., ge=0,   le=8)
    JobSatisfaction:          int   = Field(..., ge=1,   le=4)
    MaritalStatus:            int   = Field(..., ge=0,   le=2)
    MonthlyIncome:            float = Field(..., gt=0)
    MonthlyRate:              int   = Field(..., gt=0)
    NumCompaniesWorked:       int   = Field(..., ge=0)
    OverTime:                 int   = Field(..., ge=0,   le=1)
    PercentSalaryHike:        int   = Field(..., ge=0)
    PerformanceRating:        int   = Field(..., ge=1,   le=4)
    RelationshipSatisfaction: int   = Field(..., ge=1,   le=4)
    StockOptionLevel:         int   = Field(..., ge=0,   le=3)
    TotalWorkingYears:        int   = Field(..., ge=0)
    TrainingTimesLastYear:    int   = Field(..., ge=0)
    WorkLifeBalance:          int   = Field(..., ge=1,   le=4)
    YearsAtCompany:           int   = Field(..., ge=0)
    YearsInCurrentRole:       int   = Field(..., ge=0)
    YearsSinceLastPromotion:  int   = Field(..., ge=0)
    YearsWithCurrManager:     int   = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionOutput(BaseModel):
    """Output schema — prediction result."""
    attrition_risk: str    # "High" or "Low"
    probability:    float  # 0.0 to 1.0
    model_source:   str    # where the model was loaded from