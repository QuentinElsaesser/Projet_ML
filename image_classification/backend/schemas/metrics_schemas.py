from pydantic import BaseModel

class RegressionMetrics(BaseModel):
    mae: float
    mse: float
    rmse: float
    r2: float

class ClassificationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int