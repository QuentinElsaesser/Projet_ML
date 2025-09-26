from pydantic import BaseModel

class RegressionMetrics(BaseModel):
    mae: float
    mse: float
    rmse: float
    r2: float
