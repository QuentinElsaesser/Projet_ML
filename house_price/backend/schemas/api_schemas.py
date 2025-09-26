from pydantic import BaseModel
from typing import List, Optional, Any

class TrainRequest(BaseModel):
    x: List[List[float]]
    y: List[float]
    model_type: Optional[str] = None

class PredictRequest(BaseModel):
    x: List[List[float]]
    model_type: Optional[str] = None

class PredictResponse(BaseModel):
    predictions: List[float]

class MetricsResponse(BaseModel):
    metrics: Any
    
class EvaluateRequest(BaseModel):
    x: List[List[float]]
    y: List[float]
    model_type: Optional[str] = None

class EvaluateResponse(BaseModel):
    metrics: Any
