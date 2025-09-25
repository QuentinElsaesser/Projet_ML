from pydantic import BaseModel
from typing import List, Dict, Any, Union

class TrainRequest(BaseModel):
    x: List[List[float]]
    y: List[float]

class PredictRequest(BaseModel):
    x: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[float]

class MetricsResponse(BaseModel):
    metrics: Any
    # metrics: Union[Dict[str, Any], BaseModel]
    
class EvaluateRequest(BaseModel):
    x: List[List[float]]
    y: List[float]

class EvaluateResponse(BaseModel):
    metrics: Any
    # metrics: Union[Dict[str, Any], BaseModel]