from pydantic import BaseModel, Field
from typing import List, Optional, Any

# schemas regression
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
    
# Schemas Image
class TrainImageRequest(BaseModel):
    root_dir: str = "data/raw/"
    epochs: int = 5
    backbone: str = "resnet18"
    batch_size: int = 16
    model_type: Optional[str] = None
    mode: str = "balanced"

class PredictImageRequest(BaseModel):
    backbone: str = "resnet18"
    model_type: Optional[str] = None

class TrainImageResponse(BaseModel):
    metrics: Any
    
class TrainImageResponseStep(BaseModel):
    status: str
    metrics: Any
    epoch: int
    
class PredictionItem(BaseModel):
    class_: int = Field(..., alias="class")
    probs: List[float]
    
class PredictImageResponse(BaseModel):
    predictions: List[PredictionItem]