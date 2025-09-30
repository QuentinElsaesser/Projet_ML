from backend.schemas.metrics_schemas import RegressionMetrics, ClassificationMetrics
from backend.metrics.metrics import regression_metrics, classification_metrics

from typing import Dict, Union
import pandas as pd
import numpy as np

ArrayLike = Union[np.ndarray, pd.Series, list]

def evaluate_regression(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    return RegressionMetrics(**regression_metrics(y_true=y_true, y_pred=y_pred))

def evaluate_classification(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    return ClassificationMetrics(**classification_metrics(y_true=y_true, y_pred=y_pred))