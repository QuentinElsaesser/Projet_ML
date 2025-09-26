from backend.schemas.regression_metrics_schemas import RegressionMetrics
from backend.metrics.metrics import regression_metrics

from typing import Dict, Union
import pandas as pd
import numpy as np

ArrayLike = Union[np.ndarray, pd.Series, list]

def evaluate_regression(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    return RegressionMetrics(**regression_metrics(y_true=y_true, y_pred=y_pred))
