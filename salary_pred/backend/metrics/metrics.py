import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Union
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series, list]

def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """
    Calcule les métriques de régression standard (MAE, MSE, RMSE, R2).
    Args:
        y_true: valeurs réelles
        y_pred: valeurs prédites
    Returns:
        dict: métriques {"mae", "mse", "rmse", "r2"}
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }