from typing import Optional, Dict
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import os
import joblib

from backend.models.basic_model import BasicModel
from backend.metrics.metrics_eval import regression_metrics
from backend.utils.helpers import ArrayLike, my_to_numpy

class SKLearnModelWrapper(BasicModel):
    """
    Wrapper générique pour les estimateurs sklearn-like.
    Implémente train/predict/evaluate/save/load de façon standard.
    """

    def __init__(self, estimator: BaseEstimator, use_scaler: bool = True):
        """
        estimator: un estimateur sklearn-like (doit implémenter fit/predict)
        use_scaler: si True, on met un StandardScaler en pipeline (utile pour MLP, Ridge/Lasso)
        """
        self.is_trained: bool = False
        self.model: Optional[BaseEstimator] = None
        self._estimator = estimator
        self.use_scaler = use_scaler
        if self.use_scaler:
            self.model = make_pipeline(StandardScaler(), self._estimator)
        else:
            self.model = self._estimator

    def train(self, x: ArrayLike, y: ArrayLike) -> Dict[str, float]:
        X = my_to_numpy(x)
        y_arr = np.array(y).ravel()
        self.model.fit(X, y_arr)
        self.is_trained = True
        y_pred = self.predict(X)
        metrics = regression_metrics(y_true=y_arr, y_pred=y_pred)
        return metrics

    def predict(self, x: ArrayLike) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas entraîné.")
        X = my_to_numpy(x)
        preds = self.model.predict(X)
        return np.array(preds)

    def evaluate(self, x: ArrayLike, y: ArrayLike) -> Dict[str, float]:
        X = my_to_numpy(x)
        y_arr = np.array(y).ravel()
        preds = self.predict(X)
        return regression_metrics(y_true=y_arr, y_pred=preds)

    def save(self, path: str) -> None:
        if not self.is_trained or self.model is None:
            raise ValueError("Rien à sauvegarder : modèle non entraîné.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable: {path}")
        self.model = joblib.load(path)
        self.is_trained = True