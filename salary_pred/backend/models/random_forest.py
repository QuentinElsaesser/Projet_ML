import os
import joblib
import numpy as np
from typing import Dict
from sklearn.ensemble import RandomForestRegressor

from backend.models.basic_model import BasicModel
from backend.metrics.metrics_eval import evaluate_regression

class RandomForestModel(BasicModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestRegressor(**kwargs)

    def train(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self.model.fit(x, y)
        self.is_trained = True
        y_pred = self.predict(x)
        return evaluate_regression(y_true=y, y_pred=y_pred)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de prédire.")
        return self.model.predict(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(x)
        return evaluate_regression(y_true=y, y_pred=y_pred)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modèle introuvable: {path}")
        self.model = joblib.load(path)
        self.is_trained = True
