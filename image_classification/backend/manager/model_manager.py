import os
import numpy as np
from typing import Dict, Any, List

from backend.models.basic_model import BasicModel
from backend.models.model_factory import create_regressor
from backend.utils.config import config

class ModelManager:
    def __init__(self, model_type: str = None, model_path: str = None, **kwargs):
        """
        Manager de modèles ML générique.
        Args:
            model_type (str): Type de modèle (linear_regression, random_forest, ...)
                             Si None, prend dans la config.
            model_path (str): Chemin du modèle sauvegardé. Si None, prend dans la config.
            **kwargs: Paramètres à passer au constructeur du modèle.
        """
        self.available_models = config.model.available_model
        self.model_type = model_type or config.model.model_type
        self.model_path = model_path or config.model.model_path
        self.model_fullname = self.model_path + self.model_type
        self.kwargs = kwargs
        self.model: BasicModel = None
        self.train_loader = None # Tuple[DataLoader, Optional[DataLoader]]:
        self.val_loader = None # Tuple[DataLoader, Optional[DataLoader]]:
        self.current_epoch = 0
        self.metrics_history: List[Dict[str, Any]] = []  

    def set_model_type(self, model_type: str):
        """Change dynamiquement le type de modèle utilisé."""
        if model_type not in self.available_models:
            raise ValueError(f"Model {model_type} non supporté. "
                             f"Choisir parmi {self.available_models}")
        self.model_type = model_type
        self.model = None  # reset
        self.model_fullname = self.model_path + self.model_type

    def create_model(self, model_type: str = None, **kwargs) -> BasicModel:
        """Fabrique un modèle en fonction du type demandé."""
        model_type = model_type or self.model_type
        kwargs = kwargs or self.kwargs
        return create_regressor(model_type, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne des informations sur le modèle en cours."""
        if not self.model:
            return {"status": "not_trained"}
        model_type = type(self.model).__name__
        return {
            "status": "trained" if self.model.is_trained else "not_trained",
            "model_type": model_type
        }

    def train_model(self, x: np.ndarray, y: np.ndarray, model_type: str = None, **kwargs) -> Dict[str, float]:
        """Entraîne le modèle avec les données fournies."""
        # Ici on reset le modele car pas d'interet a reentrainer le modele
        # mais plus tard, il faut prendre ce cas en compte
        if model_type:
            self.set_model_type(model_type)
        self.model = self.create_model(model_type, **kwargs)
        metrics = self.model.train(x, y)
        self.metrics_history.append({
            "type":"train",
            "model":self.model_type,
            "metrics":metrics
        })
        self.save_model()
        return metrics

    def evaluate_model(self, x: np.ndarray, y: np.ndarray, model_type: str = None) -> Dict[str, float]:
        """Évalue le modèle avec des données de test."""
        if model_type:
            self.set_model_type(model_type)
        if not self.model:
            self.load_model()
        metrics = self.model.evaluate(x, y)
        self.metrics_history.append({
            "type": "eval",
            "model":self.model_type,
            "metrics": metrics
        })
        return self.model.evaluate(x, y)

    def predict(self, x: np.ndarray, model_type: str = None) -> np.ndarray:
        """Fait des prédictions avec le modèle."""
        if model_type:
            self.set_model_type(model_type)
        if not self.model:
            self.load_model()
        return self.model.predict(x)

    def save_model(self):
        """Sauvegarde le modèle courant."""
        if self.model:
            self.model.save(self.model_fullname + '.pt')

    def load_model(self):
        """Charge un modèle sauvegardé."""
        if os.path.exists(self.model_fullname + '.pt'):
            self.model = self.create_model(self.model_type)
            self.model.load(self.model_fullname + '.pt')
        else:
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_fullname + '.pt'}")
