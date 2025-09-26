from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class BasicModel(ABC):
    def __init__(self):
        """self.model peut être un modèle custom ou sklearn"""
        self.model = None  
        self.is_trained = False

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Entraîne le modèle"""
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Fait des prédictions"""
        pass

    @abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Évalue le modèle"""
        pass

    @abstractmethod
    def save(self, path: str):
        """Sauvegarde le modèle"""
        pass

    @abstractmethod
    def load(self, path: str):
        """Charge le modèle"""
        pass
