import os
import joblib
import numpy as np
from typing import Optional, Dict

from backend.models.basic_model import BasicModel
from backend.metrics.metrics_eval import evaluate_regression

class LinearRegressionModel(BasicModel):
    def __init__(self):
        """
        Régression linéaire multiple (n variables explicatives).
        Utilise la solution analytique (moindres carrés ordinaires).
        
        self.model stocke les parametres ici
        """
        super().__init__()
        self.model: Optional[dict] = None

    def train(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Entraîne le modèle en calculant les coefficients via la formule fermée :
            theta = (X^T X)^(-1) X^T y

        Args:
            x (np.ndarray): matrice des features (shape = (n_samples, n_features))
            y (np.ndarray): vecteur des labels (shape = (n_samples,))
        """
        # Ajout d'une colonne de 1 pour représenter l'intercept
        X_b = np.c_[np.ones((x.shape[0], 1)), x]
        # Conversion en vecteur colonne
        y = y.reshape(-1, 1)
        # Formule analytique
        theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.model = {"theta": theta}
        self.is_trained = True
        
        y_pred = self.predict(x)
        return evaluate_regression(y_true=y, y_pred=y_pred)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Prédit les valeurs cibles pour de nouvelles données.
        Args:
            x (np.ndarray): matrice des features (shape = (n_samples, n_features))
        Returns:
            np.ndarray: prédictions (shape = (n_samples,))
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Le modèle doit être entraîné avant de prédire.")
        theta = self.model["theta"]
        X_b = np.c_[np.ones((x.shape[0], 1)), x]
        # revel plus rapide que flatten et mieux en memoire
        return X_b.dot(theta).ravel()

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Évalue le modèle avec des métriques standard de régression.
        """
        y_pred = self.predict(x)
        return evaluate_regression(y_true=y, y_pred=y_pred)
    
    def save(self, path: str):
        """
        Sauvegarde du modèle entraîné (paramètres theta).
        """
        if not self.is_trained:
            raise ValueError("Impossible de sauvegarder : le modèle n’est pas entraîné.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        """
        Chargement du modèle (paramètres theta).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modèle introuvable: {path}")
        self.model = joblib.load(path)
        self.is_trained = True
