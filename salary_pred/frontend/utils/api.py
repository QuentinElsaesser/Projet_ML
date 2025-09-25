import requests
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

from streamlit.runtime.uploaded_file_manager import UploadedFile

from backend.utils.config import config
from backend.utils.logger import smartlog

ArrayLike = Union[np.ndarray, pd.Series, List[List[float]]]

class FastAPIClient:
    """
    Client pour interagir avec le backend FastAPI.
    """
    def __init__(self, port: str = "8000", host: str = "ml_service"):
        """
        Args:
            base_url (str): URL de base de l'API
        """
        host = config.api.dockerhost or host
        port = config.api.port or port 
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
    
    def _post(self, endpoint: str, data: Any = None, files: Any = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, json=data, files=files, timeout=5)
        response.raise_for_status()
        return response.json()
    
    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params, timeout=5)
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, Any]:
        """Retourne la santé du backend et du modèle"""
        return self._get("/health")
    
    def train_model(self, x: ArrayLike, y: ArrayLike) -> Dict[str, Any]:
        """
        Entraîne le modèle sur des données fournies.
        """
        x_list = x.tolist()
        y_list = y.tolist()
        return self._post("/ml/train", data={"x": x_list, "y": y_list})["metrics"]

    def train_model_file(self, file_path: UploadedFile) -> Dict[str, Any]:
        """
        Entraîne le modèle à partir d'un fichier CSV.
        """
        return self._post("/ml/trainfile", files={"file": file_path})["metrics"]

    def predict(self, x: ArrayLike) -> List[float]:
        """
        Prédit des valeurs pour de nouvelles données.
        """
        x_list = x.tolist()
        return self._post("/ml/predict", data={"x": x_list})["predictions"]

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne le dernier résultat des métriques.
        """
        return self._get("/ml/metrics")["metrics"]

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Retourne l'historique complet des métriques.
        """
        return self._get("/ml/metrics/history")

    def evaluate(self, x: ArrayLike, y: ArrayLike) -> Dict[str, Any]:
        """
        Évalue le modèle sur de nouvelles données.
        """
        x_list = x.tolist()
        y_list = y.tolist()
        return self._post("/ml/evaluate", data={"x": x_list, "y": y_list})["metrics"]
