import requests
from typing import List, Dict, Any, Optional

from streamlit.runtime.uploaded_file_manager import UploadedFile

from backend.utils.config import config
from backend.utils.helpers import ArrayLike

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
    
    def _post(self, endpoint: str, data: Any = None, files: Any = None, timeout: int = 5) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        if files:
            response = self.session.post(url, data=data, files=files, timeout=timeout)
        else:
            response = self.session.post(url, json=data, files=files, timeout=timeout)
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
    
    def train_model(self, x: ArrayLike, y: ArrayLike, model_type: str) -> Dict[str, Any]:
        """
        Entraîne le modèle sur des données fournies.
        """
        x_list = x.tolist()
        y_list = y.tolist()
        return self._post("/ml/train", data={"x": x_list, "y": y_list, "model_type": model_type})["metrics"]

    def train_model_file(self, file_path: UploadedFile, model_type: str) -> Dict[str, Any]:
        """
        Entraîne le modèle à partir d'un fichier CSV.
        """
        return self._post("/ml/trainfile", data={"model_type": model_type}, files={"file": file_path})["metrics"]

    def predict(self, x: ArrayLike, model_type: str) -> List[float]:
        """
        Prédit des valeurs pour de nouvelles données.
        """
        x_list = x.tolist()
        return self._post("/ml/predict", data={"x": x_list, "model_type": model_type})["predictions"]

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

    def evaluate(self, x: ArrayLike, y: ArrayLike, model_type: str) -> Dict[str, Any]:
        """
        Évalue le modèle sur de nouvelles données.
        """
        x_list = x.tolist()
        y_list = y.tolist()
        return self._post("/ml/evaluate", data={"x": x_list, "y": y_list, "model_type": model_type})["metrics"]

    def train_image(self, root_dir: str, epochs: int = 5, backbone: str = "resnet18", batch_size: int = 16, mode: str = "balanced") -> Dict[str, Any]:
        """
        """
        data = {"root_dir": root_dir, "epochs": epochs, "backbone": backbone, "batch_size": batch_size, "mode": mode}
        return self._post("/ml/train_image", data=data, timeout=600)["metrics"]

    def predict_image(self, uploaded_file: UploadedFile, backbone: str = "resnet18") -> Dict[str, Any]:
        """
        """
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        data = {"backbone": backbone}
        return self._post("/ml/predict_image", data=data, files=files, timeout=30)["predictions"]

    def train_image_step(self) -> Dict[str, Any]:
        """
        Lance une epoch de training image.
        """
        return self._post("/ml/train_step", timeout=600)
    
    def train_image_init(self, root_dir: str, epochs: int = 5, backbone: str = "resnet18", batch_size: int = 16, mode: str = "balanced") -> Dict[str, Any]:
        """
        Initialise le modèle image et prépare les dataloaders.
        """
        data = {
            "root_dir": root_dir,
            "epochs": epochs,
            "backbone": backbone,
            "batch_size": batch_size,
            "mode": mode
        }
        return self._post("/ml/train_image_init", data=data, timeout=600)