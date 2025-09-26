from fastapi import APIRouter, Request, UploadFile, File, Form
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from backend.schemas.api_schemas import (
    TrainRequest, PredictRequest, PredictResponse, 
    MetricsResponse, EvaluateRequest, EvaluateResponse
)

router = APIRouter(tags=["ML"])

@router.post("/train", response_model=MetricsResponse, 
             summary="Entraîne le modèle avec des données fournies directement")
async def train_model(request: Request, data: TrainRequest) -> Dict[str, Any]:
    """
    Entraîne le modèle stocké dans l'application à partir des données fournies
    dans le corps de la requête.
    Args:
        request (Request): Instance FastAPI pour accéder au modèle via 
        data (TrainRequest): Données d'entraînement (x et y)
    Returns:
        dict: métriques de performance calculées après entraînement
    """
    manager = request.app.state.model_manager
    x = np.array(data.x)
    y = np.array(data.y)
    metrics = manager.train_model(x, y, model_type=data.model_type)
    return {"metrics": metrics}

@router.post("/trainfile", response_model=MetricsResponse, 
             summary="Entraîne le modèle à partir d'un fichier CSV")
async def train_model_file(request: Request, model_type: Optional[str] = Form(None), file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Entraîne le modèle avec un fichier CSV.
    Le CSV doit contenir au minimum les colonnes 'x' et 'y'.
    Args:
        request (Request): Instance FastAPI
        file (UploadFile): Fichier CSV contenant les données
    Returns:
        dict: métriques de performance après entraînement
    """
    manager = request.app.state.model_manager
    df = pd.read_csv(file.file)
    x = np.array(df.drop(columns=['y']))
    y = np.array(df['y'])
    metrics = manager.train_model(x, y, model_type=model_type)
    return {"metrics": metrics}

@router.post("/predict", response_model=PredictResponse, 
             summary="Prédit des valeurs avec le modèle entraîné")
async def predict(request: Request, data: PredictRequest) -> Dict[str, List[float]]:
    """
    Fait des prédictions sur de nouvelles données.
    Args:
        request (Request): Instance FastAPI
        data (PredictRequest): Données pour la prédiction
    Returns:
        dict: liste des prédictions
    """
    manager = request.app.state.model_manager
    x = np.array(data.x)
    predictions = manager.predict(x, model_type=data.model_type)
    return {"predictions": predictions.tolist()}


@router.get("/metrics", response_model=MetricsResponse, 
            summary="Récupère le dernier résultat des métriques du modèle")
async def get_metrics(request: Request) -> Dict[str, Any]:
    """
    Retourne les métriques du dernier entraînement ou évaluation.
    Args:
        request (Request): Instance FastAPI
    Returns:
        dict: dictionnaire contenant les métriques
    """
    manager = request.app.state.model_manager
    if not manager.metrics_history:
        return {"metrics": {}, "warning": "Aucun résultat disponible"}
    return {"metrics": manager.metrics_history[-1]}

@router.get("/metrics/history", response_model=List[MetricsResponse], 
            summary="Récupère l'historique complet des métriques")
async def get_metrics_history(request: Request) -> List[Dict[str, Any]]:
    """
    Retourne l'historique de toutes les métriques obtenues lors des
    entraînements ou évaluations précédentes.
    Args:
        request (Request): Instance FastAPI
    Returns:
        List[dict]: Liste des métriques successives
    """
    manager = request.app.state.model_manager
    return [{"metrics": m} for m in manager.metrics_history]


@router.post("/evaluate", response_model=EvaluateResponse, 
             summary="Évalue le modèle sur des données fournies")
async def evaluate_model(request: Request, data: EvaluateRequest) -> Dict[str, Any]:
    """
    Évalue le modèle sur de nouvelles données et renvoie les métriques
    de performance.
    Args:
        request (Request): Instance FastAPI
        data (EvaluateRequest): Données pour l'évaluation
    Returns:
        dict: métriques d'évaluation
    """
    manager = request.app.state.model_manager
    x = np.array(data.x)
    y = np.array(data.y)
    metrics = manager.evaluate_model(x, y, model_type=data.model_type)
    return {"metrics": metrics}
