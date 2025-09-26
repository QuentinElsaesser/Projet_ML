from fastapi import APIRouter, Request
from typing import Dict, Any

router = APIRouter(tags=["Health"])

@router.get("/health", summary="Santé de l'API / état du modèle", response_model=Dict[str, Any])
async def health(request: Request) -> Dict[str, Any]:
    """
    Route pour vérifier la santé de l'application et obtenir des
    informations sur le modèle chargé.
    Args:
        request (Request): Instance FastAPI pour accéder au model_manager
    Returns:
        dict: Informations sur le modèle et son état
    """
    manager = request.app.state.model_manager
    return manager.get_model_info()
