from fastapi import APIRouter
from typing import Dict

from backend.utils.config import config

router = APIRouter(tags=["Root"])

@router.get("/", response_model=Dict[str, str], summary="Informations sur l'API")
async def root() -> Dict[str, str]:
    """
    Route principale pour récupérer les informations de l'application.
    Returns:
        dict: Contient le titre et la version de l'API
    """
    return {"message": config.server.title, "version": config.server.version}
