from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, List

from backend.utils.logger import smartlog
from backend.manager.model_manager import ModelManager
from backend.utils.config import config
            
class FastAPIapp:
    def __init__(self, title: str, version: str):
        self.title = title
        self.version = version
        self.app: Optional[FastAPI] = None
        self.model_manager: Optional[ModelManager] = None
        
    def create_app(self) -> FastAPI:
        """Creation et configuration de l'app FastAPI"""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self.model_manager = ModelManager(
                model_type=config.model.model_type,
                model_path=config.model.model_path
            )
            
            smartlog.smartp(f"Initialisation du modèle {self.model_manager.model_type}")
            
            try:
                self.model_manager.load_model()
                smartlog.success(f"Modèle chargé depuis {self.model_manager.model_path}")
            except FileNotFoundError:
                smartlog.warning(f"Modèle non trouvé, il sera créé lors du premier entraînement")
            
            app.state.model_manager = self.model_manager
            
            yield

            smartlog.warning("⏹️ Fin de l'application")
    
        self.app = FastAPI(
            title=self.title,
            version=self.version,
            lifespan=lifespan,
            docs_url="/docs",
            redoc_url="/redoc"
        )
        self.add_middleware()
    
    def add_routes(self, router: APIRouter, prefix: str = "", tags: List[str] = []):
        """Ajoute une route"""
        if not self.app:
            smartlog.error(f"App doit être configurée d'abord : self.app = {self.app}")
            raise RuntimeError("App doit être configurée d'abord")        
        self.app.include_router(router, prefix=prefix, tags=tags)
        
    def add_middleware(self):
        """Ajoute un middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], 
            allow_credentials=True,
            allow_methods=["*"], 
            allow_headers=["*"],
        )
