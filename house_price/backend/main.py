import uvicorn
from backend.app.core import FastAPIapp
from backend.utils.config import config

from backend.api.routes_root import router as root_router
from backend.api.routes_model import router as model_router
from backend.api.routes_health import router as health_router

app_builder = FastAPIapp(
    title=config.server.title or "FASTAPI APP",
    version=config.server.version or "0.0.0"
)

app_builder.create_app()
app_builder.add_routes(router=root_router, prefix="", tags=["Root"])
app_builder.add_routes(router=health_router, prefix="", tags=["Health"])
app_builder.add_routes(router=model_router, prefix="/ml", tags=["ML"])

app = app_builder.app

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host=config.api.host,
        port=config.api.port,
        reload=True,
        log_level="info"
    )