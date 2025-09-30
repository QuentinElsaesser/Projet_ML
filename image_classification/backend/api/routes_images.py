from fastapi import APIRouter, Request, UploadFile, File, Form
from typing import Dict, Any, List, Union
import torchvision.transforms as T
from PIL import Image

from backend.utils.image_train import make_dataloaders
from backend.models.image_classifier import ImageClassifier
from backend.utils.logger import smartlog

from backend.schemas.api_schemas import (
    TrainImageResponse, TrainImageRequest,
    PredictImageResponse, TrainImageResponseStep
)

router = APIRouter(tags=["Image Classification"])

@router.post("/train_image", response_model=TrainImageResponse, summary="Entraîner un classifieur d'images (chemin dossier)")
async def train_image(request: Request, req: TrainImageRequest) -> Dict[str, Any]:
    manager = request.app.state.model_manager
    smartlog.smartp(f"Training image classifier: root_dir={req.root_dir}, backbone={req.backbone}, epochs={req.epochs}")
    clf = ImageClassifier(backbone=req.backbone, epochs=req.epochs, batch_size=req.batch_size)
    train_loader, val_loader = make_dataloaders(req.root_dir, input_size=clf.input_size, batch_size=clf.batch_size, mode=req.mode)
    metrics = clf.train(train_loader, val_loader)
    manager.model = clf
    manager.model_type = "image_classifier"
    manager.train_loader = train_loader
    manager.val_loader = val_loader
    manager.current_epoch = 0
    manager.metrics_history.append({"type": "train", "model": "image_classifier", "metrics": metrics})
    manager.save_model()
    return {"metrics": metrics}

@router.post("/train_image_init", summary="Initialiser un classifieur d'images (chemin dossier)")
async def train_image_init(request: Request, req: TrainImageRequest) -> Dict[str, Any]:
    manager = request.app.state.model_manager
    smartlog.smartp(f"Init image classifier: root_dir={req.root_dir}, backbone={req.backbone}, epochs={req.epochs}")
    clf = ImageClassifier(
        backbone=req.backbone,
        epochs=req.epochs,
        batch_size=req.batch_size
    )
    train_loader, val_loader = make_dataloaders(
        req.root_dir,
        input_size=clf.input_size,
        batch_size=clf.batch_size,
        mode=req.mode
    )
    manager.model = clf
    manager.model_type = "image_classifier"
    manager.train_loader = train_loader
    manager.val_loader = val_loader
    manager.current_epoch = 0 
    return {"status": "ok", "message": "Model initialized, ready for training step"}

@router.post("/train_step", response_model=TrainImageResponseStep, summary="Entraîner un classifieur d'images sur une seule étape (chemin dossier)")
async def train_image_step(request: Request) -> Dict[str, Any]:
    manager = request.app.state.model_manager
    if not manager.model or manager.model_type != "image_classifier":
        return {"error": "No model loaded"}
    if not hasattr(manager, "train_loader") or not manager.train_loader:
        return {"error": "No training data available. Call /train_image first."}
    manager.current_epoch += 1
    clf = manager.model
    metrics = clf.train_step(manager.train_loader, manager.val_loader, epoch=manager.current_epoch)
    # Sauvegarde historique
    manager.metrics_history.append({
        "type": "train_step",
        "epoch": manager.current_epoch,
        "metrics": metrics
    })
    manager.save_model()
    return {"status": "ok", "epoch": manager.current_epoch, "metrics": metrics}

@router.post("/predict_image", response_model=PredictImageResponse, summary="Prédire une image (upload)")
async def predict_image(request: Request,
                        file: UploadFile = File(...),
                        backbone: str = Form("resnet18")) -> List[Union[int, Dict[str, float]]]:
    manager = request.app.state.model_manager
    if not manager.model or manager.model_type != "image_classifier":
        manager.model = ImageClassifier(backbone=backbone)
        try:
            manager.load_model()
        except Exception as e:
            return {"error": "No image model available", "detail": str(e)}

    img = Image.open(file.file).convert("RGB")
    transform = T.Compose([
        T.Resize((manager.model.input_size, manager.model.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # valeur standard ImageNet pour utiliser resnet18
    ])
    tensor = transform(img).unsqueeze(0)
    # predictions = manager.model.predict(tensor)
    predictions = manager.model.predict(tensor, return_probs=True)
    return {"predictions": predictions}