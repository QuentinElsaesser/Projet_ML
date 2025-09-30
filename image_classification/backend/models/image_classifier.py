import os
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import torch.nn.functional as F

from backend.models.basic_model import BasicModel
from backend.metrics.metrics_eval import evaluate_classification
from backend.utils.logger import smartlog

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageClassifier(BasicModel):
    """
    Wrapper PyTorch pour classification d'images (2 classes: NORMAL vs PNEUMONIA).
    Utilise transfer learning (ResNet18 by default).
    """

    def __init__(self,
                 num_classes: int = 2,
                 backbone: str = "resnet18",
                 pretrained: bool = True,
                 lr: float = 1e-3,
                 epochs: int = 10,
                 batch_size: int = 32,
                 input_size: int = 224,
                 device: Optional[torch.device] = None):
        self.num_classes = num_classes
        self.backbone = backbone
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_size = input_size
        self.device = device or DEVICE
        self.is_trained = False
        self.history: List[Dict[str, Any]] = []
        
        if pretrained:
            if backbone.endswith("18"):
                self.weights = ResNet18_Weights.DEFAULT
            elif backbone.endswith("50"):
                self.weights = ResNet50_Weights.DEFAULT
        else:
            self.weights = None

        # Modèle PyTorch
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None

        # build model
        self._build_model()
        
    def _build_model(self):
        # backbone
        if self.backbone == "resnet18":
            net = models.resnet18(weights=self.weights)
        elif self.backbone == "resnet50":
            net = models.resnet50(weights=self.weights)
        else:
            raise ValueError("backbone not supported")
        in_features = net.fc.in_features
        # head
        net.fc = nn.Linear(in_features, self.num_classes)
        self.model = net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_step(self, train_loader, val_loader=None, verbose: bool = True, epoch: int = 1) -> Dict[str, Any]:
        """
        Entraîne le modèle pour UNE epoch seulement.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        if verbose:
            smartlog.smartp(f"--- Epoch {epoch}/{self.epochs} ---")

        for batch_idx, (imgs, targets) in enumerate(train_loader, 1):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device).long()
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            if verbose and batch_idx % 10 == 0:
                    smartlog.smartp(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        metrics = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
        }

        if val_loader is not None:
            val_metrics = self.evaluate(val_loader)
            metrics["val"] = val_metrics

        self.history.append(metrics)
        return metrics

        
    def train(self, train_loader, val_loader=None, verbose: bool = True) -> Dict[str, Any]:
        """
        train_loader: PyTorch DataLoader for training
        val_loader: optional DataLoader for validation
        Returns last epoch metrics dict (val metrics if val_loader provided else train)
        """
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0
            
            if verbose:
                smartlog.smartp(f"--- Epoch {epoch}/{self.epochs} ---")
                
            for batch_idx, (imgs, targets) in enumerate(train_loader, 1):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device).long()
                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
                if verbose and batch_idx % 10 == 0:
                    smartlog.smartp(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")


            epoch_loss = running_loss / total
            epoch_acc = correct / total

            # validation
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.history.append({
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "train_acc": epoch_acc,
                    "val": val_metrics
                })
            else:
                self.history.append({
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "train_acc": epoch_acc
                })
            
            if verbose:
                msg = f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}"
                if val_metrics:
                    if hasattr(val_metrics, "dict"):# si c’est un Pydantic
                        val_metrics_dict = val_metrics.dict()
                    else:
                        val_metrics_dict = val_metrics
                    msg += f", Val Acc: {val_metrics_dict.get('accuracy', 0):.4f}"
                smartlog.success(msg)

        self.is_trained = True
        return self.history[-1] if self.history else {}

    def predict(self, loader_or_tensor, return_probs: bool = False) -> List[Union[int, Dict[str, float]]]:
        self.model.eval()
        results = []
        with torch.no_grad():
            # DataLoader case
            if hasattr(loader_or_tensor, "__iter__") and not isinstance(loader_or_tensor, torch.Tensor):
                for batch in loader_or_tensor:
                    imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                    imgs = imgs.to(self.device)
                    outputs = self.model(imgs)
                    probs = F.softmax(outputs, dim=1)
                    _, preds = torch.max(probs, 1)
                    if return_probs:
                        results.extend([
                            {"class": int(p.item()), "probs": prob.cpu().numpy().tolist()}
                            for p, prob in zip(preds, probs)
                        ])
                    else:
                        results.extend(preds.cpu().numpy().tolist())
                return results
            # single tensor case
            tensor = loader_or_tensor
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.to(self.device)
                outputs = self.model(tensor)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                if return_probs:
                    return [{
                        "class": int(preds.item()),
                        "probs": probs.squeeze().cpu().numpy().tolist()
                    }]
                else:
                    return preds.cpu().numpy().tolist()
            raise ValueError("Unsupported input to predict()")

    def evaluate(self, loader) -> Dict[str, float]:
        """
        Evaluate on a DataLoader. Returns dict of accuracy, precision, recall, f1.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, targets in loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
        return evaluate_classification(all_targets, all_preds)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "backbone": self.backbone,
            "input_size": self.input_size
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self._build_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        try:
            self.optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
        except Exception:
            pass
        self.is_trained = True