import os
import numpy as np
from typing import Tuple, Optional
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import random

from backend.utils.config import config
from backend.utils.logger import smartlog

class ChestXrayDataset(Dataset):
    """
    Retour: (image_tensor, label_index)
    """
    def __init__(self, root_dir: str, split: str = "", transform=None):
        """
        split : nom du dossier train / test 
        """
        if split:
            self.root = os.path.join(root_dir, split)
        else:
            self.root = root_dir
        smartlog.warning(f"Chemin : {self.root}")
        self.transform = transform
        self.samples = []
        classes = ["NORMAL", "PNEUMONIA"]
        for cls_idx, cls in enumerate(classes):
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))
        if not self.samples:
            smartlog.error(f"Pas d'image dans {self.root}")
            raise RuntimeError(f"Pas d'image dans {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms(input_size: int = 224, train: bool = True):
    if train:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(8),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def get_transforms_augmented(input_size: int = 224, train: bool = True):
    if train:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_dataloaders(root_dir: str = config.data.raw_path,
                     input_size: int = 224,
                     batch_size: int = 32,
                     val_split: float = config.data.test_size,
                     num_workers: int = 2,
                     mode: str = "balanced") -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Crée train_loader et val_loader.
    Si root_dir contient 'train' subfolder, on l'utilise.
    Sinon on split le dossier racine.
    Selon le dataset utilisé, prend en compte le dossier val
    Crée train_loader et val_loader avec plusieurs modes:
    - "normal"   : dataloaders classiques
    - "augmented": data augmentation sur train
    - "balanced" : batches équilibrés avec WeightedRandomSampler
    """
    assert mode in ["normal", "augmented", "balanced"], f"Mode inconnu: {mode}"
    
    # val_dir_name = "val"
    val_dir_name = "test"
    train_dir = f"{root_dir}train"
    # Sélection des bons transforms
    if mode == "augmented":
        train_transform = get_transforms_augmented(input_size=input_size, train=True)
    else:
        train_transform = get_transforms(input_size=input_size, train=True)
    val_transform = get_transforms(input_size=input_size, train=False)
    # Dataset
    if os.path.isdir(train_dir):
        train_ds = ChestXrayDataset(root_dir, split="train", transform=train_transform)
        if os.path.isdir(os.path.join(root_dir, val_dir_name)):
            val_ds = ChestXrayDataset(root_dir, split=val_dir_name, transform=val_transform)
        else:
            val_ds = None
    else:
        all_ds = ChestXrayDataset(root_dir, split="", transform=train_transform)
        labels = [s[1] for s in all_ds.samples]
        idx = list(range(len(all_ds)))
        train_idx, val_idx = train_test_split(idx, test_size=val_split, stratify=labels, random_state=config.data.random_state)
        train_ds = Subset(all_ds, train_idx)
        val_ds = Subset(all_ds, val_idx)
        # set val transforms
        val_ds.transform = get_transforms(input_size=input_size, train=False)
    # Dataloaders
    if mode == "balanced":
        labels = [s[1] for s in getattr(train_ds, "samples", train_ds.samples)]
        class_sample_count = np.bincount(labels)
        weights = 1. / class_sample_count
        samples_weight = [weights[t] for t in labels]
        sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_ds else None
    return train_loader, val_loader