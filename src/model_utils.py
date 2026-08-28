"""
Shared model definition + pre/post-processing utilities.
Imported by both src/train.py (training) and app/main.py (inference),
and exercised directly by tests/test_model_utils.py.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMG_SIZE = 224
CLASS_NAMES = ["cat", "dog"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SimpleCNN(nn.Module):
    """A small baseline CNN — good enough for a reproducible MLOps demo,
    not intended to be SOTA on Cats-vs-Dogs."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 224 -> 112
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 56 -> 28
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Raw image bytes -> normalized (1, 3, 224, 224) float tensor.

    This is the function unit-tested in tests/test_preprocess.py, and is
    also what the FastAPI /predict endpoint calls at inference time, so
    training-serving skew stays impossible by construction.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(arr).unsqueeze(0).float()
    return tensor


def load_model(weights_path: str, device: str = "cpu") -> SimpleCNN:
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def predict(model: SimpleCNN, tensor: torch.Tensor) -> Tuple[str, np.ndarray]:
    """Run inference and return (predicted_label, probability_array)."""
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    label = CLASS_NAMES[int(np.argmax(probs))]
    return label, probs
