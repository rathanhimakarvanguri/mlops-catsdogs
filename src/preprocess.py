"""
Data preprocessing for the Cats-vs-Dogs pipeline.

Reads raw images from data/raw/{cats,dogs}/*.jpg, resizes to 224x224 RGB,
applies light augmentation to the training split, and writes
data/processed/{train,val,test}.npz with arrays X (N,3,224,224) and y (N,).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 224
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
CLASSES = ["cats", "dogs"]  # folder names -> label 0, 1


def resize_normalize(image_bytes_or_path, size: int = IMG_SIZE) -> np.ndarray:
    """Core, independently-testable transform: load -> RGB -> resize ->
    scale to [0,1] float32 array of shape (size, size, 3).
    """
    if isinstance(image_bytes_or_path, (str, Path)):
        img = Image.open(image_bytes_or_path)
    else:
        img = image_bytes_or_path
    img = img.convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def augment(img: Image.Image) -> Image.Image:
    """Simple, dependency-light augmentation for the training split."""
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, fillcolor=(255, 255, 255))
    return img


def list_files() -> List[Tuple[Path, int]]:
    items = []
    for label, cls in enumerate(CLASSES):
        cls_dir = RAW_DIR / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append((p, label))
    return items


def split(items, train=0.8, val=0.1, seed=42):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train)
    n_val = int(n * val)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def build_split(items: List[Tuple[Path, int]], is_train: bool) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    skipped = 0
    for path, label in items:
        try:
            img = Image.open(path).convert("RGB")
            img.load()  # force decode now to catch truncated/corrupt files
        except Exception:
            skipped += 1
            continue
        if is_train:
            img = augment(img)
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        X.append(np.transpose(arr, (2, 0, 1)))  # CHW
        y.append(label)
    if skipped:
        print(f"  (skipped {skipped} unreadable/corrupt image(s))")
    if not X:
        return np.zeros((0, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(X), np.array(y, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    items = list_files()
    if not items:
        raise SystemExit(
            "No images found under data/raw/{cats,dogs}. "
            "Download the Kaggle dataset or run scripts/generate_synthetic_data.py first."
        )

    train_items, val_items, test_items = split(items, args.train_frac, args.val_frac)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, split_items, is_train in [
        ("train", train_items, True),
        ("val", val_items, False),
        ("test", test_items, False),
    ]:
        X, y = build_split(split_items, is_train)
        np.savez_compressed(OUT_DIR / f"{name}.npz", X=X, y=y)
        print(f"{name}: {len(y)} images -> {OUT_DIR / f'{name}.npz'}")


if __name__ == "__main__":
    main()
