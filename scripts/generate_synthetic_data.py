"""
Generates a tiny synthetic stand-in dataset (solid-color-ish images with
noise) so the full pipeline (preprocess -> train -> serve -> test) can be
exercised end-to-end without downloading the real Kaggle Cats-and-Dogs
dataset. Replace data/raw/{cats,dogs} with real images for an actual model.
"""
import numpy as np
from PIL import Image
from pathlib import Path

RAW_DIR = Path("data/raw")
N_PER_CLASS = 40
SIZE = (256, 256)


def make_image(base_color, seed):
    rng = np.random.RandomState(seed)
    arr = np.ones((*SIZE, 3), dtype=np.uint8) * np.array(base_color, dtype=np.uint8)
    noise = rng.randint(-30, 30, size=arr.shape)
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    classes = {"cats": (200, 150, 100), "dogs": (100, 150, 200)}
    for cls, color in classes.items():
        out_dir = RAW_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(N_PER_CLASS):
            img = make_image(color, seed=i)
            img.save(out_dir / f"{cls}_{i:03d}.jpg", quality=90)
    print(f"Wrote {N_PER_CLASS} synthetic images per class to {RAW_DIR}")


if __name__ == "__main__":
    main()
