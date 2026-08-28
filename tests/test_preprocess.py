import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import IMG_SIZE, resize_normalize  # noqa: E402


def _make_dummy_image(size=(300, 150), color=(10, 200, 30)):
    return Image.new("RGB", size, color)


def test_resize_normalize_shape_and_range():
    img = _make_dummy_image()
    arr = resize_normalize(img)

    assert arr.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_resize_normalize_handles_non_rgb_input():
    # grayscale image should still come out as 3-channel RGB after conversion
    img = Image.new("L", (50, 50), 128)
    arr = resize_normalize(img)
    assert arr.shape == (IMG_SIZE, IMG_SIZE, 3)
