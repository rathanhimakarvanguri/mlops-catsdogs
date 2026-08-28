import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from model_utils import CLASS_NAMES, SimpleCNN, predict, preprocess_image  # noqa: E402


def test_preprocess_image_returns_expected_tensor_shape():
    img = Image.new("RGB", (128, 128), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    tensor = preprocess_image(buf.getvalue())

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)


def test_predict_returns_valid_label_and_probabilities():
    torch.manual_seed(0)
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    dummy_input = torch.randn(1, 3, 224, 224)

    label, probs = predict(model, dummy_input)

    assert label in CLASS_NAMES
    assert probs.shape == (len(CLASS_NAMES),)
    assert pytest.approx(float(np.sum(probs)), abs=1e-4) == 1.0
    assert all(0.0 <= p <= 1.0 for p in probs)
