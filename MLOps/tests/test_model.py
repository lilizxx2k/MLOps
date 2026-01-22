import torch
from mlops.model import Model


def test_model_output_shape():
    model = Model(output_dim=8)
    model.eval()

    x = torch.randn(4, 3, 224, 224)
    y = model(x)

    assert y.shape == (4, 8), f"Expected (4, 8), got {y.shape}"
