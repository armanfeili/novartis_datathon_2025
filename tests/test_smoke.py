"""Smoke tests for core utilities."""

import torch
from src.utils import set_seed, get_device, SimpleNet


def test_imports_and_device():
    """Test that imports and device detection work."""
    import torch  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    device = get_device()
    assert device.type in ["cuda", "cpu"]


def test_set_seed():
    """Test that set_seed produces reproducible results."""
    set_seed(0)
    x1 = torch.randn(2, 2)

    set_seed(0)
    x2 = torch.randn(2, 2)

    assert torch.allclose(x1, x2)


def test_simple_net():
    """Test SimpleNet initialization and forward pass."""
    device = get_device()
    model = SimpleNet(num_classes=10).to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    logits = model(x)
    assert logits.shape == (2, 10)
