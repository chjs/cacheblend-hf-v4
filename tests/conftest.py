"""Global pytest config — auto-skip GPU/model tests when no CUDA available [L24]."""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip 'gpu' marked tests when no CUDA, 'requires_model' when no HF cache."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    
    skip_gpu = pytest.mark.skip(reason="no CUDA available")
    
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
