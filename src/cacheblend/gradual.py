"""Phase 8 — Gradual filtering scheme (논문 §4.3).

LMCache의 단순 check_layer=1 flat schedule을 넘어, multi-check-layer 점진적 좁히기.
Interactive multi-step discovery experiment [L22].
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GradualSchedule:
    """Schedule with multiple check_layers and decreasing ratios.
    
    Stub. Real implementation in Phase 8.
    """
    check_layers: list = field(default_factory=list)
    ratios: list = field(default_factory=list)
    warmup_layers: list = field(default_factory=list)
    metric_used: str = ""


class LayerProfiler:
    """Step 1 — measure (a) top-15% mass, (b) Spearman, (c) information gain per layer.
    
    Stub. Real implementation in Phase 8 Step 1.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LayerProfiler: implemented in Phase 8 Step 1")


class SchedulePlanner:
    """Step 2 — generate schedule instances from user-confirmed check_layers + budgets.
    
    Stub. Real implementation in Phase 8 Step 2.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SchedulePlanner: implemented in Phase 8 Step 2")
