"""Phase 0 smoke tests — all imports work, all stub classes exist [L25]."""
from __future__ import annotations


def test_imports_top_level():
    import cacheblend
    assert hasattr(cacheblend, "__version__")


def test_imports_tolerance():
    from cacheblend import Tolerance, ToleranceResult, assert_logits_close
    assert Tolerance.IDENTICAL_PATH
    assert Tolerance.SAME_SHAPE
    assert Tolerance.MIXED_SHAPE
    assert Tolerance.RECOMPUTE_PATH


def test_imports_layerwise_model():
    from cacheblend import LayerwiseModel
    # Stub class, instantiation raises in Phase 0
    assert LayerwiseModel is not None


def test_imports_kv_store():
    from cacheblend import KVStore
    store = KVStore()
    assert not store.has("anything")


def test_imports_chunker():
    from cacheblend import Chunk, chunk_texts, fused_input_ids
    chunk = Chunk(text="hello", token_ids=[1, 2, 3], chunk_id="abc", is_cached=False)
    assert chunk.text == "hello"


def test_imports_fusor():
    from cacheblend import (
        fuse_full_recompute, fuse_full_reuse, fuse_selective, fuse_prefix_cache,
    )
    assert all(callable(f) for f in [
        fuse_full_recompute, fuse_full_reuse, fuse_selective, fuse_prefix_cache,
    ])


def test_imports_hkvd():
    from cacheblend import kv_deviation, select_top_k
    assert callable(kv_deviation)
    assert callable(select_top_k)


def test_imports_controller():
    from cacheblend import LoadingController, StorageProfile
    assert StorageProfile.RAM
    assert StorageProfile.SLOW_DISK


def test_imports_gradual():
    from cacheblend import LayerProfiler, SchedulePlanner, GradualSchedule
    sched = GradualSchedule(check_layers=[2, 5, 10], ratios=[0.30, 0.15, 0.10])
    assert sched.check_layers == [2, 5, 10]


def test_imports_runners():
    """Runner stubs must be importable even before mydata is cloned [Phase 0]."""
    from cacheblend import (
        FullRecomputeRunner, FullReuseRunner, PrefixCacheRunner,
        CacheBlendV4Runner, GradualV4Runner,
    )
    # Stub runners — instantiation OK without model/tokenizer for smoke
    r = CacheBlendV4Runner(recompute_ratio=0.15, check_layer=1)
    assert r.recompute_ratio == 0.15
    assert r.check_layer == 1


# test_boundary_safe_shortcut_logic removed — was a Phase-0 stub test that
# expected fuse_selective to raise NotImplementedError. Boundary dispatch
# (ratio=0 → full_reuse, ratio>=1 → full_recompute) is now exercised end-to-end
# in tests/test_fusor_selective.py with real models.
