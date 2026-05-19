# benchmarks/musique/ — Run the original YaoJiayi/CacheBlend `blend_musique.py` unmodified

The file `blend_musique.py` here is a **verbatim copy** of
[`example/blend_musique.py`](https://github.com/YaoJiayi/CacheBlend/blob/main/example/blend_musique.py)
from the YaoJiayi/CacheBlend repo (commit at the time of import; see
`external/CacheBlend/`).

> **Rule**: do not modify `blend_musique.py`. Any plumbing changes go in
> wrappers, adapters, symlinks, import paths, or environment — never in the
> reference file itself.

The original script was written against the YaoJiayi vLLM fork (custom
`cache_fuse_metadata` + `hack_kv` + `old_kvs` hooks). This directory provides
the surrounding scaffolding that lets the same file run unmodified against
the HuggingFace-Transformers-based `cacheblend-hf-v4` impl in this repo:

```
benchmarks/musique/
├── blend_musique.py            # ORIGINAL, untouched (do not edit)
├── utils.py                  → ../../external/CacheBlend/example/utils.py   (symlink)
├── inputs/
│   └── musique_s.json        → ../../../external/CacheBlend/inputs/musique_s.json  (symlink)
├── _shim/
│   └── vllm/
│       └── __init__.py         # vllm.LLM + SamplingParams adapter
│                               # routes generate() calls to LayerwiseModel + fuse_selective
├── run_blend_musique.py        # wrapper: sets sys.path + cwd, runpy-executes blend_musique.py
└── README.md
```

## How it works

The original file relies on `vllm.LLM` plus a deeply-nested attribute chain
that lets user code mutate the model's per-forward behavior:

```python
cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
cache_fuse_metadata['collect'] = True    # capture per-chunk K/V into layers[j].self_attn.hack_kv
cache_fuse_metadata['check']   = True    # selective recompute against ...model.model.old_kvs
```

Our shim exposes the same attribute chain but routes the underlying compute
to `cacheblend-hf-v4`:

| `cache_fuse_metadata` flag | Original (vLLM fork) | Shim (HF) |
|---|---|---|
| `collect=True` | per-chunk standalone forward, hack_kv = post-RoPE K/V | per-chunk `precompute_chunk_kv` → KVStore (pre-RoPE K + V); hack_kv = zero-placeholder so user slicing doesn't crash |
| `check=True` | selective recompute using accumulated `old_kvs` | `fuse_selective(tracked_chunks, kv_store, recomp_ratio)` → greedy decode |
| both False | normal full prefill | HF model forward + greedy decode |

`old_kvs` assignments from user code are recorded but unused — the real KV
cache lives in the shim's internal `KVStore`. `hack_kv` placeholders are
sized just-enough that user-code slicing (`[:s_start_len]`,
`[s_start_1_len : len(...)+1]`) does not raise.

**Chunk-boundary parity**: chunk 0 (the `[INST]` + prefix prompt) is stored
with the BOS token included (it sits at fused position 0). Chunks 1..N
(documents and the trailing `[/INST]` query) are stored with BOS stripped,
matching the original's `[s_start_1_len : len(doc_chunk_ids[i])+1]` slice
which drops BOS for non-first chunks.

## Setup

The shim expects `external/CacheBlend/` to be present (for the symlinked
`utils.py` and `inputs/musique_s.json`). If missing:

```bash
git clone https://github.com/YaoJiayi/CacheBlend external/CacheBlend
```

Symlinks (already created in this branch):

```bash
cd benchmarks/musique
ln -s ../../external/CacheBlend/example/utils.py utils.py
mkdir -p inputs
ln -s ../../../external/CacheBlend/inputs/musique_s.json inputs/musique_s.json
```

## Usage

### CPU smoke test (no GPU, no model load)

Verifies the entire scaffolding — sys.path injection, attribute chain access,
collect/check/normal dispatch, output plumbing — without loading Mistral-7B:

```bash
CACHEBLEND_MOCK_MODEL=1 CACHEBLEND_MUSIQUE_N=2 python benchmarks/musique/run_blend_musique.py
```

Expected output (truncated):

```
[run_blend_musique] mock model: 1
Loading dataset: inputs/musique_s.json
[run_blend_musique] CACHEBLEND_MUSIQUE_N=2 → slicing dataset to first 2 examples
Cached generation: [mock check 12 chunks]
TTFT with cache: 3.83e-06
Normal generation: [mock normal generation]
TTFT with full prefill: 2.92e-07
------------
...
---------------Result Summary---------------------
TTFT with cache: 7.85e-06
TTFT with full prefill: 3.69e-06
F1 with cache: 0.0
F1 with full prefill: 0.0
```

F1 = 0 in mock mode is expected — mock generation returns stub text.

### Real run on GPU

Requires a GPU with ≥16GB VRAM (Mistral-7B FP16 ≈ 14GB + activations + KV).

```bash
# Full 150-example sweep:
python benchmarks/musique/run_blend_musique.py

# Truncated to 20 examples:
CACHEBLEND_MUSIQUE_N=20 python benchmarks/musique/run_blend_musique.py
```

## Environment variables

All read by `_shim/vllm/__init__.py` at import time, plus the truncation
hook in `run_blend_musique.py`:

| Variable | Default | Effect |
|---|---|---|
| `CACHEBLEND_MOCK_MODEL` | `0` | If `1`: skip model load; `generate()` returns stub text. Scaffolding-only test. |
| `CACHEBLEND_DEVICE` | auto (`cuda` if available else `cpu`) | Device to load Mistral-7B on. |
| `CACHEBLEND_DTYPE` | `float16` | Model dtype. |
| `CACHEBLEND_CHECK_LAYER` | `1` | `check_layer` arg to `fuse_selective`. |
| `CACHEBLEND_RECOMP_RATIO` | `0.15` | Default `recomp_ratio` (used when `cache_fuse_metadata['recomp_ratio']` not set — musique default). |
| `CACHEBLEND_ATTN_IMPL` | `sdpa` | `attn_implementation` for the HF model. musique prompts reach ~7K tokens; eager attention OOMs on 24GB GPUs. |
| `CACHEBLEND_MUSIQUE_N` | (unset = all 150) | Slice `utils.load_dataset()` to first N examples. Original file untouched; truncation done via `utils` rebinding before import. |

## Original ↔ shim mapping

| Original line | Original op | Shim handling |
|---|---|---|
| `from vllm import LLM, SamplingParams` | import vllm | `_shim/vllm/` injected onto sys.path |
| `LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5)` | spin up vLLM engine | `LayerwiseModel(...)` load + build fake attribute chain |
| `llm.set_tokenizer(tokenizer)` | install tokenizer | adopt user-passed tokenizer in mock mode (real mode uses LayerwiseModel's own) |
| `llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata` | mutable per-forward flags | dict in `_FakeInnerModel` |
| `cache_fuse_metadata['collect']=True; llm.generate([chunk_text], SamplingParams(max_tokens=1))` | per-chunk K/V capture | re-encode prompt, `precompute_chunk_kv` → KVStore; populate `hack_kv` placeholder per layer |
| `llm_layers[j].self_attn.hack_kv` | per-layer (K, V) post-forward | zero-tensor placeholder of shape `(L, hidden_kv)` per layer — user slicing works; real K/V lives in shim's KVStore |
| `llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values` | install fused KV cache | recorded but unused (shim uses its KVStore) |
| `cache_fuse_metadata['check']=True; llm.generate([input_prompt], SamplingParams(max_tokens=32))` | selective recompute + decode | `fuse_selective(tracked_chunks, kv_store, recompute_ratio=0.15, check_layer=1)` + greedy decode |
| `cache_fuse_metadata['check']=False; llm.generate(...)` | full prefill + decode | HF `model(...)` + greedy decode |
| `output[0].outputs[0].text` | generated text | `_Completion(text=...)` |
| `output[0].metrics.first_token_time - .first_scheduled_time` | TTFT | `_Metrics(first_scheduled_time, first_token_time)` from `time.perf_counter()` brackets |

## Known differences from the original vLLM fork

1. **TTFT semantics**: original measures TTFT inside vLLM's request scheduler.
   Shim measures `time.perf_counter()` brackets around prefill → first decoded
   token. Approximation, not identical.
2. **`old_kvs` is a no-op**: user-code accumulator is ignored. Real cache
   flows through the shim's internal `KVStore`, keyed by chunker `_stable_id`.
3. **`hack_kv` contents are zeros**: user-code reads are not consumed for our
   compute path. If you need actual post-RoPE K/V back to the user code (e.g.
   to cross-check against vLLM), modify the shim's `_populate_hack_kv` to
   expose real `K` after RoPE — non-trivial but mechanical.
4. **Tokenizer**: original uses YaoJiayi's vLLM tokenizer install path which
   may differ from HF's `AutoTokenizer.from_pretrained` in edge cases (e.g.
   whitespace handling). For Mistral-7B-Instruct-v0.2 these agree on the
   ~99% of strings encountered in musique_s.json.
5. **Greedy decoding**: shim uses standard HF argmax decoding. vLLM with
   `temperature=0` is also argmax-equivalent, but PagedAttention numerics
   may differ from eager attention at the last few bits.

See `_shim/vllm/__init__.py` docstring for shim-internal notes.

## Files outside this directory that matter

- `src/cacheblend/{model.py, fusor.py, precompute.py, chunker.py, kv_store.py, hkvd.py}` — actual CacheBlend mechanism
- `external/CacheBlend/example/{blend_musique.py, utils.py}` — symlink targets (and the reference)
- `external/CacheBlend/inputs/musique_s.json` — symlink target

## Branch

This wiring was developed on `step/musique-original-runnable`.
