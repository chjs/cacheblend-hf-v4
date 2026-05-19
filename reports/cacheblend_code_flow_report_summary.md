# CacheBlend HF v4 코드 흐름 분석 — 요약

> Full HTML 보고서: `reports/cacheblend_code_flow_email_report.html`
> 분석 대상: `cacheblend-hf-v4` @ commit `4689713` (2026-05-19)
> 분석 방식: read-only static analysis. 코드 수정 없음.

## 한 줄 요약

CacheBlend (EuroSys '25, paper §4 selective KV recompute) 를 HuggingFace Transformers 위에 재구현한 v4 codebase. 핵심 라이브러리 `src/cacheblend/` (11 파일, ~1,650 LOC) + 실험 드라이버 `benchmarks/` (run_phase6 / run_loong / run_cacheblend_official). 검증된 정확성: paper §4 의 모든 핵심 메커니즘 (pre-RoPE 저장, RoPE shift, HKVD 수식, top-K 선정, sparse forward). LMCache 와의 1:1 비교 완료 (수식/단계 등가) — vLLM 바이너리 직접 비교는 인프라 부재로 불가.

## 핵심 파일 (한 눈에)

| 파일 | 역할 |
|---|---|
| `src/cacheblend/model.py` | `LayerwiseModel` — HF wrap + layer-wise forward + pre-RoPE K capture |
| `src/cacheblend/chunker.py` | `Chunk`, `chunk_texts`, `fused_input_ids`, `chunk_offsets`. chunk_id = SHA256[:16] |
| `src/cacheblend/kv_store.py` | LRU chunk KV cache (in-memory) |
| `src/cacheblend/precompute.py` | `precompute_chunk_kv` (chunk-local), `precompute_from_cache_prompt` (cross-chunk) |
| `src/cacheblend/rope.py` | `apply_rope_shift` — pre-RoPE K → post-RoPE K at fused positions |
| `src/cacheblend/hkvd.py` | `kv_deviation` (squared L2 per token, fp32), `select_top_k` |
| `src/cacheblend/fusor.py` | **5 fuse 함수**: full_recompute / full_reuse / selective (legacy) / **selective_lmc_parity** / prefix_cache |
| `benchmarks/run_phase6.py` | Phase 6/7 Musique 평가 driver (4 runner × N samples) |
| `benchmarks/run_loong.py` | Loong cache-then-reverse driver (HKVD capture + boundary enrichment) |
| `benchmarks/run_cacheblend_official.py` | YaoJiayi/CacheBlend 3 dataset 포팅 (실행 대기) |
| `external/CacheBlend/example/utils.py` | F1 / ROUGE-L (paper convention) |
| `external/mydata/cacheblend_fig12/harness/metrics.py` | mydata harness 의 동일 metric (Phase 6/7 사용) |
| `external/LMCache/lmcache/v1/compute/blend/blender.py` | LMCache `process_qkv` (paper §4 reference 구현) |

## 데이터 흐름 (간략)

1. **CLI** → driver 진입.
2. **Dataset** load (mydata prompts.jsonl / framolfese/Loong / CacheBlend inputs).
3. **Chunk 생성** (`chunker.chunk_texts(tokenizer, [str, ...])` → list[Chunk]).
4. **LayerwiseModel** wrap (HF AutoModelForCausalLM + k_proj forward-hook).
5. **Cache population**:
   - `precompute_chunk_kv` (Phase 6/7, official): chunk-local standalone forward
   - `precompute_from_cache_prompt` (Loong): full forward + per-chunk slice (cross-chunk)
6. **Prefill fusion** (5 전략):
   - `fuse_full_recompute` — cache 무시
   - `fuse_full_reuse` — 모든 chunk cached K/V hook 주입
   - `fuse_selective` (legacy) — full forward + K/V hook merge (non-HKVD 만 cached)
   - **`fuse_selective_lmc_parity`** — sparse forward at top-K, paper §4 충실
   - `fuse_prefix_cache` — 첫 chunk 만 reused
7. **Greedy decode** (max_new_tokens=32 또는 128, EOS guard).
8. **Metric**: F1 (max over aliases) / ROUGE-L + paired bootstrap CI (vs FullReuse).
9. **출력**: `results.jsonl` (per-row), `summary.json` (집계), `reports/*.md`, PNG (Loong).

## CacheBlend 단계 → 우리 코드 매핑 (요약)

| Step | 개념 | 구현 위치 |
|---|---|---|
| A | chunk KV cache 저장 | `precompute.py` + `kv_store.py` |
| B | 재사용 chunk 탐색 | `chunker._stable_id` + `KVStore.has` |
| C | cached KV 새 prompt 순서 에 재조합 | `fusor.py` 의 K_override 빌드 |
| D | position/RoPE mismatch 처리 | `rope.py` + HF `apply_rotary_pos_emb` (RoPE 가역성 활용) |
| E | top-K HKVD token 선정 | `hkvd.py` |
| F | sparse forward 로 selected K/V 갱신 | `fuse_selective_lmc_parity` (paper §4 충실) |
| G | full-length past_key_values 로 decoding | driver `_greedy_decode` |
| H | F1/Rouge-L + paired CI | mydata harness + `benchmarks/metrics/bootstrap.py` |

## 검증 상태

- **검증됨**: RoPE 재적용 비트 등가 (commit de6d818), CPU 단위 테스트 8/8 PASS, HKVD 수식 LMCache 등가, Phase 6c Mistral 통과, Loong CB 우위 (+0.03 F1) + HKVD boundary enrichment 3.13x.
- **부분 검증**: legacy vs lmc_parity 의 F1 거의 동일 (의미 다름), SDPA GPU 추가 검증 필요, LMCache vLLM 바이너리 직접 비교 불가능.
- **확인 필요**: Llama-3.1-8B Musique 의 CB < FullReuse 원인, Loong sample 확장, controller.py 사용처, samsum max_ctx_len 정확도.
- **리스크**: HF 4.50+ kwarg 변경, SDPA kernel 결과 변동, tokenizer 버전 변경 시 cache 무효화.

## 다음 액션

1. Phase 8 step 2-4 (gradual filtering) — Llama Musique 원인 검증 후보.
2. CacheBlend official 3 dataset GPU 실행 (~$0.90).
3. Loong n 확장 (min_docs 완화).
4. multi-check_layer + multi-ratio 지원 (LMCache 와 정합).
5. fuse_selective_lmc_parity GPU 정확성 추가 비교.

## 보고서 파일

- `reports/cacheblend_code_flow_email_report.html` — 풀 HTML, 이메일/PR 본문 직접 paste 가능
- `reports/cacheblend_code_flow_email_subject.txt` — 이메일 제목 1 줄
- `reports/cacheblend_code_flow_report_summary.md` — 본 요약 (이 파일)
