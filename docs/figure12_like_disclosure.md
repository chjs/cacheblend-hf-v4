# Figure 12-like Reproduction — Disclosure of Differences

> CacheBlend (EuroSys '25) Figure 12 와 우리 v4 평가의 차이점을 명시. **bit-identical 재현 아님**. paper 와 우리가 다른 substitutes 를 채택한 모든 항목을 한 자리에서 정리.

## 0. Why this document

논문 Figure 12 는 RAG-style multi-document QA 에서 KV reuse 의 quality 효과를 보고한다. 핵심 데이터 (200 sample MuSiQue) 와 metric (F1) 은 명확하지만, **재현에 필요한 hyperparam 의 일부가 paper 에서 비공개**이며 우리는 합리적 substitute 를 선택했다. 이 disclosure 는 (a) paper claim 과 우리 number 를 바로 비교하지 못하게 하고 (b) 차후 paper authors 와의 alignment 시도에 필요한 정보를 모은다.

데이터 빌드 디테일은 `external/mydata/cacheblend_fig12/README.md` 참조. 본 문서는 **차이 / substitute** 만 정리.

## 1. 12개 차이점 — 한눈에

| # | Aspect | Paper Figure 12 | v4 (mydata + this repo) | 채택 근거 |
|---|---|---|---|---|
| 1 | Embedding 모델 | 비공개 (paper 에 모델명 없음) | `sentence-transformers/all-mpnet-base-v2` (768-d) | 가장 표준적 RAG embedding. mydata 단일 source. |
| 2 | Embedding 정규화 | 비공개 | **raw L2** (no L2-normalization on embedding) | CacheBlend 평가 코드 관행 (mydata README §3). |
| 3 | Retrieval similarity | 비공개 (cosine 추정) | `‖p_emb − q_emb‖_2` ascending | mydata pipeline 그대로 사용. |
| 4 | Retrieval query | GPT-4 simulated query (paper "Musique extended" 부분) | MuSiQue dev question text 직접 사용 | GPT-4 비공개 prompt + cost. |
| 5 | Top-K docs/sample | 비공개 (paper Figure 12: 5 또는 10 추정) | **6** | mydata 고정값. MuSiQue paragraph pool 20 중 top-6. |
| 6 | Document order | 비공개 (random 추정) | `random.Random(42)` **단일 인스턴스** sequential shuffle (per-sample 다른 순서, sample 별 deterministic) | L20 — 단일 RNG 의 의도적 선택. Prefix cache baseline 약화. |
| 7 | Sample count / dataset | Figure 12: MuSiQue 200, 2WikiMQA 200, SAMSum 200, MultiNews 200 | **MuSiQue 200 만** | 2WikiMQA / SAMSum / MultiNews 는 v5. |
| 8 | Models | Mistral-7B-Instruct (그 외 paper 명세 부분 비공개) | Mistral-7B-Instruct-v0.2 + Llama-3.1-8B-Instruct (Phase 7) + Llama-3.1-70B-Instruct **8-bit bitsandbytes** (Phase 7) | Llama-3.1 family 추가 + 70B 는 8-bit (cost). |
| 9 | Precision | 비공개 (FP16 추정) | **FP16** for Mistral / Llama-3.1-8B; **8-bit** for Llama-3.1-70B | bitsandbytes 8-bit 는 FP16 baseline 대비 small drift 가능. |
| 10 | Reported metric | TTFT 절감 + F1 quality | **F1 + Rouge-L only** (TTFT 비목표 L27) | quality-only milestone. Hook-injection 디자인 (design-decisions.md §3). |
| 11 | Statistical test | Figure 12 absolute F1 차이 (CI 비공개) | **paired bootstrap CI** (n=1000, 95%) — pass 조건 ci_low > 0 (L28) | F1 noise 위에서 significance 확보. |
| 12 | Tolerance / equivalence | 비공개 | 4 categories frozen: `IDENTICAL_PATH` / `SAME_SHAPE` / `MIXED_SHAPE` / `RECOMPUTE_PATH` (design-decisions.md §1) | Phase 시작 전 freeze, retroactive 변경 금지 (L05, L13, L16). |

## 2. 채택한 substitute — 상세

### 2.1 Embedding: `all-mpnet-base-v2`

- 768-d, sentence-transformers 의 사실상 default.
- Dense retrieval 의 baseline. paper 가 더 최신 모델 (E5, BGE) 을 썼을 가능성은 있으나 알 길 없음.
- 라이선스: Apache 2.0.

### 2.2 L2 raw distance, no normalization

- `np.linalg.norm(p_emb − q_emb, axis=1)` (mydata README §3).
- 정규화 끔 — `all-mpnet-base-v2` 의 raw vector magnitude 가 의미 있다고 가정.
- Paper 가 cosine 을 썼다면 L2(normalized) 와 동치이지만, raw L2 는 다른 ranking 을 줄 수 있다.

### 2.3 GPT-4 simulated query 제외

- Paper Figure 12 의 일부 결과는 "MuSiQue extended" 라며 GPT-4 가 simulate 한 query 로 retrieval 을 다시 함.
- 이 prompt 도 GPT-4 cost 도 우리 v4 범위 밖. 원래 question text 그대로 retrieval.
- 결과: paper 보다 retrieval 이 더 noisy 할 수 있음. F1 절대값 비교 불가.

### 2.4 Random shuffle: `random.Random(42)` 단일 인스턴스

- `mydata/cacheblend_fig12/build_prompts.py` 가 200 sample 을 순회하며 **하나의 RNG 인스턴스** 로 sequential shuffle.
- 결과: same sample = same order (deterministic). different samples = different orders (RNG state 가 advance 되므로).
- 이 디자인은 prefix-caching baseline 을 약화시킴. 모든 sample 이 같은 doc order 라면 첫 doc 의 prefix cache hit 이 우연히 cacheblend 와 같은 quality 를 줄 수 있음 (false equivalence). RNG 가 sample 별 다른 순서를 강제하므로 prefix cache 가 random subset 만 hit.

### 2.5 데이터셋 축소: MuSiQue 200 만

- Paper Figure 12 는 4 dataset 보고 (MuSiQue / 2WikiMQA / SAMSum / MultiNews).
- v4 는 MuSiQue 200 만 (1/4). 이유:
  - mydata 가 MuSiQue 만 사전 빌드.
  - 2WikiMQA 는 retrieval pipeline 을 우리가 다시 짜야 함.
  - SAMSum (dialog summarization) / MultiNews (multi-news summarization) 는 metric (Rouge-L) 평가 의미 검증이 추가로 필요.
- v5 에서 4 dataset coverage 회복.

### 2.6 Llama-3.1-70B 8-bit

- bitsandbytes 8-bit quantization. `BitsAndBytesConfig(load_in_8bit=True)`.
- FP16 baseline 대비 small drift 가능 (논문 식 §4.2 KV deviation 자체가 실제 weight 의 함수가 아니라 hidden_state 함수이므로 quantization 영향이 deviation 비교에는 제한적이지만, generation quality 에는 영향).
- Cost trade-off: 80GB GPU 1 장으로 70B 가능 vs FP16 은 multi-GPU.

### 2.7 TTFT 비목표 (L27)

- 우리 v4 는 hook-injection 디자인 (design-decisions.md §3) — q_proj/MLP/RMSNorm 모두 fully 계산. TTFT 절감 없음.
- 따라서 TTFT 보고는 sanity log 로만, gate 조건 아님.
- Paper Figure 12 가 quality + TTFT joint trade-off 를 보였다면, 우리는 quality 만 본다.

### 2.8 Statistical test: paired bootstrap CI (L28)

- F1 자체는 multi-hop QA 에서 가변성 큼 (sample 200 정도면 sampling noise 만으로 F1 ±0.03~0.05).
- v4 는 CacheBlend vs baseline 의 per-sample F1 차이를 1000-bootstrap → 95% CI low > 0 인지 확인.
- Pass condition: `ci_low > 0` AND absolute diff < 0.05 (design-decisions.md §6).
- Paper 가 CI 를 어떻게 보고했는지 비공개 — 우리는 statistical significance 를 명시한다.

### 2.9 Tolerance freeze

- 4 카테고리 (IDENTICAL_PATH / SAME_SHAPE / MIXED_SHAPE / RECOMPUTE_PATH).
- Phase 시작 전 카테고리 결정. Retroactive 변경 금지 (L05, L13, L16).
- Paper 는 tolerance 를 정의하지 않음 (FP16 cuBLAS path 차이 등 numerical concern 명시 없음).

## 3. 비교 가능성에 대한 명시

- **F1 절대값** 을 paper 와 직접 비교 금지. 위 12 substitute 모두 F1 에 영향.
- 비교 가능한 것:
  - **알고리즘 동등성**: cacheblend vs full_recompute 의 F1 차이가 (a) 작고 (b) statistically significant 한지 (paired bootstrap CI). 이는 paper claim 과 정성적으로 비교 가능.
  - **HKVD elbow shape**: paper Figure 6 (recompute_ratio vs F1 curve) 의 elbow 위치를 우리가 같은 데이터/모델/HKVD 정의로 측정한 것과 비교. paper 가 정의한 ratio 0.10~0.20 elbow 가 우리 data 에서도 나타나는가 — qualitative claim.

## 4. Cross-references

- 데이터 빌드 디테일: `external/mydata/cacheblend_fig12/README.md`
- LMCache (paper 의 spirit 을 implement 한 production system) 분석: `docs/lmcache-analysis.md`
- 디자인 결정 (tolerance, hook-injection, mydata 사용 정당화): `docs/design-decisions.md`
- v4 작업 명세: `tasks/phase-N-*.md`
- v5-lessons 누적 로그: `docs/notes/v5-lessons.md`
