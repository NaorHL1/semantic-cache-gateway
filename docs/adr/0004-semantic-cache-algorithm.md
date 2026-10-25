# ADR-0004: Two-Layer Semantic Cache Algorithm & Threshold Policy

* **Status:** Accepted (similarity threshold value is provisional until the calibration in §7 is run)
* **Date:** 2026-10-16
* **Deciders:** Lead Architect & FDE Team

---

### 1. Context & Problem Statement
ADR-0001 to ADR-0003 fixed the storage engine (Redis 8 Open Source with built-in vector search), the embedding model (`all-MiniLM-L12-v2`, 384-dim) and the runtime (FastAPI + Granian). This ADR defines the algorithm that sits on top of them: for every `POST /v1/chat/completions` request, the gateway must decide whether it may answer from the cache, which stored entry it may use, and what it may store afterwards.

For a cache, a wrong hit is far more expensive than a miss. A miss costs one upstream call (seconds and fractions of a cent). A wrong hit silently serves another question's answer, and the caller has no way to detect it. Six problems make this decision hard:

1. **Raw cosine scores are miscalibrated.** Anisotropy concentrates pretrained embedding scores in a narrow high-similarity band, so absolute values carry little meaning and differ between models [1].
2. **No single static threshold separates correct from incorrect hits.** In the vCache evaluation, correct and incorrect hits have nearly identical similarity distributions (means 0.84 vs 0.85), and per-entry optimal thresholds range from 0.71 to 1.0 [2]. Their own example: *"Where was Roger Federer born"* vs *"In which town did Roger Federer grow up"* (same question) scores above 0.75, while *"Explain Round Robin scheduling"* vs *"Explain Shortest Job First scheduling"* (different questions) scores above 0.92 [3].
3. **Multi-turn context.** A follow-up such as *"Yes, go ahead"* means different things in different conversations. Embedding it alone collides unrelated chats.
4. **Generation parameters change the answer.** `stop`, `temperature`, `response_format`, `reasoning_effort` and future OpenAI parameters all alter the output. A key that ignores any of them serves wrong answers.
5. **Streaming.** Most chat traffic is SSE. The cache must capture streamed answers without delaying the client, and replay hits in the format the client asked for.
6. **Shared caches are an attack surface.** In 2026, semantic cache poisoning was confirmed on all three major public clouds [4], and key-collision attacks reached 80–88% hit rates against AWS Bedrock and Azure API Management [5].

### 2. Decision Drivers
* **Correctness over hit rate (fail-closed):** when in doubt, miss. Unknown inputs must cause a miss, never a wrong hit.
* **Latency budget:** exact hits in <1ms; the full semantic path (embedding + KNN) inside the <25ms cache-hit SLA.
* **Tenant isolation:** one caller must never read or poison another caller's entries.
* **Measurability:** the threshold must be a measured result on our own model, not a borrowed number.
* **Availability:** a cache failure must degrade to a slower response, never to an outage.
* **Simplicity & reversibility:** prefer `.env`-level (Type 2) knobs; keep the Type 1 surface small and explicit.

### 3. Considered Options

**3.1 Cache layering**
* **Option A: Semantic layer only** (`redisvl` `SemanticCache` as-is). One vector search per request; no exact layer, no tenant or parameter scoping built in.
* **Option B: Two layers, two records.** An exact-match key-value entry plus a separate vector entry per response. Doubles response storage; two TTLs and two delete paths to keep in sync.
* **Option C: Two layers, one record.** One Redis HASH per response, addressed directly by its exact hash (Layer 1) and indexed for vector search (Layer 2).

**3.2 Parameter scoping**
* **Option A: Allowlist** of fields that enter the cache key. Fail-open: any forgotten or newly released parameter silently causes collisions.
* **Option B: Exclusion denylist.** Every request field enters the key except a fixed list known not to change the answer. Fail-closed: an unknown parameter costs a miss, never a wrong hit.

**3.3 Threshold strategy**
* **Option A: Fixed 0.94**, justified by citation alone.
* **Option B: Global threshold with 0.94 as a provisional default**, replaced by the value measured on MiniLM-L12 (§7).
* **Option C: Adaptive per-entry thresholds** (vCache-style online learning [2]). Requires per-entry observation state and background LLM-judge calls.

### 4. Decision Outcome
**Chosen Options:** **3.1-C** (two layers over one record), **3.2-B** (fail-closed exclusion denylist), **3.3-B** (calibrated global threshold).

**Primary Rationale:**
Option 3.1-C gives sub-millisecond exact hits without duplicating responses, and keeps TTL, eviction and deletion in one place. In caching, an allowlist of key parameters is a fail-open design: any parameter you forgot, or any parameter OpenAI ships next Tuesday, silently causes cache collisions and serves the wrong data. An exclusion denylist makes the cache fail-closed: an unknown parameter causes a cache miss, never data corruption. Finally, anisotropy makes raw scores model-specific [1], so the threshold is measured on our own model; adaptive thresholds (3.3-C) are deferred because they add Type 1 state and an LLM judge to the hot path.

`redisvl` is still used, through its lower-level `SearchIndex` and `VectorQuery` primitives. `SemanticCache` is not used directly, because it has no exact layer and no tenant or parameter scoping.

#### 4.1 Request Lifecycle

```
[Incoming Request]
      │
      ▼
 1. Eligibility Filter ──(n>1, logprobs, non-text input, audio)──────► BYPASS ─► Upstream
      │
      ▼
 2. Canonicalize ─► exact_hash, context_hash, param_fp
      │
      ▼
 3. Layer 1: HGETALL cache:v1:{tenant}:{exact_hash} ──(hit)─────────► HIT-EXACT
      │ miss
      ▼
 4. Layer 2 eligible? (last msg = user text, ≤256 word pieces) ──(no)─► Upstream
      │ yes
      ▼
 5. Embed last user message ─► KNN(k=1) filtered by @tenant + @scope
      │
      ├── distance ≤ 1 − τ and completion_tokens ≤ max_tokens ──────► HIT-SEMANTIC
      │
      └── otherwise ─► Upstream (stream to client) ─► Write-Path Filter ─► Store (background)
```

#### 4.2 Eligibility Filter (before canonicalization or embedding)
The request bypasses both layers (`X-Cache-Status: BYPASS`) if any of the following hold:
* `n > 1`: the caller explicitly wants varied generations.
* `logprobs` is true: the caller needs live token distributions.
* Any message contains a non-text content part (`image_url`, `input_audio`, `file`): the embedding sees only the text, so two different images with the same question would collide.
* `modalities` contains `"audio"`, or an `audio` parameter is present.
* `tools` / `tool_choice` are present **and** `CACHE_ALLOW_TOOLS=false`. The default is `true`: tool definitions enter the fingerprint automatically (§4.3), and tool-call responses are never stored (§4.7).

#### 4.3 Canonicalization & Parameter Fingerprint
One canonicalization function serves both layers:
1. Parse the JSON body.
2. Fill provider defaults for absent fields (`temperature=1.0`, `top_p=1.0`, `n=1`, `presence_penalty=0`, `frequency_penalty=0`), so an omitted field and its explicit default hash identically.
3. Round `temperature` and `top_p` to 2 decimals.
4. Remove the **exclusion denylist**: `stream`, `stream_options`, `user`, `metadata`, `store`, `max_tokens`, `max_completion_tokens`.
5. **Conditional seed rule:** if `temperature == 0.0`, decoding is greedy and the seed has no effect, so `seed` is removed. Otherwise `seed` is kept.
6. Serialize with sorted keys and compact separators. Message text is never altered (no trimming, no case folding).

Every field not on the denylist, including fields unknown today, stays in the hash. Unknown fields are logged so that harmless ones can be added to the denylist deliberately.

Three hashes are derived from the canonical form `C`:

| Hash | Input | Used for |
|---|---|---|
| `exact_hash` | SHA-256 of all of `C`, including all messages | Layer 1 key |
| `context_hash` | SHA-256 of all messages before the last user message | Layer 2 scope |
| `param_fp` | SHA-256 of `C` without `messages` | Layer 2 scope |

#### 4.4 Key & Index Schema
* **Tenant:** `tenant = sha256(api_key)[:16]`. Raw API keys are never stored. Consequences: rotating an API key starts that caller on an empty cache (old entries expire within one TTL), and two keys belonging to the same organization do not share entries. Mapping keys to a stable organization ID is a later Type 2 change.
* **Key:** `cache:{schema_version}:{tenant}:{exact_hash}`, e.g. `cache:v1:3f9a…:c41e…`
* **Index:** `idx:cache:{schema_version}`, created `ON HASH PREFIX cache:{schema_version}:`
* **Schema version** `v1` = the embedding model **and its exact ONNX file** (`all-MiniLM-L12-v2`, `onnx/model.onnx`; ADR-0002) + the canonicalization rules in §4.3. Changing the embedding model, the ONNX file (e.g. fp32 → qint8), the canonicalization or the denylist requires bumping the version (`CACHE_SCHEMA_VERSION`). Two models with the same dimension (e.g. MiniLM and `bge-small-en`, both 384) would otherwise be mixed silently in one index.

| Field | Type | Purpose |
|---|---|---|
| `tenant` | TAG | Isolation filter; per-tenant deletion and metrics |
| `scope` | TAG | `sha256(context_hash \| param_fp)`; Layer 2 filter |
| `embedding` | VECTOR (HNSW, FLOAT32, 384, COSINE) | Layer 2 search; absent for entries not eligible for Layer 2 |
| `response` | TEXT (not indexed) | Normalized response record (§4.8) |
| `completion_tokens` | NUMERIC | `max_tokens` check at read time |
| `created_at` | NUMERIC | `X-Cache-Age` |

#### 4.5 Layer 1: Exact Match
`HGETALL cache:v1:{tenant}:{exact_hash}`. A direct key read with no search and no embedding. A hit is served if `completion_tokens ≤ request.max_tokens` (when the request sets one). Otherwise the request continues to upstream.

#### 4.6 Layer 2: Semantic Match
* **Eligibility:** the last message must have role `user` with text content of at most 256 word pieces (the MiniLM truncation limit). Otherwise Layer 2 is skipped and Layer 1 alone applies. This covers, for example, a final answer that follows a tool result, whose last message has role `tool`.
* **Embedded text:** only the last user message. All earlier context (system prompt, prior turns, tool results) must match exactly through `context_hash`, so a follow-up can only match follow-ups to an identical conversation.
* **Query:** KNN with `k=1`, filtered by `@tenant:{t} @scope:{s}`. A hit requires Redis COSINE distance `≤ 1 − τ` (0–2 scale; τ = 0.94 → distance ≤ 0.06) and `completion_tokens ≤ request.max_tokens`.

#### 4.7 Write-Path Filter
A response is stored only if all of the following hold:
* Upstream returned HTTP 200 with `finish_reason == "stop"`.
* `message.refusal` is empty.
* The message contains no `tool_calls`.
* The request did not carry `Cache-Control: no-store`.

Responses with `finish_reason` of `length` (truncated), `content_filter` (blocked) or `tool_calls` are never stored. Writes run as background tasks after the response is delivered. A failed write is logged and never affects the client.

#### 4.8 Streaming
* **One normalized record per entry:** full text, `finish_reason`, `usage`, `model`. A hit is rendered as a `chat.completion` JSON or as SSE, according to the request's `stream` flag, so streaming and non-streaming requests share entries.
* **Streamed misses** are teed by the `StreamInterceptor` (ADR-0003). The record is written only after `data: [DONE]` arrives. To record `completion_tokens`, the gateway requests `stream_options.include_usage` upstream and strips the usage chunk if the client did not ask for it.
* **SSE replay (v1):** one delta chunk containing the full text, then `data: [DONE]`. Finer chunking is a later Type 2 change.
* **Client disconnect:** the upstream connection is cancelled immediately and nothing is stored. A user pressing "Stop" is often rejecting the answer, and continuing to read would also let a caller run up full-length upstream costs by opening and abandoning requests.

#### 4.9 Threshold Policy
* **Default:** τ = 0.94 (`CACHE_SIMILARITY_THRESHOLD`), provisional until §7 is run. After calibration, τ becomes the **lowest threshold whose measured error rate is ≤ 2%**.
* **Per-request override:** `X-Cache-Threshold` can only tighten matching: `τ_effective = clamp(max(τ_default, header), ≤ 1.0)`. Loosening is refused, because applications often forward client headers, which would let end users degrade answer quality and bypass the calibrated value.

#### 4.10 TTL & Eviction
* **TTL:** 7 days (`CACHE_TTL_SECONDS=604800`), **fixed**. Set on write and never refreshed on a hit, so even the most popular answer is regenerated at least once a week. This bounds staleness from changing facts and from model aliases such as `gpt-4o` moving to newer snapshots.
* **Eviction:** `maxmemory-policy allkeys-lfu`, `maxmemory 2gb` (development default). Most cache writes are one-off prompts that will never be hit again. LFU evicts those first, while LRU would let a burst of one-off writes push out popular answers.
* **Memory estimate:** about 4KB per entry (1.5KB vector + HNSW links + ~1.5KB average response + fields), so roughly 500k entries per 2GB. ADR-0002's 1.5GB per 1M figure counts vectors only.

#### 4.11 Failure Handling (fail-open on availability)
The whole lookup path (Layer 1 + embedding + Layer 2) runs under `CACHE_LOOKUP_TIMEOUT_MS=50`. Any Redis error or timeout is treated as a miss: the request goes upstream and the response carries `X-Cache-Status: ERROR`. This does not contradict §4.3. The fingerprint is fail-closed on **correctness** (when unsure, do not serve from the cache); this rule is fail-open on **availability** (when the cache is down, serve from the source). Neither ever serves a wrong answer.

#### 4.12 Response & Request Headers
| Header | Direction | Behavior |
|---|---|---|
| `X-Cache-Status` | Response | `HIT-EXACT` · `HIT-SEMANTIC` · `MISS` · `BYPASS` · `ERROR` |
| `X-Cache-Age` | Response | Seconds since the entry was written (hits only) |
| `X-Cache-Similarity` | Response | Similarity score on semantic hits, **only when `CACHE_DEBUG_HEADERS=true`**. Always written to logs. |
| `X-Cache-Threshold` | Request | Tighten-only override (§4.9) |
| `Cache-Control: no-cache` | Request | Skip lookup, call upstream, store the fresh answer (refresh) |
| `Cache-Control: no-store` | Request | Normal lookup, never store the response |

The similarity score is hidden by default because key-collision attacks refine adversarial prompts using hit signals [5]. An exact score per attempt would give an attacker a precise "warmer/colder" signal.

#### 4.13 Configuration Summary (`.env`)
| Variable | Default |
|---|---|
| `CACHE_SCHEMA_VERSION` | `v1` |
| `CACHE_SIMILARITY_THRESHOLD` | `0.94` (provisional, §7) |
| `CACHE_TTL_SECONDS` | `604800` |
| `CACHE_LOOKUP_TIMEOUT_MS` | `50` |
| `CACHE_ALLOW_TOOLS` | `true` |
| `CACHE_DEBUG_HEADERS` | `false` |

Redis (`docker-compose.yml`): `maxmemory 2gb`, `maxmemory-policy allkeys-lfu`.

### 5. Pros & Cons of the Chosen Option

* **Good (Positive Outcomes):**
  * [+] **Fail-closed correctness:** parameters that are forgotten or released in the future cause misses, not wrong answers.
  * [+] **Sub-millisecond exact hits** without loading the embedding model, plus semantic hits within the SLA.
  * [+] **Tenant isolation by construction:** every lookup is filtered by `tenant`. Per-tenant deletion and metrics come for free.
  * [+] **Single record per entry:** no duplicated responses; one TTL, one eviction and one delete path.
  * [+] **Format-agnostic hits:** streaming and non-streaming requests share entries.
  * [+] **Measurable threshold:** the production value is backed by a reproducible benchmark.
  * [+] **Safe schema evolution:** versioned prefixes make it impossible to mix incompatible vector spaces or hash rules.

* **Bad / Known Trade-offs (Consequences Accepted):**
  * [-] **Lower hit rate than a permissive cache:** exact context matching limits multi-turn semantic hits mostly to first turns. Fail-closed fingerprints split entries on any differing parameter.
  * [-] **Global threshold is not optimal per entry:** some equivalent paraphrases will miss and some near-boundary pairs will hit wrongly. This is bounded by the 2% error budget, not eliminated.
  * [-] **No request coalescing (thundering herd):** N identical concurrent misses produce N upstream calls until the first write lands. Planned upgrade: in-process single-flight via `asyncio.Future`.
  * [-] **Cold cache after any `v1 → v2` bump,** until entries are rebuilt (up to one TTL period for full turnover).
  * [-] **Cold cache per tenant after API key rotation,** because the tenant is derived from the key (§4.4).
  * [-] **Up to 7 days of staleness** for time-sensitive answers. Callers can force a refresh with `Cache-Control: no-cache`.
  * [-] **Flat SSE replay:** streamed hits arrive as one chunk, not token by token.
  * [-] **Interface change:** ADR-0001's `BaseVectorCache.check_similarity(vector, threshold)` has no tenant or scope parameters and must be extended (§6).

### 6. Mitigation & Reversibility

* **Interface Contract (supersedes the sketch in ADR-0001):**

```python
from abc import ABC, abstractmethod
from typing import Optional

class BaseVectorCache(ABC):
    @abstractmethod
    async def get_exact(self, tenant: str, exact_hash: str) -> Optional["CacheEntry"]:
        """Layer 1: direct key read. Returns None on miss."""

    @abstractmethod
    async def search_similar(
        self, tenant: str, scope: str, vector: list[float], max_distance: float
    ) -> Optional["CacheHit"]:
        """Layer 2: KNN(k=1) filtered by tenant and scope. Returns None if distance > max_distance."""

    @abstractmethod
    async def store(
        self, tenant: str, exact_hash: str, scope: str,
        entry: "CacheEntry", vector: Optional[list[float]], ttl: int
    ) -> None:
        """Write one record serving both layers. vector=None stores a Layer 1-only entry."""
```

* **Reversibility per decision:**

| Decision | Type | How to change |
|---|---|---|
| Key/index schema, tenant and scope fields | **Type 1** | New `CACHE_SCHEMA_VERSION` + index; old entries expire |
| Canonicalization rules, exclusion denylist | **Type 1** | Version bump (hash meaning changes) |
| Layer 2 embeds last user message + exact context | **Type 1** | Version bump |
| Normalized record shared by stream and non-stream | **Type 1** | Version bump + `StreamInterceptor` change |
| Similarity threshold value | Type 2 | `CACHE_SIMILARITY_THRESHOLD` |
| Header override direction | Type 2 | Code flag |
| TTL, eviction policy, `maxmemory` | Type 2 | `.env` / Redis config |
| Tools allowed, debug headers, lookup timeout | Type 2 | `.env` |
| Request coalescing | Type 2 | Add single-flight; no schema change |
| Adaptive per-entry thresholds | Type 1 | Future ADR; adds per-entry state |

* **Note on Type 1 cost:** because every entry expires within 7 days, a Type 1 change never requires a data migration. Its cost is a cold cache during turnover, and the version prefix guarantees old and new entries never mix.

### 7. Validation: Threshold Calibration
The provisional τ = 0.94 is replaced by a measured value before the first release.

* **Dataset:** `vCache/SemBenchmarkLmArena` [6] (Apache-2.0). 51,147 chat prompts in 3,500 groups (`ID_Set`), each group being one original LM-Arena prompt plus 1–23 paraphrases generated by GPT-4.1-nano. Only the `prompt` and `ID_Set` columns are used. The shipped precomputed embeddings do not include MiniLM, so embeddings are computed locally (about 51k × ~7ms ≈ 6 min on CPU).
* **Protocol (sequential replay, reproducible):**
  1. Shuffle once with a fixed seed.
  2. For each prompt in order: find the nearest cached neighbor (k=1). If similarity ≥ τ, record a hit, which is correct when the neighbor's `ID_Set` equals the prompt's. Otherwise record a miss and insert the prompt.
  3. Repeat for each τ in **{1.00, 0.99, 0.98, 0.97, 0.95, 0.94, 0.92, 0.89, 0.86, 0.80}**.
  4. Repeat for both `all-MiniLM-L6-v2` and `all-MiniLM-L12-v2`. The model with the higher hit rate within the error budget is adopted (ADR-0002 §6); if they tie, L6 is adopted for its lower CPU cost.
* **Metrics:** **Hit rate** = hits / all requests. **Error rate** = wrong hits / all requests (vCache's definition, for comparability).
* **Selection rule:** production τ = the lowest τ with error rate ≤ 2%. If it falls between two rows, that one value is added to the table.
* **Reference points:** τ = 1.00 approximates Layer 1 alone. The theoretical ceiling is ≈ 93% ((51,147 − 3,500) / 51,147), since the first prompt of each group must miss.
* **Artifacts:** `benchmarks/threshold_sweep.py` (`--models L6 L12`); results table in the README:

| Threshold | Hit Rate | Error Rate |
|---|---|---|
| 1.00 | TBD | TBD |
| 0.99 | TBD | TBD |
| 0.98 | TBD | TBD |
| 0.97 | TBD | TBD |
| 0.95 | TBD | TBD |
| 0.94 | TBD | TBD |
| 0.92 | TBD | TBD |
| 0.89 | TBD | TBD |
| 0.86 | TBD | TBD |
| 0.80 | TBD | TBD |

* **Limitations:**
  * The paraphrases are LLM-generated and cleaner than real traffic, so the result is an estimate that should be re-checked against production logs.
  * For context, vCache measured static thresholds on E5-large-v2 at 0.99 → 2.5% error / 37% hit, 0.98 → 4.1% / 53%, and 0.97 → 5.2% / 67% [2]. Results for MiniLM-L12 are expected to differ.
* **Licensing note:** only the vCache method and datasets are used. The vCache source code is licensed CC BY-NC-ND 3.0, and none of it is copied or adapted.

### 8. References & Benchmarks
1. **Tacheny, N. (2026).** *Calibrated Similarity for Reliable Geometric Analysis of Embedding Spaces.* arXiv:2601.16907. Raw cosine scores concentrate in a narrow high-similarity band due to anisotropy; *"Different models require separate calibration."* Cited for model-specificity only; the paper does not address caching or recommend cache thresholds.
2. **Schroeder, L. G., et al. (2026).** *vCache: Verified Semantic Prompt Caching.* ICLR 2026. [arXiv:2502.03771](https://arxiv.org/abs/2502.03771). Overlapping similarity distributions for correct and incorrect hits; static-threshold error/hit trade-offs; per-entry adaptive thresholds.
3. **vCache ICLR 2026 slides.** [iclr.cc/media/iclr-2026/Slides/10006471.pdf](https://iclr.cc/media/iclr-2026/Slides/10006471.pdf). Federer / Round Robin threshold examples.
4. **Wu, G., et al. (2026).** *When Cache Poisoning Meets LLM Systems: Semantic Cache Poisoning and Its Countermeasures.* NDSS 2026. Poisoning confirmed on three major public clouds.
5. **From Similarity to Vulnerability: Key Collision Attack on LLM Semantic Caching (2026).** [arXiv:2601.23088](https://arxiv.org/abs/2601.23088). 86% response-hijack rate on GPTCache; 80–88% on AWS Bedrock and Azure API Management; per-user isolation listed as a mitigation.
6. **vCache/SemBenchmarkLmArena.** Hugging Face dataset, Apache-2.0. [huggingface.co/datasets/vCache/SemBenchmarkLmArena](https://huggingface.co/datasets/vCache/SemBenchmarkLmArena).
7. **Singh, A. K., et al. (2026).** *Asynchronous Verified Semantic Caching for Tiered LLM Architectures.* [arXiv:2602.13165](https://arxiv.org/abs/2602.13165). Static thresholds tuned to a 1–2% error budget in production-style tiered caches.
8. **Apache APISIX `ai-cache` plugin.** [apisix.apache.org](https://apisix.apache.org/docs/apisix/plugins/ai-cache/). Reference two-layer design (SHA-256 exact layer + Redis Stack semantic layer); SSE caching only after the terminal event.
9. **Bifrost semantic caching.** [docs.getbifrost.ai](https://docs.getbifrost.ai/features/semantic-caching.md). Tighten-only per-request threshold header; identity-scoped cache entries.
10. **Redis Key Eviction.** [redis.io/docs/latest/develop/reference/eviction](https://redis.io/docs/latest/develop/reference/eviction/). LRU vs LFU policies and approximated eviction.
11. **RedisVL Cache API.** [redis.io/docs/latest/develop/ai/redisvl/api/cache](https://redis.io/docs/latest/develop/ai/redisvl/api/cache/). COSINE distance units (0–2) and `SemanticCache` limitations.
