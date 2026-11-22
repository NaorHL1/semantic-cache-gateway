# ⚡ Semantic Caching AI Gateway (`semantic-cache-gateway`)

> **OpenAI-compatible AI proxy that answers repeated and paraphrased questions from an in-memory semantic cache, in under 25ms and at zero token cost, with a measured error budget instead of a guessed one.**
>
> *Design goals. The design is documented in the ADRs below; no gateway code or measurements exist yet (see Project Status).*

---

## 🎯 Overview

Modern Large Language Model (LLM) applications face a triple constraint in production:
1. **Severe Latency:** Generation takes seconds per query.
2. **Repeated Token Costs:** Repetitive queries drain API budgets.
3. **Traditional Caches Fail:** Exact-string matching misses semantically identical questions phrased differently (*"How do I reset my password?"* vs. *"Forgot password, how to recover access?"*).

**Semantic Cache Gateway** intercepts `POST /v1/chat/completions` and checks two cache layers before calling the upstream LLM:
* **Layer 1 (exact):** a canonical SHA-256 of the request, looked up directly in Redis (<1ms, no embedding).
* **Layer 2 (semantic):** a local embedding of the last user message (~7ms on CPU), searched in a Redis HNSW index within the caller's tenant and conversation scope.
* **Cache HIT:** the stored answer is returned as JSON or SSE, at $0 token cost.
* **Cache MISS:** the request goes upstream, the answer streams to the client, and it is cached in the background only if it completed cleanly.

Correctness comes first: any request parameter the gateway does not recognize causes a miss, never a wrong hit (ADR-0004).

---

## 📍 Project Status

**Phase: design complete, implementation starting.**

| Done | Not yet |
|---|---|
| ADR-0001 to ADR-0005 (storage, embedding, runtime, cache algorithm, upstream client) | Gateway logic: `canonicalize()`, Layer 1, Layer 2, streaming interceptor |
| `docker-compose.yml` for Redis 8.10.2 (no persistence, 2GB `allkeys-lfu`) | Measurements: threshold sweep, L6 vs L12, Hebrew check, load tests |
| Python package scaffold (`src/semantic_cache_gateway/`, `uv`, `pyproject.toml`) | Shadow verification (ADR-0004 §4.14) |
| First failing tests for canonicalization (`tests/test_canonical.py`, test-first) | Tenant isolation tests, PII handling, observability |

`src/semantic_cache_gateway/main.py` is still the default FastAPI placeholder app.

### Decisions at a glance

Values marked *provisional* are starting points to be replaced by measurements.

| Area | Decision | Source |
|---|---|---|
| Vector store | Redis 8.10.2 Open Source, HNSW (M=16, EF_CONSTRUCTION=200, EF_RUNTIME=10) | ADR-0001 |
| Persistence | None (`--save ""`, `--appendonly no`); cold cache after a restart is accepted | ADR-0001 |
| Memory | `maxmemory 2gb`, `allkeys-lfu`, fixed 7-day TTL | ADR-0004 |
| Embedding | `all-MiniLM-L12-v2` via FastEmbed (ONNX), 1 thread per worker, off the event loop (L6 vs L12 pending sweep) | ADR-0002 |
| Runtime | FastAPI on Granian | ADR-0003 |
| Cache algorithm | Two layers over one record; fail-closed fingerprint; tenant and scope filters | ADR-0004 |
| Similarity threshold | 0.94, *provisional* until the calibration sweep (error budget 2%) | ADR-0004 |
| Lookup timeout | 50ms, then fail-open to the upstream | ADR-0004 |
| Upstream client | aiohttp behind `UpstreamClient` | ADR-0005 |
| Pool, timeouts, retries | 200 connections, connect 5s, read 30s, 2 retries before the first byte; all *provisional* | ADR-0005 |
| Extension (planned) | Shadow verification: measure the live wrong-hit rate per tenant | ADR-0004 §4.14 |

---

## 🏗️ High-Level Architecture

```
[ Client / App ]
       │
       ▼
[ FastAPI on Granian ] ── eligibility filter ──► BYPASS ──► Upstream LLM
       │
       ▼
 Layer 1: exact canonical hash ──(hit)──► HIT-EXACT  (<1ms)
       │ miss
       ▼
 Layer 2: FastEmbed (~7ms) ─► Redis HNSW KNN, filtered by tenant + scope
       │
       ├──► similarity ≥ τ ──► HIT-SEMANTIC  (<25ms, $0)
       │
       └──► miss ──► Upstream LLM (OpenAI / vLLM) ──► stream to client
                                       │
                                       └──► store in background (finish_reason = stop only)
```

---

## ⏱️ Latency Budget

All documents refer to this table. Figures marked *est.* will be replaced by measurements from the planned `benchmarks/` suite.

| Step | Budget |
|---|---|
| Layer 1 lookup (`HGETALL`) | <1ms |
| Embedding, MiniLM on 1 CPU thread | ~7ms *(est.)* |
| KNN search in Redis (HNSW) | 1–3ms *(est.)* |
| **Semantic hit, end to end** | **<25ms** |
| Cache lookup timeout (fail-open to upstream) | 50ms |
| Upstream miss | seconds, depending on model and output length |

Throughput on the semantic path is bounded by embedding CPU: about 140 lookups per second per core (ADR-0002).

---

## 📊 Benchmark: Hit Rate vs. Error Rate

To be measured on [vCache/SemBenchmarkLmArena](https://huggingface.co/datasets/vCache/SemBenchmarkLmArena) (51,147 chat prompts, 3,500 groups) with `benchmarks/threshold_sweep.py` (planned; see ADR-0004 §7). Error rate = wrong hits / all requests. The production threshold is the lowest one with an error rate ≤ 2%.

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

Theoretical hit-rate ceiling on this dataset: ≈93% (the first prompt of each group must miss).

---

## 📑 Architecture Decision Records (ADRs)

Key architectural trade-offs and structural choices are formally tracked and documented in [`docs/adr/`](docs/adr/):

| ADR | Title | Status | Chosen Solution | Primary Trade-off |
| :--- | :--- | :--- | :--- | :--- |
| **[ADR-0001](docs/adr/0001-in-memory-vector-store.md)** | In-Memory Storage & Vector Engine | **Accepted** | Redis 8 Open Source (`redis:8.10.2`) + `redisvl` | Client-library support over raw multi-core throughput |
| **[ADR-0002](docs/adr/0002-local-embedding-engine.md)** | Local Embedding Engine & Architecture | **Accepted** *(L6 vs. L12 pending sweep)* | `all-MiniLM-L12-v2` via FastEmbed (ONNX) | ~7ms CPU and 384-dim density over model breadth; English-centric |
| **[ADR-0003](docs/adr/0003-gateway-runtime-framework.md)** | Gateway Runtime & ASGI Server | **Accepted** | FastAPI + Granian | Python ML synergy; small, reversible server-speed gain |
| **[ADR-0004](docs/adr/0004-semantic-cache-algorithm.md)** | Two-Layer Semantic Cache Algorithm & Threshold Policy | **Accepted** *(threshold provisional)* | Two layers over one record, fail-closed fingerprint, calibrated threshold | Correctness (≤2% error) over hit rate |
| **[ADR-0005](docs/adr/0005-upstream-http-client.md)** | Upstream HTTP Client, Pool, Timeouts & Retries | **Accepted** *(values provisional)* | aiohttp behind an `UpstreamClient` wrapper | Streaming and concurrency fit over HTTP/2 and API convenience |

**Planned extension:** shadow verification, which measures the live wrong-hit rate per tenant by re-checking a small sample of cache hits against the upstream (ADR-0004 §4.14, not yet implemented).

---

## 🚀 Quick Start

Requires Python 3.13, [`uv`](https://docs.astral.sh/uv/) and Docker. Steps 1 to 3 work today. Step 4 starts only the placeholder app until the gateway is implemented.

```bash
# 1. Clone & enter repository
git clone https://github.com/NaorHL1/semantic-cache-gateway.git
cd semantic-cache-gateway

# 2. Install dependencies
uv sync

# 3. Launch Redis 8 (vector search built in)
docker compose up -d
docker compose exec redis redis-cli ping    # expected: PONG

# 4. Start the app (Rust-powered Granian ASGI server)
uv run granian --interface asgi semantic_cache_gateway.main:app --reload --port 8000
```

### Benchmarks (planned)

The evaluation suite will add `benchmarks/threshold_sweep.py` (threshold calibration on L6 and L12, ADR-0004 §7) and `benchmarks/hebrew_check.py` (non-English safety check, ADR-0002 §6).
