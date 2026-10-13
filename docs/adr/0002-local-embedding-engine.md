# ADR-0002: Local Embedding Engine & Model Architecture Selection

* **Status:** Accepted
* **Date:** 2026-10-13
* **Deciders:** Lead Architect & FDE Team

---

### 1. Context & Problem Statement
The Semantic Caching AI Gateway requires an embedding generation pipeline that converts incoming user prompts into high-dimensional vector representations in real time. Because every incoming query must be embedded before querying the in-memory cache, this embedding stage directly dictates the gateway's p99 latency baseline, operational cost, and overall availability.

Using cloud-hosted embedding APIs (e.g., OpenAI, Google Gemini, Voyage AI) incurs an unacceptable network round-trip penalty (60ms–200ms+), costs money on every cache lookup (including misses), and introduces external rate limits. Therefore, embedding inference must run locally in-process on CPU with a strict latency budget of <10ms.

### 2. Decision Drivers
* **Inference Latency & Budget:** Sub-10ms CPU inference per query to guarantee an end-to-end cache hit SLA of <25ms.
* **In-Memory Cache RAM Footprint:** Vector dimensionality directly scales Redis RAM consumption (`Dimension * 4 bytes` per float32 vector + HNSW indexing graph overhead). Lower dimensions significantly reduce operational memory costs.
* **Zero Network Overhead & Offline Operation:** In-process embedding generation eliminates external HTTP calls, API token fees, and vendor rate limits.
* **Semantic Clustering Quality:** High semantic clustering fidelity on MTEB benchmarks to accurately differentiate subtle prompt nuances without false positive cache hits.
* **Lightweight Runtime Footprint:** Seamless deployment on standard CPU containers using quantized ONNX Runtime (`fastembed`), avoiding heavy PyTorch dependencies and dedicated GPU hardware.

### 3. Considered Options
* **Option 1: Sentence-Transformers `all-MiniLM-L12-v2` via FastEmbed (ONNX Runtime)** — 33M parameter bi-encoder (12 layers), generating 384-dimensional dense vectors.
* **Option 2: Sentence-Transformers `all-MiniLM-L6-v2` via FastEmbed (ONNX Runtime)** — 22M parameter bi-encoder (6 layers), generating 384-dimensional dense vectors.
* **Option 3: BAAI `bge-base-en-v1.5` via FastEmbed (ONNX Runtime)** — 110M parameter bi-encoder (12 layers, 768 width), generating 768-dimensional dense vectors.
* **Option 4: Cloud-Hosted Proprietary APIs (`text-embedding-3-small`, `gemini-embedding-2`)** — Remote managed APIs generating 768–1536 dimensional vectors over HTTPS.

### 4. Decision Outcome
**Chosen Option:** **Option 1: `sentence-transformers/all-MiniLM-L12-v2` executed via FastEmbed (ONNX Runtime)**

**Primary Rationale:**
Semantic caching requires balancing inference speed, memory density, and semantic fidelity:
1. **Latency & Execution:** Running quantized `all-MiniLM-L12-v2` via ONNX Runtime achieves 6–8ms CPU execution per prompt without requiring GPUs or external network calls.
2. **Optimal Memory Density:** With 384 dimensions, 1,000,000 cached prompts consume only ~1.5 GB of RAM in Redis Stack (compared to ~3.1 GB for 768-dim models like BGE-Base and ~6.1 GB for 1,536-dim cloud models).
3. **Semantic Fidelity over L6:** Compared to `all-MiniLM-L6-v2` (6 layers), the 12-layer L12 architecture provides significantly higher semantic separation and clustering stability (MTEB Clustering score 59.4 vs 56.8), drastically reducing false-positive cache collisions at similarity thresholds of 0.92–0.96 for only a negligible 2–3ms compute trade-off.
4. **Cloud APIs Disqualified:** Remote APIs introduce 60–200ms latency and per-query operational costs, defeating the purpose of a sub-30ms cache proxy.

### 5. Pros & Cons of the Chosen Option

* **Good (Positive Outcomes):**
  * [+] **Deterministic Sub-10ms Latency:** Local in-process execution on CPU completely eliminates network jitter, TLS handshakes, and third-party API throttling.
  * [+] **Compact Redis Memory Footprint:** 384-dimensional vectors reduce Redis vector index memory consumption by 50% vs 768-dim models and 75% vs 1536-dim models.
  * [+] **Zero Cloud API Costs:** Embeddings are generated at zero marginal token cost on both cache hits and cache misses.
  * [+] **Lean Deployment Container:** `fastembed` uses ONNX Runtime directly, avoiding bulky PyTorch GPU binaries (>2GB reduced container size).

* **Bad / Known Trade-offs (Consequences Accepted):**
  * [-] **CPU Contention under High Concurrency:** Running local model inference on the same CPU instances as the FastAPI routing layer requires worker process tuning to prevent event loop thread-pool saturation.
  * [-] **English-Centric Vocabulary:** `all-MiniLM-L12-v2` is optimized primarily for English text; multilingual queries exhibit degraded semantic resolution compared to multilingual models (e.g. `bge-m3`).
  * [-] **Sequence Length Limit:** Context window is bounded to 256 tokens (sufficient for chat prompts, but unsuitable for full-document caching).

### 6. Mitigation & Reversibility
* **Encapsulation via Interface Contract:** Embedding generation is isolated behind an abstract `BaseEmbeddingEngine` contract:

```python
from abc import ABC, abstractmethod

class BaseEmbeddingEngine(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality (e.g. 384)."""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate a normalized dense vector representation for a query string."""
        pass
```

* **Mitigating CPU Thread Starvation:** FastEmbed inference calls are offloaded from FastAPI's async event loop using `asyncio.to_thread` or a dedicated ProcessPoolExecutor to avoid blocking concurrent HTTP connections.
* **Reversibility (Type 2 Decision):** If multilingual support or higher dimensionality becomes necessary, swapping to `bge-small-en-v1.5` or `multilingual-e5-small` only requires updating the adapter class and rebuilding the Redis index schema (`DIM 384` to target dimension), without altering gateway proxy logic.
