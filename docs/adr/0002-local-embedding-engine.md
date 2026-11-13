# ADR-0002: Local Embedding Engine & Model Architecture Selection

* **Status:** Accepted (revised 2026-10-16: loading path, benchmark evidence and throughput figures corrected; L6 vs. L12 and Hebrew handling pending the measurements in §6)
* **Date:** 2026-10-13
* **Deciders:** Lead Architect & FDE Team

---

### 1. Context & Problem Statement
The Semantic Caching AI Gateway requires an embedding generation pipeline that converts incoming user prompts into dense vectors in real time. Because every request that reaches Layer 2 (ADR-0004) must be embedded before querying the cache, this stage sets the gateway's p99 latency baseline, its throughput ceiling, and part of its operational cost.

Cloud-hosted embedding APIs (OpenAI, Google Gemini, Voyage AI) add a network round-trip of 60ms–200ms+, cost money on every lookup (including misses), and introduce external rate limits. Embedding inference must therefore run locally, in-process, on CPU, within a budget of <10ms per prompt.

### 2. Decision Drivers
* **Inference Latency & Budget:** <10ms CPU inference per query, to keep a semantic hit under the 25ms SLA.
* **In-Memory Cache RAM Footprint:** Vector dimensionality directly scales Redis memory (`dimension × 4 bytes` per float32 vector, plus HNSW graph overhead).
* **Zero Network Overhead & Offline Operation:** No external HTTP calls, API fees or vendor rate limits.
* **Semantic Separation Quality:** The model must separate paraphrases from different questions, because false positives become wrong cache hits.
* **Lightweight Runtime Footprint:** CPU-only ONNX Runtime via `fastembed`, with no PyTorch dependency or GPU.

### 3. Considered Options
* **Option 1: `sentence-transformers/all-MiniLM-L12-v2` via FastEmbed (ONNX Runtime)**: 33M parameters, 12 layers, 384 dimensions.
* **Option 2: `sentence-transformers/all-MiniLM-L6-v2` via FastEmbed (ONNX Runtime)**: 22M parameters, 6 layers, 384 dimensions, roughly half the compute of L12.
* **Option 3: BAAI `bge-base-en-v1.5` via FastEmbed (ONNX Runtime)**: 110M parameters, 768 dimensions.
* **Option 4: Cloud-hosted APIs (`text-embedding-3-small`, `gemini-embedding-2`)**: Remote managed APIs, 768–1536 dimensions, over HTTPS.

### 4. Decision Outcome
**Chosen Option:** **Option 1: `all-MiniLM-L12-v2` executed via FastEmbed (ONNX Runtime), provisionally.** The final choice between L12 and L6 is made by the calibration sweep in ADR-0004 §7, run on both models (see §6).

**Primary Rationale:**
1. **Latency & Execution:** A local ONNX forward pass runs in single-digit milliseconds on CPU, with no GPU and no network call. The exact figure must be measured on target hardware (§6).
2. **Memory Density:** 384 float32 dimensions = 1.5KB per vector, so 1M vectors take ~1.5GB **for the vectors alone**. A full cache entry (vector + HNSW links + response + fields) is about 4KB (ADR-0004 §4.10). 768-dim and 1536-dim models double or quadruple the vector share.
3. **L12 vs. L6, measured:** Recomputed from the official MTEB raw results [1]:

| MTEB category (English) | L6 | L12 |
|---|---|---|
| Clustering (11 tasks) | 41.91 | 41.82 |
| Pair classification / duplicate detection (3 tasks) | 82.37 | 82.46 |
| STS, semantic similarity (9 comparable tasks) | 80.2 | 81.4 |

   L12 has a small advantage in semantic similarity (+1.2 STS) and is tied on clustering and duplicate detection, at roughly twice the compute. This is too small a margin to decide by benchmark tables alone, so the decision is made on our own task (§6).
4. **Cloud APIs Disqualified:** Remote APIs add 60–200ms latency and per-query cost, defeating the purpose of a sub-30ms cache.

**Loading the model (L12 is not in FastEmbed's built-in catalog):**
FastEmbed 0.9.0 ships `all-MiniLM-L6-v2` and `paraphrase-multilingual-MiniLM-L12-v2`, but not `all-MiniLM-L12-v2`. `TextEmbedding("sentence-transformers/all-MiniLM-L12-v2")` raises `ValueError: Model ... is not supported`. The model's own Hugging Face repository includes ONNX exports [2], so it is registered as a custom model:

```python
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

MODEL_ID = "sentence-transformers/all-MiniLM-L12-v2"
TextEmbedding.add_custom_model(
    model=MODEL_ID,
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf=MODEL_ID),
    dim=384,
    model_file="onnx/model.onnx",  # pinned: part of cache schema v1 (ADR-0004 §4.4)
)
embedder = TextEmbedding(MODEL_ID, threads=1)  # one ONNX thread per Granian worker
```

The exact `model_file` is part of the cache schema version. fp32 (`model.onnx`) and quantized variants (`model_qint8_avx512.onnx`, `model_quint8_avx2.onnx`, …) produce slightly different similarity scores, so a threshold calibrated on one does not carry over to another.

### 5. Pros & Cons of the Chosen Option

* **Good (Positive Outcomes):**
  * [+] **Deterministic Local Latency:** In-process CPU execution removes network jitter, TLS handshakes and third-party throttling.
  * [+] **Compact Memory Footprint:** 384-dim vectors use 50% of the vector memory of 768-dim models and 25% of 1536-dim models.
  * [+] **Zero Cloud API Costs:** Embeddings cost nothing at the margin, on hits and misses alike.
  * [+] **Lean Deployment Container:** ONNX Runtime without PyTorch.

* **Bad / Known Trade-offs (Consequences Accepted):**
  * [-] **Embedding Is the Throughput Ceiling:** At ~7ms of CPU per prompt, one core sustains about **140 Layer 2 lookups per second** (estimated from a load test where a 6.6ms NumPy workload stood in for one embedding: 140 req/s on a single event loop, §6; the real MiniLM figure is pending measurement). This, not the HTTP server, bounds gateway throughput on the semantic path, and it scales with cores.
  * [-] **English-Centric Model:** `all-MiniLM-L12-v2` was trained on English data. For other languages, Hebrew included, this is a **correctness** risk, not only a quality one: unrelated non-English prompts may receive high similarity scores and produce false cache hits. Pending measurement (§6).
  * [-] **Sequence Length Limit:** Input is truncated at 256 word pieces, and the model was fine-tuned on sequences of only 128 tokens [3]. Quality beyond 128 tokens is untested. ADR-0004 skips Layer 2 above 256 word pieces.

### 6. Mitigation & Reversibility
* **Encapsulation via Interface Contract:** Embedding generation is isolated behind an abstract `BaseEmbeddingEngine` contract:

```python
from abc import ABC, abstractmethod

class BaseEmbeddingEngine(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality (e.g. 384)."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate a normalized dense vector for a query string."""
```

* **Mandatory Thread Offloading (correctness, not optimization):** Every FastEmbed call runs via `asyncio.to_thread`. ONNX Runtime releases the GIL during inference, so offloading keeps the event loop free. In a local load test with a 6.6ms CPU workload at 32 concurrent requests, a trivial request issued alongside waited **p50 220ms** when the work ran on the event loop, versus **p50 2.6ms** when it was offloaded. Without offloading, one embedding blocks every sub-millisecond Layer 1 hit.
* **CPU Oversubscription:** Each worker creates its ONNX session with `threads=1`, so `workers × ONNX threads ≤ cores`.
* **Pending measurements (planned scripts under `benchmarks/`, added in the evaluation phase):**
  * `threshold_sweep.py --models L6 L12`: hit rate and error rate per threshold on SemBenchmarkLmArena for both models, plus per-prompt embedding time on target hardware. The model that reaches the higher hit rate within the 2% error budget is selected. If they are tied, L6 wins, being twice as fast and built into FastEmbed.
  * `hebrew_check.py`: compares similarity of unrelated English pairs, unrelated Hebrew pairs, Hebrew paraphrases and Hebrew opposite-intent pairs across L6, L12 and `paraphrase-multilingual-MiniLM-L12-v2`. The result decides between: skipping Layer 2 for non-English prompts (fail-closed), switching to the multilingual model, or declaring English-only support.
* **Reversibility (Type 1 at the data level):** Swapping the model or the ONNX file only requires a new adapter class or a configuration change, with no gateway logic changes. However, vectors from different models (even with the same dimension) are not comparable, so a swap requires a new cache schema version and starts from a cold cache (ADR-0004 §4.4). Entries from the old version expire within one TTL period.

### 7. References
1. **MTEB raw results.** `results/sentence-transformers__all-MiniLM-L6-v2` and `__all-MiniLM-L12-v2` (main scores averaged per category; STS22 excluded because the L12 result uses a different subset). [github.com/embeddings-benchmark/results](https://github.com/embeddings-benchmark/results)
2. **all-MiniLM-L12-v2 ONNX exports.** [huggingface.co/sentence-transformers/all-MiniLM-L12-v2/tree/main/onnx](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/tree/main/onnx)
3. **all-MiniLM-L12-v2 model card.** [huggingface.co/sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
4. **FastEmbed 0.9.0.** Supported models and `TextEmbedding.add_custom_model`. [github.com/qdrant/fastembed](https://github.com/qdrant/fastembed)
