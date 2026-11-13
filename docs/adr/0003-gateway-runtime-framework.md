# ADR-0003: Gateway Runtime Framework & ASGI Server Selection

* **Status:** Accepted (revised 2026-10-16: performance rationale corrected with workload-specific measurements; streaming contract moved to ADR-0004)
* **Date:** 2026-10-13
* **Deciders:** Lead Architect & FDE Team

---

### 1. Context & Problem Statement
The Semantic Caching AI Gateway is a reverse proxy in front of upstream LLM providers (OpenAI, Anthropic, vLLM). It must intercept `POST /v1/chat/completions`, validate the request, run in-process ONNX embeddings (ADR-0002), query Redis asynchronously (ADR-0001), and stream Server-Sent Events (SSE) both from upstream and when replaying cache hits.

Its workload has two distinct cost profiles:
* **Cache miss:** dominated by waiting on the upstream LLM, from hundreds of milliseconds to several seconds. The server must hold many concurrent streams open cheaply.
* **Semantic hit:** dominated by about 7ms of CPU for one embedding, plus a few milliseconds of Redis I/O.

Neither profile is limited by HTTP parsing speed. The runtime must keep Python's ML ecosystem (`fastembed`, `onnxruntime`, `redisvl`, `pydantic`) in-process, and keep CPU work off the event loop.

### 2. Decision Drivers
* **AI & In-Process ML Interoperability:** Native, zero-FFI use of `fastembed`, `onnxruntime`, `redisvl` and `pydantic-v2`.
* **Many Concurrent Long-Lived Streams:** Cheap concurrency for thousands of open SSE connections waiting on upstream.
* **CPU/Async Isolation:** CPU-bound embedding must never block the event loop.
* **Streaming Primitives:** Async generators to tee upstream SSE and to synthesize SSE on cache hits.
* **Development Velocity & Maintainability:** Typed, modular Python suitable for portfolio review.

### 3. Considered Options
* **Option 1: FastAPI + Granian**: FastAPI on Granian, an ASGI server implemented in Rust (`hyper` + `tokio` via `PyO3`).
* **Option 2: FastAPI + Uvicorn**: The standard Python ASGI stack (`uvloop` + `httptools`).
* **Option 3: Go (`net/http` / Bifrost-style)**: Compiled proxy with very low overhead, but ONNX inference requires Cgo bindings.
* **Option 4: Rust (`Axum` + `ort`)**: Maximum efficiency, but no high-level Redis vector client and much slower iteration.

### 4. Decision Outcome
**Chosen Option:** **Option 1: FastAPI on Granian**, with Uvicorn as a drop-in fallback.

**Primary Rationale:**
Python is chosen for in-process ML interoperability, which rules out Options 3 and 4. Between the two Python servers, Granian is chosen for a modest, measurable advantage, not for an order-of-magnitude one.

Granian's own benchmark suite (run 2026-08-05; AMD Ryzen 7 5700X, 16 CPUs, Python 3.13, Granian 2.8.1) shows the gap shrinking to nothing as request handling becomes wait-bound [1]:

| ASGI scenario (c128 / c512) | Granian RPS | Uvicorn (httptools) RPS | Advantage |
|---|---|---|---|
| 10KB plain-text GET | 125,539 | 51,051 | ×2.5 |
| 10KB streamed echo POST | 63,181 | 50,665 | ×1.25 |
| Long I/O, 10ms wait | 45,859 | 43,903 | ~4% |
| Long I/O, 100ms wait | 5,067 | 5,053 | ~0% |

A local load test with a gateway-shaped ASGI app (Uvicorn, 2 CPUs) confirms that the server is not the bottleneck for either cost profile [2]:

| Endpoint | Result |
|---|---|
| Trivial response | 7,109 req/s |
| 1s upstream wait, 500 concurrent | 480 req/s, p50 1,022ms: all requests in flight at once; throughput = concurrency / upstream latency |
| 6.6ms CPU on the event loop | 140 req/s; a concurrent trivial request waited p50 220ms |
| 6.6ms CPU via `asyncio.to_thread` | 203 req/s; a concurrent trivial request waited p50 2.6ms |

Gateway throughput is therefore set by upstream latency (misses) and embedding CPU (semantic hits), not by the ASGI server. Granian remains the default because it is never slower, lowers overhead on the cheapest paths (Layer 1 hits, health checks), and switching back is a single command.

### 5. Pros & Cons of the Chosen Option

* **Good (Positive Outcomes):**
  * [+] **Lower Server Overhead:** HTTP parsing and socket management run in Rust, which helps most on cheap, high-rate paths such as exact-match hits.
  * [+] **First-Class ML Integration:** FastEmbed, ONNX Runtime and `redisvl` run in-process without FFI.
  * [+] **Fast Validation:** Pydantic v2 validates OpenAI payloads through its compiled core (`pydantic-core`).
  * [+] **Streaming Primitives:** `StreamingResponse` and async generators support SSE teeing and synthesis.

* **Bad / Known Trade-offs (Consequences Accepted):**
  * [-] **The GIL Still Applies to Application Code:** Granian removes HTTP handling from Python, but routing, canonicalization, hashing and validation glue still run under each worker's GIL. Scaling is by worker processes.
  * [-] **Memory Footprint per Worker:** The Python runtime plus loaded ONNX weights need roughly 150–220MB per worker, compared with ~20MB for Go or Rust.
  * [-] **Small Real-World Gain:** On this workload the measured advantage over Uvicorn is small. The choice is low-risk because it is fully reversible.

### 6. Mitigation & Reversibility
* **Thread Offloading for CPU Tasks:** All FastEmbed calls run via `asyncio.to_thread` (ADR-0002 §6). The load test above shows this is required for correctness of the latency SLA, not just for throughput.
* **Streaming Interceptor:** Upstream SSE will be teed by `StreamInterceptor` (planned: `src/gateway/stream_interceptor.py`). Its storage rules are defined in **ADR-0004 §4.7–4.8**:
  * Forward bytes immediately, buffering partial SSE events across TCP chunks.
  * Store only after `data: [DONE]` with `finish_reason == "stop"`, no refusal, no tool calls, and recorded usage.
  * Close upstream and store nothing on client disconnect or upstream error.
  * Write to the cache as a background task, never inline.
  
  Each of these behaviors must be covered by `tests/test_stream_interceptor.py` (planned, implementation phase).
* **Open dependency:** The upstream HTTP client (library, connection pool, timeouts, retries) is not decided here and will be covered by ADR-0005. Note that the OpenAI Python SDK's default transport (httpx/httpcore) has a documented connection-pool scaling problem at high concurrency [3].
* **Reversibility (Type 2 Decision):** The gateway is a standard ASGI app (`app = FastAPI()`), so the server can be swapped with one command (`granian --interface asgi src.main:app` ↔ `uvicorn src.main:app`) without changing application code.

### 7. References & Benchmarks
1. **Granian benchmarks: comparison with other servers.** Run 2026-08-05. [github.com/emmett-framework/granian/blob/master/benchmarks/vs.md](https://github.com/emmett-framework/granian/blob/master/benchmarks/vs.md)
2. **Local gateway-shaped load test.** Scripts to be committed under `benchmarks/server_load/` (`app.py`, `load.py`). Uvicorn 0.53, Python 3.13, 2 CPUs, 6.6ms single-threaded NumPy workload standing in for one embedding.
3. **NVIDIA NeMo Gym.** *Engineering notes: aiohttp vs httpx* (httpcore connection pool is O(n²); a 16,000-request rollout hung for 40 minutes). [docs.nvidia.com/nemo/gym](https://docs.nvidia.com/nemo/gym/infrastructure/engineering-notes/aiohttp-vs-httpx)
4. **FastAPI documentation.** [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
5. **Granian.** [github.com/emmett-framework/granian](https://github.com/emmett-framework/granian)
