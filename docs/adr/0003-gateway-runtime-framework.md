# ADR-0003: Gateway Runtime Framework & ASGI Server Selection

* **Status:** Accepted
* **Date:** 2026-10-13
* **Deciders:** Lead Architect & FDE Team

---

### 1. Context & Problem Statement
The Semantic Caching AI Gateway operates as a high-concurrency reverse proxy sitting in front of upstream LLM providers (e.g., OpenAI, Anthropic, vLLM). It must intercept `POST /v1/chat/completions` requests, validate schemas, execute in-process ONNX embeddings (`all-MiniLM-L12-v2`), query Redis Stack asynchronously, and handle bi-directional Server-Sent Events (SSE) streaming with sub-25ms cache hit latency.

While compiled system proxies (like Go-based Bifrost) offer raw networking throughput, they suffer from poor ML ecosystem interoperability (requiring brittle Cgo bindings for local ONNX inference). Conversely, standard Python ASGI stacks (like LiteLLM on Uvicorn) frequently become I/O and GIL bottlenecks under concurrent streaming traffic. We must select a runtime architecture that bridges native Python AI library synergy with low-latency, multi-threaded network I/O.

### 2. Decision Drivers
* **AI & In-Process ML Interoperability:** Native, zero-FFI integration with Python AI tooling (`fastembed`, `onnxruntime`, `redisvl`, `pydantic-v2`).
* **High-Throughput Network I/O:** Ability to handle 40,000+ requests/sec with minimal TCP socket overhead and zero event-loop starvation.
* **Streaming & SSE Interception:** Robust, async streaming capabilities to tee upstream LLM token chunks (forwarding to client while background-buffering for cache storage) and synthesize SSE events on cache hits.
* **CPU/Async Concurrency Isolation:** Clean separation between CPU-bound embedding generation and async network I/O to avoid blocking the main event loop.
* **Development Velocity & Maintainability:** Modular, typed Python code that is maintainable, extensible, and clean for technical portfolio evaluation.

### 3. Considered Options
* **Option 1: FastAPI + Granian (Rust-Powered ASGI Server)** — Asynchronous Python framework paired with a high-performance HTTP server written in Rust (`hyper` + `tokio` via `PyO3`), using `pydantic-v2` for Rust-based JSON serialization.
* **Option 2: FastAPI + Uvicorn (Standard Python ASGI)** — Traditional Python asynchronous stack (`uvicorn` with `uvloop` and `httptools`).
* **Option 3: Go (`net/http` / `Chi` / Bifrost Architecture)** — Compiled Go binary delivering microsecond network overhead but requiring Cgo bridges for ONNX inference and manual C-struct parsing.
* **Option 4: Rust (`Axum` / `Actix-web` + `ort`)** — Pure systems programming stack offering maximum throughput and minimal memory, but with high cognitive complexity, lack of high-level Redis vector libraries, and slower iteration velocity.

### 4. Decision Outcome
**Chosen Option:** **Option 1: FastAPI executed via Granian (Rust-powered ASGI Server)**

**Primary Rationale:**
FastAPI + Granian provides the optimal architectural convergence: the high-performance network throughput and multi-threaded Tokio runtime of Rust, combined with the first-class machine learning ecosystem of Python.

By running FastAPI on Granian instead of standard Uvicorn, the gateway offloads HTTP/1.1, HTTP/2 parsing, socket management, and async task scheduling to Rust's battle-tested `hyper` and `tokio` engines. Meanwhile, the application retains native Python bindings for `fastembed` (ONNX Runtime) and `redisvl`. Pydantic v2 (itself compiled from Rust) provides near-instant JSON parsing of OpenAI request payloads.

### 5. Pros & Cons of the Chosen Option

* **Good (Positive Outcomes):**
  * [+] **Rust-Grade Network Concurrency:** Granian delivers 50,000–80,000 req/sec and handles thousands of open SSE connections without GIL-bound socket bottlenecks.
  * [+] **First-Class ML & Vector Integration:** Seamless in-process execution of FastEmbed ONNX models and RedisVL without foreign function interface (FFI) or Cgo fragility.
  * [+] **Fast JSON Serialization:** Pydantic v2 handles validation and serialization via `pydantic-core` in compiled Rust, eliminating Python JSON parsing overhead.
  * [+] **Rich Streaming Primitives:** Native `StreamingResponse` and async generators allow straightforward SSE stream teeing and synthesis.

* **Bad / Known Trade-offs (Consequences Accepted):**
  * [-] **Memory Footprint per Worker:** Python runtime overhead plus loaded ONNX model weights require ~150MB–220MB RAM per worker process (higher than Go/Rust's ~20MB).
  * [-] **CPU GIL Contention on In-Process Inference:** If CPU-bound embedding computation is run directly on the async event loop, it will block concurrent I/O. (Mitigated via dedicated thread offloading).

### 6. Mitigation & Reversibility
* **Thread Offloading for CPU Tasks:** All FastEmbed ONNX calls are strictly encapsulated and executed via `asyncio.to_thread` or worker thread pools, completely isolating CPU-bound tensor operations from the Granian/FastAPI async event loop.
* **Streaming Interceptor Pattern:** The proxy streaming layer is encapsulated in a reusable `StreamInterceptor` pipeline:

```python
from typing import AsyncGenerator
import json

class StreamInterceptor:
    @staticmethod
    async def tee_stream(
        upstream_generator: AsyncGenerator[bytes, None],
        on_complete: callable
    ) -> AsyncGenerator[bytes, None]:
        """Tee incoming SSE chunks: stream immediately to client while accumulating response."""
        accumulated_chunks: list[str] = []
        async for chunk in upstream_generator:
            yield chunk
            # Buffer delta tokens in memory
            # ...
        await on_complete("".join(accumulated_chunks))
```

* **Reversibility (Type 2 Decision):** Because the Gateway adheres strictly to standard ASGI specifications (`app = FastAPI()`), the runtime server can be swapped from Granian to Uvicorn or Hypercorn via a single CLI argument (`granian --interface asgi src.main:app` vs `uvicorn src.main:app`) without modifying a single line of application logic.

### 7. References & Benchmarks
1. **Paul, K. (2026).** *Top AI Gateways with Semantic Caching and Dynamic Routing (2026 Guide).* DEV Community. [dev.to/kuldeep_paul](https://dev.to/kuldeep_paul/top-ai-gateways-with-semantic-caching-and-dynamic-routing-2026-guide-4a0g) — Analyzed architectural trade-offs in LiteLLM (external API embedding latency) and Go-based Bifrost (two-layer caching patterns).
2. **McKinney, D. (2026).** *Best AI Gateways for Semantic Caching to Cut LLM Costs.* Medium. [medium.com/@hi.debmckinney](https://medium.com/@hi.debmckinney/best-ai-gateways-for-semantic-caching-to-cut-llm-costs-7d31c880bef6) — Evaluated Portkey, LiteLLM, Bifrost, Kong, and Helicone; established the architectural requirement for streaming cache interception (SSE teeing).
3. **Granian HTTP Server.** *A high performance HTTP server for Python applications written in Rust.* GitHub: [emmett-framework/granian](https://github.com/emmett-framework/granian).
4. **FastAPI & Pydantic v2 Performance Benchmarks.** Tiangolo & Pydantic Core. [FastAPI Documentation](https://fastapi.tiangolo.com).
5. **LiteLLM Open Source Architecture.** BerriAI. [GitHub: BerriAI/litellm](https://github.com/BerriAI/litellm) — Examined ASGI proxy implementation patterns (`litellm/proxy/proxy_server.py`).
