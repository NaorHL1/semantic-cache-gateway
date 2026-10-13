# ADR-0001: In-Memory Storage & Vector Search Engine Selection

* **Status:** Accepted
* **Date:** 2026-10-13
* **Deciders:** Lead Architect & FDE Team

---

### 1. Context & Problem Statement
The Semantic Caching AI Gateway requires an in-memory data store capable of sub-5ms query response times. Unlike traditional string-matching key-value caches, the engine must natively perform Approximate Nearest Neighbor (ANN) vector similarity search across prompt embeddings, support Cosine distance metrics, and manage automatic key expiration (TTL) without introducing disk I/O bottlenecks.

### 2. Decision Drivers
* **Vector Search Maturity & Parity:** Complete support for vector indexing commands (`KNN`, Cosine distance, HNSW parameters) without dialect syntax exceptions.
* **Python AI Ecosystem Integration:** Turnkey integration with production-grade semantic caching client libraries (specifically `redisvl`).
* **Low Latency & High Throughput:** Sub-5ms query execution under concurrent gateway traffic.
* **Operational Simplicity:** Official single-container deployment with zero custom compilation or out-of-tree module builds.
* **Licensing Clarity:** Acceptable terms for standalone microservices and internal enterprise deployments.

### 3. Considered Options
* **Option 1: Redis Stack (`redis/redis-stack:latest`)** — In-memory engine with first-party C modules (RediSearch, RedisJSON) and deep Python AI library support.
* **Option 2: Valkey 8 (`valkey-io/valkey` + `valkey-search`)** — Linux Foundation's permissive open-source fork (BSD-3), backed by AWS/Google, featuring an actively developed C++ vector search module.
* **Option 3: Dragonfly (`dragonflydb.io`)** — Multi-threaded C++20 engine delivering up to 25x raw throughput and fork-less snapshotting, featuring a custom Global HNSW index.

### 4. Decision Outcome
**Chosen Option:** **Redis Stack (`redis/redis-stack:latest`)**

**Primary Rationale:**
For an AI Gateway, vector calculation precision and complete query engine compatibility take precedence over raw multi-million key-value operations per second. 

While Dragonfly delivers unmatched raw throughput on basic `GET`/`SET` pipelines, high-level AI client libraries such as `redisvl` fail during index creation (`FT.CREATE`) on Dragonfly due to its strict reliance on RediSearch Dialect 2/3 parameters, payload structuring, and unsupported schema flags. Valkey's `valkey-search` module, while strategically promising, is nascent and lacks official turnkey container distributions and specialized semantic cache abstractions. Redis Stack remains the only engine providing 100% vector feature parity out of the box.

### 5. Pros & Cons of the Chosen Option

* **Good (Positive Outcomes):**
  * [+] **Ecosystem Maturity:** Native integration with `redisvl.extensions.llmcache.SemanticCache`, eliminating custom search dialect wrappers.
  * [+] **Turnkey Deployment:** Official Docker image with pre-compiled search capabilities and bundled management UI (RedisInsight).
  * [+] **Full Feature Parity:** 100% support for HNSW parameters (`M`, `EF_CONSTRUCTION`, `EF_RUNTIME`), payload filters, and distance metrics.

* **Bad / Known Trade-offs (Consequences Accepted):**
  * [-] **Single-Threaded Command Execution:** Command processing is bound to a single CPU core, requiring cluster sharding if vertical single-core limits are exceeded.
  * [-] **Memory Spikes during `BGSAVE`:** Standard fork-based snapshotting can temporarily double RAM usage under high-write persistence operations.
  * [-] **Licensing Constraints:** Governed by RSALv2 / SSPLv1 (and AGPLv3 in Redis 8), preventing offering the engine as a competing managed cloud database.

### 6. Mitigation & Reversibility
* **Encapsulation via Interface Contract:** Direct Redis calls will not leak into the FastAPI routing layer. The gateway enforces a strict interface boundary:

```python
from abc import ABC, abstractmethod
from typing import Optional

class BaseVectorCache(ABC):
    @abstractmethod
    async def check_similarity(self, vector: list[float], threshold: float) -> Optional[dict]:
        """Query nearest vector and return cached response if distance <= threshold."""
        pass

    @abstractmethod
    async def store(self, prompt: str, response: str, vector: list[float], ttl: int) -> None:
        """Store prompt, response, and embedding vector with an expiration TTL."""
        pass
```

* **Reversibility (Type 2 Decision):** Because Valkey and Dragonfly share the identical RESP wire protocol, migrating to Valkey 8 when `valkey-search` reaches operational parity requires implementing an alternative `ValkeyVectorCache` adapter and updating the environment configuration, with zero changes to gateway logic or routing endpoints.
