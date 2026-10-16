# ADR-0001: In-Memory Storage & Vector Search Engine Selection

* **Status:** Accepted (revised 2026-10-16: distribution updated to Redis 8, evidence corrected; interface amended by ADR-0004)
* **Date:** 2026-10-13
* **Deciders:** Lead Architect & FDE Team

---

### 1. Context & Problem Statement
The Semantic Caching AI Gateway requires an in-memory data store capable of sub-5ms query response times. Unlike traditional string-matching key-value caches, the engine must natively perform Approximate Nearest Neighbor (ANN) vector similarity search across prompt embeddings, support Cosine distance metrics, filter by tag fields (tenant and scope, per ADR-0004), and manage automatic key expiration (TTL) without introducing disk I/O bottlenecks.

### 2. Decision Drivers
* **Vector Search Maturity & Parity:** Complete support for vector indexing commands (`KNN`, Cosine distance, HNSW parameters, TAG pre-filters).
* **Python AI Ecosystem Integration:** Turnkey integration with a production-grade Python vector client (specifically `redisvl`).
* **Low Latency & High Throughput:** Sub-5ms query execution under concurrent gateway traffic.
* **Operational Simplicity:** An official, pinned, single-container deployment with zero custom compilation.
* **Licensing Clarity:** Acceptable terms for standalone microservices and internal enterprise deployments.

### 3. Considered Options
* **Option 1: Redis 8 Open Source (`redis:8.10.2`)**: Since Redis 8, the former Redis Stack modules (Redis Query Engine, JSON) are built into Redis Open Source itself. The official Docker image builds with `BUILD_WITH_MODULES=yes` on amd64 and arm64 [1][2].
* **Option 2: Valkey 8 + valkey-search (`valkey/valkey-bundle`)**: The Linux Foundation fork (BSD-3). Since June 2025 the official `valkey-bundle` image ships valkey-search 1.0 with HNSW, COSINE, TAG and NUMERIC indexes [3].
* **Option 3: Dragonfly (`dragonflydb.io`)**: Multi-threaded C++ engine with high raw key-value throughput and its own search implementation.

### 4. Decision Outcome
**Chosen Option:** **Option 1: Redis 8 Open Source, pinned to `redis:8.10.2`**

**Primary Rationale:**
For an AI gateway, vector-query compatibility with the client library matters more than raw key-value throughput. `redisvl`, the Python library this gateway builds on, is developed and tested only against Redis. Its source (v0.28.0) contains no reference to Valkey or Dragonfly [4], so compatibility with either is unverified. Both alternatives are viable engines, and Valkey has the more permissive license, but adopting either would mean writing and maintaining our own search client.

Redis 8 removes the reason to use the separate `redis/redis-stack` image, which is superseded: "Redis 8 merges Redis Stack and Redis Community Edition into a single unified distribution" [1]. The image is pinned to an exact version so benchmarks and builds are reproducible. Verify that the tag exists on Docker Hub before pinning; update deliberately, never via `:latest`.

**Index configuration (explicit, matching Redis and `redisvl` defaults [5]):**

| Parameter | Value | Effect |
|---|---|---|
| Algorithm | HNSW | Approximate KNN |
| `DISTANCE_METRIC` | COSINE | Distance on a 0–2 scale (ADR-0004 §4.6) |
| `M` | 16 | Graph connectivity (memory vs. recall) |
| `EF_CONSTRUCTION` | 200 | Build-time accuracy |
| `EF_RUNTIME` | 10 | Query-time accuracy (latency vs. recall) |

HNSW is approximate: with `EF_RUNTIME=10` it can occasionally miss the true nearest neighbor. For a cache this mostly produces extra misses, not wrong hits, because every candidate must still pass the similarity threshold. The calibration in ADR-0004 §7 uses exact search, so its hit rate is an upper bound for production.

### 5. Pros & Cons of the Chosen Option

* **Good (Positive Outcomes):**
  * [+] **Client Library Support:** `redisvl` `SearchIndex` and `VectorQuery` work out of the box, with no custom search dialect wrappers.
  * [+] **Turnkey, Reproducible Deployment:** One official, pinned image with search built in.
  * [+] **Full Feature Parity:** HNSW parameters, TAG pre-filters, NUMERIC fields and TTL on indexed hashes, all of which ADR-0004 requires.

* **Bad / Known Trade-offs (Consequences Accepted):**
  * [-] **Single-Threaded Command Execution:** Command processing is bound to a single CPU core, requiring cluster sharding if vertical single-core limits are exceeded.
  * [-] **Persistence vs. Memory:** Fork-based snapshotting (`BGSAVE`) can temporarily raise memory use. See the open decision in §6.
  * [-] **Licensing:** Redis 8 is offered under RSALv2, SSPLv1 or AGPLv3. None of these restrict self-hosting the unmodified server inside this gateway; they matter only if the database itself were offered as a managed service. Valkey (BSD-3) would be simpler on this point.

### 6. Mitigation & Reversibility
* **Encapsulation via Interface Contract:** Direct Redis calls never leak into the FastAPI routing layer. The original `check_similarity(vector, threshold)` contract has no tenant or scope parameters and is **superseded by the `BaseVectorCache` interface in ADR-0004 §6** (`get_exact`, `search_similar`, `store`).
* **Client library path:** use `redisvl.extensions.cache.llm` (or the lower-level `redisvl.index.SearchIndex`). Importing from `redisvl.extensions.llmcache` is deprecated [4].
* **Open decision: persistence (Type 2).** A cache can run without persistence (`save ""`, `appendonly no`). That removes `BGSAVE` memory spikes and disk I/O, at the cost of an empty cache after a restart. To be decided; the default Redis configuration keeps RDB snapshots.
* **Reversibility (Type 2 Decision):** Valkey and Dragonfly speak the same RESP protocol, but their search commands are not verified against `redisvl`. Migrating means implementing a new `BaseVectorCache` adapter and validating it against the ADR-0004 test suite, with zero changes to gateway logic or routing endpoints.

### 7. References
1. **Redis.** *What's new in Redis 8.0.* [redis.io/docs/latest/develop/whats-new/8-0](https://redis.io/docs/latest/develop/whats-new/8-0/)
2. **Redis Docker official image source.** `release/8.10` branch, `debian/Dockerfile` (`redis-full.tar.gz` 8.10.2, `BUILD_WITH_MODULES=yes`). [github.com/redis/docker-library-redis](https://github.com/redis/docker-library-redis)
3. **Valkey.** *valkey-bundle: One stop shop for real-time applications* (2025-06-23). [valkey.io/blog](https://valkey.io/blog/valkey-bundle-one-stop-shop-for-low-latency-modern-applications/)
4. **RedisVL source, v0.28.0.** `redisvl/schema/fields.py` (HNSW defaults), `redisvl/extensions/llmcache/__init__.py` (deprecation). [github.com/redis/redis-vl-python](https://github.com/redis/redis-vl-python)
5. **Redis.** *Vector search: HNSW index parameters.* [redis.io/docs/latest/develop/ai/search-and-query/vectors](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)
