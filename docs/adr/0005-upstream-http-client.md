# ADR-0005: Upstream HTTP Client (aiohttp), Pool, Timeouts & Retries

* **Status:** Accepted (pool size, timeout values and retry count are provisional until measured, see §7)
* **Date:** 2026-11-22
* **Deciders:** Lead Architect & FDE Team

---

### 1. Context & Problem Statement
ADR-0003 chose FastAPI + Granian and left one dependency open: the HTTP client that calls the upstream LLM provider (library, connection pool, timeouts, retries). ADR-0004 adds requirements on that client: stream SSE bytes unchanged, request `stream_options.include_usage`, and cancel the upstream call when the client disconnects.

The workload is unusual for an HTTP client. Every cache miss holds one upstream connection open for the whole generation (seconds), so the number of open connections is roughly *miss rate × response duration* (Little's law). The pool, not the server, is the likely bottleneck.

### 2. Decision Drivers
* Long-lived streaming connections at high concurrency.
* Raw byte pass-through (no parsing into SDK objects) so the `StreamInterceptor` stays in control.
* Fast failure when the pool is full, instead of silent queueing.
* Retries must never duplicate or corrupt a stream already started.
* Every operational value must be changeable without a code change.

### 3. Considered Options
1. **aiohttp.** Async-native client.
2. **httpx.** Friendlier API, HTTP/2 support. Its `httpcore` pool was reported to scale badly at high concurrency (NVIDIA NeMo Gym notes, cited in ADR-0003).
3. **OpenAI Python SDK.** Parses responses into typed objects and sits on httpx.
4. **LiteLLM as a library.** Multi-provider abstraction. Not needed with a single OpenAI-compatible upstream, and it adds a normalization layer between us and the bytes. Its streaming pass-through behaviour was not verified.

### 4. Decision Outcome
**Option 1: aiohttp, wrapped in a single `UpstreamClient` class.**

#### 4.1 Wrapper
One `aiohttp.ClientSession` per Granian worker, created at application startup and closed at shutdown. All upstream calls go through `UpstreamClient`; no other module imports aiohttp.

#### 4.2 Connection pool
`UPSTREAM_POOL_LIMIT=200` (provisional). aiohttp's own default is `limit=100` and `limit_per_host=0` (no per-host limit) [1]. With the upstream being one host, `limit` is the effective cap. Illustration of the sizing logic (hypothetical numbers): 30 misses/s × 10 s ≈ 300 open connections, which already exceeds the default.

#### 4.3 Timeouts
| Setting | Value | Meaning |
|---|---|---|
| `UPSTREAM_CONNECT_TIMEOUT_S` | 5 | Maximum time to acquire a connection from the pool, **including waiting for a free slot** when the limit is reached [1] |
| `UPSTREAM_SOCK_READ_TIMEOUT_S` | 30 | Maximum silence between data portions; detects a stalled stream [1] |
| `total` | not set | A whole-request cap would cut long, healthy streams. The aiohttp documentation does not state whether `total` includes the response body, so it is not relied on. |

An optional separate time-to-first-chunk timeout (an idea LiteLLM exposes as `stream_timeout` [2]) is a deferred Type 2 addition, to be considered once measurements exist. A pool-full failure (connect timeout) returns an error to the client quickly with a clear status instead of hanging.

These timeouts are independent of `CACHE_LOOKUP_TIMEOUT_MS` (ADR-0004), which bounds only the cache lookup.

#### 4.4 Retries
* **Only before the first byte is sent to the client.** After streaming has started, a failure ends the stream and nothing is stored (ADR-0004 §4.8).
* `UPSTREAM_MAX_RETRIES=2` (provisional), with exponential backoff.
* Retry on: connection errors, timeouts, 5xx.
* Retry on 429 **only if the upstream sends `Retry-After`**, and honour it.
* No retry on other 4xx (a client error will not fix itself).
* **Open details, to be fixed at implementation:** backoff base and jitter, and an upper bound on how long a `Retry-After` value we are willing to wait.

#### 4.5 Cancellation
Client disconnect closes the upstream connection immediately (ADR-0004 §4.8).

#### 4.6 Configuration Summary (`.env`)
| Variable | Default (provisional) |
|---|---|
| `UPSTREAM_POOL_LIMIT` | `200` |
| `UPSTREAM_CONNECT_TIMEOUT_S` | `5` |
| `UPSTREAM_SOCK_READ_TIMEOUT_S` | `30` |
| `UPSTREAM_MAX_RETRIES` | `2` |

#### 4.7 Note: Multiple Models and Providers (out of scope for v1)
v1 talks to one OpenAI-compatible upstream. The `model` field is part of `param_fp` (ADR-0004), so each model already has its own cache entries. If more providers are needed later, the intended path is to point `UpstreamClient` at a multi-provider gateway (for example a LiteLLM Proxy, assumed to expose an OpenAI-compatible endpoint; not verified here) by changing the base URL only. Provider-specific parameter defaults must be added to ADR-0004 §4.3 only after being verified for that provider. A routing table inside this gateway was considered and deferred. No separate ADR is planned unless this becomes a requirement.

### 5. Pros & Cons of the Chosen Option
* [+] Built for async and long-lived streams; no `httpcore` pool.
* [+] Raw bytes in and out; `StreamInterceptor` stays simple.
* [+] LiteLLM, a widely used gateway, moved its default transport from httpx to aiohttp. In their own test (200 RPS, 200 concurrent users, 2 vCPUs) throughput rose from 50.2 to 224 RPS and median latency fell from 2,500 ms to 74 ms [3]. This is their workload, not ours.
* [-] No HTTP/2 (not needed for a single upstream today).
* [-] Lower-level API than httpx; more code to write.
* [-] Retrying a POST can bill twice if the upstream processed the request but the connection dropped before we saw the answer. Accepted, because retries are limited to two and only before streaming starts.
* [-] All numeric values above are estimates, not measurements.

### 6. Mitigation & Reversibility
| Decision | Type | How to change |
|---|---|---|
| Client library (aiohttp) | **Type 1**, reduced by the wrapper | Replace the body of `UpstreamClient` only |
| Pool limit, timeouts, retry count | Type 2 | `.env` |
| Time-to-first-chunk timeout | Type 2 | Add one setting |
| Retry policy details | Type 2 | Code in `UpstreamClient` |

### 7. Validation
Before the first release, measure with a gateway-shaped load test against a mock upstream that streams slowly:
* Connections in use at the target miss rate, to calibrate `UPSTREAM_POOL_LIMIT`.
* Behaviour when the pool is full (time to failure should match `UPSTREAM_CONNECT_TIMEOUT_S`).
* A stalled stream is closed after `UPSTREAM_SOCK_READ_TIMEOUT_S`.
* Retries never fire after the first byte reaches the client.
* Our own aiohttp vs httpx comparison under the same load, to replace the borrowed figures in §5.

### 8. References
1. **aiohttp Client Reference.** `TCPConnector` and `ClientTimeout` definitions. [docs.aiohttp.org/en/stable/client_reference.html](https://docs.aiohttp.org/en/stable/client_reference.html)
2. **LiteLLM: Timeouts.** [docs.litellm.ai/docs/proxy/timeout](https://docs.litellm.ai/docs/proxy/timeout)
3. **LiteLLM: aiohttp benchmarks.** [docs.litellm.ai/docs/aiohttp_benchmarks](https://docs.litellm.ai/docs/aiohttp_benchmarks)
4. **NVIDIA NeMo Gym.** *Engineering notes: aiohttp vs httpx*, as cited in ADR-0003 (not re-verified when this ADR was written). [docs.nvidia.com/nemo/gym](https://docs.nvidia.com/nemo/gym/infrastructure/engineering-notes/aiohttp-vs-httpx)
