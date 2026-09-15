# High-Performance Python Reverse Proxy & Inference Gateway

A production-grade, asynchronous reverse proxy and caching inference gateway written in Python (`aiohttp`) for machine learning REST APIs.

Designed to sit directly in front of the Customer Churn Prediction Flask service (`POST /predict`), offloading repeated inference requests, enforcing client rate limits, tracking live operational metrics, and protecting backend ML workers from overload.

---

## Architecture Overview

```
                      Client Request
                            │
                            ▼
              ┌───────────────────────────┐
              │      Async Web Server     │ (Port 8080 - aiohttp)
              └─────────────┬─────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
        [GET /health]             [POST /predict]
        [GET /metrics]                   │
               │                         ▼
               │              ┌─────────────────────┐
               │              │ Per-IP Rate Limiter │
               │              └──────────┬──────────┘
               │                         │ OK (or 429)
               │                         ▼
               │              ┌─────────────────────┐
               │              │   LRU Cache Lookup  │
               │              │ (OrderedDict + TTL) │
               │              └──────────┬──────────┘
               │                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
             [Cache HIT (200)]                       [Cache MISS]
                     │                                       │
                     │                                       ▼
                     │                             ┌───────────────────┐
                     │                             │ Forward to Flask  │ (Port 5000)
                     │                             │  XGBoost Worker   │
                     │                             └─────────┬─────────┘
                     │                                       │ Response
                     │                                       ▼
                     │                             ┌───────────────────┐
                     │                             │ Cache Put & Store │
                     │                             └─────────┬─────────┘
                     │                                       │
                     └───────────────────┬───────────────────┘
                                         ▼
                             Structured Request Log
                             & Metrics Update
                                         │
                                         ▼
                                  Client Response
```

---

## Key Features

1. **Asynchronous Non-Blocking Architecture**:
   - Built on Python's native `asyncio` event loop and `aiohttp`.
   - Low memory footprint and high concurrency without manual thread management.

2. **Sub-millisecond LRU Prediction Cache**:
   - O(1) in-memory LRU cache (`OrderedDict`) with configurable capacity and TTL (`CACHE_CAPACITY`, `CACHE_TTL_SECONDS`).
   - Content-addressable cache keys: SHA-256 hash of JSON payload + model version.
   - Caches only successful (200 OK) responses; automatically bypassed on non-POST requests.

3. **Per-Client Rate Limiter**:
   - Sliding window counter tracked per client IP address.
   - Configurable limits (`RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`).
   - Returns standard `HTTP/1.1 429 Too Many Requests` with informative JSON.

4. **Health & Readiness Probes**:
   - `GET /health` — Immediate liveness probe returning `{ "status": "ok", "proxy": "running", "version": "1.0.0" }`.
   - `GET /health/backend` — Active upstream probe that checks connection with the Flask backend.

5. **Operational Metrics**:
   - `GET /metrics` — Real-time metrics:
     - `proxy_uptime_seconds`: Time proxy has been running.
     - `proxy_total_requests`: Total requests handled.
     - `proxy_success_requests`: Successful responses delivered.
     - `proxy_failed_requests`: Upstream or client errors.
     - `proxy_cache_hits` & `proxy_cache_misses`: Cache performance metrics.
     - `proxy_rate_limited`: Number of 429 rejections.
     - `proxy_avg_latency_ms`: Rolling average latency.

6. **Safety & Zero-PII Structured Logging**:
   - Thread-safe structured log lines with timestamp, client IP, method, path, HTTP status, cache status (`HIT`/`MISS`/`BYPASS`), backend status (`OK`/`FAIL`/`SKIP`), and latency in ms.
   - **Strict Privacy**: Customer feature payloads (credit score, balance, salary, etc.) are never written to logs or disk.

7. **Resilience & Fault Tolerance**:
   - Graceful fallback: If Flask is down or restarts, proxy immediately returns `503 Service Unavailable` with clean error JSON.
   - Cached queries continue to be served with `200 OK` (Cache HIT) even when the backend is offline.

---

## Configuration

All parameters can be configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `8080` | Port the proxy listens on |
| `BACKEND_HOST` | `127.0.0.1` | Hostname or IP of upstream Flask service |
| `BACKEND_PORT` | `5000` | Port of upstream Flask service |
| `CACHE_CAPACITY` | `1000` | Maximum entries in LRU prediction cache |
| `CACHE_TTL_SECONDS` | `300` | TTL in seconds for cached predictions |
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per sliding window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Duration of rate-limit window in seconds |
| `BACKEND_TIMEOUT_SEC` | `10` | Upstream socket timeout in seconds |
| `PROXY_LOG_FILE` | `""` | File path for request log (empty = stdout) |

---

## Quick Start

### Running Locally

```bash
python proxy/gateway.py
```

### Running Tests

```bash
# Unit tests
python -m unittest discover -s proxy/tests -p "test_*.py"

# End-to-end verification
python proxy/tests/verify_gateway.py
```
