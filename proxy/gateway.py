#!/usr/bin/env python3
"""
gateway.py — High-Performance Python Reverse Proxy & Inference Gateway
for Customer Churn Prediction.

Features:
- LRU caching with TTL and SHA-256 model-version-aware hashing
- Per-client IP sliding window rate limiting
- Real-time Prometheus-style /metrics endpoint
- Instant /health and upstream /health/backend probes
- Zero-PII structured request logging
- Resilient upstream forwarding to Flask (port 5000)
"""

import os
import sys
import time
import json
import hashlib
import asyncio
import threading
from datetime import datetime, timezone
from collections import OrderedDict
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROXY_PORT = int(os.environ.get("PROXY_PORT", 8080))
BACKEND_HOST = os.environ.get("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", 5000))
BACKEND_TIMEOUT_SEC = float(os.environ.get("BACKEND_TIMEOUT_SEC", 10.0))
CACHE_CAPACITY = int(os.environ.get("CACHE_CAPACITY", 1000))
CACHE_TTL_SECONDS = float(os.environ.get("CACHE_TTL_SECONDS", 300.0))
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", 100))
RATE_LIMIT_WINDOW_SECONDS = float(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", 60.0))
MODEL_METADATA_PATH = os.environ.get("MODEL_METADATA_PATH", "data/output/model_metadata.json")
PROXY_LOG_FILE = os.environ.get("PROXY_LOG_FILE", "")


def get_model_version(metadata_path: str) -> str:
    """Compute SHA-256 hex digest of model_metadata.json for cache busting."""
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "rb") as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()[:16]
        except Exception:
            pass
    return "v1-default"


MODEL_VERSION = get_model_version(MODEL_METADATA_PATH)


# -----------------------------------------------------------------------------
# Zero-PII Structured Logger
# -----------------------------------------------------------------------------
class StructuredLogger:
    def __init__(self, log_file: str = ""):
        self.log_file = log_file
        self.lock = threading.Lock()

    def log_request(
        self,
        client_ip: str,
        method: str,
        path: str,
        status_code: int,
        latency_ms: int,
        cache_result: str,   # "HIT", "MISS", "BYPASS"
        backend_result: str, # "OK", "FAIL", "SKIP"
    ):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = (
            f"{ts} client={client_ip or '-'} method={method or '-'} path={path or '-'} "
            f"status={status_code} latency_ms={latency_ms} cache={cache_result} "
            f"backend={backend_result}\n"
        )
        with self.lock:
            sys.stdout.write(line)
            sys.stdout.flush()
            if self.log_file:
                try:
                    with open(self.log_file, "a") as f:
                        f.write(line)
                except Exception:
                    pass

    def info(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{ts} [INFO] {msg}\n"
        with self.lock:
            sys.stdout.write(line)
            sys.stdout.flush()


logger = StructuredLogger(PROXY_LOG_FILE)


# -----------------------------------------------------------------------------
# Real-Time Operational Metrics
# -----------------------------------------------------------------------------
class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.success_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.rate_limited = 0
        self.active_connections = 0
        self.total_latency_ms = 0
        self.latency_count = 0

    def inc_total(self):
        with self.lock:
            self.total_requests += 1

    def inc_success(self):
        with self.lock:
            self.success_requests += 1

    def inc_failed(self):
        with self.lock:
            self.failed_requests += 1

    def inc_cache_hit(self):
        with self.lock:
            self.cache_hits += 1

    def inc_cache_miss(self):
        with self.lock:
            self.cache_misses += 1

    def inc_rate_limited(self):
        with self.lock:
            self.rate_limited += 1

    def inc_active(self):
        with self.lock:
            self.active_connections += 1

    def dec_active(self):
        with self.lock:
            if self.active_connections > 0:
                self.active_connections -= 1

    def add_latency(self, ms: int):
        with self.lock:
            self.total_latency_ms += ms
            self.latency_count += 1

    def render(self) -> str:
        with self.lock:
            uptime = int(time.time() - self.start_time)
            avg_latency = (
                self.total_latency_ms // self.latency_count
                if self.latency_count > 0
                else 0
            )
            cache_total = self.cache_hits + self.cache_misses
            hit_rate_pct = (
                (self.cache_hits * 100 // cache_total) if cache_total > 0 else 0
            )

            lines = [
                "# Customer Churn Inference Gateway Metrics",
                f"proxy_uptime_seconds={uptime}",
                f"proxy_total_requests={self.total_requests}",
                f"proxy_success_requests={self.success_requests}",
                f"proxy_failed_requests={self.failed_requests}",
                f"proxy_cache_hits={self.cache_hits}",
                f"proxy_cache_misses={self.cache_misses}",
                f"proxy_cache_hit_rate_pct={hit_rate_pct}",
                f"proxy_rate_limited={self.rate_limited}",
                f"proxy_active_connections={self.active_connections}",
                f"proxy_avg_latency_ms={avg_latency}",
                f"proxy_total_latency_ms={self.total_latency_ms}",
            ]
            return "\n".join(lines) + "\n"


metrics = Metrics()


# -----------------------------------------------------------------------------
# LRU Prediction Cache with TTL and Model Versioning
# -----------------------------------------------------------------------------
class CacheEntry:
    __slots__ = ("body", "status_code", "content_type", "expiry")

    def __init__(self, body: str, status_code: int, content_type: str, expiry: float):
        self.body = body
        self.status_code = status_code
        self.content_type = content_type
        self.expiry = expiry


class LruCache:
    def __init__(self, capacity: int, ttl_seconds: float):
        self.capacity = max(1, capacity)
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> CacheEntry | None:
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            if time.time() > entry.expiry:
                del self.cache[key]
                return None
            # Move to end (MRU)
            self.cache.move_to_end(key)
            return entry

    def put(self, key: str, body: str, status_code: int = 200, content_type: str = "application/json"):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.capacity:
                # Evict oldest (LRU)
                self.cache.popitem(last=False)
            expiry = time.time() + self.ttl_seconds
            self.cache[key] = CacheEntry(body, status_code, content_type, expiry)

    @staticmethod
    def build_key(body_bytes: bytes, model_version: str) -> str:
        """Hash body bytes + model version with SHA-256 for consistent keying."""
        h = hashlib.sha256()
        h.update(body_bytes)
        h.update(b"|")
        h.update(model_version.encode("utf-8"))
        return h.hexdigest()


cache = LruCache(CACHE_CAPACITY, CACHE_TTL_SECONDS)


# -----------------------------------------------------------------------------
# Sliding-Window Per-IP Rate Limiter
# -----------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # client_ip -> list of timestamps
        self.history: dict[str, list[float]] = {}
        self.lock = threading.Lock()

    def is_rate_limited(self, ip: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        with self.lock:
            timestamps = self.history.get(ip)
            if timestamps is None:
                self.history[ip] = [now]
                return False

            # Filter timestamps older than the sliding window
            valid_ts = [t for t in timestamps if t > window_start]
            if len(valid_ts) >= self.max_requests:
                self.history[ip] = valid_ts
                return True

            valid_ts.append(now)
            self.history[ip] = valid_ts
            return False


rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


# -----------------------------------------------------------------------------
# HTTP Handlers
# -----------------------------------------------------------------------------
async def handle_health(request: web.Request) -> web.Response:
    data = {"status": "ok", "proxy": "running", "version": "1.0.0"}
    return web.json_response(data)


async def handle_health_backend(request: web.Request) -> web.Response:
    """Probe the upstream Flask backend via a quick TCP connection."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(BACKEND_HOST, BACKEND_PORT),
            timeout=2.0,
        )
        writer.close()
        await writer.wait_closed()
        return web.json_response({"status": "ok", "backend": "reachable"}, status=200)
    except Exception:
        return web.json_response(
            {"status": "error", "backend": "unreachable"}, status=503
        )


async def handle_metrics(request: web.Request) -> web.Response:
    rendered = metrics.render()
    return web.Response(text=rendered, content_type="text/plain")


async def handle_predict(request: web.Request) -> web.Response:
    """Handle POST /predict with caching and rate limiting."""
    body_bytes = await request.read()
    cache_key = LruCache.build_key(body_bytes, MODEL_VERSION)

    # 1. Check LRU Cache
    cached = cache.get(cache_key)
    if cached is not None:
        request["cache_result"] = "HIT"
        request["backend_result"] = "SKIP"
        metrics.inc_cache_hit()
        metrics.inc_success()
        return web.Response(
            text=cached.body,
            status=cached.status_code,
            content_type=cached.content_type,
            headers={"X-Cache": "HIT"},
        )

    # 2. Cache MISS — forward to Flask backend
    request["cache_result"] = "MISS"
    metrics.inc_cache_miss()

    backend_url = f"http://{BACKEND_HOST}:{BACKEND_PORT}/predict"
    client_session: ClientSession = request.app["client_session"]

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    client_ip = request.remote or "127.0.0.1"
    headers["X-Forwarded-For"] = client_ip

    try:
        async with client_session.post(
            backend_url,
            data=body_bytes,
            headers=headers,
            timeout=ClientTimeout(total=BACKEND_TIMEOUT_SEC),
        ) as upstream_resp:
            resp_body = await upstream_resp.text()
            status = upstream_resp.status
            content_type = upstream_resp.content_type or "application/json"

            # Cache successful predictions
            if status == 200 and resp_body:
                cache.put(cache_key, resp_body, status, content_type)

            request["backend_result"] = "OK"
            if 200 <= status < 400:
                metrics.inc_success()
            else:
                metrics.inc_failed()

            return web.Response(
                text=resp_body,
                status=status,
                content_type=content_type,
                headers={"X-Cache": "MISS"},
            )
    except Exception:
        request["backend_result"] = "FAIL"
        metrics.inc_failed()
        return web.json_response(
            {
                "error": "Backend unavailable",
                "detail": "The ML inference service is unreachable.",
            },
            status=503,
            headers={"X-Cache": "MISS"},
        )


async def handle_fallback_proxy(request: web.Request) -> web.Response:
    """Generic reverse proxy forwarder for all other routes."""
    request["cache_result"] = "BYPASS"
    backend_url = f"http://{BACKEND_HOST}:{BACKEND_PORT}{request.rel_url}"
    client_session: ClientSession = request.app["client_session"]

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    headers["X-Forwarded-For"] = request.remote or "127.0.0.1"
    body_bytes = await request.read()

    try:
        async with client_session.request(
            method=request.method,
            url=backend_url,
            data=body_bytes if body_bytes else None,
            headers=headers,
            timeout=ClientTimeout(total=BACKEND_TIMEOUT_SEC),
        ) as upstream_resp:
            resp_body = await upstream_resp.read()
            request["backend_result"] = "OK"
            if 200 <= upstream_resp.status < 400:
                metrics.inc_success()
            else:
                metrics.inc_failed()

            fwd_headers = {
                k: v
                for k, v in upstream_resp.headers.items()
                if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
            }
            return web.Response(
                body=resp_body,
                status=upstream_resp.status,
                headers=fwd_headers,
            )
    except Exception:
        request["backend_result"] = "FAIL"
        metrics.inc_failed()
        return web.json_response({"error": "Backend unavailable"}, status=503)


# -----------------------------------------------------------------------------
# Middleware: Metrics, Rate Limiting & Zero-PII Logging
# -----------------------------------------------------------------------------
@web.middleware
async def proxy_middleware(request: web.Request, handler):
    metrics.inc_total()
    metrics.inc_active()
    start_time = time.perf_counter()

    client_ip = request.remote or "127.0.0.1"
    request["cache_result"] = "BYPASS"
    request["backend_result"] = "SKIP"

    # Rate limiting check
    if rate_limiter.is_rate_limited(client_ip):
        metrics.inc_rate_limited()
        metrics.inc_failed()
        metrics.dec_active()
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        metrics.add_latency(duration_ms)
        logger.log_request(
            client_ip=client_ip,
            method=request.method,
            path=request.path,
            status_code=429,
            latency_ms=duration_ms,
            cache_result="BYPASS",
            backend_result="SKIP",
        )
        return web.json_response(
            {
                "error": "Too Many Requests",
                "detail": "Rate limit exceeded. Please slow down.",
            },
            status=429,
        )

    try:
        response = await handler(request)
        status_code = response.status
    except web.HTTPException as ex:
        response = ex
        status_code = ex.status
    except Exception:
        response = web.json_response({"error": "Internal Server Error"}, status=500)
        status_code = 500

    metrics.dec_active()
    duration_ms = int((time.perf_counter() - start_time) * 1000)
    metrics.add_latency(duration_ms)

    # Privacy-preserving structured log
    logger.log_request(
        client_ip=client_ip,
        method=request.method,
        path=request.path,
        status_code=status_code,
        latency_ms=duration_ms,
        cache_result=request.get("cache_result", "BYPASS"),
        backend_result=request.get("backend_result", "SKIP"),
    )

    return response


# -----------------------------------------------------------------------------
# Application Setup
# -----------------------------------------------------------------------------
async def on_startup(app: web.Application):
    connector = TCPConnector(limit=500, enable_cleanup_closed=True)
    app["client_session"] = ClientSession(connector=connector)
    logger.info(
        f"Python Inference Gateway started on port {PROXY_PORT} "
        f"(Upstream: {BACKEND_HOST}:{BACKEND_PORT}, ModelVersion: {MODEL_VERSION})"
    )


async def on_cleanup(app: web.Application):
    await app["client_session"].close()
    logger.info("Python Inference Gateway shut down.")


def create_app() -> web.Application:
    app = web.Application(middlewares=[proxy_middleware])
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # Defined endpoints
    app.router.add_get("/health", handle_health)
    app.router.add_get("/health/backend", handle_health_backend)
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_post("/predict", handle_predict)

    # Catch-all fallback proxy
    app.router.add_route("*", "/{tail:.*}", handle_fallback_proxy)

    return app


if __name__ == "__main__":
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=PROXY_PORT, access_log=None)
