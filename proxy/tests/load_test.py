#!/usr/bin/env python3
"""
load_test.py — Benchmark and load test comparison for Customer Churn Prediction.

Compares:
  1. Direct Flask API (http://127.0.0.1:5000/predict)
  2. C Inference Gateway (http://127.0.0.1:8080/predict) — Cache MISS
  3. C Inference Gateway (http://127.0.0.1:8080/predict) — Cache HIT

Reports:
  - Throughput (req/sec)
  - Mean latency (ms)
  - Latency percentiles: p50, p95, p99 (ms)
  - Cache hit rate & speedup factor
"""

import argparse
import concurrent.futures
import json
import random
import time
import sys
import statistics
import requests

DEFAULT_FLASK_URL = "http://127.0.0.1:5000/predict"
DEFAULT_PROXY_URL = "http://127.0.0.1:8080/predict"

SAMPLE_PAYLOAD_TEMPLATE = {
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 45,
    "Tenure": 5,
    "Balance": 100000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 60000.0,
}

def generate_payload(variant_id: int = 0) -> dict:
    p = dict(SAMPLE_PAYLOAD_TEMPLATE)
    p["CreditScore"] = 500 + (variant_id % 350)
    p["Age"] = 20 + (variant_id % 60)
    p["Balance"] = float((variant_id * 1000) % 200000)
    p["EstimatedSalary"] = float(30000 + (variant_id * 500) % 150000)
    return p

def send_request(url: str, payload: dict, timeout: float = 5.0):
    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        duration_ms = (time.perf_counter() - start) * 1000.0
        cache_header = resp.headers.get("X-Cache", "NONE")
        return {
            "status_code": resp.status_code,
            "duration_ms": duration_ms,
            "cache": cache_header,
            "success": (resp.status_code == 200),
            "error": None,
        }
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "status_code": 0,
            "duration_ms": duration_ms,
            "cache": "ERROR",
            "success": False,
            "error": str(e),
        }

def run_benchmark(name: str, url: str, payloads: list, concurrency: int) -> dict:
    print(f"\n--- Running Benchmark: {name} ---")
    print(f"Target URL: {url} | Total Requests: {len(payloads)} | Concurrency: {concurrency}")

    start_total = time.perf_counter()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, url, p) for p in payloads]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    total_time_sec = time.perf_counter() - start_total
    successful = [r for r in results if r["success"]]
    durations = [r["duration_ms"] for r in successful]

    if not durations:
        print(f"  [ERROR] All {len(results)} requests failed.")
        return {
            "name": name,
            "total": len(results),
            "success": 0,
            "throughput_rps": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    durations.sort()
    p50 = statistics.median(durations)
    p95 = durations[int(len(durations) * 0.95)]
    p99 = durations[int(len(durations) * 0.99)]
    mean_ms = statistics.mean(durations)
    rps = len(successful) / total_time_sec if total_time_sec > 0 else 0.0

    cache_hits = sum(1 for r in successful if r["cache"] == "HIT")
    cache_misses = sum(1 for r in successful if r["cache"] == "MISS")

    print(f"  Completed: {len(successful)}/{len(results)} ok in {total_time_sec:.3f}s")
    print(f"  Throughput: {rps:.1f} req/s")
    print(f"  Latency: mean={mean_ms:.2f}ms | p50={p50:.2f}ms | p95={p95:.2f}ms | p99={p99:.2f}ms")
    if cache_hits or cache_misses:
        hit_rate = (cache_hits / (cache_hits + cache_misses) * 100.0) if (cache_hits + cache_misses) > 0 else 0.0
        print(f"  Cache: {cache_hits} hits, {cache_misses} misses ({hit_rate:.1f}% hit rate)")

    return {
        "name": name,
        "total": len(results),
        "success": len(successful),
        "throughput_rps": rps,
        "mean_ms": mean_ms,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
    }

def print_summary_table(benchmarks: list):
    print("\n" + "=" * 80)
    print("                      BENCHMARK COMPARISON SUMMARY                      ")
    print("=" * 80)
    header = f"{'Target':<28} | {'RPS':<9} | {'Mean (ms)':<10} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | {'P99 (ms)':<9}"
    print(header)
    print("-" * 80)
    for b in benchmarks:
        print(f"{b['name']:<28} | {b['throughput_rps']:<9.1f} | {b['mean_ms']:<10.2f} | {b['p50_ms']:<9.2f} | {b['p95_ms']:<9.2f} | {b['p99_ms']:<9.2f}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Inference Gateway Load Test")
    parser.add_argument("--flask-url", default=DEFAULT_FLASK_URL, help="Direct Flask URL")
    parser.add_argument("--proxy-url", default=DEFAULT_PROXY_URL, help="C Proxy URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests per scenario")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent workers")
    parser.add_argument("--skip-flask", action="store_true", help="Skip direct Flask test")
    args = parser.parse_args()

    benchmarks = []

    # 1. Direct Flask test (if requested and accessible)
    if not args.skip_flask:
        try:
            r = requests.get("http://127.0.0.1:5000/", timeout=1.0)
            flask_up = (r.status_code == 200)
        except Exception:
            flask_up = False

        if flask_up:
            flask_payloads = [generate_payload(i) for i in range(args.requests)]
            res_flask = run_benchmark("Direct Flask API", args.flask_url, flask_payloads, args.concurrency)
            benchmarks.append(res_flask)
        else:
            print("[INFO] Direct Flask not reachable at http://127.0.0.1:5000/ — skipping direct Flask test.")

    # Check if proxy is up
    try:
        r = requests.get("http://127.0.0.1:8080/health", timeout=1.0)
        proxy_up = (r.status_code == 200)
    except Exception:
        proxy_up = False

    if not proxy_up:
        print("[ERROR] C Proxy is not reachable at http://127.0.0.1:8080/health. Start it first.")
        sys.exit(1)

    # 2. Proxy Cache MISS (each request has a unique payload)
    miss_payloads = [generate_payload(10000 + i) for i in range(args.requests)]
    res_miss = run_benchmark("Proxy (Cache MISS / Forwarded)", args.proxy_url, miss_payloads, args.concurrency)
    benchmarks.append(res_miss)

    # 3. Proxy Cache HIT (repeated requests with the same small set of payloads)
    sample_pool = [generate_payload(i) for i in range(5)]
    hit_payloads = [random.choice(sample_pool) for _ in range(args.requests)]
    # Warm up cache first
    for p in sample_pool:
        requests.post(args.proxy_url, json=p, timeout=2.0)

    res_hit = run_benchmark("Proxy (Cache HIT / Cached)", args.proxy_url, hit_payloads, args.concurrency)
    benchmarks.append(res_hit)

    print_summary_table(benchmarks)

if __name__ == "__main__":
    main()
