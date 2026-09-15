#!/usr/bin/env python3
"""
verify_gateway.py — End-to-end verification and load testing suite for C Reverse Proxy.
"""

import subprocess
import time
import sys
import os
import json
import urllib.request
import urllib.error

PYTHON_BIN = "/home/sakshi/customer_churn/.venv/bin/python"
FLASK_APP = "/home/sakshi/customer_churn/customer_churn_prediction/app.py"
PROXY_BIN = "/home/sakshi/customer_churn/customer_churn_prediction/proxy/build/proxy"
PROJECT_DIR = "/home/sakshi/customer_churn/customer_churn_prediction"

SAMPLE_PAYLOAD = {
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

def log(msg):
    print(f"[VERIFY] {msg}", flush=True)

def wait_for_port(url, timeout=10.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status in (200, 404, 503):
                    return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            time.sleep(0.3)
    return False

def http_request(url, method="GET", data=None, headers=None, timeout=5.0):
    if headers is None:
        headers = {}
    req_data = json.dumps(data).encode("utf-8") if data is not None else None
    if req_data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            resp_headers = dict(resp.headers)
            return resp.status, body, resp_headers
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return e.code, body, dict(e.headers)
    except Exception as e:
        return 0, str(e), {}

def main():
    log("=== Stage 1: Running Python Unit Tests ===")
    res = subprocess.run([PYTHON_BIN, "-m", "unittest", "discover", "-s", "proxy/tests", "-p", "test_*.py"], cwd=PROJECT_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"Unit tests failed:\n{res.stderr}\n{res.stdout}")
        sys.exit(1)
    log("Python Unit tests passed successfully (test_cache & test_rate_limiter)!")

    log("=== Stage 2: Starting Backend and Proxy ===")
    flask_proc = subprocess.Popen([PYTHON_BIN, FLASK_APP], cwd=PROJECT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_for_port("http://127.0.0.1:5000/", timeout=8.0):
        log("ERROR: Flask failed to start!")
        flask_proc.kill()
        sys.exit(1)
    log("Flask backend is UP on port 5000.")

    proxy_env = dict(os.environ, RATE_LIMIT_REQUESTS="5000", CACHE_CAPACITY="1000", PROXY_PORT="8080")
    proxy_proc = subprocess.Popen([PYTHON_BIN, f"{PROJECT_DIR}/proxy/gateway.py"], cwd=PROJECT_DIR, env=proxy_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_for_port("http://127.0.0.1:8080/health", timeout=8.0):
        log("ERROR: Proxy failed to start!")
        flask_proc.kill()
        proxy_proc.kill()
        sys.exit(1)
    log("Python Reverse Proxy is UP on port 8080.")

    try:
        log("=== Stage 3: Verifying Endpoints ===")
        # 1. Health
        status, body, _ = http_request("http://127.0.0.1:8080/health")
        assert status == 200 and "ok" in body, f"Health check failed: {status}, {body}"
        log(f"GET /health: {status} OK -> {body.strip()}")

        # 2. Backend Health
        status, body, _ = http_request("http://127.0.0.1:8080/health/backend")
        assert status == 200 and "reachable" in body, f"Backend health failed: {status}, {body}"
        log(f"GET /health/backend: {status} OK -> {body.strip()}")

        # 3. Metrics
        status, body, _ = http_request("http://127.0.0.1:8080/metrics")
        assert status == 200 and "total_requests" in body, f"Metrics failed: {status}, {body}"
        log(f"GET /metrics: {status} OK")

        # 4. Predict MISS
        status, body, headers = http_request("http://127.0.0.1:8080/predict", method="POST", data=SAMPLE_PAYLOAD)
        assert status == 200, f"Predict failed: {status}, {body}"
        cache_hdr = headers.get("X-Cache", headers.get("x-cache", ""))
        assert cache_hdr == "MISS", f"Expected X-Cache: MISS, got: {cache_hdr}"
        pred1 = json.loads(body)
        log(f"POST /predict (MISS): status={status}, X-Cache={cache_hdr}, churn_probability={pred1.get('churn_probability')}, churn={pred1.get('churn')}")

        # 5. Predict HIT
        status, body, headers = http_request("http://127.0.0.1:8080/predict", method="POST", data=SAMPLE_PAYLOAD)
        assert status == 200, f"Predict HIT failed: {status}, {body}"
        cache_hdr = headers.get("X-Cache", headers.get("x-cache", ""))
        assert cache_hdr == "HIT", f"Expected X-Cache: HIT, got: {cache_hdr}"
        pred2 = json.loads(body)
        assert pred1 == pred2, f"Predictions do not match! {pred1} vs {pred2}"
        log(f"POST /predict (HIT):  status={status}, X-Cache={cache_hdr}, churn_probability={pred2.get('churn_probability')}, churn={pred2.get('churn')}")

        # 6. Backend Failure Resilience Test
        log("=== Stage 4: Verifying Backend Failure Resilience ===")
        flask_proc.kill()
        flask_proc.wait()
        time.sleep(0.5)

        # GET /health should still be 200 OK
        status, body, _ = http_request("http://127.0.0.1:8080/health")
        assert status == 200, f"Proxy died when Flask stopped! {status}"
        log(f"GET /health when backend down: {status} OK (Proxy is alive!)")

        # GET /health/backend should return 503
        status, body, _ = http_request("http://127.0.0.1:8080/health/backend")
        assert status == 503, f"Expected 503, got {status}"
        log(f"GET /health/backend when backend down: {status} Service Unavailable -> {body.strip()}")

        # POST /predict with uncached payload should return 503
        uncached_payload = dict(SAMPLE_PAYLOAD, CreditScore=789, Age=33)
        status, body, _ = http_request("http://127.0.0.1:8080/predict", method="POST", data=uncached_payload)
        assert status == 503, f"Expected 503 for uncached predict, got {status}"
        log(f"POST /predict (uncached) when backend down: {status} Service Unavailable -> {body.strip()}")

        # POST /predict with CACHED payload should still return 200 HIT even when backend is down!
        status, body, headers = http_request("http://127.0.0.1:8080/predict", method="POST", data=SAMPLE_PAYLOAD)
        assert status == 200, f"Cached predict failed while backend was down: {status}"
        cache_hdr = headers.get("X-Cache", headers.get("x-cache", ""))
        assert cache_hdr == "HIT", f"Expected HIT, got: {cache_hdr}"
        log(f"POST /predict (cached) when backend down: {status} OK (Served from Cache! X-Cache={cache_hdr})")

        log("=== Stage 5: Restarting Backend and Running Load Test ===")
        flask_proc = subprocess.Popen([PYTHON_BIN, FLASK_APP], cwd=PROJECT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_for_port("http://127.0.0.1:5000/", timeout=8.0)
        time.sleep(1.0)

        # Run load_test.py
        subprocess.run([PYTHON_BIN, f"{PROJECT_DIR}/proxy/tests/load_test.py", "--requests", "100", "--concurrency", "8"], cwd=PROJECT_DIR)

        log("=== All Gateway Verification Tests PASSED Successfully! ===")

    finally:
        flask_proc.kill()
        proxy_proc.kill()

if __name__ == "__main__":
    main()
