#!/usr/bin/env bash
#
# test_runner.sh — Integration test suite for Customer Churn Python Inference Gateway
#
# Usage:
#   bash proxy/tests/test_runner.sh [PROXY_HOST] [PROXY_PORT]
#
# Default: 127.0.0.1:8080
#

set -uo pipefail

PROXY_HOST="${1:-127.0.0.1}"
PROXY_PORT="${2:-8080}"
BASE_URL="http://${PROXY_HOST}:${PROXY_PORT}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0
TOTAL=0

assert_equal() {
    local label="$1"
    local expected="$2"
    local actual="$3"
    TOTAL=$((TOTAL + 1))
    if [ "$expected" = "$actual" ]; then
        echo -e "  ${GREEN}[PASS]${NC} ${label} (expected: '${expected}', got: '${actual}')"
        PASSED=$((PASSED + 1))
    else
        echo -e "  ${RED}[FAIL]${NC} ${label} (expected: '${expected}', got: '${actual}')"
        FAILED=$((FAILED + 1))
    fi
}

assert_contains() {
    local label="$1"
    local needle="$2"
    local haystack="$3"
    TOTAL=$((TOTAL + 1))
    if echo "$haystack" | grep -q "$needle"; then
        echo -e "  ${GREEN}[PASS]${NC} ${label} (found: '${needle}')"
        PASSED=$((PASSED + 1))
    else
        echo -e "  ${RED}[FAIL]${NC} ${label} (missing: '${needle}')"
        FAILED=$((FAILED + 1))
    fi
}

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN} Customer Churn Proxy Integration Test Suite          ${NC}"
echo -e "${CYAN} Target: ${BASE_URL}                                  ${NC}"
echo -e "${CYAN}======================================================${NC}"

# Check proxy connectivity
echo -e "\n${YELLOW}[Test 1] Proxy Health Endpoint (GET /health)${NC}"
HTTP_CODE=$(curl -s -o /tmp/health_resp.json -w "%{http_code}" "${BASE_URL}/health" || echo "000")
assert_equal "HTTP status code" "200" "$HTTP_CODE"
if [ "$HTTP_CODE" = "200" ]; then
    assert_contains "Response has status ok" '"status":"ok"' "$(cat /tmp/health_resp.json)"
fi

echo -e "\n${YELLOW}[Test 2] Backend Health Endpoint (GET /health/backend)${NC}"
HTTP_CODE=$(curl -s -o /tmp/backend_resp.json -w "%{http_code}" "${BASE_URL}/health/backend" || echo "000")
assert_equal "Backend status code" "200" "$HTTP_CODE"
if [ "$HTTP_CODE" = "200" ]; then
    assert_contains "Backend reachable message" '"status":"ok"' "$(cat /tmp/backend_resp.json)"
fi

echo -e "\n${YELLOW}[Test 3] Metrics Endpoint (GET /metrics)${NC}"
METRICS_RESP=$(curl -s "${BASE_URL}/metrics" || echo "")
assert_contains "Contains total_requests metric" "total_requests" "$METRICS_RESP"
assert_contains "Contains cache_hits metric" "cache_hits" "$METRICS_RESP"
assert_contains "Contains cache_misses metric" "cache_misses" "$METRICS_RESP"

echo -e "\n${YELLOW}[Test 4] Prediction via Proxy (Cache MISS on first call)${NC}"
SAMPLE_PAYLOAD='{
  "CreditScore": 650,
  "Geography": "Germany",
  "Gender": "Female",
  "Age": 45,
  "Tenure": 5,
  "Balance": 100000.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 60000.0
}'

PRED_RESP_1=$(curl -s -i -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "$SAMPLE_PAYLOAD")

assert_contains "HTTP 200 OK in response" "200 OK" "$PRED_RESP_1"
assert_contains "X-Cache header is MISS" "X-Cache: MISS" "$PRED_RESP_1"
assert_contains "Contains churn_probability" "churn_probability" "$PRED_RESP_1"
assert_contains "Contains prediction" "prediction" "$PRED_RESP_1"

echo -e "\n${YELLOW}[Test 5] Prediction via Proxy (Cache HIT on second identical call)${NC}"
PRED_RESP_2=$(curl -s -i -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "$SAMPLE_PAYLOAD")

assert_contains "HTTP 200 OK in response" "200 OK" "$PRED_RESP_2"
assert_contains "X-Cache header is HIT" "X-Cache: HIT" "$PRED_RESP_2"
assert_contains "Contains churn_probability" "churn_probability" "$PRED_RESP_2"

echo -e "\n${YELLOW}[Test 6] Non-existent Endpoint (404 Not Found)${NC}"
HTTP_CODE_404=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/nonexistent-path")
assert_equal "404 for unknown route" "404" "$HTTP_CODE_404"

echo -e "\n${YELLOW}[Test 7] Forwarding to Flask Root Endpoint (GET /)${NC}"
ROOT_RESP=$(curl -s -i "${BASE_URL}/")
assert_contains "HTTP 200 OK on forwarded root" "200 OK" "$ROOT_RESP"
assert_contains "Contains Flask root message" "status" "$ROOT_RESP"

echo -e "\n${YELLOW}[Test 8] Rate Limiting Behavior${NC}"
# Send requests until either 429 is received or 200 requests attempted
RATE_LIMITED=0
for i in $(seq 1 150); do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health")
    if [ "$CODE" = "429" ]; then
        RATE_LIMITED=1
        break
    fi
done
assert_equal "Rate limiter triggered 429 Too Many Requests" "1" "$RATE_LIMITED"

# Final summary
echo -e "\n${CYAN}======================================================${NC}"
echo -e " Tests: ${TOTAL} | ${GREEN}Passed: ${PASSED}${NC} | ${RED}Failed: ${FAILED}${NC}"
echo -e "${CYAN}======================================================${NC}"

# Clean up temporary response files
rm -f /tmp/health_resp.json /tmp/backend_resp.json

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}All integration tests passed successfully!${NC}\n"
    exit 0
else
    echo -e "${RED}Some integration tests failed.${NC}\n"
    exit 1
fi
