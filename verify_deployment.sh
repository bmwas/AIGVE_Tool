#!/usr/bin/env bash
# Comprehensive verification script for AIGVE deployment
# Tests all aspects of the permission fix

set -euo pipefail

CONTAINER_NAME="${1:-aigve-1}"
BASE_URL="${2:-http://localhost:2200}"

echo "=============================================================="
echo "AIGVE Deployment Verification"
echo "=============================================================="
echo ""
echo "Container: $CONTAINER_NAME"
echo "Base URL:  $BASE_URL"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0
WARNINGS=0

pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
    ((WARNINGS++))
}

# Test 1: Check container exists and is running
echo "Test 1: Container Status"
echo "--------------------------------------------------------------"
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    STATUS=$(docker ps --format '{{.Status}}' --filter name=${CONTAINER_NAME})
    pass "Container is running - $STATUS"
else
    fail "Container is not running"
    echo ""
    echo "Available containers:"
    docker ps -a --format 'table {{.Names}}\t{{.Status}}'
    exit 1
fi
echo ""

# Test 2: Check host directory ownership
echo "Test 2: Host Directory Ownership"
echo "--------------------------------------------------------------"
if [ -d "./uploads" ]; then
    OWNER=$(stat -c '%U:%G' ./uploads 2>/dev/null || echo "unknown")
    PERMS=$(stat -c '%a' ./uploads 2>/dev/null || echo "unknown")
    EXPECTED_UID="${UID:-1000}"
    EXPECTED_USER=$(id -un ${EXPECTED_UID} 2>/dev/null || echo "${EXPECTED_UID}")
    
    echo "Directory: ./uploads"
    echo "Owner:     $OWNER"
    echo "Perms:     $PERMS"
    echo "Expected:  $EXPECTED_USER:... (UID $EXPECTED_UID)"
    
    if [ -w "./uploads" ]; then
        pass "Host directory is writable"
    else
        fail "Host directory is not writable"
        echo "Fix: sudo chown ${EXPECTED_UID}:${EXPECTED_UID} ./uploads"
    fi
else
    fail "Host ./uploads directory doesn't exist"
    echo "Fix: mkdir -p uploads && sudo chown ${UID:-1000}:${GID:-1000} uploads"
fi
echo ""

# Test 3: Check container directory ownership
echo "Test 3: Container Directory Ownership"
echo "--------------------------------------------------------------"
CONTAINER_OWNER=$(docker exec ${CONTAINER_NAME} stat -c '%u:%g' /app/uploads 2>/dev/null || echo "unknown")
CONTAINER_PERMS=$(docker exec ${CONTAINER_NAME} stat -c '%a' /app/uploads 2>/dev/null || echo "unknown")
CONTAINER_USER=$(docker exec ${CONTAINER_NAME} id -u)

echo "Container user: $CONTAINER_USER"
echo "Directory:      /app/uploads"
echo "Owner:          $CONTAINER_OWNER"
echo "Perms:          $CONTAINER_PERMS"

if [ "$CONTAINER_OWNER" = "${CONTAINER_USER}:${CONTAINER_USER}" ] || [ "$CONTAINER_OWNER" = "1000:1000" ]; then
    pass "Container directory has correct ownership"
else
    warn "Container directory ownership mismatch (but may still work)"
fi
echo ""

# Test 4: Test write access from container
echo "Test 4: Container Write Access"
echo "--------------------------------------------------------------"
if docker exec ${CONTAINER_NAME} touch /app/uploads/.test_write 2>/dev/null; then
    docker exec ${CONTAINER_NAME} rm /app/uploads/.test_write 2>/dev/null
    pass "Container can write to /app/uploads"
else
    fail "Container cannot write to /app/uploads"
    echo "Fix: sudo chown ${UID:-1000}:${GID:-1000} ./uploads && docker-compose restart aigve"
fi
echo ""

# Test 5: Check entrypoint logs
echo "Test 5: Entrypoint Permission Checks"
echo "--------------------------------------------------------------"
if docker logs ${CONTAINER_NAME} 2>&1 | grep -q "Creating /app/uploads directory"; then
    pass "Entrypoint directory check executed"
else
    warn "Entrypoint directory check not found in logs (container may be old)"
fi

if docker logs ${CONTAINER_NAME} 2>&1 | grep -q "not writable"; then
    fail "Entrypoint detected permission issues"
    echo ""
    echo "Recent logs:"
    docker logs ${CONTAINER_NAME} 2>&1 | grep -A 10 "not writable" | tail -15
else
    pass "No permission warnings in entrypoint"
fi
echo ""

# Test 6: Check API startup logs
echo "Test 6: API Startup Permission Checks"
echo "--------------------------------------------------------------"
if docker logs ${CONTAINER_NAME} 2>&1 | grep -q "aigve.api.*Uploads directory ready"; then
    pass "API startup checks executed"
else
    warn "API startup checks not found (API may not have started)"
fi

if docker logs ${CONTAINER_NAME} 2>&1 | grep -q "aigve.api.*Uploads directory is writable"; then
    pass "API confirmed uploads directory is writable"
elif docker logs ${CONTAINER_NAME} 2>&1 | grep -q "aigve.api.*is NOT writable"; then
    fail "API detected uploads directory is NOT writable"
    echo ""
    echo "API logs:"
    docker logs ${CONTAINER_NAME} 2>&1 | grep "aigve.api.*writable" | tail -5
else
    warn "Cannot find API writable status in logs"
fi
echo ""

# Test 7: Test API healthz endpoint
echo "Test 7: API Health Endpoint"
echo "--------------------------------------------------------------"
if command -v curl >/dev/null 2>&1; then
    HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" ${BASE_URL}/healthz 2>/dev/null || echo "000")
    
    if [ "$HEALTH_RESPONSE" = "200" ]; then
        pass "API healthz endpoint responding (HTTP 200)"
        
        # Try to get JSON response
        HEALTH_JSON=$(curl -s ${BASE_URL}/healthz 2>/dev/null || echo "{}")
        STATUS=$(echo "$HEALTH_JSON" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
        
        if [ "$STATUS" = "ok" ]; then
            pass "API status is 'ok'"
        else
            warn "API status is '$STATUS' (expected 'ok')"
        fi
    else
        fail "API healthz endpoint not responding (HTTP $HEALTH_RESPONSE)"
        echo "Check if API is running: docker logs ${CONTAINER_NAME} | tail -20"
    fi
else
    warn "curl not installed, skipping API health check"
fi
echo ""

# Test 8: Test upload endpoint (if curl is available)
echo "Test 8: Upload Endpoint Test"
echo "--------------------------------------------------------------"
if command -v curl >/dev/null 2>&1; then
    # Try to upload a minimal test (without actual files)
    UPLOAD_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST ${BASE_URL}/run_upload \
        -F "compute=false" \
        2>/dev/null || echo "000")
    
    # We expect 422 (validation error) because we didn't send files
    # This is actually GOOD - it means the endpoint is working
    if [ "$UPLOAD_RESPONSE" = "422" ]; then
        pass "Upload endpoint is accessible (validation working)"
    elif [ "$UPLOAD_RESPONSE" = "200" ]; then
        pass "Upload endpoint responding (HTTP 200)"
    elif [ "$UPLOAD_RESPONSE" = "500" ] || [ "$UPLOAD_RESPONSE" = "000" ]; then
        fail "Upload endpoint returned error (HTTP $UPLOAD_RESPONSE)"
    else
        warn "Upload endpoint returned unexpected status (HTTP $UPLOAD_RESPONSE)"
    fi
else
    warn "curl not installed, skipping upload endpoint test"
fi
echo ""

# Test 9: Check for recent errors in logs
echo "Test 9: Recent Error Check"
echo "--------------------------------------------------------------"
ERROR_COUNT=$(docker logs ${CONTAINER_NAME} 2>&1 | grep -c "PermissionError" || echo 0)

if [ "$ERROR_COUNT" -eq 0 ]; then
    pass "No PermissionError found in logs"
else
    fail "Found $ERROR_COUNT PermissionError(s) in logs"
    echo ""
    echo "Recent permission errors:"
    docker logs ${CONTAINER_NAME} 2>&1 | grep -B 2 -A 2 "PermissionError" | tail -20
fi
echo ""

# Summary
echo "=============================================================="
echo "Summary"
echo "=============================================================="
echo ""
echo -e "Passed:   ${GREEN}$PASSED${NC}"
echo -e "Failed:   ${RED}$FAILED${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All critical tests passed!${NC}"
    echo ""
    echo "Your AIGVE deployment is ready for use."
    echo ""
    echo "API Endpoints:"
    echo "  - Health:  ${BASE_URL}/healthz"
    echo "  - Docs:    ${BASE_URL}/docs"
    echo "  - Upload:  ${BASE_URL}/run_upload"
    echo ""
    
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}Note: There are $WARNINGS warning(s). Review them above.${NC}"
    fi
    
    exit 0
else
    echo -e "${RED}❌ $FAILED test(s) failed${NC}"
    echo ""
    echo "Common fixes:"
    echo "  1. Fix directory ownership:"
    echo "     sudo chown ${UID:-1000}:${GID:-1000} ./uploads"
    echo "     docker-compose restart aigve"
    echo ""
    echo "  2. Run pre-startup script:"
    echo "     bash docker-compose-pre-start.sh"
    echo "     docker-compose up -d --force-recreate aigve"
    echo ""
    echo "  3. Check detailed logs:"
    echo "     docker logs ${CONTAINER_NAME} | tail -50"
    echo ""
    echo "See DOCKER_COMPOSE_SETUP.md for detailed troubleshooting."
    exit 1
fi

