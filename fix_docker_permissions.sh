#!/usr/bin/env bash
# Quick fix script for Docker upload permission errors
# Usage: bash fix_docker_permissions.sh [container-name]

set -euo pipefail

CONTAINER_NAME="${1:-aigve-1}"

echo "==================================================================="
echo "Docker Upload Permissions Fix Script"
echo "==================================================================="
echo ""
echo "Target container: $CONTAINER_NAME"
echo ""

# Check if container exists
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ Error: Container '$CONTAINER_NAME' not found"
    echo ""
    echo "Available containers:"
    docker ps -a --format 'table {{.Names}}\t{{.Status}}'
    echo ""
    echo "Usage: $0 [container-name]"
    exit 1
fi

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "⚠️  Warning: Container '$CONTAINER_NAME' is not running"
    echo "   Starting container..."
    docker start "$CONTAINER_NAME" || {
        echo "❌ Failed to start container"
        exit 1
    }
    sleep 2
fi

echo "Step 1: Checking current permissions..."
echo "-------------------------------------------------------------------"
docker exec "$CONTAINER_NAME" sh -c "ls -la /app/ | grep -E 'uploads|total'" || true
echo ""

echo "Step 2: Checking current user..."
echo "-------------------------------------------------------------------"
CURRENT_UID=$(docker exec "$CONTAINER_NAME" id -u)
CURRENT_GID=$(docker exec "$CONTAINER_NAME" id -g)
echo "Container running as: uid=$CURRENT_UID, gid=$CURRENT_GID"
echo ""

echo "Step 3: Creating uploads directory if missing..."
echo "-------------------------------------------------------------------"
docker exec "$CONTAINER_NAME" mkdir -p /app/uploads 2>&1 || {
    echo "⚠️  Cannot create as current user, trying as root..."
    docker exec -u root "$CONTAINER_NAME" mkdir -p /app/uploads
}
echo "✅ Directory exists"
echo ""

echo "Step 4: Fixing ownership..."
echo "-------------------------------------------------------------------"
# Try to fix as root
if docker exec -u root "$CONTAINER_NAME" chown -R "${CURRENT_UID}:${CURRENT_GID}" /app/uploads 2>/dev/null; then
    echo "✅ Fixed ownership as root"
else
    echo "⚠️  Could not fix as root, trying chmod..."
    docker exec -u root "$CONTAINER_NAME" chmod -R 777 /app/uploads 2>/dev/null || {
        echo "❌ Could not fix permissions even as root"
    }
fi
echo ""

echo "Step 5: Setting permissions..."
echo "-------------------------------------------------------------------"
docker exec -u root "$CONTAINER_NAME" chmod 755 /app/uploads 2>/dev/null || {
    echo "⚠️  Could not set 755, trying 777..."
    docker exec -u root "$CONTAINER_NAME" chmod 777 /app/uploads 2>/dev/null || true
}
echo "✅ Permissions updated"
echo ""

echo "Step 6: Verifying fix..."
echo "-------------------------------------------------------------------"
docker exec "$CONTAINER_NAME" sh -c "ls -la /app/ | grep uploads"
echo ""

echo "Step 7: Testing write access..."
echo "-------------------------------------------------------------------"
if docker exec "$CONTAINER_NAME" touch /app/uploads/.test_write_access 2>/dev/null; then
    docker exec "$CONTAINER_NAME" rm /app/uploads/.test_write_access
    echo "✅ Write test PASSED - uploads directory is writable!"
else
    echo "❌ Write test FAILED - still cannot write to uploads directory"
    echo ""
    echo "This might be due to:"
    echo "  1. SELinux or AppArmor restrictions"
    echo "  2. Volume mount ownership issues"
    echo "  3. Filesystem restrictions"
    echo ""
    echo "Recommended actions:"
    echo "  1. Add volume mount to docker-compose.yml:"
    echo "     volumes:"
    echo "       - ./uploads:/app/uploads:rw"
    echo ""
    echo "  2. Create host directory:"
    echo "     mkdir -p uploads && chown \${UID:-1000}:\${GID:-1000} uploads"
    echo ""
    echo "  3. Restart container:"
    echo "     docker-compose restart $CONTAINER_NAME"
    echo ""
    echo "See DOCKER_PERMISSIONS_FIX.md for detailed solutions"
    exit 1
fi
echo ""

echo "Step 8: Checking API server status..."
echo "-------------------------------------------------------------------"
if docker logs "$CONTAINER_NAME" 2>&1 | tail -20 | grep -q "Uploads directory is writable"; then
    echo "✅ API server confirmed uploads directory is writable"
elif docker logs "$CONTAINER_NAME" 2>&1 | tail -20 | grep -q "Uploads directory is NOT writable"; then
    echo "⚠️  API server reports uploads directory is NOT writable"
    echo "    (This may be from before the fix - restart to verify)"
    echo ""
    echo "    Run: docker restart $CONTAINER_NAME"
else
    echo "ℹ️  Cannot confirm API server status (may not be started yet)"
fi
echo ""

echo "==================================================================="
echo "✅ Fix Complete!"
echo "==================================================================="
echo ""
echo "Next steps:"
echo "  1. Test the API endpoint:"
echo "     curl http://localhost:2200/healthz"
echo ""
echo "  2. Try uploading a file:"
echo "     curl -X POST http://localhost:2200/run_upload \\"
echo "       -F 'videos=@./data/test_video.mp4' \\"
echo "       -F 'videos=@./data/test_video_synthetic.mp4' \\"
echo "       -F 'compute=false'"
echo ""
echo "  3. For permanent fix, see: DOCKER_PERMISSIONS_FIX.md"
echo ""
echo "If the issue persists after container restart, you'll need to:"
echo "  - Add volume mount for /app/uploads in docker-compose.yml"
echo "  - Or rebuild the image with updated Dockerfile"
echo "==================================================================="

