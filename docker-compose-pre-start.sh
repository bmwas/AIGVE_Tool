#!/usr/bin/env bash
# Pre-startup script for docker-compose
# Run this before 'docker-compose up' to ensure proper permissions

set -euo pipefail

echo "============================================================"
echo "AIGVE Docker Compose Pre-Startup Checks"
echo "============================================================"
echo ""

# Determine UID/GID (same logic as docker-compose)
TARGET_UID="${UID:-1000}"
TARGET_GID="${GID:-1000}"

echo "Target user: UID=$TARGET_UID, GID=$TARGET_GID"
echo ""

# List of required directories (from docker-compose volumes)
REQUIRED_DIRS=(
    "data"
    "results"
    "out"
    "grid_runs"
    "uploads"
)

echo "Checking required directories..."
echo "------------------------------------------------------------"

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "  📁 Creating: $dir"
        mkdir -p "$dir"
    else
        echo "  ✅ Exists: $dir"
    fi
    
    # Check ownership
    CURRENT_OWNER=$(stat -c '%u:%g' "$dir" 2>/dev/null || echo "unknown")
    
    if [ "$CURRENT_OWNER" != "${TARGET_UID}:${TARGET_GID}" ]; then
        echo "     ⚠️  Owner: $CURRENT_OWNER (expected ${TARGET_UID}:${TARGET_GID})"
        echo "     🔧 Fixing ownership..."
        
        if [ "$(id -u)" -eq 0 ]; then
            # Running as root, can fix directly
            chown "${TARGET_UID}:${TARGET_GID}" "$dir"
            echo "     ✅ Fixed as root"
        elif command -v sudo >/dev/null 2>&1; then
            # Try with sudo
            sudo chown "${TARGET_UID}:${TARGET_GID}" "$dir" 2>/dev/null || {
                echo "     ❌ Failed to fix (sudo required or failed)"
                echo "     Run: sudo chown ${TARGET_UID}:${TARGET_GID} $dir"
            }
        else
            echo "     ❌ Cannot fix: need root or sudo"
            echo "     Run: sudo chown ${TARGET_UID}:${TARGET_GID} $dir"
        fi
    else
        echo "     ✅ Owner: $CURRENT_OWNER"
    fi
    
    # Check permissions
    CURRENT_PERMS=$(stat -c '%a' "$dir" 2>/dev/null || echo "unknown")
    if [ "$CURRENT_PERMS" != "unknown" ] && [ "$CURRENT_PERMS" -lt 755 ]; then
        echo "     ⚠️  Permissions: $CURRENT_PERMS (should be at least 755)"
        chmod 755 "$dir" 2>/dev/null || {
            echo "     ❌ Failed to fix permissions"
        }
    fi
done

echo ""
echo "Checking for existing containers..."
echo "------------------------------------------------------------"

if docker ps -a --format '{{.Names}}' | grep -q '^aigve-1$'; then
    CONTAINER_STATUS=$(docker ps -a --format '{{.Status}}' --filter name=aigve-1)
    echo "  Found: aigve-1 - $CONTAINER_STATUS"
    
    if docker ps --format '{{.Names}}' | grep -q '^aigve-1$'; then
        echo "  ⚠️  Container is running - will need to recreate"
    else
        echo "  ℹ️  Container is stopped"
    fi
else
    echo "  ℹ️  No existing aigve container"
fi

echo ""
echo "Summary"
echo "------------------------------------------------------------"

ALL_OK=true
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -w "$dir" ]; then
        echo "  ❌ $dir is not writable by current user"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = true ]; then
    echo "  ✅ All directories are ready"
    echo ""
    echo "You can now run:"
    echo "    docker-compose up -d --force-recreate aigve"
    echo ""
    echo "Or if you made Dockerfile changes:"
    echo "    docker-compose build aigve --no-cache"
    echo "    docker-compose up -d --force-recreate aigve"
else
    echo ""
    echo "  ⚠️  Some directories need manual fixing"
    echo ""
    echo "Run these commands to fix ownership:"
    for dir in "${REQUIRED_DIRS[@]}"; do
        if [ ! -w "$dir" ]; then
            echo "    sudo chown ${TARGET_UID}:${TARGET_GID} $dir"
        fi
    done
    echo ""
    echo "Then run this script again: bash $0"
fi

echo "============================================================"

