# Docker Upload Permission Error - Complete Fix Guide

## Problem
```
PermissionError: [Errno 13] Permission denied: '/app/uploads/2c731950-f6ca-4ac2-96a0-1bea9c9a9f63'
```

This error occurs when the Docker container cannot create subdirectories in `/app/uploads` due to permission issues.

## Root Causes

1. **Docker Compose with user directive**: Running with `user: "${UID:-1000}:${GID:-1000}"` but uploads directory owned by root
2. **Volume mounts**: External volume mounts overriding built-in permissions
3. **Missing directory**: Uploads directory not existing at runtime
4. **Permission mismatch**: Container user doesn't match directory ownership

## Solutions (Choose Based on Your Setup)

### ✅ Solution 1: Volume Mount (Best for docker-compose)

If you're using `docker-compose.yml`, add a volume mount for uploads:

```yaml
services:
  aigve:
    image: ghcr.io/bmwas/aigve:latest
    volumes:
      - ./data:/app/data:rw
      - ./results:/app/results:rw
      - ./uploads:/app/uploads:rw  # ← ADD THIS LINE
    # ... rest of config
```

Then create the directory with proper permissions:

```bash
# Create directory
mkdir -p uploads

# Set ownership (use your actual UID/GID)
sudo chown ${UID:-1000}:${GID:-1000} uploads
chmod 755 uploads

# Restart container
docker-compose down
docker-compose up -d --force-recreate
```

**Why this works**: The host directory has correct ownership and is mounted into the container.

---

### ✅ Solution 2: Runtime Fix (Immediate/Temporary)

If the container is already running:

```bash
# Replace 'aigve-1' with your actual container name/id
CONTAINER_NAME="aigve-1"

# Option A: Run as root to fix permissions
docker exec -u root $CONTAINER_NAME mkdir -p /app/uploads
docker exec -u root $CONTAINER_NAME chown -R 1000:1000 /app/uploads
docker exec -u root $CONTAINER_NAME chmod -R 755 /app/uploads

# Option B: If running as non-root, recreate directory at user level
docker exec $CONTAINER_NAME mkdir -p /app/uploads

# Verify permissions
docker exec $CONTAINER_NAME ls -la /app/ | grep uploads
```

**Why this works**: Fixes permissions in running container, but won't persist after restart.

---

### ✅ Solution 3: Rebuild Image (Permanent)

The Dockerfile has been updated with enhanced permission handling. Rebuild your image:

```bash
# Pull latest changes
git pull origin main

# Rebuild image (no cache to ensure clean build)
docker build --no-cache -t ghcr.io/bmwas/aigve:latest .

# If using docker-compose
docker-compose build aigve --no-cache
docker-compose up -d
```

**Why this works**: Updated Dockerfile ensures proper ownership from build time.

---

### ✅ Solution 4: Remove user Directive (docker-compose)

If you don't need specific UID/GID, remove the `user:` directive:

```yaml
services:
  aigve:
    image: ghcr.io/bmwas/aigve:latest
    # user: "${UID:-1000}:${GID:-1000}"  # ← COMMENT OUT OR REMOVE
    volumes:
      - ./data:/app/data:rw
      - ./results:/app/results:rw
```

```bash
docker-compose down
docker-compose up -d
```

**Why this works**: Container runs as USER 1000 defined in Dockerfile, which has proper permissions.

---

### ✅ Solution 5: Using docker run (No Compose)

If running with `docker run`:

```bash
# Option A: Let container use built-in user (recommended)
docker run --rm --gpus '"device=1"' -p 2200:2200 \
  -v "$PWD/data":/app/data \
  -v "$PWD/results":/app/results \
  -v "$PWD/uploads":/app/uploads \
  ghcr.io/bmwas/aigve:latest

# Option B: Run as root (not recommended for production)
docker run --rm --gpus '"device=1"' -p 2200:2200 \
  --user root \
  -v "$PWD/data":/app/data \
  -v "$PWD/results":/app/results \
  ghcr.io/bmwas/aigve:latest
```

---

## Verification Steps

After applying any fix, verify it works:

### 1. Check Directory Exists and is Writable

```bash
# Check if directory exists
docker exec aigve-1 ls -la /app/ | grep uploads

# Should show something like:
# drwxr-xr-x  2 1000 1000  4096 Oct 17 12:00 uploads

# Test write permissions
docker exec aigve-1 touch /app/uploads/test_file
docker exec aigve-1 ls -la /app/uploads/test_file
docker exec aigve-1 rm /app/uploads/test_file
```

### 2. Check Server Logs

```bash
docker logs aigve-1 | grep -i upload
```

You should see:
```
INFO | aigve.api | Uploads directory ready: /app/uploads
INFO | aigve.api | Uploads directory is writable
```

If you see errors:
```
ERROR | aigve.api | Uploads directory is NOT writable: [Errno 13] Permission denied
ERROR | aigve.api | Container user: uid=1000, gid=1000
ERROR | aigve.api | Directory permissions: 755
```

This means the directory owner doesn't match the running user.

### 3. Test API Endpoint

```bash
# Test with curl
curl -X POST http://localhost:2200/run_upload \
  -F "videos=@./data/test_video.mp4" \
  -F "videos=@./data/test_video_synthetic.mp4" \
  -F "compute=false"

# Should succeed without PermissionError
```

---

## Understanding the Updates

### Updated Dockerfile
The Dockerfile now includes:
- `chown -R 1000:1000 /app` to ensure entire app directory is owned by user 1000
- Enhanced permission settings on `/app/uploads`

### Updated entrypoint.sh
The entrypoint now:
- Checks if `/app/uploads` exists and creates it if missing
- Tests write permissions at startup
- Logs warnings if permissions are incorrect

### Updated server/main.py
The FastAPI server now:
- Creates `/app/uploads` on startup if missing
- Tests write permissions before accepting requests
- Logs detailed permission information for debugging

---

## Docker Compose Example

Complete working `docker-compose.yml`:

```yaml
version: "3.8"

services:
  aigve:
    image: ghcr.io/bmwas/aigve:latest
    container_name: aigve
    restart: unless-stopped
    
    # Option 1: Run as default user (recommended)
    # No user directive needed - Dockerfile sets USER 1000
    
    # Option 2: Run as specific user
    # user: "${UID:-1000}:${GID:-1000}"
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    environment:
      - PORT=2200
      - REQUIRE_GPU=1
      - AIGVE_LOG_LEVEL=INFO
    
    volumes:
      - ./data:/app/data:rw
      - ./results:/app/results:rw
      - ./uploads:/app/uploads:rw  # Important!
    
    ports:
      - "2200:2200"
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:2200/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Start with:
```bash
# Create directories
mkdir -p data results uploads
sudo chown ${UID:-1000}:${GID:-1000} data results uploads

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f aigve
```

---

## Prevention Checklist

When deploying AIGVE with Docker:

- [ ] Create host directories for volume mounts (`mkdir -p data results uploads`)
- [ ] Set proper ownership (`sudo chown $UID:$GID data results uploads`)
- [ ] Add volume mount for uploads in docker-compose.yml
- [ ] Don't run as root unless absolutely necessary
- [ ] Check logs on startup for permission warnings
- [ ] Test with a simple upload before production use

---

## Need More Help?

If none of these solutions work:

1. **Check your actual docker-compose.yml or docker run command**
   - Look for `user:` directives
   - Check volume mounts
   - Verify UID/GID values

2. **Inspect the container**
   ```bash
   docker exec aigve-1 id                        # Check current user
   docker exec aigve-1 ls -la /app/              # Check ownership
   docker exec aigve-1 stat /app/uploads         # Detailed permissions
   ```

3. **Check Docker logs**
   ```bash
   docker logs aigve-1 2>&1 | grep -E "upload|permission|error" -i
   ```

4. **File an issue** with:
   - Your docker-compose.yml or docker run command
   - Output of the inspection commands above
   - Full error message from logs

---

## Summary of Changes Made

### Files Modified:
1. **Dockerfile** - Enhanced ownership settings for /app/uploads
2. **entrypoint.sh** - Added startup checks for uploads directory
3. **server/main.py** - Added FastAPI startup event to verify permissions

### What These Changes Do:
- **Build time**: Dockerfile ensures correct ownership (1000:1000)
- **Container startup**: entrypoint.sh creates and checks uploads directory
- **Application startup**: FastAPI tests write permissions and logs status
- **Runtime**: Better error messages if permissions are still wrong

These changes make the container more resilient to various deployment scenarios while providing clear diagnostic information when issues occur.

