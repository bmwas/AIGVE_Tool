# Docker Compose Setup Guide for AIGVE

This guide is specifically for deployments using docker-compose with volume mounts and the `user:` directive.

## Your Configuration

```yaml
aigve:
  image: ghcr.io/bmwas/aigve:latest
  build:
    context: .
    dockerfile: Dockerfile
  ports:
    - "2200:2200"
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
    - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    - PORT=2200
    - REQUIRE_GPU=1
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  volumes:
    - ./data:/app/data:rw
    - ./results:/app/results:rw
    - ./out:/app/out:rw
    - ./grid_runs:/app/grid_runs:rw
    - .:/app/host:rw
    - ./uploads:/app/uploads:rw  # ← Critical for upload functionality
  user: "${UID:-1000}:${GID:-1000}"
  restart: unless-stopped
  networks:
    - multistatic-network
```

## ⚠️ Important: Volume Mounts Override Container Permissions

**Key Insight**: When you use `- ./uploads:/app/uploads:rw`, the **host directory's ownership/permissions** override whatever was set in the Dockerfile.

This means:
- ✅ Dockerfile sets `/app/uploads` owned by 1000:1000
- ❌ But `./uploads` on host might be owned by root
- ❌ Volume mount replaces container directory with host directory
- ❌ Container running as user 1000 can't write to root-owned directory

## 🔧 Correct Setup Procedure

### Step 1: Run Pre-Startup Script (Recommended)

```bash
# From the directory containing docker-compose.yml
bash docker-compose-pre-start.sh
```

This script will:
- Create all required directories (`data`, `results`, `out`, `grid_runs`, `uploads`)
- Set correct ownership (`${UID:-1000}:${GID:-1000}`)
- Verify write permissions
- Provide fix commands if issues found

### Step 2: Start/Restart Container

**If building from local Dockerfile:**
```bash
# Build with no cache to ensure latest fixes
docker-compose build aigve --no-cache

# Start with force-recreate to ensure fresh container
docker-compose up -d --force-recreate aigve
```

**If using pre-built image:**
```bash
# Pull latest image
docker-compose pull aigve

# Start with force-recreate
docker-compose up -d --force-recreate aigve
```

### Step 3: Verify

```bash
# Check container logs for permission checks
docker-compose logs aigve | grep -E "upload|permission" -i

# Should see:
# [INFO] Creating /app/uploads directory...
# [INFO] Uploads directory ready: /app/uploads
# [INFO] Uploads directory is writable

# Test write access
docker-compose exec aigve touch /app/uploads/.test
docker-compose exec aigve rm /app/uploads/.test
echo "✅ Upload directory is writable!"

# Test API endpoint
curl http://localhost:2200/healthz
```

## 🚨 Manual Fix (If Pre-Startup Script Not Used)

If you started the container without running the pre-startup script:

### Quick Fix (Current Session Only)

```bash
# Stop container
docker-compose stop aigve

# Fix host directory ownership
sudo chown ${UID:-1000}:${GID:-1000} ./uploads
sudo chmod 755 ./uploads

# Restart container
docker-compose start aigve

# Verify
docker-compose logs aigve | tail -20
```

### Fix for All Directories

```bash
# Stop all services
docker-compose down

# Fix all volume mount directories
for dir in data results out grid_runs uploads; do
  sudo chown ${UID:-1000}:${GID:-1000} ./$dir
  sudo chmod 755 ./$dir
done

# Restart services
docker-compose up -d
```

## 🔍 Troubleshooting

### Problem: "Permission denied: '/app/uploads/...'"

**Symptom**: API fails with `PermissionError: [Errno 13]`

**Cause**: Host `./uploads` directory owned by wrong user

**Solution**:
```bash
# Check current ownership
ls -la ./uploads

# Fix ownership
sudo chown ${UID:-1000}:${GID:-1000} ./uploads

# Restart container
docker-compose restart aigve
```

### Problem: Container logs show "uploads is not writable"

**Symptom**: Entrypoint shows error about non-writable directory

**Diagnosis**: Volume mount ownership issue

**Solution**:
```bash
# The entrypoint will show you the exact commands needed
docker-compose logs aigve | grep -A 10 "not writable"

# Follow the suggested commands, typically:
sudo chown $(id -u):$(id -g) ./uploads
docker-compose restart aigve
```

### Problem: Directory doesn't exist

**Symptom**: Docker creates `./uploads` as root

**Cause**: Docker auto-creates missing volume mount directories as root

**Prevention**:
```bash
# Always create directories BEFORE first run
mkdir -p data results out grid_runs uploads
chown ${UID:-1000}:${GID:-1000} data results out grid_runs uploads
```

**Fix if already created as root**:
```bash
docker-compose down
sudo chown -R ${UID:-1000}:${GID:-1000} ./uploads
docker-compose up -d
```

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Pre-startup script ran without errors
- [ ] All directories exist and have correct ownership
- [ ] Container starts without permission errors
- [ ] Logs show "Uploads directory is writable"
- [ ] Can touch/remove files in /app/uploads
- [ ] API healthz endpoint responds
- [ ] Can successfully upload files via API

### Comprehensive Test

```bash
#!/bin/bash
echo "Running AIGVE verification tests..."

# Test 1: Check directory ownership
echo "1. Checking host directory ownership..."
stat -c "uploads: %U:%G (perms: %a)" ./uploads

# Test 2: Check container can write
echo "2. Testing container write access..."
docker-compose exec aigve touch /app/uploads/.test && \
  docker-compose exec aigve rm /app/uploads/.test && \
  echo "✅ Write test passed" || echo "❌ Write test failed"

# Test 3: Check API startup
echo "3. Checking API startup logs..."
docker-compose logs aigve | grep -E "Uploads directory (ready|writable)" && \
  echo "✅ API startup checks passed" || echo "❌ API startup checks failed"

# Test 4: Test healthz endpoint
echo "4. Testing API healthz endpoint..."
curl -s http://localhost:2200/healthz | jq -r '.status' | grep -q "ok" && \
  echo "✅ API healthz passed" || echo "❌ API healthz failed"

# Test 5: Test upload endpoint (if you have test files)
if [ -f "./data/test_video.mp4" ]; then
  echo "5. Testing upload endpoint..."
  curl -s -X POST http://localhost:2200/run_upload \
    -F "videos=@./data/test_video.mp4" \
    -F "compute=false" | jq -r '.status' && \
    echo "✅ Upload endpoint passed" || echo "❌ Upload endpoint failed"
else
  echo "5. Skipping upload test (no test video found)"
fi

echo "Verification complete!"
```

Save as `verify_aigve_setup.sh`, make executable, and run:
```bash
chmod +x verify_aigve_setup.sh
bash verify_aigve_setup.sh
```

## 📋 Best Practices

1. **Always run pre-startup script before first deployment**
2. **Create volume directories before docker-compose up**
3. **Use correct UID:GID matching your user**
4. **Check logs after every restart**
5. **Test with a simple upload before production use**

## 🔄 Deployment Workflow

### Initial Deployment

```bash
# 1. Clone/update repository
git pull origin main

# 2. Run pre-startup checks
bash docker-compose-pre-start.sh

# 3. Build if needed
docker-compose build aigve --no-cache  # Only if building locally

# 4. Start services
docker-compose up -d

# 5. Verify
docker-compose logs -f aigve  # Watch for startup messages
```

### Updates/Rebuilds

```bash
# 1. Pull changes
git pull origin main

# 2. Verify directories (quick check)
bash docker-compose-pre-start.sh

# 3. Rebuild
docker-compose build aigve --no-cache

# 4. Recreate container
docker-compose up -d --force-recreate aigve

# 5. Verify
docker-compose logs aigve | tail -30
```

### Production Restart

```bash
# If only restarting (no code changes)
docker-compose restart aigve

# Full restart (cleans state)
docker-compose stop aigve
docker-compose up -d aigve

# Check status
docker-compose ps aigve
docker-compose logs aigve | tail -20
```

## 🎯 Summary

Your docker-compose configuration **is correct** and includes all necessary components:
- ✅ Volume mount for `/app/uploads`
- ✅ User directive matching container user (1000:1000)
- ✅ GPU configuration
- ✅ Network configuration

**The only requirement** is ensuring the **host directories** have correct ownership **before** starting the container.

**My fixes enhance this by**:
1. ✅ Detecting permission issues at startup
2. ✅ Providing clear error messages with solutions
3. ✅ Automated pre-startup checks
4. ✅ Diagnostic logging for troubleshooting

**You will NOT see the original error again if**:
- You run `docker-compose-pre-start.sh` before deployment
- Or manually ensure `./uploads` is owned by your UID:GID
- And restart the container after fixing ownership

## 🆘 Getting Help

If you still encounter issues:

1. **Collect diagnostic information**:
   ```bash
   # Container info
   docker-compose ps
   docker-compose logs aigve | tail -50 > aigve_logs.txt
   
   # Host directory info
   ls -la . | grep -E "data|results|uploads|out|grid_runs" > dir_perms.txt
   
   # User info
   id > user_info.txt
   
   # Test write access
   docker-compose exec aigve sh -c "id && ls -la /app/uploads" > container_info.txt
   ```

2. **Check these files**:
   - DOCKER_PERMISSIONS_FIX.md (general fixes)
   - DOCKER_COMPOSE_SETUP.md (this file)
   - README.md (lines 935-979)

3. **Common issues and solutions**:
   - Directory owned by root → `sudo chown ${UID}:${GID} ./uploads`
   - SELinux issues → Add `:z` or `:Z` to volume mounts
   - NFS mounts → May need different permissions (777)
   - Multiple users → Create shared group with write access

