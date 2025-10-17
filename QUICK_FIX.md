# Quick Fix for Your Docker Compose Setup

## TL;DR - The Problem

Your `docker-compose.yml` has volume mounts, and Docker may have created `./uploads` as **root**. Your container runs as **user 1000** and can't write to it.

## ⚡ Immediate Fix (30 seconds)

```bash
# Stop container
docker-compose stop aigve

# Fix ownership (run from directory with docker-compose.yml)
sudo chown ${UID:-1000}:${GID:-1000} ./uploads
sudo chmod 755 ./uploads

# Restart
docker-compose start aigve

# Verify
docker-compose logs aigve | grep "Uploads directory is writable"
```

If you see `✅ Uploads directory is writable`, you're done! Test the API:

```bash
curl http://localhost:2200/healthz
```

## 🛡️ Permanent Prevention

**Before EVERY fresh deployment**, run:

```bash
# Automated check and fix
bash docker-compose-pre-start.sh

# Then start
docker-compose up -d --force-recreate aigve
```

## 📊 Your Configuration Analysis

Your docker-compose.yml already has:
- ✅ `./uploads:/app/uploads:rw` (correct volume mount)
- ✅ `user: "${UID:-1000}:${GID:-1000}"` (correct user)

**The issue**: Host directory ownership, NOT configuration.

## 🔍 Why This Happens

1. `docker-compose up` runs
2. `./uploads` doesn't exist on host
3. Docker creates it (as root by default)
4. Container tries to write as user 1000
5. ❌ Permission denied

## ✅ What My Fixes Do

### 1. Enhanced Dockerfile
- Sets correct ownership inside image
- **BUT**: Volume mounts override this

### 2. Enhanced entrypoint.sh
- Detects permission issues at startup
- Shows clear error message with fix commands
- **What you'll see if permissions are wrong**:
  ```
  [ERROR] Cannot fix permissions - running as non-root user 1000
  
  SOLUTION 1: Fix host directory ownership (recommended)
    On the host machine, run:
      sudo chown 1000:1000 ./uploads
      sudo chmod 755 ./uploads
      docker-compose restart aigve
  ```

### 3. Enhanced server/main.py
- Tests write permissions on API startup
- Logs diagnostic info
- **What you'll see in logs**:
  ```
  INFO | aigve.api | Uploads directory ready: /app/uploads
  INFO | aigve.api | Uploads directory is writable
  ```
  OR (if broken):
  ```
  ERROR | aigve.api | Uploads directory is NOT writable: [Errno 13]
  ERROR | aigve.api | Container user: uid=1000, gid=1000
  ERROR | aigve.api | Directory permissions: 755
  ```

### 4. Pre-Startup Script
- Checks ALL your volume mount directories
- Creates them if missing
- Sets correct ownership
- Verifies write access

## 🎯 Will This Fix Your Error?

**YES**, if you:
1. Run `bash docker-compose-pre-start.sh` before deployment, **OR**
2. Manually run `sudo chown ${UID:-1000}:${GID:-1000} ./uploads`
3. Then restart the container

**You will NEVER see this error again** because:
- Pre-startup script ensures correct ownership
- Entrypoint detects and warns at container start
- API startup logs permission status
- All before any upload attempts

## 📝 Deployment Checklist

```bash
# Step 1: Fix directories (one-time or after fresh clone)
bash docker-compose-pre-start.sh

# Step 2: Build if you made Dockerfile changes
docker-compose build aigve --no-cache

# Step 3: Start (force recreate to ensure fresh state)
docker-compose up -d --force-recreate aigve

# Step 4: Verify (should see "Uploads directory is writable")
docker-compose logs aigve | grep -E "upload|permission" -i

# Step 5: Test API
curl http://localhost:2200/healthz
```

## 🔧 If You're Still Getting the Error

Check these in order:

### 1. Verify host directory ownership
```bash
ls -la ./uploads
# Should show: drwxr-xr-x ... YOUR_USER YOUR_GROUP ... uploads
```

### 2. Verify container user
```bash
docker-compose exec aigve id
# Should show: uid=1000 gid=1000
```

### 3. Test write access from inside container
```bash
docker-compose exec aigve touch /app/uploads/.test
docker-compose exec aigve rm /app/uploads/.test
# Should succeed without errors
```

### 4. Check container logs
```bash
docker-compose logs aigve | tail -50
# Look for permission-related messages
```

### 5. If nothing works
```bash
# Nuclear option: remove and recreate
docker-compose down
sudo rm -rf ./uploads
mkdir -p ./uploads
sudo chown ${UID:-1000}:${GID:-1000} ./uploads
docker-compose up -d
```

## 📞 Need More Help?

See detailed guides:
- **DOCKER_COMPOSE_SETUP.md** - Complete guide for your docker-compose setup
- **DOCKER_PERMISSIONS_FIX.md** - All possible permission scenarios
- **README.md** (lines 935-979) - Original documentation

Or run the automated diagnostic:
```bash
bash fix_docker_permissions.sh aigve-1
```

## ✨ One-Liner for Future Deployments

```bash
bash docker-compose-pre-start.sh && docker-compose up -d --force-recreate aigve && docker-compose logs aigve | tail -20
```

This will:
1. ✅ Check/fix directories
2. ✅ Start container fresh
3. ✅ Show startup logs
4. ✅ Verify permissions

**Save this as an alias**:
```bash
echo 'alias aigve-deploy="bash docker-compose-pre-start.sh && docker-compose up -d --force-recreate aigve && docker-compose logs aigve | tail -20"' >> ~/.bashrc
source ~/.bashrc

# Then just run:
aigve-deploy
```

---

**Bottom line**: Your docker-compose config is perfect. Just ensure host directories have correct ownership before starting. My fixes make this obvious and easy to fix.

