#!/usr/bin/env python3
"""
Test script to verify Docker permission fixes for CD-FVD model files.
Run this inside the Docker container to validate the radical fix.
"""
import os
import sys
import subprocess
from pathlib import Path

def test_permission_fix():
    """Comprehensive test of permission fixes."""
    print("=" * 60)
    print("PERMISSION FIX VALIDATION TEST")
    print("=" * 60)
    
    # Test 1: Check environment variables
    print("\n[TEST 1] Environment Variables:")
    env_vars = {
        "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
        "HF_HOME": os.getenv("HF_HOME"),
        "TORCH_HOME": os.getenv("TORCH_HOME"),
        "CDFVD_MODEL_DIR": os.getenv("CDFVD_MODEL_DIR")
    }
    for var, value in env_vars.items():
        status = "✅" if value else "❌"
        print(f"  {status} {var}: {value}")
    
    # Test 2: Check directory ownership
    print("\n[TEST 2] Directory Ownership (should be 1000:1000):")
    dirs_to_check = [
        "/app/models",
        "/app/models/cdfvd",
        "/app/models/cdfvd/third_party",
        "/app/uploads",
        "/app/.cache",
        "/app/.cache/huggingface",
        "/app/.cache/torch"
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            stat = os.stat(dir_path)
            uid, gid = stat.st_uid, stat.st_gid
            status = "✅" if (uid == 1000 and gid == 1000) else "❌"
            print(f"  {status} {dir_path}: uid={uid}, gid={gid}")
        else:
            print(f"  ❌ {dir_path}: DOES NOT EXIST")
    
    # Test 3: Check symlink
    print("\n[TEST 3] Symlink Verification:")
    symlink_path = "/usr/local/lib/python3.10/dist-packages/cdfvd/third_party"
    if os.path.islink(symlink_path):
        target = os.readlink(symlink_path)
        expected = "/app/models/cdfvd/third_party"
        status = "✅" if target == expected else "❌"
        print(f"  {status} Symlink exists: {symlink_path}")
        print(f"      Points to: {target}")
        print(f"      Expected: {expected}")
    else:
        print(f"  ❌ NOT A SYMLINK: {symlink_path}")
        if os.path.exists(symlink_path):
            print(f"      It's a: {'directory' if os.path.isdir(symlink_path) else 'file'}")
    
    # Test 4: Check model files
    print("\n[TEST 4] Model Files Accessibility:")
    model_files = [
        "/app/models/cdfvd/third_party/VideoMAEv2/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
        "/app/models/cdfvd/third_party/i3d/i3d_pretrained_400.pt"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            stat = os.stat(model_file)
            uid, gid = stat.st_uid, stat.st_gid
            mode = oct(stat.st_mode)[-3:]
            
            # Check if current user can read
            try:
                with open(model_file, 'rb') as f:
                    f.read(1)  # Try to read 1 byte
                can_read = True
            except PermissionError:
                can_read = False
            
            status = "✅" if (uid == 1000 and can_read) else "❌"
            print(f"  {status} {os.path.basename(model_file)}")
            print(f"      Full path: {model_file}")
            print(f"      Owner: uid={uid}, gid={gid}, mode={mode}")
            print(f"      Readable: {'YES' if can_read else 'NO'}")
        else:
            print(f"  ❌ NOT FOUND: {model_file}")
    
    # Test 5: Write test
    print("\n[TEST 5] Write Permission Tests:")
    write_tests = [
        ("/app/uploads/test_write.txt", "uploads"),
        ("/app/.cache/huggingface/test_write.txt", "huggingface cache"),
        ("/app/.cache/torch/test_write.txt", "torch cache")
    ]
    
    for test_path, name in write_tests:
        try:
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            with open(test_path, 'w') as f:
                f.write("test")
            os.remove(test_path)
            print(f"  ✅ Can write to {name}: {test_path}")
        except Exception as e:
            print(f"  ❌ Cannot write to {name}: {e}")
    
    # Test 6: CD-FVD import test
    print("\n[TEST 6] CD-FVD Import Test:")
    try:
        from cdfvd import fvd as cdfvd
        print("  ✅ CD-FVD imported successfully")
        
        # Check if monkey-patch is applied
        if hasattr(cdfvd, 'cdfvd'):
            print("  ✅ CD-FVD class found")
            
            # Try to check if our paths are being used
            model_dir = os.getenv("CDFVD_MODEL_DIR", "/app/models/cdfvd/third_party")
            if os.path.exists(model_dir):
                print(f"  ✅ Model directory exists: {model_dir}")
            else:
                print(f"  ❌ Model directory missing: {model_dir}")
        else:
            print("  ⚠️  CD-FVD class not found (might be ok)")
            
    except Exception as e:
        print(f"  ❌ CD-FVD import failed: {e}")
    
    # Test 7: Current user info
    print("\n[TEST 7] Current User Info:")
    uid = os.getuid()
    gid = os.getgid()
    print(f"  Running as: uid={uid}, gid={gid}")
    status = "✅" if (uid == 1000 and gid == 1000) else "⚠️"
    print(f"  {status} Expected: uid=1000, gid=1000")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_permission_fix()
