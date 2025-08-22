#!/usr/bin/env python3
"""
Diagnose actual permission issues in the Docker container.
Run this inside the container to see what's really failing.
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return stdout, stderr, returncode."""
    print(f"\n=== {description} ===")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        print("Command timed out!")
        return "", "timeout", -1
    except Exception as e:
        print(f"Command failed: {e}")
        return "", str(e), -1

def main():
    print("DIAGNOSING PERMISSION ISSUES")
    print("=" * 60)
    
    # Basic info
    print(f"Running as UID: {os.getuid()}, GID: {os.getgid()}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if we're in container
    if os.path.exists("/.dockerenv"):
        print("✅ Running inside Docker container")
    else:
        print("❌ NOT running in container")
    
    # Check critical directories and files
    critical_paths = [
        "/app/models/cdfvd/third_party",
        "/app/.cache/huggingface", 
        "/app/.cache/torch",
        "/usr/local/lib/python3.10/dist-packages/cdfvd/third_party",
        "/usr/local/lib/python3.10/dist-packages/cdfvd/third_party/VideoMAEv2/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
        "/usr/local/lib/python3.10/dist-packages/cdfvd/third_party/i3d/i3d_pretrained_400.pt"
    ]
    
    print(f"\n=== CHECKING CRITICAL PATHS ===")
    for path in critical_paths:
        if os.path.exists(path):
            stat = os.stat(path)
            is_link = os.path.islink(path)
            link_target = os.readlink(path) if is_link else "N/A"
            print(f"✅ {path}")
            print(f"   UID: {stat.st_uid}, GID: {stat.st_gid}, Mode: {oct(stat.st_mode)}")
            if is_link:
                print(f"   → SYMLINK to: {link_target}")
        else:
            print(f"❌ MISSING: {path}")
    
    # Test write permissions
    print(f"\n=== WRITE PERMISSION TESTS ===")
    test_dirs = [
        "/app/.cache/huggingface",
        "/app/.cache/torch", 
        "/app/uploads"
    ]
    
    for test_dir in test_dirs:
        try:
            test_file = os.path.join(test_dir, "write_test.tmp")
            os.makedirs(test_dir, exist_ok=True)
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✅ Can write to: {test_dir}")
        except Exception as e:
            print(f"❌ Cannot write to {test_dir}: {e}")
    
    # Test CD-FVD model file access
    print(f"\n=== CD-FVD MODEL FILE ACCESS ===")
    model_files = [
        "/usr/local/lib/python3.10/dist-packages/cdfvd/third_party/VideoMAEv2/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
        "/usr/local/lib/python3.10/dist-packages/cdfvd/third_party/i3d/i3d_pretrained_400.pt"
    ]
    
    for model_file in model_files:
        try:
            with open(model_file, 'rb') as f:
                data = f.read(1024)  # Read first 1KB
            print(f"✅ Can read: {model_file} ({len(data)} bytes)")
        except Exception as e:
            print(f"❌ Cannot read {model_file}: {e}")
    
    # Test Huggingface cache access
    print(f"\n=== HUGGINGFACE CACHE TEST ===")
    run_command(
        "python3 -c \"import transformers; print('Transformers cache:', transformers.utils.hub.TRANSFORMERS_CACHE)\"",
        "Check transformers cache location"
    )
    
    # Try to actually trigger the error by running prepare_annotations
    print(f"\n=== REPRODUCE THE ACTUAL ERROR ===")
    
    # Create a minimal test case
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy video files
        test_video = os.path.join(tmpdir, "test.mp4")
        test_synthetic = os.path.join(tmpdir, "test_synthetic.mp4")
        
        # Create minimal video files (just empty for permission test)
        with open(test_video, 'wb') as f:
            f.write(b'\x00' * 1024)
        with open(test_synthetic, 'wb') as f:
            f.write(b'\x00' * 1024)
            
        print(f"Created test files in: {tmpdir}")
        
        # Run the actual command that's failing
        cmd = f"cd /app && python3 /app/scripts/prepare_annotations.py --input-dir {tmpdir} --generated-suffixes synthetic --compute --metrics fid --categories distribution_based"
        
        stdout, stderr, rc = run_command(cmd, "Run prepare_annotations.py with test data")
        
        if rc != 0:
            print(f"\n🔥 FOUND THE ERROR! Return code: {rc}")
            print("This is likely the root cause of your issue.")

if __name__ == "__main__":
    main()
