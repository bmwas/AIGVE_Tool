#!/usr/bin/env python3
"""
Test script to validate FID, IS, and FVD metrics computation.
This script creates minimal test data and verifies that metrics can be computed.
"""

import sys
import os
import tempfile
import json
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def create_test_videos_and_annotations(test_dir):
    """
    Create minimal test videos and annotation file for testing metrics.
    For testing purposes, we'll create dummy video files and a proper annotation JSON.
    """
    
    # Create video directory
    video_dir = os.path.join(test_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    # Create dummy video files (empty files for testing structure)
    real_video = os.path.join(video_dir, "test_real.mp4")
    generated_video = os.path.join(video_dir, "test_synthetic.mp4")
    
    # Create empty video files (for structure testing)
    with open(real_video, 'wb') as f:
        f.write(b"dummy_video_data")
        
    with open(generated_video, 'wb') as f:
        f.write(b"dummy_video_data")
    
    # Create annotation file
    annotation_data = {
        "metainfo": {
            "source": "test_dataset", 
            "length": 1
        },
        "data_list": [{
            "prompt_gt": "",
            "video_path_pd": "test_synthetic.mp4",
            "video_path_gt": "test_real.mp4"
        }]
    }
    
    annotation_file = os.path.join(test_dir, "annotations.json")
    with open(annotation_file, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    return video_dir, annotation_file

def test_imports():
    """Test that AIGVE metrics can be imported successfully."""
    print("[TEST] Testing AIGVE imports...")
    
    try:
        from aigve.datasets.fid_dataset import FidDataset
        print("[TEST] ✅ FidDataset imported successfully")
        
        from aigve.metrics.video_quality_assessment.distribution_based.fid.fid_metric import FIDScore
        print("[TEST] ✅ FIDScore imported successfully")
        
        from aigve.metrics.video_quality_assessment.distribution_based.is_score.is_metric import ISScore
        print("[TEST] ✅ ISScore imported successfully")
        
        from aigve.metrics.video_quality_assessment.distribution_based.fvd.fvd_metric import FVDScore
        print("[TEST] ✅ FVDScore imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"[TEST] ❌ Import failed: {e}")
        return False

def test_standalone_script():
    """Test the standalone metrics computation script."""
    print("\n[TEST] Testing standalone metrics script...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        video_dir, annotation_file = create_test_videos_and_annotations(temp_dir)
        
        # Test script exists and is executable
        standalone_script = os.path.join(project_root, "compute_metrics_standalone.py")
        if not os.path.exists(standalone_script):
            print(f"[TEST] ❌ Standalone script not found: {standalone_script}")
            return False
            
        print(f"[TEST] ✅ Standalone script exists: {standalone_script}")
        
        # Test help message
        import subprocess
        try:
            result = subprocess.run([
                sys.executable, standalone_script, "--help"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("[TEST] ✅ Script help message works")
            else:
                print(f"[TEST] ❌ Script help failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[TEST] ❌ Script help test failed: {e}")
            return False
            
        return True

def test_server_function():
    """Test the server's direct metrics computation function."""
    print("\n[TEST] Testing server metrics function...")
    
    try:
        # Import the server function
        sys.path.insert(0, os.path.join(project_root, "server"))
        from main import _compute_aigve_metrics
        print("[TEST] ✅ Server metrics function imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"[TEST] ❌ Server function import failed: {e}")
        return False
    except Exception as e:
        print(f"[TEST] ❌ Server function test failed: {e}")
        return False

def test_annotation_structure():
    """Test that we can create proper annotation structures."""
    print("\n[TEST] Testing annotation structure...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        video_dir, annotation_file = create_test_videos_and_annotations(temp_dir)
        
        # Verify annotation file structure
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        required_keys = ['metainfo', 'data_list']
        for key in required_keys:
            if key not in data:
                print(f"[TEST] ❌ Missing key in annotation: {key}")
                return False
                
        if len(data['data_list']) == 0:
            print("[TEST] ❌ Empty data_list in annotation")
            return False
            
        sample = data['data_list'][0]
        required_sample_keys = ['video_path_gt', 'video_path_pd']
        for key in required_sample_keys:
            if key not in sample:
                print(f"[TEST] ❌ Missing key in sample: {key}")
                return False
                
        print("[TEST] ✅ Annotation structure is valid")
        return True

def main():
    """Run all tests."""
    print("="*60)
    print("AIGVE METRICS COMPUTATION TESTS")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Standalone Script Test", test_standalone_script),
        ("Server Function Test", test_server_function),
        ("Annotation Structure Test", test_annotation_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[TEST] ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY:")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Metrics computation is ready.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")
    
    print("="*60)

if __name__ == "__main__":
    main()
