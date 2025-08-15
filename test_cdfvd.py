#!/usr/bin/env python3
"""
Test script for CD-FVD integration.
This script tests the cd-fvd functionality added to the AIGVE API.
"""

import json
import os
import tempfile
import shutil
import numpy as np
import cv2
from pathlib import Path


def create_test_video(filename, num_frames=16, width=128, height=128, is_synthetic=False):
    """Create a simple test video file."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 25.0, (width, height))
    
    for i in range(num_frames):
        # Create different patterns for real vs synthetic videos
        if is_synthetic:
            # Synthetic: uniform color changing over time
            frame = np.ones((height, width, 3), dtype=np.uint8) * (i * 255 // num_frames)
        else:
            # Real: random noise pattern
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    return filename


def test_cdfvd_locally():
    """Test cd-fvd functionality directly without API."""
    print("Testing cd-fvd package locally...")
    
    # Create temporary directory for test videos
    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = Path(tmpdir) / "real"
        fake_dir = Path(tmpdir) / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()
        
        # Create test videos
        print("Creating test videos...")
        for i in range(3):
            create_test_video(str(real_dir / f"real_{i}.mp4"), is_synthetic=False)
            create_test_video(str(fake_dir / f"fake_{i}.mp4"), is_synthetic=True)
        
        # Test cd-fvd import and basic functionality
        try:
            from cdfvd import fvd
            print("✓ cd-fvd package imported successfully")
            
            # Create evaluator
            evaluator = fvd.cdfvd()
            print(f"✓ CD-FVD evaluator created (device: {evaluator.device})")
            
            # Compute FVD
            print("Computing FVD score...")
            fvd_score = evaluator.compute_fvd(str(real_dir), str(fake_dir))
            print(f"✓ FVD Score computed: {fvd_score:.2f}")
            
            return True
            
        except ImportError as e:
            print(f"✗ Failed to import cd-fvd: {e}")
            print("  Please install: pip install cd-fvd")
            return False
        except Exception as e:
            print(f"✗ Error during cd-fvd test: {e}")
            return False


def test_api_integration():
    """Test cd-fvd integration through the API (requires server to be running)."""
    print("\nTesting cd-fvd API integration...")
    print("Note: This requires the API server to be running at http://localhost:2200")
    
    try:
        import requests
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:2200/healthz", timeout=2)
            if response.status_code != 200:
                print("✗ API server not accessible")
                return False
            print("✓ API server is running")
        except requests.exceptions.RequestException:
            print("✗ API server is not running. Start it with:")
            print("  python -m uvicorn server.main:app --port 2200")
            return False
        
        # Create test videos
        with tempfile.TemporaryDirectory() as tmpdir:
            print("Creating test videos for API upload...")
            video_files = []
            for i in range(2):
                # Real video
                real_path = Path(tmpdir) / f"test_real_{i}.mp4"
                create_test_video(str(real_path), is_synthetic=False)
                video_files.append(str(real_path))
                
                # Synthetic video
                fake_path = Path(tmpdir) / f"test_real_{i}_synthetic.mp4"
                create_test_video(str(fake_path), is_synthetic=True)
                video_files.append(str(fake_path))
            
            # Upload and compute with cd-fvd
            print("Uploading videos and computing FVD with cd-fvd...")
            
            files = []
            for vf in video_files:
                files.append(('videos', (os.path.basename(vf), open(vf, 'rb'), 'video/mp4')))
            
            form_data = {
                'compute': 'true',
                'use_cdfvd': 'true',
                'cdfvd_model': 'videomae',
                'cdfvd_resolution': '128',
                'cdfvd_sequence_length': '16',
                'max_seconds': '2',
                'generated_suffixes': 'synthetic'
            }
            
            response = requests.post(
                "http://localhost:2200/run_upload",
                files=files,
                data=form_data,
                timeout=60
            )
            
            # Close file handles
            for _, file_tuple in files:
                file_tuple[1].close()
            
            if response.status_code == 200:
                result = response.json()
                if 'cdfvd_result' in result:
                    cdfvd_res = result['cdfvd_result']
                    print(f"✓ CD-FVD API test successful!")
                    print(f"  FVD Score: {cdfvd_res.get('fvd_score', 'N/A')}")
                    print(f"  Model: {cdfvd_res.get('model', 'N/A')}")
                    print(f"  Real Videos: {cdfvd_res.get('num_real_videos', 'N/A')}")
                    print(f"  Fake Videos: {cdfvd_res.get('num_fake_videos', 'N/A')}")
                    return True
                elif 'cdfvd_error' in result:
                    print(f"✗ CD-FVD computation failed: {result['cdfvd_error']}")
                    return False
                else:
                    print("✗ No CD-FVD result in response")
                    return False
            else:
                print(f"✗ API request failed with status {response.status_code}")
                return False
                
    except Exception as e:
        print(f"✗ API integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("CD-FVD Integration Test")
    print("=" * 60)
    
    # Test 1: Local cd-fvd functionality
    local_success = test_cdfvd_locally()
    
    # Test 2: API integration (optional, requires running server)
    api_success = test_api_integration()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Local cd-fvd test: {'✓ PASSED' if local_success else '✗ FAILED'}")
    print(f"  API integration test: {'✓ PASSED' if api_success else '✗ FAILED (or server not running)'}")
    print("=" * 60)
