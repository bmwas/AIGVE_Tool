#!/usr/bin/env python3
"""
Test script to verify CD-FVD upload mode functionality
Creates test videos and uploads them via the API
"""

import os
import sys
import json
import requests
import tempfile
import argparse
from pathlib import Path

# Add scripts directory to path for client import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

def create_test_videos(output_dir):
    """Create minimal test videos for testing"""
    try:
        import cv2
    except ImportError:
        print("Warning: opencv-python not installed, using dummy files")
        # Create dummy video files for testing
        real_path = os.path.join(output_dir, "test_video.mp4")
        fake_path = os.path.join(output_dir, "test_video_synthetic.mp4")
        
        # Create dummy files
        with open(real_path, 'wb') as f:
            f.write(b'dummy video content')
        with open(fake_path, 'wb') as f:
            f.write(b'dummy synthetic video content')
        
        return real_path, fake_path
    
    # Create actual test videos if cv2 is available
    import numpy as np
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25
    frame_count = 32
    resolution = (128, 128)
    
    # Real video path
    real_path = os.path.join(output_dir, "test_video.mp4")
    out_real = cv2.VideoWriter(real_path, fourcc, fps, resolution)
    
    # Generate frames for real video (random but stable pattern)
    np.random.seed(42)
    for i in range(frame_count):
        frame = np.random.randint(0, 255, (*resolution[::-1], 3), dtype=np.uint8)
        out_real.write(frame)
    out_real.release()
    
    # Fake/synthetic video path
    fake_path = os.path.join(output_dir, "test_video_synthetic.mp4")
    out_fake = cv2.VideoWriter(fake_path, fourcc, fps, resolution)
    
    # Generate frames for fake video (different pattern)
    np.random.seed(123)
    for i in range(frame_count):
        frame = np.random.randint(0, 255, (*resolution[::-1], 3), dtype=np.uint8)
        out_fake.write(frame)
    out_fake.release()
    
    return real_path, fake_path

def test_upload_with_cdfvd(base_url, real_video, fake_video):
    """Test uploading videos and computing CD-FVD"""
    
    # Check server health first
    health_url = f"{base_url}/healthz"
    try:
        response = requests.get(health_url)
        if response.status_code == 200:
            health = response.json()
            print("Server health check:")
            print(f"  - Python: {health.get('python', 'unknown')}")
            print(f"  - CUDA available: {health.get('cuda_available', False)}")
            print(f"  - Torch version: {health.get('torch', 'unknown')}")
        else:
            print(f"Warning: Health check failed with status {response.status_code}")
    except Exception as e:
        print(f"Warning: Could not connect to server: {e}")
        return False
    
    # Prepare the upload
    upload_url = f"{base_url}/run_upload"
    
    files = [
        ('videos', ('test_video.mp4', open(real_video, 'rb'), 'video/mp4')),
        ('videos', ('test_video_synthetic.mp4', open(fake_video, 'rb'), 'video/mp4'))
    ]
    
    data = {
        'compute': 'true',
        'generated_suffixes': 'synthetic',
        'categories': 'distribution_based',
        'max_seconds': '8',
        'fps': '25',
        'cdfvd_resolution': '128',
        'cdfvd_sequence_length': '16'
    }
    
    print("\nTesting merged AIGVE + CD-FVD computation with following parameters:")
    print(f"  - CD-FVD Models: VideMAE + I3D (automatic)")
    print(f"  - Resolution: 128")
    print(f"  - Sequence length: 16")
    print(f"  - Max seconds: 8")
    print(f"  - Real video: {os.path.basename(real_video)}")
    print(f"  - Fake video: {os.path.basename(fake_video)}")
    
    try:
        response = requests.post(upload_url, files=files, data=data)
        
        # Close file handles
        for _, file_info in files:
            file_info[1].close()
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ Upload successful!")
            print(f"Return code: {result.get('returncode', 'N/A')}")
            
            # Check for CD-FVD results (new format: multiple models)
            if 'cdfvd_results' in result:
                cdfvd_results = result['cdfvd_results']
                print("\n📊 CD-FVD Results (Merged Ecosystem):")
                for model, model_result in cdfvd_results.items():
                    if 'error' in model_result:
                        print(f"  - {model.upper()}: ERROR - {model_result['error']}")
                    else:
                        score = model_result.get('fvd_score', 'N/A')
                        print(f"  - {model.upper()}: FVD Score = {score}")
                        print(f"    Real videos: {model_result.get('n_real', 0)}")
                        print(f"    Fake videos: {model_result.get('n_fake', 0)}")
            elif 'cdfvd_error' in result:
                print(f"\n⚠️ CD-FVD Error: {result['cdfvd_error']}")
            else:
                print("\n⚠️ No CD-FVD results in response")
            
            # Check for AIGVE native results
            native_results = []
            for artifact in result.get('artifacts', []):
                name = artifact.get('name', '')
                if name in ['fid_results.json', 'is_results.json', 'fvd_results.json']:
                    native_results.append(name)
            
            if native_results:
                print(f"\n📊 AIGVE Native Results: {', '.join(native_results)}")
            else:
                print("\n⚠️ No AIGVE native results found")
            
            # Check artifacts
            if 'artifacts' in result:
                print("\n📦 Artifacts returned:")
                for artifact in result['artifacts']:
                    name = artifact.get('name', 'unknown')
                    print(f"  - {name}")
                    if name == 'cdfvd_results.json' and 'json' in artifact:
                        print(f"    Content: {json.dumps(artifact['json'], indent=4)}")
            
            # Check session info
            if 'session' in result:
                session = result['session']
                print(f"\n📁 Session info:")
                print(f"  - Session ID: {session.get('id', 'N/A')}")
                print(f"  - Uploaded files: {session.get('files', [])}")
            
            # Print any stderr for debugging
            if result.get('stderr'):
                stderr_lines = result['stderr'].strip().split('\n')
                if stderr_lines:
                    print("\n⚠️ Server stderr (last 5 lines):")
                    for line in stderr_lines[-5:]:
                        print(f"  {line}")
            
            return True
            
        else:
            print(f"\n❌ Upload failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during upload: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test CD-FVD upload functionality')
    parser.add_argument('--base_url', default='http://localhost:2200',
                        help='Base URL of the AIGVE server (default: http://localhost:2200)')
    
    args = parser.parse_args()
    base_url = args.base_url
    
    print(f"🎬 CD-FVD Upload Test")
    print(f"Server: {base_url}")
    print("-" * 50)
    
    # Use real videos from aigve/data if available
    real_videos_dir = "aigve/data/AIGVE_Bench_toy/videos"
    if os.path.exists(real_videos_dir):
        videos = [f for f in os.listdir(real_videos_dir) if f.endswith('.mp4')]
        if len(videos) >= 2:
            print(f"\n📹 Using real videos from {real_videos_dir}")
            # Use first video as "real", rename second as synthetic for testing
            real_video = os.path.join(real_videos_dir, videos[0])
            
            # Copy second video with _synthetic suffix to temp dir
            with tempfile.TemporaryDirectory() as tmpdir:
                fake_name = videos[1].replace('.mp4', '_synthetic.mp4')
                fake_video = os.path.join(tmpdir, fake_name)
                import shutil
                shutil.copy(os.path.join(real_videos_dir, videos[1]), fake_video)
                
                # Also copy the real video to maintain naming convention
                real_copy = os.path.join(tmpdir, videos[0])
                shutil.copy(real_video, real_copy)
                
                print(f"  ✓ Using real: {os.path.basename(real_copy)}")
                print(f"  ✓ Using fake: {os.path.basename(fake_video)}")
                
                # Test the upload with CD-FVD
                success = test_upload_with_cdfvd(base_url, real_copy, fake_video)
        else:
            print(f"\n⚠️ Not enough videos in {real_videos_dir}, creating dummy files...")
            with tempfile.TemporaryDirectory() as tmpdir:
                real_video, fake_video = create_test_videos(tmpdir)
                success = test_upload_with_cdfvd(base_url, real_video, fake_video)
    else:
        # Fallback to creating test videos
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\n📹 Creating test videos in {tmpdir}...")
            real_video, fake_video = create_test_videos(tmpdir)
            print(f"  ✓ Created: {os.path.basename(real_video)}")
            print(f"  ✓ Created: {os.path.basename(fake_video)}")
            
            # Test the upload with CD-FVD
            success = test_upload_with_cdfvd(base_url, real_video, fake_video)
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        print("\n🔧 Troubleshooting steps:")
        print("1. Ensure the server is running: docker ps")
        print("2. Check server logs: docker logs <container_id>")
        print("3. Verify server health: curl http://localhost:2200/healthz")
        print("4. Verify GPU access: docker exec <container_id> python -c 'import torch; print(torch.cuda.is_available())'")
        print("5. Check cd-fvd installation: docker exec <container_id> python -c 'import cdfvd; print(\"OK\")')")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
