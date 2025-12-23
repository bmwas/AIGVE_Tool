#!/usr/bin/env python3
"""
Download model checkpoints for AIGVE metrics.

This script downloads the required model checkpoints for:
- SimpleVQA (UGC_BVQA_model.pth)
- LightVQA+ (model checkpoint + swin weights)
- FVD (optional, uses default torchvision weights if not present)

Usage:
    python scripts/download_model_checkpoints.py --all
    python scripts/download_model_checkpoints.py --simplevqa
    python scripts/download_model_checkpoints.py --lightvqa
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Known checkpoint URLs (from official repos and Hugging Face)
# Note: Some models require manual download due to licensing
CHECKPOINT_INFO = {
    "simplevqa": {
        "name": "SimpleVQA (UGC_BVQA_model.pth)",
        "repo": "https://github.com/sunwei925/SimpleVQA.git",
        "submodule_path": "aigve/metrics/video_quality_assessment/nn_based/simplevqa/SimpleVQA",
        "checkpoint_path": "aigve/metrics/video_quality_assessment/nn_based/simplevqa/SimpleVQA/ckpts/UGC_BVQA_model.pth",
        "google_drive_id": "137XJdq3reNMJ9tkBNqKUYTY_dTlcwXc3",
        "manual_download_url": "https://drive.google.com/file/d/137XJdq3reNMJ9tkBNqKUYTY_dTlcwXc3/view?usp=sharing",
        "instructions": """
SimpleVQA model checkpoint not found!

To download MANUALLY:
1. Visit: https://drive.google.com/file/d/137XJdq3reNMJ9tkBNqKUYTY_dTlcwXc3/view?usp=sharing
2. Download UGC_BVQA_model.pth
3. Create directory: mkdir -p {ckpt_dir}
4. Place file in: {checkpoint_path}

Or use gdown (pip install gdown):
   gdown 137XJdq3reNMJ9tkBNqKUYTY_dTlcwXc3 -O {checkpoint_path}
"""
    },
    "lightvqa": {
        "name": "LightVQA+ (checkpoint + swin weights)",
        "repo": "https://github.com/SaMMyCHoo/Light-VQA-plus.git",
        "submodule_path": "aigve/metrics/video_quality_assessment/nn_based/lightvqa_plus/Light_VQA_plus",
        "checkpoint_path": "aigve/metrics/video_quality_assessment/nn_based/lightvqa_plus/Light_VQA_plus/ckpts/last2_SI+TI_epoch_19_SRCC_0.925264.pth",
        "swin_path": "aigve/metrics/video_quality_assessment/nn_based/lightvqa_plus/Light_VQA_plus/swin_small_patch4_window7_224.pth",
        "swin_url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
        "manual_download_url": "https://jbox.sjtu.edu.cn/l/S1bbm1",
        "baidu_url": "https://pan.baidu.com/s/1JZMsibiVDDSQVdrRob1clw",
        "baidu_password": "ui9v",
        "instructions": """
LightVQA+ model checkpoint not found!

To download MANUALLY:
1. Option A - JBOX (Shanghai Jiao Tong University):
   Visit: https://jbox.sjtu.edu.cn/l/S1bbm1
   
2. Option B - Baidu Netdisk:
   Visit: https://pan.baidu.com/s/1JZMsibiVDDSQVdrRob1clw
   Password: ui9v

3. Download the checkpoint file (last2_SI+TI_epoch_19_SRCC_0.925264.pth)
4. Create directory: mkdir -p {ckpt_dir}
5. Place file in: {checkpoint_path}

Swin weights (auto-downloaded):
- URL: https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
- Path: {swin_path}
"""
    },
    "gstvqa": {
        "name": "GSTVQA",
        "repo": "https://github.com/Baoliang93/GSTVQA.git",
        "submodule_path": "aigve/metrics/video_quality_assessment/nn_based/gstvqa/GSTVQA",
        "checkpoint_paths": [
            "aigve/metrics/video_quality_assessment/nn_based/gstvqa/GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/models/training-all-data-GSTVQA-konvid-EXP0-best"
        ],
        "instructions": """
GSTVQA model checkpoints are bundled with the repo.
If missing, the submodule will be cloned automatically.
"""
    }
}


def get_project_root() -> Path:
    """Get the project root directory."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def clone_submodule(repo_url: str, target_path: Path) -> bool:
    """Clone a git repository as a submodule."""
    if target_path.exists() and any(target_path.iterdir()):
        print(f"  ✅ Submodule already exists: {target_path}")
        return True
    
    print(f"  📥 Cloning {repo_url} to {target_path}...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"  ✅ Cloned successfully")
            return True
        else:
            print(f"  ❌ Clone failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Clone error: {e}")
        return False


def download_file(url: str, target_path: Path) -> bool:
    """Download a file from URL."""
    import urllib.request
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  📥 Downloading {url}...")
    
    try:
        urllib.request.urlretrieve(url, str(target_path))
        print(f"  ✅ Downloaded to {target_path}")
        return True
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return False


def download_from_google_drive(file_id: str, target_path: Path) -> bool:
    """Try to download from Google Drive using gdown."""
    try:
        import gdown
        target_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  📥 Downloading from Google Drive (file_id: {file_id})...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(target_path), quiet=False)
        if target_path.exists():
            print(f"  ✅ Downloaded successfully: {target_path}")
            return True
        else:
            print(f"  ❌ Download failed - file not created")
            return False
    except ImportError:
        print(f"  ⚠️  gdown not installed. Install with: pip install gdown")
        return False
    except Exception as e:
        print(f"  ❌ Download error: {e}")
        return False


def setup_simplevqa(root: Path) -> bool:
    """Setup SimpleVQA model."""
    info = CHECKPOINT_INFO["simplevqa"]
    print(f"\n{'='*60}")
    print(f"Setting up {info['name']}")
    print(f"{'='*60}")
    
    submodule_path = root / info["submodule_path"]
    checkpoint_path = root / info["checkpoint_path"]
    
    # Clone submodule if needed
    if not clone_submodule(info["repo"], submodule_path):
        print(info["instructions"].format(
            checkpoint_path=checkpoint_path,
            ckpt_dir=checkpoint_path.parent
        ))
        return False
    
    # Check if checkpoint exists
    if checkpoint_path.exists():
        print(f"  ✅ Checkpoint found: {checkpoint_path}")
        return True
    
    # Try to auto-download from Google Drive
    print(f"  📥 Attempting auto-download from Google Drive...")
    if download_from_google_drive(info["google_drive_id"], checkpoint_path):
        return True
    
    # Manual download required
    print(f"  ⚠️  Checkpoint NOT found: {checkpoint_path}")
    print(info["instructions"].format(
        checkpoint_path=checkpoint_path,
        ckpt_dir=checkpoint_path.parent
    ))
    return False


def setup_lightvqa(root: Path) -> bool:
    """Setup LightVQA+ model."""
    info = CHECKPOINT_INFO["lightvqa"]
    print(f"\n{'='*60}")
    print(f"Setting up {info['name']}")
    print(f"{'='*60}")
    
    submodule_path = root / info["submodule_path"]
    checkpoint_path = root / info["checkpoint_path"]
    swin_path = root / info["swin_path"]
    
    # Clone submodule if needed
    if not clone_submodule(info["repo"], submodule_path):
        print(info["instructions"].format(
            checkpoint_path=checkpoint_path,
            swin_path=swin_path,
            ckpt_dir=checkpoint_path.parent
        ))
        return False
    
    success = True
    
    # Download swin weights if not present
    if not swin_path.exists():
        if not download_file(info["swin_url"], swin_path):
            success = False
    else:
        print(f"  ✅ Swin weights found: {swin_path}")
    
    # Check if main checkpoint exists
    if checkpoint_path.exists():
        print(f"  ✅ Checkpoint found: {checkpoint_path}")
    else:
        (checkpoint_path.parent).mkdir(parents=True, exist_ok=True)
        print(f"  ⚠️  Checkpoint NOT found: {checkpoint_path}")
        print(f"\n  LightVQA+ requires MANUAL download (Chinese cloud storage):")
        print(f"  ─────────────────────────────────────────────────────────")
        print(f"  Option 1 - JBOX: {info['manual_download_url']}")
        print(f"  Option 2 - Baidu: {info['baidu_url']} (password: {info['baidu_password']})")
        print(f"  ─────────────────────────────────────────────────────────")
        print(f"  After download, place in: {checkpoint_path}")
        success = False
    
    return success


def setup_gstvqa(root: Path) -> bool:
    """Setup GSTVQA model."""
    info = CHECKPOINT_INFO["gstvqa"]
    print(f"\n{'='*60}")
    print(f"Setting up {info['name']}")
    print(f"{'='*60}")
    
    submodule_path = root / info["submodule_path"]
    
    # Clone submodule if needed
    if not clone_submodule(info["repo"], submodule_path):
        print(info["instructions"])
        return False
    
    # GSTVQA checkpoints should be bundled
    print(f"  ✅ GSTVQA setup complete (checkpoints bundled with repo)")
    return True


def check_all_models(root: Path):
    """Check status of all models."""
    print("\n" + "="*60)
    print("MODEL CHECKPOINT STATUS")
    print("="*60)
    
    models = [
        ("GSTVQA", root / CHECKPOINT_INFO["gstvqa"]["submodule_path"]),
        ("SimpleVQA", root / CHECKPOINT_INFO["simplevqa"]["checkpoint_path"]),
        ("LightVQA+ Model", root / CHECKPOINT_INFO["lightvqa"]["checkpoint_path"]),
        ("LightVQA+ Swin", root / CHECKPOINT_INFO["lightvqa"]["swin_path"]),
    ]
    
    for name, path in models:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024) if path.is_file() else 0
            if path.is_file():
                print(f"  ✅ {name}: {path.name} ({size_mb:.1f} MB)")
            else:
                print(f"  ✅ {name}: {path} (directory)")
        else:
            print(f"  ❌ {name}: NOT FOUND")
            print(f"      Expected: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download model checkpoints for AIGVE metrics"
    )
    parser.add_argument("--all", action="store_true", help="Setup all models")
    parser.add_argument("--simplevqa", action="store_true", help="Setup SimpleVQA")
    parser.add_argument("--lightvqa", action="store_true", help="Setup LightVQA+")
    parser.add_argument("--gstvqa", action="store_true", help="Setup GSTVQA")
    parser.add_argument("--check", action="store_true", help="Check model status only")
    
    args = parser.parse_args()
    
    root = get_project_root()
    print(f"Project root: {root}")
    
    if args.check:
        check_all_models(root)
        return 0
    
    if not any([args.all, args.simplevqa, args.lightvqa, args.gstvqa]):
        # Default: check status and show help
        check_all_models(root)
        print("\nTo setup models, run with --all or specific model flags:")
        print("  python scripts/download_model_checkpoints.py --all")
        print("  python scripts/download_model_checkpoints.py --simplevqa")
        print("  python scripts/download_model_checkpoints.py --lightvqa")
        return 0
    
    results = {}
    
    if args.all or args.gstvqa:
        results["GSTVQA"] = setup_gstvqa(root)
    
    if args.all or args.simplevqa:
        results["SimpleVQA"] = setup_simplevqa(root)
    
    if args.all or args.lightvqa:
        results["LightVQA+"] = setup_lightvqa(root)
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "✅ Ready" if success else "⚠️  Manual download required"
        print(f"  {name}: {status}")
    
    check_all_models(root)
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

