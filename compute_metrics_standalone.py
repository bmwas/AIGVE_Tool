#!/usr/bin/env python3
"""
Standalone script to compute FID, IS, and FVD metrics from AIGVE.
This script MUST compute all three metrics and print their values to console.
NO try-except blocks are used to ensure any errors are visible.
"""

import sys
import os
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
        
    try:
        import mmengine
    except ImportError:
        missing.append("mmengine")
        
    try:
        from aigve.datasets.fid_dataset import FidDataset
    except ImportError:
        missing.append("aigve.datasets")
        
    if missing:
        print("❌ MISSING DEPENDENCIES:")
        for dep in missing:
            print(f"   - {dep}")
        print("\n💡 SOLUTION: Run this script in Docker environment:")
        print("   docker run --gpus all -p 2200:2200 your-image python compute_metrics_standalone.py --help")
        return False
    return True

# Only import AIGVE components if dependencies are available
if check_dependencies():
    import torch
    from aigve.datasets.fid_dataset import FidDataset
    from aigve.metrics.video_quality_assessment.distribution_based.fid.fid_metric import FIDScore
    from aigve.metrics.video_quality_assessment.distribution_based.is_score.is_metric import ISScore
    from aigve.metrics.video_quality_assessment.distribution_based.fvd.fvd_metric import FVDScore


def compute_all_metrics(video_dir: str, annotation_file: str, max_len: int = 64, 
                       use_cpu: bool = False, fvd_model: str = None):
    """
    Compute FID, IS, and FVD metrics.
    
    Args:
        video_dir: Directory containing videos
        annotation_file: JSON annotation file path
        max_len: Maximum frames per video
        use_cpu: Force CPU usage
        fvd_model: Path to FVD model checkpoint
    
    Returns:
        Dictionary containing all metric results
    """
    
    # Determine device
    device_available = torch.cuda.is_available()
    use_gpu = not use_cpu and device_available
    device_str = 'cuda' if use_gpu else 'cpu'
    
    print(f"[METRICS] Device: {device_str} (CUDA available: {device_available})")
    print(f"[METRICS] Video directory: {video_dir}")
    print(f"[METRICS] Annotation file: {annotation_file}")
    print(f"[METRICS] Max frames: {max_len}")
    
    # Build dataset
    print(f"[METRICS] Building FidDataset...")
    dataset = FidDataset(
        video_dir=str(video_dir),
        prompt_dir=str(annotation_file),
        max_len=max_len,
        if_pad=False
    )
    
    print(f"[METRICS] Dataset size: {len(dataset)} samples")
    
    # Results storage
    results = {}
    
    # Helper to iterate dataset samples
    def iter_samples():
        for idx in range(len(dataset)):
            gt_tensor, gen_tensor, gt_name, gen_name = dataset[idx]
            yield gt_tensor, gen_tensor, gt_name, gen_name
    
    # ===============================
    # COMPUTE FID - NO TRY-EXCEPT
    # ===============================
    print(f"\n[METRICS] Computing FID...")
    fid_metric = FIDScore(is_gpu=use_gpu)
    
    sample_count = 0
    for gt_tensor, gen_tensor, gt_name, gen_name in iter_samples():
        data_samples = ((gt_tensor,), (gen_tensor,), (gt_name,), (gen_name,))
        fid_metric.process(data_batch={}, data_samples=data_samples)
        sample_count += 1
    
    print(f"[METRICS] Processed {sample_count} samples for FID")
    fid_summary = fid_metric.compute_metrics([])
    
    # Extract FID value and store
    fid_score = fid_summary.get('FID', fid_summary.get('fid_score', 'UNKNOWN'))
    results['fid'] = {
        'score': fid_score,
        'summary': fid_summary
    }
    
    print(f"[METRICS RESULT] ✅ FID COMPUTED: {fid_score}")
    
    # ===============================
    # COMPUTE IS - NO TRY-EXCEPT  
    # ===============================
    print(f"\n[METRICS] Computing IS...")
    is_metric = ISScore(is_gpu=use_gpu)
    
    sample_count = 0
    for _, gen_tensor, _, gen_name in iter_samples():
        data_samples = ((), (gen_tensor,), (), (gen_name,))
        is_metric.process(data_batch={}, data_samples=data_samples)
        sample_count += 1
        
    print(f"[METRICS] Processed {sample_count} samples for IS")
    is_summary = is_metric.compute_metrics([])
    
    # Extract IS value and store
    is_score = is_summary.get('IS', is_summary.get('is_score', 'UNKNOWN'))
    results['is'] = {
        'score': is_score,
        'summary': is_summary
    }
    
    print(f"[METRICS RESULT] ✅ IS COMPUTED: {is_score}")
    
    # ===============================
    # COMPUTE FVD - NO TRY-EXCEPT
    # ===============================
    print(f"\n[METRICS] Computing FVD...")
    
    # Determine FVD model path
    if fvd_model:
        model_path = Path(fvd_model).expanduser().resolve()
    else:
        model_path = project_root / 'aigve/metrics/video_quality_assessment/distribution_based/fvd/model_rgb.pth'
    
    print(f"[METRICS] FVD model path: {model_path}")
    fvd_metric = FVDScore(model_path=str(model_path), is_gpu=use_gpu)
    
    sample_count = 0
    for gt_tensor, gen_tensor, gt_name, gen_name in iter_samples():
        data_samples = ((gt_tensor,), (gen_tensor,), (gt_name,), (gen_name,))
        fvd_metric.process(data_batch={}, data_samples=data_samples)
        sample_count += 1
        
    print(f"[METRICS] Processed {sample_count} samples for FVD")
    fvd_summary = fvd_metric.compute_metrics([])
    
    # Extract FVD value and store
    fvd_score = fvd_summary.get('FVD', fvd_summary.get('fvd_score', 'UNKNOWN'))
    results['fvd'] = {
        'score': fvd_score,
        'summary': fvd_summary
    }
    
    print(f"[METRICS RESULT] ✅ FVD COMPUTED: {fvd_score}")
    
    # ===============================
    # FINAL RESULTS
    # ===============================
    print(f"\n" + "="*60)
    print(f"FINAL METRIC RESULTS:")
    print(f"="*60)
    print(f"FID Score: {results['fid']['score']}")
    print(f"IS Score:  {results['is']['score']}")  
    print(f"FVD Score: {results['fvd']['score']}")
    print(f"="*60)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute FID, IS, and FVD metrics")
    parser.add_argument("--video-dir", help="Directory containing videos")
    parser.add_argument("--annotation-file", help="JSON annotation file")
    parser.add_argument("--max-len", type=int, default=64, help="Maximum frames per video")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--fvd-model", help="Path to FVD model checkpoint")
    parser.add_argument("--output-json", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n🐳 This script requires AIGVE dependencies that are installed in Docker.")
        print("   Use the Docker container to run metrics computation.")
        sys.exit(1)
    
    # Validate inputs if provided
    if args.video_dir and not os.path.exists(args.video_dir):
        print(f"ERROR: Video directory does not exist: {args.video_dir}")
        sys.exit(1)
        
    if args.annotation_file and not os.path.exists(args.annotation_file):
        print(f"ERROR: Annotation file does not exist: {args.annotation_file}")
        sys.exit(1)
    
    if not args.video_dir or not args.annotation_file:
        print("ERROR: Both --video-dir and --annotation-file are required")
        sys.exit(1)
    
    # Compute metrics
    print(f"Starting metric computation...")
    sys.stdout.flush()  # Force output
    
    results = compute_all_metrics(
        video_dir=args.video_dir,
        annotation_file=args.annotation_file,
        max_len=args.max_len,
        use_cpu=args.use_cpu,
        fvd_model=args.fvd_model
    )
    
    # Save results if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {args.output_json}")
    
    print(f"\nMetric computation completed successfully!")
    sys.stdout.flush()  # Force output


if __name__ == "__main__":
    main()
