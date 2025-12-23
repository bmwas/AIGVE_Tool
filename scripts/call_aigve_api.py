#!/usr/bin/env python3
"""
A client to call the AIGVE API (server/main.py) to compute ALL metrics
(FID/IS/FVD + CD-FVD variants) on video pairs.

REQUIREMENTS (enforced by server):
- Upload mode: EXACTLY 2 videos (1 real + 1 generated) via /run_upload
- Generated video filename MUST contain 'synthetic' or 'generated' (configurable)
- ALL metrics computed: FID, IS, FVD (legacy) + CD-FVD (videomae, i3d)

Assumptions
- The AIGVE Docker container is already running and exposes the API at
  http://localhost:2200 (override via env AIGVE_API_URL or --base-url).
- Your host folder ./data is mounted into the container at /app/data.
  Example container run (GPU):
    docker run -d --name aigve --restart unless-stopped \
      --gpus '"device=1"' -p 2200:2200 \
      -v "$PWD/data":/app/data -v "$PWD/out":/app/out \
      ghcr.io/bmwas/aigve:latest

What this script does
1) GET /healthz to verify the API is up.
2a) POST /run_upload (recommended): Upload exactly 2 local videos
2b) POST /run (legacy): Use server-side paths with video validation
3) Computes ALL metrics with comprehensive retry logic and error handling

Output Metrics
- FID (Fréchet Inception Distance)
- IS (Inception Score) 
- FVD (Fréchet Video Distance)
- CD-FVD (8 flavors): i3d/videomae models × 128/256 resolution × 16/128 sequence length
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


def get_video_properties(video_path: str) -> Tuple[float, float]:
    """
    Get FPS and duration (in seconds) from a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (fps, duration_seconds)
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0.0
        
        cap.release()
        
        if fps <= 0 or duration <= 0:
            raise ValueError(f"Invalid video properties: fps={fps}, duration={duration}")
        
        return float(fps), float(duration)
    except ImportError:
        raise ImportError("opencv-python is required for automatic video property detection. Install with: pip install opencv-python")
    except Exception as e:
        raise ValueError(f"Failed to read video properties from {video_path}: {e}")


def check_health(base_url: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/healthz"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


def get_help(base_url: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/help"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def run_distribution_metrics(
    base_url: str,
    input_dir: str = "/app/data",
    stage_dataset: str = "/app/out/staged",
    max_seconds: float | None = 8.0,
    fps: float = 25.0,
    use_cpu: bool = False,
    generated_suffixes: str = "synthetic,generated",
    cdfvd_resolution: int = 128,
    cdfvd_sequence_length: int = 16,
    cdfvd_all_flavors: bool = False,  # Default: single flavor for speed
) -> Dict[str, Any]:
    """
    Calls POST /run with the minimal JSON body to stage and compute
    distribution-based metrics. CD-FVD is computed by default with both
    videomae and i3d models. See server/main.py and
    scripts/prepare_annotations.py for field semantics.
    """
    payload: Dict[str, Any] = {
        "input_dir": input_dir,
        "stage_dataset": stage_dataset,
        "compute": True,
        "categories": "distribution_based",
        "generated_suffixes": generated_suffixes,
    }
    if max_seconds is not None:
        payload.update({"max_seconds": float(max_seconds), "fps": float(fps)})
    else:
        # Fallback to frame-based control if needed
        payload.update({"max_len": 64})

    if use_cpu:
        payload["use_cpu"] = True
    
    # CD-FVD is computed by default with all 8 flavors, but allow single-flavor mode
    payload["cdfvd_resolution"] = cdfvd_resolution
    payload["cdfvd_sequence_length"] = cdfvd_sequence_length
    payload["cdfvd_all_flavors"] = cdfvd_all_flavors

    url = f"{base_url.rstrip('/')}/run"
    r = requests.post(url, json=payload, timeout=3600)
    r.raise_for_status()
    return r.json()


ALLOWED_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def _iter_video_files_from_dir(path: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if os.path.isdir(p):
            continue
        if os.path.splitext(name)[1].lower() in ALLOWED_EXTS:
            out.append(p)
    return out


def run_distribution_metrics_upload(
    base_url: str,
    upload_files: Optional[Iterable[str]] = None,
    upload_dir: Optional[str] = None,
    stage_dataset: Optional[str] = None,
    max_seconds: float | None = 8.0,
    fps: float = 25.0,
    use_cpu: bool = False,
    generated_suffixes: str = "synthetic,generated",
    categories: str = "distribution_based",
    metrics: str = "",
    cdfvd_resolution: int = 128,
    cdfvd_sequence_length: int = 16,
    cdfvd_all_flavors: bool = False,  # Default: single flavor for speed
) -> Dict[str, Any]:
    """
    Uploads local video files to the server and calls POST /run_upload.
    
    REQUIREMENTS (enforced by server):
    - EXACTLY 2 videos must be uploaded (1 real + 1 generated)
    - Generated video must contain one of the suffixes in filename
    - ALL metrics computed: FID, IS, FVD (legacy) + CD-FVD (8 flavors)
    """
    files_to_send: List[str] = []
    if upload_files:
        files_to_send.extend(list(upload_files))
    if upload_dir:
        files_to_send.extend(_iter_video_files_from_dir(upload_dir))
    # De-dup and keep order
    seen = set()
    files_to_send = [f for f in files_to_send if not (f in seen or seen.add(f))]
    if not files_to_send:
        raise ValueError("No video files to upload. Provide --upload-files or --upload-dir with supported extensions.")

    # VALIDATE EXACTLY 2 VIDEOS REQUIREMENT
    if len(files_to_send) != 2:
        raise ValueError(f"Server requires exactly 2 videos (1 real + 1 generated), got {len(files_to_send)}. "
                        f"Files found: {[os.path.basename(f) for f in files_to_send]}")

    # VALIDATE NAMING CONVENTION
    suffixes = [s.strip().lower() for s in generated_suffixes.split(',') if s.strip()]
    
    def _is_generated_video(filename: str) -> bool:
        base = filename.lower()
        return any(suffix in base for suffix in suffixes)
    
    real_videos = [f for f in files_to_send if not _is_generated_video(os.path.basename(f))]
    generated_videos = [f for f in files_to_send if _is_generated_video(os.path.basename(f))]
    
    if len(real_videos) != 1 or len(generated_videos) != 1:
        print(f"\n⚠️  NAMING CONVENTION WARNING:")
        print(f"   Expected: 1 real + 1 generated video")
        print(f"   Found: {len(real_videos)} real, {len(generated_videos)} generated")
        print(f"   Real videos: {[os.path.basename(f) for f in real_videos]}")
        print(f"   Generated videos: {[os.path.basename(f) for f in generated_videos]}")
        print(f"   Generated suffixes: {suffixes}")
        print(f"   Note: Generated video filename must contain: {' or '.join(suffixes)}")
        print(f"   Server will validate and may reject the request.\n")

    form_data: Dict[str, Any] = {
        "compute": True,
        "categories": categories,
        "generated_suffixes": generated_suffixes,
        "fps": float(fps),
        "pad": False,
    }
    if stage_dataset:
        form_data["stage_dataset"] = stage_dataset
    if max_seconds is not None:
        form_data["max_seconds"] = float(max_seconds)
    else:
        form_data["max_len"] = 64
    if use_cpu:
        form_data["use_cpu"] = True
    if metrics:
        form_data["metrics"] = metrics
    # CD-FVD is computed by default with all flavors (I3D + VideoMAE)
    form_data["use_cdfvd"] = True  # Always compute CD-FVD for complete metrics
    form_data["cdfvd_resolution"] = cdfvd_resolution
    form_data["cdfvd_sequence_length"] = cdfvd_sequence_length
    form_data["cdfvd_all_flavors"] = cdfvd_all_flavors

    url = f"{base_url.rstrip('/')}/run_upload"
    opened: List[Any] = []
    try:
        files_param = []
        for p in files_to_send:
            fname = os.path.basename(p)
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ALLOWED_EXTS:
                continue
            fobj = open(p, "rb")
            opened.append(fobj)
            files_param.append(("videos", (fname, fobj, "application/octet-stream")))

        if not files_param:
            raise ValueError("No acceptable files to upload after filtering by extension.")

        # Final validation that we're sending exactly 2 videos
        if len(files_param) != 2:
            raise ValueError(f"Server requires exactly 2 videos, but {len(files_param)} valid files remain after filtering.")

        print(f"\n{'='*60}", flush=True)
        print(f"[UPLOAD] Sending {len(files_param)} files to server", flush=True)
        print(f"{'='*60}", flush=True)
        for _, (fname, _, _) in files_param:
            print(f"   📁 {fname}", flush=True)
        print(f"\n[UPLOAD] Configuration:", flush=True)
        print(f"   URL: {url}", flush=True)
        print(f"   Generated suffixes: '{generated_suffixes}'", flush=True)
        print(f"   Categories: {categories}", flush=True)
        print(f"   Max seconds: {max_seconds}", flush=True)
        print(f"   FPS: {fps}", flush=True)
        print(f"   CD-FVD all flavors: {cdfvd_all_flavors}", flush=True)
        print(f"\n[UPLOAD] Form data being sent:", flush=True)
        for key, value in form_data.items():
            print(f"   {key}: {value}", flush=True)
        print(f"\n[UPLOAD] Sending request (timeout: 7200s)...", flush=True)

        r = requests.post(url, data=form_data, files=files_param, timeout=7200)
        
        print(f"[UPLOAD] Response status code: {r.status_code}", flush=True)
        
        if r.status_code != 200:
            print(f"\n{'='*60}", flush=True)
            print(f"[ERROR] Server returned error: {r.status_code}", flush=True)
            print(f"{'='*60}", flush=True)
            try:
                error_detail = r.json()
                print(f"[ERROR] Response JSON:", flush=True)
                print(json.dumps(error_detail, indent=2), flush=True)
            except Exception:
                print(f"[ERROR] Response text: {r.text[:1000]}", flush=True)
            print(f"{'='*60}\n", flush=True)
        
        r.raise_for_status()
        return r.json()
    finally:
        for f in opened:
            try:
                f.close()
            except Exception:
                pass


def save_artifacts_locally(result: Dict[str, Any], save_dir: str) -> list[str]:
    """Save artifacts locally and return list of saved file paths."""
    artifacts = result.get("artifacts") or []
    if not artifacts:
        print("[artifacts] No artifacts returned by server.", flush=True)
        return []
    os.makedirs(save_dir, exist_ok=True)
    saved: list[str] = []
    for art in artifacts:
        name = art.get("name") or "artifact.json"
        base = os.path.basename(name)
        target = os.path.join(save_dir, base)
        content: str | None = None
        
        # Debug: print artifact structure
        print(f"[artifacts] Processing {name}, keys: {art.keys()}", flush=True)
        
        if isinstance(art.get("json"), (dict, list)):
            content = json.dumps(art["json"], indent=2)
            print(f"[artifacts] Found json field for {name}", flush=True)
        elif isinstance(art.get("text"), str):
            content = art["text"]
            print(f"[artifacts] Found text field for {name}", flush=True)
        elif isinstance(art.get("content"), (dict, list)):
            # Handle content field when it's a parsed JSON object (dict/list)
            content = json.dumps(art["content"], indent=2)
            print(f"[artifacts] Found content field (JSON) for {name}", flush=True)
        elif isinstance(art.get("content"), str):
            # Handle content field when it's a string
            content = art["content"]
            print(f"[artifacts] Found content field (string) for {name}, length={len(content)}", flush=True)
        # Skip if no readable content
        if content is None:
            print(f"[artifacts] Skipping {name} - no readable content found", flush=True)
            continue
        try:
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            saved.append(target)
        except Exception as e:
            print(f"[artifacts] Failed to write {target}: {e}", flush=True)
    if saved:
        print("[artifacts] Saved locally:")
        for p in saved:
            print(" -", p)
    return saved


def extract_and_print_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metric scores from artifacts and result data, print them prominently.
    Returns a dict of metric_name -> score for easy access.
    """
    metrics_summary: Dict[str, Any] = {}
    artifacts = result.get("artifacts") or []
    
    print("\n" + "=" * 60, flush=True)
    print("📊 METRIC RESULTS", flush=True)
    print("=" * 60, flush=True)
    
    # Define metric file mappings (distribution-based + NN-based)
    # Note: Key names must match exactly what the metric classes output
    metric_files = {
        # Distribution-based metrics
        "fid_results.json": ("FID", ["FID_Mean_Score", "fid_score", "FID"]),
        "is_results.json": ("IS", ["IS_Mean_Score", "is_score", "IS"]),
        "fvd_results.json": ("FVD", ["FVD_Mean_Score", "fvd_score", "FVD"]),
        # NN-based video quality metrics (exact keys from aigve metric classes)
        "gstvqa_results.json": ("GSTVQA", ["GSTVQA_Mean_Score", "gstvqa_score"]),
        "simplevqa_results.json": ("SimpleVQA", ["SimpleVQA_Mean_Score", "simplevqa_score"]),
        # LightVQA+ uses "LightVQAPlus_Mean_Score" (no underscore before Plus)
        "lightvqa_plus_results.json": ("LightVQA+", ["LightVQAPlus_Mean_Score", "LightVQA_Plus_Mean_Score"]),
        "lightvqaplus_results.json": ("LightVQA+", ["LightVQAPlus_Mean_Score", "LightVQA_Plus_Mean_Score"]),
    }
    
    print(f"\n[DEBUG] Checking {len(artifacts)} artifacts for metric results...", flush=True)
    
    found_any = False
    found_metrics = []
    missing_metrics = []
    
    # Extract from artifacts
    for art in artifacts:
        name = art.get("name", "")
        print(f"[DEBUG]   Checking artifact: {name}", flush=True)
        
        if name in metric_files:
            metric_label, score_keys = metric_files[name]
            # Content can be dict/list (parsed JSON) or the "json" field
            content = art.get("content") or art.get("json")
            print(f"[DEBUG]     Content type: {type(content).__name__}", flush=True)
            
            if isinstance(content, dict):
                # Try each possible key name
                score = None
                used_key = None
                for key in score_keys:
                    if key in content:
                        score = content[key]
                        used_key = key
                        print(f"[DEBUG]     Found score with key '{key}': {score}", flush=True)
                        break
                
                if score is not None:
                    metrics_summary[metric_label] = score
                    found_any = True
                    found_metrics.append(metric_label)
                    # Format nicely
                    if isinstance(score, float):
                        if abs(score) > 1e6 or (abs(score) < 1e-3 and score != 0):
                            score_str = f"{score:.6e}"
                        else:
                            score_str = f"{score:.4f}"
                    else:
                        score_str = str(score)
                    print(f"   ✅ {metric_label}: {score_str}", flush=True)
                else:
                    print(f"[DEBUG]     No score found for {metric_label}, tried keys: {score_keys}", flush=True)
                    print(f"[DEBUG]     Available keys in content: {list(content.keys())}", flush=True)
                    missing_metrics.append(metric_label)
            else:
                print(f"[DEBUG]     Content is not a dict, skipping", flush=True)
    
    # Also check for CD-FVD results
    print(f"\n[DEBUG] Checking for CD-FVD results in response...", flush=True)
    if "cdfvd_results" in result:
        cdfvd = result["cdfvd_results"]
        print(f"[DEBUG]   Found cdfvd_results with {len(cdfvd) if isinstance(cdfvd, dict) else 0} models", flush=True)
        if isinstance(cdfvd, dict):
            print("\n   📹 CD-FVD Results:", flush=True)
            for model, model_data in cdfvd.items():
                print(f"[DEBUG]     Processing model: {model}", flush=True)
                if isinstance(model_data, dict):
                    if "error" in model_data:
                        print(f"      ❌ {model}: ERROR - {model_data['error']}", flush=True)
                    elif "flavors" in model_data:
                        # New all-flavors format
                        print(f"[DEBUG]       Model has {len(model_data['flavors'])} flavors", flush=True)
                        for flavor_key, flavor_data in model_data["flavors"].items():
                            if "error" in flavor_data:
                                print(f"      ❌ {model}/{flavor_key}: ERROR - {flavor_data['error']}", flush=True)
                            else:
                                fvd_score = flavor_data.get("fvd_score", "N/A")
                                metrics_summary[f"CD-FVD_{model}_{flavor_key}"] = fvd_score
                                found_any = True
                                found_metrics.append(f"CD-FVD_{model}_{flavor_key}")
                                print(f"      ✅ {model}/{flavor_key}: {fvd_score}", flush=True)
                    elif "fvd_score" in model_data:
                        # Single score format
                        fvd_score = model_data["fvd_score"]
                        metrics_summary[f"CD-FVD_{model}"] = fvd_score
                        found_any = True
                        found_metrics.append(f"CD-FVD_{model}")
                        print(f"      ✅ {model}: {fvd_score}", flush=True)
                    else:
                        print(f"[DEBUG]       Unknown model_data format: {list(model_data.keys())}", flush=True)
    else:
        print(f"[DEBUG]   No cdfvd_results in response", flush=True)
    
    if not found_any:
        print("   ⚠️  No metric scores extracted from artifacts", flush=True)
        print("   Check server logs for computation details", flush=True)
        
        # Try to extract from stdout as fallback
        stdout = result.get("stdout", "")
        if stdout:
            import re
            print(f"\n[DEBUG] Attempting to extract metrics from stdout ({len(stdout)} chars)...", flush=True)
            # Look for patterns like "FID mean score: X", "GSTVQA summary: {...}" etc.
            patterns = [
                # Distribution-based metrics
                (r'FID mean score:\s*([-\d.eE+]+)', 'FID'),
                (r'IS mean score:\s*([-\d.eE+]+)', 'IS'),
                (r'FVD mean score:\s*([-\d.eE+]+)', 'FVD'),
                # NN-based video quality metrics
                (r'GSTVQA mean score:\s*([-\d.eE+]+)', 'GSTVQA'),
                (r"GSTVQA summary:.*'GSTVQA_Mean_Score':\s*([-\d.eE+]+)", 'GSTVQA'),
                (r'SimpleVQA mean score:\s*([-\d.eE+]+)', 'SimpleVQA'),
                (r"SimpleVQA summary:.*'SimpleVQA_Mean_Score':\s*([-\d.eE+]+)", 'SimpleVQA'),
                (r'LightVQA\+ mean score:\s*([-\d.eE+]+)', 'LightVQA+'),
                (r"LightVQA\+ summary:.*'LightVQAPlus_Mean_Score':\s*([-\d.eE+]+)", 'LightVQA+'),
            ]
            print("\n   📄 Extracted from stdout:", flush=True)
            extracted_labels = set()
            for pattern, label in patterns:
                if label in extracted_labels:
                    continue  # Skip duplicate patterns for same metric
                match = re.search(pattern, stdout)
                if match:
                    try:
                        score = float(match.group(1))
                        metrics_summary[label] = score
                        print(f"      ✅ {label}: {score}", flush=True)
                        found_any = True
                        found_metrics.append(label)
                        extracted_labels.add(label)
                    except ValueError:
                        pass
    
    # Final summary
    print("\n" + "-" * 60, flush=True)
    print("📋 METRICS EXTRACTION SUMMARY", flush=True)
    print("-" * 60, flush=True)
    
    # List all expected metrics
    all_expected = ["FID", "IS", "FVD", "GSTVQA", "SimpleVQA", "LightVQA+", "CD-FVD"]
    
    print(f"   ✅ Successfully extracted: {len(found_metrics)}", flush=True)
    for m in found_metrics:
        print(f"      • {m}", flush=True)
    
    if missing_metrics:
        print(f"   ⚠️  Missing from artifacts: {len(missing_metrics)}", flush=True)
        for m in missing_metrics:
            print(f"      • {m}", flush=True)
    
    # Check which expected metrics weren't found
    found_base_metrics = set()
    for m in found_metrics:
        # Extract base metric name (e.g., "CD-FVD_i3d_res224_len16" -> "CD-FVD")
        if m.startswith("CD-FVD"):
            found_base_metrics.add("CD-FVD")
        else:
            found_base_metrics.add(m)
    
    not_computed = [m for m in all_expected if m not in found_base_metrics]
    if not_computed:
        print(f"   ℹ️  Not computed:", flush=True)
        for m in not_computed:
            if m == "LightVQA+":
                print(f"      • {m} (requires manual model download - optional)", flush=True)
            else:
                print(f"      • {m}", flush=True)
    
    print("=" * 60, flush=True)
    
    return metrics_summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Call AIGVE API to run distribution-based metrics.")
    ap.add_argument("--base-url", default=os.getenv("AIGVE_API_URL", "http://localhost:2200"),
                    help="Base URL for the AIGVE API (default: http://localhost:2200 or env AIGVE_API_URL)")
    ap.add_argument("--input-dir", default=os.getenv("AIGVE_INPUT_DIR", "/app/data"),
                    help="Path to mixed videos. Docker default: /app/data. Local example: ./data")
    ap.add_argument("--stage-dataset", default=os.getenv("AIGVE_STAGE_DATASET", "/app/out/staged"),
                    help="Destination dataset path. Docker default: /app/out/staged. Local example: ./out/staged")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="Clip duration in seconds (overrides max_len). If not set and --auto-detect is used, uses full video length. Default: None")
    ap.add_argument("--fps", type=float, default=None,
                    help="FPS used with --max-seconds. If not set and --auto-detect is used, detects from reference video. Default: None (uses 25.0)")
    ap.add_argument("--auto-detect", action="store_true",
                    help="Automatically detect FPS and duration from the FIRST video (reference video). "
                         "FPS is always taken from the reference video. "
                         "Duration uses full video length unless --max-seconds is specified.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--no-help", action="store_true", help="Skip calling /help before /run")
    ap.add_argument("--save-dir", default="./results", help="Directory to save returned result files locally")
    ap.add_argument("--local", action="store_true", help="Use local host defaults (./data, ./out/staged)")
    # Upload mode options - EXPLICIT video pair (recommended)
    ap.add_argument("--reference-video", "--ref", default=None,
                    help="Path to the reference (real/ground-truth) video file")
    ap.add_argument("--generated-video", "--gen", default=None,
                    help="Path to the generated (synthetic/AI-generated) video file")
    # Legacy upload options (still supported)
    ap.add_argument("--upload-dir", default=None,
                    help="[Legacy] Upload mode: directory of local videos to send to the server")
    ap.add_argument("--upload-files", nargs="+", default=None,
                    help="[Legacy] Upload mode: explicit list of local video files (use --reference-video and --generated-video instead)")
    ap.add_argument("--generated-suffixes", default="synthetic,generated",
                    help="Suffix tokens for pairing (used by server script). Default: synthetic,generated")
    ap.add_argument("--second-is-generated", action="store_true",
                    help="[Legacy] Treat the SECOND video as generated. Use --reference-video and --generated-video instead.")
    ap.add_argument("--categories", default="distribution_based",
                    help="Metric categories CSV (e.g., distribution_based,nn_based_video). Default: distribution_based")
    ap.add_argument("--all-metrics", action="store_true",
                    help="Compute ALL metrics: distribution_based (FID/IS/FVD) + nn_based_video (GSTVQA/SimpleVQA/LightVQA+) + CD-FVD (single fast flavor)")
    ap.add_argument("--fast", action="store_true",
                    help="Fast mode: use optimized settings for speed (lower resolution, single CD-FVD flavor)")
    ap.add_argument("--metrics", default="",
                    help="Specific metric names CSV (optional). Example: fid,is,fvd or lightvqa+")
    # CD-FVD options
    ap.add_argument("--cdfvd-resolution", type=int, default=128,
                    help="Resolution for CD-FVD video processing. Default: 128 (fast). Use 224 for higher accuracy.")
    ap.add_argument("--cdfvd-sequence-length", type=int, default=16,
                    help="Sequence length for CD-FVD video processing. Default: 16")
    ap.add_argument("--cdfvd-all-flavors", action="store_true",
                    help="Compute ALL 8 CD-FVD flavors (2 models × 2 resolutions × 2 seq lengths). SLOW! Default: single flavor")
    ap.add_argument("--cdfvd-single-flavor", action="store_true",
                    help="[Deprecated] Compute only single CD-FVD flavor (now the default)")

    args = ap.parse_args(argv)

    # Handle --fast flag: optimize for speed
    if args.fast:
        print(f"\n⚡ --fast mode enabled: Optimizing for speed", flush=True)
        args.cdfvd_resolution = 128  # Lower resolution
        args.cdfvd_sequence_length = 8  # Fewer frames
        args.cdfvd_all_flavors = False  # Single flavor only
        print(f"   📊 CD-FVD: resolution=128, sequence_length=8, single flavor only", flush=True)
    
    # Handle --all-metrics flag: override categories to include all metric types
    if args.all_metrics:
        args.categories = "distribution_based,nn_based_video"
        # By default, use single fast CD-FVD flavor unless --cdfvd-all-flavors is set
        if not args.cdfvd_all_flavors:
            print(f"\n🎯 --all-metrics flag detected: Computing ALL metrics (FAST mode)", flush=True)
            print(f"   📊 Categories: {args.categories}", flush=True)
            print(f"   🔧 Metrics included:", flush=True)
            print(f"      - distribution_based: FID, IS, FVD (AIGVE native)", flush=True)
            print(f"      - nn_based_video: GSTVQA, SimpleVQA", flush=True)
            print(f"      - CD-FVD: single flavor (videomae, res={args.cdfvd_resolution}, len={args.cdfvd_sequence_length})", flush=True)
            print(f"   💡 For all 8 CD-FVD flavors (SLOW), add --cdfvd-all-flavors", flush=True)
        else:
            print(f"\n🎯 --all-metrics + --cdfvd-all-flavors: Computing ALL metrics with ALL CD-FVD flavors", flush=True)
            print(f"   ⚠️  This will be SLOW (8 CD-FVD combinations)", flush=True)

    base_url = args.base_url

    # Handle explicit --reference-video and --generated-video (recommended method)
    # This creates properly named temp files for server-side pairing
    temp_dir_to_cleanup = None
    
    if args.reference_video or args.generated_video:
        print(f"\n{'='*60}", flush=True)
        print(f"📹 EXPLICIT VIDEO PAIR MODE", flush=True)
        print(f"{'='*60}", flush=True)
        
        if not args.reference_video:
            print("[ERROR] --reference-video is required when using --generated-video", flush=True)
            print("[DEBUG] args.reference_video = None", flush=True)
            print("[DEBUG] args.generated_video =", args.generated_video, flush=True)
            sys.exit(1)
        if not args.generated_video:
            print("[ERROR] --generated-video is required when using --reference-video", flush=True)
            print("[DEBUG] args.reference_video =", args.reference_video, flush=True)
            print("[DEBUG] args.generated_video = None", flush=True)
            sys.exit(1)
        
        # Log full paths
        print(f"\n[INPUT] Reference video path: {args.reference_video}", flush=True)
        print(f"[INPUT] Generated video path: {args.generated_video}", flush=True)
        
        # Validate files exist
        if not os.path.exists(args.reference_video):
            print(f"[ERROR] Reference video not found: {args.reference_video}", flush=True)
            print(f"[DEBUG] os.path.exists() returned False", flush=True)
            print(f"[DEBUG] Current working directory: {os.getcwd()}", flush=True)
            sys.exit(1)
        if not os.path.exists(args.generated_video):
            print(f"[ERROR] Generated video not found: {args.generated_video}", flush=True)
            print(f"[DEBUG] os.path.exists() returned False", flush=True)
            print(f"[DEBUG] Current working directory: {os.getcwd()}", flush=True)
            sys.exit(1)
        
        # Log file sizes
        ref_size = os.path.getsize(args.reference_video)
        gen_size = os.path.getsize(args.generated_video)
        print(f"\n[VALIDATE] Reference video exists: ✅", flush=True)
        print(f"           Size: {ref_size / 1024 / 1024:.2f} MB", flush=True)
        print(f"[VALIDATE] Generated video exists: ✅", flush=True)
        print(f"           Size: {gen_size / 1024 / 1024:.2f} MB", flush=True)
        
        # Create temp directory with properly named files for server-side pairing
        # Server expects: basename.mp4 (ref) and basename_generated.mp4 (gen)
        print(f"\n[PAIRING] Creating properly named temp files for server...", flush=True)
        print(f"[PAIRING] Server requires: <basename>.mp4 + <basename>_generated.mp4", flush=True)
        
        temp_dir = tempfile.mkdtemp(prefix="aigve_upload_")
        temp_dir_to_cleanup = temp_dir
        print(f"[PAIRING] Temp directory: {temp_dir}", flush=True)
        
        # Get reference video base name (without extension)
        ref_basename = os.path.splitext(os.path.basename(args.reference_video))[0]
        ref_ext = os.path.splitext(args.reference_video)[1]
        gen_ext = os.path.splitext(args.generated_video)[1]
        
        print(f"[PAIRING] Reference basename: '{ref_basename}'", flush=True)
        print(f"[PAIRING] Reference extension: '{ref_ext}'", flush=True)
        print(f"[PAIRING] Generated extension: '{gen_ext}'", flush=True)
        
        # Create properly named copies
        ref_temp = os.path.join(temp_dir, f"{ref_basename}{ref_ext}")
        gen_temp = os.path.join(temp_dir, f"{ref_basename}_generated{gen_ext}")
        
        print(f"\n[PAIRING] Creating paired files:", flush=True)
        print(f"          Reference: {os.path.basename(ref_temp)}", flush=True)
        print(f"          Generated: {os.path.basename(gen_temp)}", flush=True)
        
        print(f"\n[COPY] Copying reference video...", flush=True)
        shutil.copy2(args.reference_video, ref_temp)
        print(f"[COPY] ✅ Reference copied: {ref_temp}", flush=True)
        
        print(f"[COPY] Copying generated video...", flush=True)
        shutil.copy2(args.generated_video, gen_temp)
        print(f"[COPY] ✅ Generated copied: {gen_temp}", flush=True)
        
        # Verify copies
        if os.path.exists(ref_temp) and os.path.exists(gen_temp):
            print(f"\n[VERIFY] Both temp files created successfully ✅", flush=True)
            print(f"         Reference: {os.path.getsize(ref_temp)} bytes", flush=True)
            print(f"         Generated: {os.path.getsize(gen_temp)} bytes", flush=True)
        else:
            print(f"[ERROR] Failed to create temp files!", flush=True)
            sys.exit(1)
        
        # Use temp files for upload
        args.upload_files = [ref_temp, gen_temp]
        args.generated_suffixes = "generated"
        
        print(f"\n[CONFIG] Upload files set to: {[os.path.basename(f) for f in args.upload_files]}", flush=True)
        print(f"[CONFIG] Generated suffix set to: '{args.generated_suffixes}'", flush=True)
        
        # Enable auto-detect by default when using explicit video pair
        if args.fps is None and args.max_seconds is None:
            args.auto_detect = True
            print(f"[CONFIG] Auto-detect enabled (FPS and duration from reference)", flush=True)
        
        print(f"{'='*60}\n", flush=True)

    # 1) Health
    print(f"\n[1/3] Checking health at {base_url}/healthz ...", flush=True)
    health = check_health(base_url)
    print(json.dumps(health, indent=2))

    # If server cwd is not an /app path, it is likely running locally (no Docker)
    cwd = str(health.get("cwd", ""))
    is_container = cwd.startswith("/app")
    if not is_container and (str(args.input_dir).startswith("/app/") or str(args.stage_dataset).startswith("/app/")):
        print("[WARN] Server is running locally (cwd: {}), but input paths are '/app/...'.".format(cwd), flush=True)
        print("       For local runs, use host paths (e.g., ./data, ./out/staged) or pass --local.", flush=True)

    # Convenience: --local switches defaults to repo-relative paths when not explicitly overridden
    if args.local:
        default_in = os.getenv("AIGVE_INPUT_DIR", "/app/data")
        default_out = os.getenv("AIGVE_STAGE_DATASET", "/app/out/staged")
        if args.input_dir == default_in:
            args.input_dir = "./data"
        if args.stage_dataset == default_out:
            args.stage_dataset = "./out/staged"
        print(f"[local] Using input_dir={args.input_dir} stage_dataset={args.stage_dataset}", flush=True)

    # Auto-detect video properties from the FIRST video (reference video)
    # The first video given is always treated as the reference video
    if args.auto_detect:
        reference_video = None
        
        # The FIRST video in the list is the reference video
        if args.upload_files and len(args.upload_files) > 0:
            reference_video = args.upload_files[0]
            print(f"\n[auto-detect] First video is treated as reference: {os.path.basename(reference_video)}", flush=True)
        elif args.upload_dir:
            video_exts = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
            # Get first video file in directory (sorted alphabetically)
            for video_file in sorted(Path(args.upload_dir).iterdir()):
                if video_file.suffix.lower() in video_exts:
                    reference_video = str(video_file)
                    print(f"\n[auto-detect] First video in directory is treated as reference: {video_file.name}", flush=True)
                    break
        
        if reference_video:
            if not os.path.exists(reference_video):
                print(f"[ERROR] Reference video does not exist: {reference_video}", flush=True)
                sys.exit(1)
            
            try:
                print(f"[auto-detect] Reading video properties from: {os.path.basename(reference_video)}", flush=True)
                detected_fps, detected_duration = get_video_properties(reference_video)
                
                # Calculate total frames for clarity
                total_frames = int(detected_fps * detected_duration)
                
                print(f"[auto-detect] ✅ Detected properties:", flush=True)
                print(f"              FPS: {detected_fps:.2f}", flush=True)
                print(f"              Duration: {detected_duration:.2f} seconds", flush=True)
                print(f"              Total frames: {total_frames}", flush=True)
                
                # Always use detected FPS (this is mandatory per user request)
                args.fps = detected_fps
                print(f"[auto-detect] Using detected FPS: {args.fps:.2f}", flush=True)
                
                # Use full duration if not explicitly overridden
                if args.max_seconds is None:
                    args.max_seconds = detected_duration
                    print(f"[auto-detect] Using full video duration: {args.max_seconds:.2f} seconds ({total_frames} frames)", flush=True)
                else:
                    # User specified max_seconds, use it but with detected FPS
                    effective_frames = int(args.fps * args.max_seconds)
                    print(f"[auto-detect] Using user-specified duration: {args.max_seconds:.2f} seconds ({effective_frames} frames)", flush=True)
                    
            except ImportError as e:
                print(f"[ERROR] {e}", flush=True)
                print(f"        Install opencv-python: pip install opencv-python", flush=True)
                sys.exit(1)
            except Exception as e:
                print(f"[ERROR] Failed to auto-detect video properties: {e}", flush=True)
                sys.exit(1)
        else:
            print(f"[ERROR] No video files found for auto-detection.", flush=True)
            print(f"        Provide videos via --upload-files or --upload-dir", flush=True)
            sys.exit(1)
    
    # Set defaults if not using auto-detect
    if args.fps is None:
        args.fps = 25.0
    if args.max_seconds is None:
        args.max_seconds = 8.0

    # Handle --second-is-generated: auto-detect a unique suffix from second video
    if args.second_is_generated and args.upload_files and len(args.upload_files) >= 2:
        first_video = os.path.basename(args.upload_files[0]).lower()
        second_video = os.path.basename(args.upload_files[1]).lower()
        
        # Remove extension for comparison
        first_stem = os.path.splitext(first_video)[0]
        second_stem = os.path.splitext(second_video)[0]
        
        print(f"\n[second-is-generated] Finding unique identifier for second video...", flush=True)
        print(f"   Reference (1st): {os.path.basename(args.upload_files[0])}", flush=True)
        print(f"   Generated (2nd): {os.path.basename(args.upload_files[1])}", flush=True)
        
        # Split second video name into parts and find one NOT in first video name
        # Try splitting by common delimiters
        parts = re.split(r'[_\-\s\.]+', second_stem)
        
        unique_suffix = None
        for part in parts:
            if len(part) >= 3 and part not in first_stem:  # At least 3 chars to be meaningful
                unique_suffix = part
                break
        
        if unique_suffix:
            args.generated_suffixes = unique_suffix
            print(f"   ✅ Auto-detected unique suffix: '{unique_suffix}'", flush=True)
            print(f"   Using --generated-suffixes {unique_suffix}", flush=True)
        else:
            # Fallback: use the entire second filename stem
            args.generated_suffixes = second_stem
            print(f"   ⚠️  Could not find unique part, using full filename: '{second_stem}'", flush=True)
            print(f"   Using --generated-suffixes {second_stem}", flush=True)

    # 2) Help (optional)
    if not args.no_help:
        print(f"\n[2/3] Fetching CLI help via {base_url}/help ...", flush=True)
        help_info = get_help(base_url)
        # Only print the command and first ~20 lines of stdout to keep it short
        print("cmd:", help_info.get("cmd"))
        stdout = help_info.get("stdout", "")
        lines = stdout.splitlines()
        preview = "\n".join(lines[:20]) + ("\n..." if len(lines) > 20 else "")
        print("stdout (truncated):\n" + preview)

    # 3) Run distribution metrics (upload mode or server-path mode)
    if args.upload_dir or args.upload_files:
        print(f"\n[3/3] Running distribution metrics via {base_url}/run_upload ...", flush=True)
        if args.cdfvd_single_flavor:
            print(f"[CD-FVD] Single flavor mode: resolution={args.cdfvd_resolution}, sequence_length={args.cdfvd_sequence_length}")
        else:
            print(f"[CD-FVD] All flavors mode: computing 8 combinations (2 models × 2 resolutions × 2 sequence lengths)")
        result = run_distribution_metrics_upload(
            base_url=base_url,
            upload_files=args.upload_files,
            upload_dir=args.upload_dir,
            stage_dataset=(None if args.stage_dataset in (None, "", "/app/out/staged") else args.stage_dataset),
            max_seconds=args.max_seconds,
            fps=args.fps,
            use_cpu=args.cpu,
            generated_suffixes=args.generated_suffixes,
            categories=args.categories,
            metrics=args.metrics,
            cdfvd_resolution=args.cdfvd_resolution,
            cdfvd_sequence_length=args.cdfvd_sequence_length,
            cdfvd_all_flavors=args.cdfvd_all_flavors,  # Default: False (single flavor for speed)
        )
    else:
        print(f"\n[3/3] Running distribution metrics via {base_url}/run ...", flush=True)
        if args.cdfvd_single_flavor:
            print(f"[CD-FVD] Single flavor mode: resolution={args.cdfvd_resolution}, sequence_length={args.cdfvd_sequence_length}")
        else:
            print(f"[CD-FVD] All flavors mode: computing 8 combinations (2 models × 2 resolutions × 2 sequence lengths)")
        result = run_distribution_metrics(
            base_url=base_url,
            input_dir=args.input_dir,
            stage_dataset=args.stage_dataset,
            max_seconds=args.max_seconds,
            fps=args.fps,
            use_cpu=args.cpu,
            generated_suffixes=args.generated_suffixes,
            cdfvd_resolution=args.cdfvd_resolution,
            cdfvd_sequence_length=args.cdfvd_sequence_length,
            cdfvd_all_flavors=args.cdfvd_all_flavors,  # Default: False (single flavor for speed)
        )

    print("\n--- /run result ---")
    print("cmd:", result.get("cmd"))
    print("returncode:", result.get("returncode"))

    # Print tail of stdout/stderr for quick visibility
    stdout = result.get("stdout", "").rstrip()
    stderr = result.get("stderr", "").rstrip()

    def tail(text: str, n: int = 40) -> str:
        lines = text.splitlines()
        if len(lines) <= n:
            return text
        return "\n".join(lines[-n:])

    if stdout:
        print("\nstdout (last 40 lines):\n" + tail(stdout, 40))
    if stderr:
        print("\nstderr (last 40 lines):\n" + tail(stderr, 40))

    # ========================================
    # EXTRACT AND PRINT ALL METRIC RESULTS
    # ========================================
    metrics_summary = extract_and_print_metrics(result)
    
    # Processing summary if available
    if "processing_summary" in result:
        summary = result["processing_summary"]
        print(f"\n--- Processing Summary ---")
        print(f"Total Duration: {summary.get('total_duration_ms', 0):.1f} ms")
        print(f"Script Success: {'✅' if summary.get('script_success') else '❌'}")
        print(f"CD-FVD Models: {summary.get('cdfvd_models_successful', 0)}/{summary.get('cdfvd_models_total', 0)} successful")
        print(f"Videos Processed: {summary.get('videos_processed', 0)}")
    
    # Save any returned artifacts locally
    try:
        save_artifacts_locally(result, args.save_dir)
        # Also save CD-FVD results if present (multiple models)
        if "cdfvd_results" in result:
            cdfvd_path = os.path.join(args.save_dir, "cdfvd_results.json")
            os.makedirs(args.save_dir, exist_ok=True)
            with open(cdfvd_path, "w") as f:
                json.dump(result["cdfvd_results"], f, indent=2)
            print(f"\n[artifacts] CD-FVD results saved to {cdfvd_path}")
    except Exception as e:
        print(f"[artifacts] Error while saving artifacts: {e}", flush=True)

    # Cleanup temp directory if we created one
    if temp_dir_to_cleanup and os.path.exists(temp_dir_to_cleanup):
        try:
            shutil.rmtree(temp_dir_to_cleanup)
            print(f"[cleanup] Removed temp directory", flush=True)
        except Exception as e:
            print(f"[cleanup] Failed to remove temp directory: {e}", flush=True)

    rc = int(result.get("returncode", 0) or 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
