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
import sys
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
    cdfvd_all_flavors: bool = True,
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
    cdfvd_all_flavors: bool = True,
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
    # CD-FVD is computed by default with all 8 flavors, but allow single-flavor mode
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

        print(f"[upload] Sending {len(files_param)} files (server requires exactly 2):")
        for _, (fname, _, _) in files_param:
            print(" -", fname)
        print(f"[upload] ALL metrics will be computed: FID, IS, FVD (legacy) + CD-FVD (8 flavors)")
        print(f"[upload] Generated suffixes for pairing: {generated_suffixes}")

        r = requests.post(url, data=form_data, files=files_param, timeout=7200)
        r.raise_for_status()
        return r.json()
    finally:
        for f in opened:
            try:
                f.close()
            except Exception:
                pass


def save_artifacts_locally(result: Dict[str, Any], save_dir: str) -> list[str]:
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
            print(f"[artifacts] Found json field for {name}, content preview: {content[:200] if content else 'EMPTY'}", flush=True)
        elif isinstance(art.get("text"), str):
            content = art["text"]
            print(f"[artifacts] Found text field for {name}", flush=True)
        elif isinstance(art.get("content"), str):
            # Handle content field (for CD-FVD results)
            content = art["content"]
            print(f"[artifacts] Found content field for {name}, length={len(content)}", flush=True)
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
    # Upload mode options
    ap.add_argument("--upload-dir", default=None,
                    help="Upload mode: directory of local videos to send to the server via /run_upload")
    ap.add_argument("--upload-files", nargs="+", default=None,
                    help="Upload mode: explicit list of local video files to send via /run_upload")
    ap.add_argument("--generated-suffixes", default="synthetic,generated",
                    help="Suffix tokens for pairing (used by server script). Default: synthetic,generated")
    ap.add_argument("--categories", default="distribution_based",
                    help="Metric categories CSV (e.g., distribution_based,nn_based_video). Default: distribution_based")
    ap.add_argument("--metrics", default="",
                    help="Specific metric names CSV (optional). Example: fid,is,fvd or lightvqa+")
    # CD-FVD options (CD-FVD computes all 8 flavors by default)
    ap.add_argument("--cdfvd-resolution", type=int, default=224,
                    help="Resolution for CD-FVD video processing (single-flavor mode). Default: 224 (min safe for i3d)")
    ap.add_argument("--cdfvd-sequence-length", type=int, default=16,
                    help="Sequence length for CD-FVD video processing (single-flavor mode). Default: 16")
    ap.add_argument("--cdfvd-single-flavor", action="store_true",
                    help="Compute only single CD-FVD flavor instead of all 8 combinations")

    args = ap.parse_args(argv)

    base_url = args.base_url

    # 1) Health
    print(f"[1/3] Checking health at {base_url}/healthz ...", flush=True)
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
            cdfvd_all_flavors=not args.cdfvd_single_flavor,
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
            cdfvd_all_flavors=not args.cdfvd_single_flavor,
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

    # Print ALL metric results
    print("\n--- ALL METRICS RESULTS ---")
    
    # Legacy metrics summary
    artifacts = result.get("artifacts", [])
    legacy_metrics_found = []
    for art in artifacts:
        name = art.get("name", "")
        if name in ["fid_results.json", "is_results.json", "fvd_results.json"]:
            legacy_metrics_found.append(name.replace("_results.json", "").upper())
    
    if legacy_metrics_found:
        print(f"Legacy metrics computed: {', '.join(legacy_metrics_found)}")
    else:
        print("Legacy metrics: No results found (check script execution)")
    
    # CD-FVD results (all flavors or legacy models)
    if "cdfvd_results" in result:
        print("\n--- CD-FVD Results ---")
        cdfvd_results = result["cdfvd_results"]
        
        # Check if this is the new all-flavors format
        if isinstance(cdfvd_results, dict) and any("flavors" in res for res in cdfvd_results.values() if isinstance(res, dict)):
            # New all-flavors format: each model contains a "flavors" dict
            successful_flavors = 0
            total_flavors = 0
            
            for model, model_result in cdfvd_results.items():
                if "error" in model_result:
                    print(f"\n{model.upper()} Model: ❌ ERROR - {model_result['error']}")
                    continue
                    
                flavors = model_result.get("flavors", {})
                if not flavors:
                    continue
                    
                print(f"\n{model.upper()} Model - All Flavors:")
                for flavor_key, flavor_result in flavors.items():
                    total_flavors += 1
                    if "error" in flavor_result:
                        print(f"  {flavor_key}: ❌ ERROR - {flavor_result['error']}")
                    else:
                        successful_flavors += 1
                        fvd_score = flavor_result.get('fvd_score', 'N/A')
                        print(f"  {flavor_key}: ✅ {fvd_score}")
            
            print(f"\nCD-FVD Summary: {successful_flavors}/{total_flavors} flavors successful")
            
        elif isinstance(cdfvd_results, dict) and "flavors" in cdfvd_results:
            # Single all-flavors result format
            flavors = cdfvd_results["flavors"]
            successful_flavors = 0
            total_flavors = len(flavors)
            
            print("All FVD Flavors:")
            for flavor_key, flavor_result in flavors.items():
                if "error" in flavor_result:
                    print(f"  {flavor_key}: ❌ ERROR - {flavor_result['error']}")
                else:
                    successful_flavors += 1
                    fvd_score = flavor_result.get('fvd_score', 'N/A')
                    model = flavor_result.get('model', '')
                    res = flavor_result.get('resolution', '')
                    seq = flavor_result.get('sequence_length', '')
                    print(f"  {flavor_key}: ✅ {fvd_score} (model={model}, res={res}, seq={seq})")
            
            print(f"\nCD-FVD Summary: {successful_flavors}/{total_flavors} flavors successful")
            
        else:
            # Legacy format: individual model results
            successful_models = 0
            total_models = len(cdfvd_results)
            
            for model, cdfvd_res in cdfvd_results.items():
                if "error" in cdfvd_res:
                    print(f"\n{model.upper()} Model: ❌ ERROR - {cdfvd_res['error']}")
                    if "attempts" in cdfvd_res:
                        print(f"  Failed after {cdfvd_res['attempts']} attempts")
                else:
                    successful_models += 1
                    print(f"\n{model.upper()} Model: ✅ SUCCESS")
                    print(f"  FVD Score: {cdfvd_res.get('fvd_score', 'N/A')}")
                    print(f"  Real Videos: {cdfvd_res.get('num_real_videos', 'N/A')}")
                    print(f"  Fake Videos: {cdfvd_res.get('num_fake_videos', 'N/A')}")
                    # If server returned length metadata, show it
                    if "max_seconds" in cdfvd_res:
                        ms = cdfvd_res.get("max_seconds")
                        fps_v = cdfvd_res.get("fps")
                        max_len = cdfvd_res.get("max_len")
                        if fps_v is not None and max_len is not None:
                            print(f"  Clip: {ms} s at {fps_v} fps (~{max_len} frames)")
                        else:
                            print(f"  Clip: {ms} s")
            
            print(f"\nCD-FVD Summary: {successful_models}/{total_models} models successful")
    elif "cdfvd_error" in result:
        print(f"\n❌ CD-FVD Error: {result['cdfvd_error']}")
    
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

    rc = int(result.get("returncode", 0) or 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
