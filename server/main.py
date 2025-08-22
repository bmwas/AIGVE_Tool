from __future__ import annotations

import os
import uuid
import shutil
import sys
import shlex
import subprocess
import json
import time
import re
import tempfile
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from pydantic import BaseModel, Field

# Optional torch import (server runs inside conda env where torch is present)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - tolerate missing/failed import
    torch = None  # type: ignore

# Optional cd-fvd import
try:
    from cdfvd import fvd as cdfvd  # type: ignore
except Exception:  # pragma: no cover - tolerate missing/failed import
    cdfvd = None  # type: ignore

APP_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
SCRIPT_PATH = os.path.join(APP_ROOT, "scripts", "prepare_annotations.py")

app = FastAPI(title="AIGVE Prepare Annotations API", version="1.0.0")

# Logger setup (works alongside uvicorn's logging)
logger = logging.getLogger("aigve.api")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _handler.setFormatter(_fmt)
    logger.addHandler(_handler)
logger.setLevel(os.getenv("AIGVE_LOG_LEVEL", "INFO").upper())
logger.propagate = False


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    
    # Log detailed request information
    logger.info("[%s] Request started: %s %s", rid, request.method, request.url.path)
    logger.debug("[%s] Request details: URL=%s, client=%s, headers=%s", 
                rid, str(request.url), getattr(request.client, 'host', 'unknown'), 
                dict(request.headers) if hasattr(request, 'headers') else {})
    
    # Track request size if applicable
    content_length = request.headers.get('content-length')
    if content_length:
        logger.debug("[%s] Request content-length: %s bytes", rid, content_length)
    
    # attach request id for handlers
    try:
        request.state.rid = rid
        request.state.start_time = start
    except Exception as e:
        logger.warning("[%s] Failed to attach request state: %s", rid, e)
    
    try:
        response = await call_next(request)
        dur_ms = (time.perf_counter() - start) * 1000.0
        
        # Log successful response with performance metrics
        status_code = getattr(response, "status_code", "unknown")
        content_length_resp = getattr(response, "headers", {}).get("content-length", "unknown")
        
        logger.info("[%s] Request completed: %s %s -> %s in %.1f ms", 
                   rid, request.method, request.url.path, status_code, dur_ms)
        logger.debug("[%s] Response details: status=%s, content-length=%s, duration=%.1f ms", 
                    rid, status_code, content_length_resp, dur_ms)
        
        # Performance warnings for slow requests
        if dur_ms > 30000:  # 30 seconds
            logger.warning("[%s] Slow request detected: %.1f ms for %s %s", 
                          rid, dur_ms, request.method, request.url.path)
        elif dur_ms > 10000:  # 10 seconds
            logger.info("[%s] Long request: %.1f ms for %s %s", 
                       rid, dur_ms, request.method, request.url.path)
        
        return response
        
    except Exception as e:
        dur_ms = (time.perf_counter() - start) * 1000.0
        logger.exception("[%s] Request failed: %s %s after %.1f ms - %s", 
                        rid, request.method, request.url.path, dur_ms, e)
        logger.debug("[%s] Exception type: %s, args: %s", rid, type(e).__name__, getattr(e, 'args', ()))
        raise


class PrepareAnnotationsRequest(BaseModel):
    # Core IO
    input_dir: Optional[str] = Field(None, description="Directory containing mixed GT and generated videos")
    out_json: Optional[str] = Field(None, description="Output JSON path; ignored if stage_dataset is set")
    generated_suffixes: Optional[str] = Field(
        "synthetic,generated",
        description="Comma-separated suffixes appended to GT basenames (e.g., 'synthetic,generated')",
    )
    stage_dataset: Optional[str] = Field(None, description="Destination root to create dataset layout")
    link: Optional[bool] = Field(False, description="When staging, symlink instead of copy")

    # Metric execution
    compute: Optional[bool] = Field(False, description="Compute metrics after preparing annotations")
    metrics: Optional[str] = Field(
        "all",
        description="CSV of metric names to run: fid,is,fvd,gstvqa,simplevqa,lightvqa+",
    )
    categories: Optional[str] = Field(
        "",
        description="CSV of categories: distribution_based, nn_based_video. Merged with --metrics",
    )
    list_metrics: Optional[bool] = Field(False, description="List available categories and metrics, then exit")

    # Length control
    max_len: Optional[int] = Field(64, description="Max frames per video (overridden by max_seconds if set)")
    max_seconds: Optional[float] = Field(None, description="Max seconds per video; overrides max_len if provided")
    fps: Optional[float] = Field(25.0, description="Frames-per-second assumption used with max_seconds")
    pad: Optional[bool] = Field(False, description="Pad videos to exactly max_len frames")

    # Device / models
    use_cpu: Optional[bool] = Field(False, description="Force CPU even if CUDA is available")
    fvd_model: Optional[str] = Field(None, description="Path to FVD checkpoint")
    gstvqa_model: Optional[str] = Field(None, description="Path to GSTVQA checkpoint")
    simplevqa_model: Optional[str] = Field(None, description="Path to SimpleVQA checkpoint")
    lightvqa_plus_model: Optional[str] = Field(None, description="Path to LightVQA+ checkpoint")
    lightvqa_plus_swin: Optional[str] = Field(None, description="Path to Swin weights for LightVQA+")

    # Forward-compatibility: extra CLI args (list of tokens)
    extra_args: Optional[List[str]] = Field(None, description="Additional raw CLI tokens to pass through")
    
    # CD-FVD specific option
    use_cdfvd: Optional[bool] = Field(False, description="Use cd-fvd package for FVD computation instead of default")
    cdfvd_model: Optional[str] = Field("videomae", description="CD-FVD model to use: 'videomae' or 'i3d'")
    cdfvd_resolution: Optional[int] = Field(128, description="Resolution for CD-FVD video processing")
    cdfvd_sequence_length: Optional[int] = Field(16, description="Sequence length for CD-FVD video processing")


def _build_cli_args(req: PrepareAnnotationsRequest) -> List[str]:
    args: List[str] = []

    if req.list_metrics:
        args.append("--list-metrics")

    if req.input_dir:
        args += ["--input-dir", req.input_dir]
    if req.out_json:
        args += ["--out-json", req.out_json]
    if req.generated_suffixes:
        args += ["--generated-suffixes", req.generated_suffixes]
    if req.stage_dataset:
        args += ["--stage-dataset", req.stage_dataset]
    if req.link:
        args.append("--link")

    if req.compute:
        args.append("--compute")
    if req.metrics:
        args += ["--metrics", req.metrics]
    if req.categories:
        args += ["--categories", req.categories]

    # Length controls
    if req.max_seconds is not None:
        args += ["--max-seconds", str(req.max_seconds)]
        if req.fps is not None:
            args += ["--fps", str(req.fps)]
    elif req.max_len is not None:
        args += ["--max-len", str(req.max_len)]
    if req.pad:
        args.append("--pad")

    # Device & model paths
    if req.use_cpu:
        args.append("--use-cpu")
    if req.fvd_model:
        args += ["--fvd-model", req.fvd_model]
    if req.gstvqa_model:
        args += ["--gstvqa-model", req.gstvqa_model]
    if req.simplevqa_model:
        args += ["--simplevqa-model", req.simplevqa_model]
    if req.lightvqa_plus_model:
        args += ["--lightvqa-plus-model", req.lightvqa_plus_model]
    if req.lightvqa_plus_swin:
        args += ["--lightvqa-plus-swin", req.lightvqa_plus_swin]

    if req.extra_args:
        args += list(req.extra_args)

    return args


def _compute_cdfvd(upload_dir: str, generated_suffixes: str, model: str = "videomae", 
                   resolution: int = 128, sequence_length: int = 16,
                   max_seconds: Optional[float] = None, fps: Optional[float] = 25.0) -> Dict[str, float]:
    """
    Compute FVD using cd-fvd package.
    
    Args:
        upload_dir: Directory containing videos
        generated_suffixes: Comma-separated suffixes for synthetic videos
        model: CD-FVD model type ('videomae' or 'i3d')
        resolution: Video resolution for processing
        sequence_length: Number of frames to process
    
    Returns:
        Dictionary with FVD score
    """
    logger.info("[CD-FVD] Starting FVD computation with model=%s, resolution=%d, seq_len=%d", 
                model, resolution, sequence_length)
    logger.debug("[CD-FVD] Parameters: upload_dir=%s, suffixes=%s, max_seconds=%s, fps=%s", 
                upload_dir, generated_suffixes, max_seconds, fps)
    
    if cdfvd is None:
        logger.error("[CD-FVD] cd-fvd package is not installed")
        raise RuntimeError("cd-fvd package is not installed. Run: pip install cd-fvd")
    
    # Parse suffixes and build a robust checker for synthetic naming
    suffixes = [s.strip() for s in generated_suffixes.split(',') if s.strip()]
    logger.debug("[CD-FVD] Parsed suffixes for synthetic detection: %s", suffixes)
    
    def _is_synthetic_name(name: str) -> bool:
        base = name.lower()
        for s in suffixes:
            tok = s.strip().lower()
            if not tok:
                continue
            if base.endswith("_" + tok) or base.endswith("-" + tok) or base.endswith(tok):
                return True
        return False
    
    logger.info("[CD-FVD] Analyzing video directory: %s", upload_dir)
    print(f"[CD-FVD Debug] Looking for videos in: {upload_dir}")
    print(f"[CD-FVD Debug] Suffixes to identify synthetic videos: {suffixes}")
    
    # Check if directory exists and list contents
    upload_path = Path(upload_dir)
    if not upload_path.exists():
        logger.error("[CD-FVD] Upload directory does not exist: %s", upload_dir)
        raise RuntimeError(f"Upload directory does not exist: {upload_dir}")
    
    try:
        all_files = list(upload_path.glob("*"))
        logger.info("[CD-FVD] Found %d total files in directory", len(all_files))
        logger.debug("[CD-FVD] Directory contents: %s", [f.name for f in all_files])
        print(f"[CD-FVD Debug] Files in directory: {[f.name for f in all_files]}")
    except Exception as e:
        logger.error("[CD-FVD] Failed to list directory contents: %s", e)
        raise RuntimeError(f"Failed to access upload directory: {e}")
    
    # Organize videos into real and fake
    real_videos = []
    fake_videos = []
    
    # Check all video formats
    video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.m4v"]
    all_videos = []
    for ext in video_extensions:
        videos = list(Path(upload_dir).glob(ext))
        all_videos.extend(videos)
        if videos:
            print(f"[CD-FVD Debug] Found {len(videos)} {ext} files")
    
    print(f"[CD-FVD Debug] Total videos found: {len(all_videos)}")
    
    for video_file in all_videos:
        video_name = video_file.stem
        is_synthetic = _is_synthetic_name(video_name)

        print(f"[CD-FVD Debug] Processing: {video_file.name}, stem={video_name}, is_synthetic={is_synthetic}")

        if is_synthetic:
            fake_videos.append(str(video_file))
            print(f"[CD-FVD Debug] Added to fake_videos: {video_file.name}")
        else:
            # For real videos, we add them regardless of whether they have synthetic counterpart
            # The pairing will be handled by CD-FVD itself
            real_videos.append(str(video_file))
            print(f"[CD-FVD Debug] Added to real_videos: {video_file.name}")
    
    if not real_videos or not fake_videos:
        raise ValueError(f"Insufficient videos for FVD computation. Found {len(real_videos)} real and {len(fake_videos)} fake videos")
    
    logger.info("[CD-FVD] Found %d real videos and %d fake videos", len(real_videos), len(fake_videos))
    
    # Create temporary directories for organized videos
    with tempfile.TemporaryDirectory() as temp_dir:
        real_dir = Path(temp_dir) / "real"
        fake_dir = Path(temp_dir) / "fake"
        real_dir.mkdir(exist_ok=True)
        fake_dir.mkdir(exist_ok=True)
        
        # Helper: trim or copy videos according to max_seconds
        def _trim_or_copy(src: str, dst: Path) -> None:
            src_size = os.path.getsize(src) if os.path.exists(src) else 0
            logger.debug("[CD-FVD] Processing video: %s (%.1f MB) -> %s", 
                        os.path.basename(src), src_size / 1024 / 1024, dst.name)
            
            try:
                if max_seconds is not None and max_seconds > 0:
                    logger.debug("[CD-FVD] Trimming video to %.1f seconds: %s", max_seconds, os.path.basename(src))
                    # Prefer stream copy for speed; fall back to re-encode on failure
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", src,
                        "-t", str(float(max_seconds)),
                        "-c", "copy",
                        str(dst),
                    ]
                    logger.debug("[CD-FVD] Trim command: %s", ' '.join(cmd))
                    print(f"[CD-FVD Trim] {' '.join(cmd)}")
                    
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    if proc.returncode != 0 or (not dst.exists() or os.path.getsize(dst) == 0):
                        logger.warning("[CD-FVD] Stream copy failed (rc=%d), trying re-encode for %s", 
                                     proc.returncode, os.path.basename(src))
                        # Fallback: fast re-encode of the clipped segment
                        cmd = [
                            "ffmpeg", "-y",
                            "-i", src,
                            "-t", str(float(max_seconds)),
                            "-an",
                            "-c:v", "libx264",
                            "-preset", "veryfast",
                            "-crf", "23",
                            str(dst),
                        ]
                        logger.debug("[CD-FVD] Fallback re-encode command: %s", ' '.join(cmd))
                        print(f"[CD-FVD Trim Fallback] {' '.join(cmd)}")
                        proc2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                        
                        if proc2.returncode != 0:
                            logger.error("[CD-FVD] Re-encode also failed (rc=%d) for %s: %s", 
                                       proc2.returncode, os.path.basename(src), proc2.stderr[:200])
                        else:
                            logger.info("[CD-FVD] Successfully re-encoded %s", os.path.basename(src))
                    else:
                        logger.info("[CD-FVD] Successfully trimmed with stream copy: %s", os.path.basename(src))
                else:
                    logger.debug("[CD-FVD] Copying video without trimming: %s", os.path.basename(src))
                    shutil.copy2(src, dst)
                    logger.debug("[CD-FVD] Copy completed: %s -> %s", os.path.basename(src), dst.name)
                
                # Verify result
                if dst.exists():
                    dst_size = os.path.getsize(dst)
                    logger.debug("[CD-FVD] Output file created: %s (%.1f MB)", dst.name, dst_size / 1024 / 1024)
                else:
                    logger.error("[CD-FVD] Output file was not created: %s", dst.name)
                    
            except Exception as e:
                logger.error("[CD-FVD] Video processing failed for %s: %s", os.path.basename(src), e)
                print(f"[CD-FVD Trim Error] {e}; copying original instead.")
                try:
                    shutil.copy2(src, dst)
                    logger.warning("[CD-FVD] Fallback copy successful: %s", os.path.basename(src))
                except Exception as e2:
                    logger.error("[CD-FVD] Fallback copy also failed for %s: %s", os.path.basename(src), e2)
                    print(f"[CD-FVD Copy Error] {e2}")

        # Copy/trim videos to temporary directories
        for i, video_path in enumerate(real_videos):
            dest = real_dir / f"video_{i:04d}{Path(video_path).suffix}"
            _trim_or_copy(video_path, dest)

        for i, video_path in enumerate(fake_videos):
            dest = fake_dir / f"video_{i:04d}{Path(video_path).suffix}"
            _trim_or_copy(video_path, dest)
        
        # Initialize CD-FVD evaluator
        device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
        logger.info("[CD-FVD] Using model='%s' on device='%s'", model, device)
        evaluator = cdfvd.cdfvd(model=model, n_real='full', n_fake='full', device=device)
        
        # Load videos
        logger.info("[CD-FVD] Loading real videos from %s", real_dir)
        real_loader = evaluator.load_videos(str(real_dir), data_type='video_folder',
                                           resolution=resolution, sequence_length=sequence_length)
        
        logger.info("[CD-FVD] Loading fake videos from %s", fake_dir)
        fake_loader = evaluator.load_videos(str(fake_dir), data_type='video_folder',
                                           resolution=resolution, sequence_length=sequence_length)
        
        # Compute FVD
        logger.info("[CD-FVD] Computing real video statistics...")
        evaluator.compute_real_stats(real_loader)
        
        logger.info("[CD-FVD] Computing fake video statistics...")
        evaluator.compute_fake_stats(fake_loader)
        
        logger.info("[CD-FVD] Computing FVD score...")
        score = evaluator.compute_fvd_from_stats()
        
        result = {
            "fvd_score": float(score),
            "num_real_videos": len(real_videos),
            "num_fake_videos": len(fake_videos),
            "model": model,
            "resolution": resolution,
            "sequence_length": sequence_length,
            "device": device
        }
        # Attach length info if requested
        try:
            if max_seconds is not None:
                ms = float(max_seconds)
                result["max_seconds"] = ms
                fv = float(fps) if fps is not None else None
                if fv is not None and fv > 0:
                    result["fps"] = fv
                    result["max_len"] = int(round(ms * fv))
        except Exception:
            pass
        
        logger.info("[CD-FVD] FVD Score: %.4f", score)
        return result


def _collect_artifacts(base_dir: str, stdout: str) -> List[dict]:
    """
    Collect known result JSON files produced by the metrics scripts and include their contents.
    """
    logger.info("[Artifacts] Starting artifact collection from directory: %s", base_dir)
    logger.debug("[Artifacts] Stdout length for analysis: %d characters", len(stdout))
    
    candidate_names = [
        "fid_results.json",
        "is_results.json",
        "fvd_results.json",
        "gstvqa_results.json",
        "simplevqa_results.json",
        "lightvqa_plus_results.json",
        "cdfvd_results.json",  # Add CD-FVD results to artifacts
    ]
    
    logger.debug("[Artifacts] Searching for %d candidate artifact files: %s", 
                len(candidate_names), candidate_names)
    
    artifacts: List[dict] = []
    files_found = 0
    total_size = 0
    
    for name in candidate_names:
        path = os.path.join(base_dir, name)
        logger.debug("[Artifacts] Checking for artifact: %s", path)
        
        if os.path.exists(path):
            try:
                file_size = os.path.getsize(path)
                total_size += file_size
                files_found += 1
                logger.info("[Artifacts] Found artifact: %s (%.1f KB)", name, file_size / 1024)
                
                item: dict = {"name": name, "path": path, "size_bytes": file_size}
                
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                        logger.debug("[Artifacts] Read %d characters from %s", len(text), name)
                    
                    try:
                        parsed_json = json.loads(text)
                        item["json"] = parsed_json
                        logger.debug("[Artifacts] Successfully parsed JSON from %s with %d keys", 
                                   name, len(parsed_json) if isinstance(parsed_json, dict) else 1)
                    except json.JSONDecodeError as je:
                        logger.warning("[Artifacts] JSON parsing failed for %s: %s", name, je)
                        item["text"] = text
                        
                except (IOError, OSError) as fe:
                    logger.error("[Artifacts] File read error for %s: %s", name, fe)
                    item["error"] = f"File read error: {fe}"
                except UnicodeDecodeError as ue:
                    logger.error("[Artifacts] Unicode decode error for %s: %s", name, ue)
                    item["error"] = f"Unicode decode error: {ue}"
                except Exception as e:
                    logger.error("[Artifacts] Unexpected error reading %s: %s", name, e)
                    item["error"] = f"Unexpected error: {e}"
                    
                artifacts.append(item)
                
            except Exception as e:
                logger.error("[Artifacts] Failed to process artifact %s: %s", name, e)
                artifacts.append({"name": name, "path": path, "error": f"Processing error: {e}"})
        else:
            logger.debug("[Artifacts] Artifact not found: %s", name)
    
    logger.info("[Artifacts] Collection summary: %d/%d files found, total size %.1f KB", 
                files_found, len(candidate_names), total_size / 1024)
    
    # Best-effort: detect staged dataset path from stdout
    try:
        logger.debug("[Artifacts] Analyzing stdout for staged dataset information")
        m = re.search(r"Staged dataset at:\s*(.+)", stdout)
        if m:
            staged_path = m.group(1).strip()
            logger.info("[Artifacts] Found staged dataset path in stdout: %s", staged_path)
            artifacts.append({
                "name": "stage_info.txt", 
                "path": None, 
                "text": f"staged_dataset: {staged_path}",
                "source": "stdout_analysis"
            })
        else:
            logger.debug("[Artifacts] No staged dataset path found in stdout")
    except Exception as e:
        logger.warning("[Artifacts] Failed to analyze stdout for staged dataset: %s", e)
    
    logger.info("[Artifacts] Final artifact count: %d items", len(artifacts))
    return artifacts


@app.get("/healthz")
def healthz(request: Request):
    rid = getattr(getattr(request, "state", object()), "rid", "-")
    logger.info("[%s] Health check requested", rid)
    
    cuda_available = False
    cuda_version: Optional[str] = None
    device_count = 0
    torch_version: Optional[str] = None
    torch_error: Optional[str] = None

    logger.debug("[%s] Checking torch and CUDA availability", rid)
    try:
        if torch is not None:
            torch_version = getattr(torch, "__version__", None)
            logger.debug("[%s] Torch version detected: %s", rid, torch_version)
            
            cuda_available = bool(torch.cuda.is_available())
            logger.debug("[%s] CUDA available: %s", rid, cuda_available)
            
            if cuda_available:
                cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
                device_count = torch.cuda.device_count()
                logger.debug("[%s] CUDA version: %s, device count: %d", rid, cuda_version, device_count)
            else:
                logger.debug("[%s] CUDA not available, skipping device enumeration", rid)
        else:
            torch_error = "torch not importable"
            logger.warning("[%s] Torch is not importable", rid)
    except Exception as e:  # be resilient: health should not fail
        torch_error = str(e)
        logger.error("[%s] Error checking torch/CUDA: %s", rid, e)

    # Check additional system information
    logger.debug("[%s] Gathering system information", rid)
    script_exists = os.path.exists(SCRIPT_PATH)
    current_dir = os.getcwd()
    
    logger.debug("[%s] Script exists at %s: %s", rid, SCRIPT_PATH, script_exists)
    logger.debug("[%s] Current working directory: %s", rid, current_dir)

    info = {
        "status": "ok",
        "python": sys.version,
        "cwd": current_dir,
        "script_exists": script_exists,
        "script_path": SCRIPT_PATH,
        "torch": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_count": device_count,
        "torch_error": torch_error,
    }
    
    logger.info("[%s] Health check complete: torch=%s, cuda=%s, devices=%d, script_exists=%s", 
                rid, torch_version, cuda_available, device_count, script_exists)
    return info


@app.get("/help")
def cli_help(request: Request):
    rid = getattr(getattr(request, "state", object()), "rid", "-")
    logger.info("[%s] Help request initiated", rid)
    
    logger.debug("[%s] Checking script existence: %s", rid, SCRIPT_PATH)
    if not os.path.exists(SCRIPT_PATH):
        logger.error("[%s] Script not found at path: %s", rid, SCRIPT_PATH)
        raise HTTPException(status_code=500, detail=f"Script not found at {SCRIPT_PATH}")
    
    cmd = [sys.executable, SCRIPT_PATH, "--help"]
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    
    logger.info("[%s] Executing help command: %s", rid, cmd_str)
    logger.debug("[%s] Working directory: %s", rid, APP_ROOT)
    
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=APP_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        dur = (time.perf_counter() - t0) * 1000.0
        
        stdout_len = len(proc.stdout or "")
        stderr_len = len(proc.stderr or "")
        
        logger.info("[%s] Help command completed: rc=%d, duration=%.1f ms", rid, proc.returncode, dur)
        logger.debug("[%s] Command output: stdout=%d bytes, stderr=%d bytes", rid, stdout_len, stderr_len)
        
        if proc.returncode != 0:
            logger.warning("[%s] Help command returned non-zero exit code: %d", rid, proc.returncode)
            if proc.stderr:
                logger.debug("[%s] Help command stderr preview: %s", rid, proc.stderr[:200])
                
    except Exception as e:
        dur = (time.perf_counter() - t0) * 1000.0
        logger.error("[%s] Failed to execute help command after %.1f ms: %s", rid, dur, e)
        raise HTTPException(status_code=500, detail=f"Failed to execute: {e}")
    
    response = {
        "cmd": cmd_str,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_ms": dur,
    }
    
    # Attach artifacts (result JSON files) with detailed logging
    logger.debug("[%s] Collecting artifacts from help command", rid)
    try:
        artifacts = _collect_artifacts(APP_ROOT, proc.stdout or "")
        response["artifacts"] = artifacts
        logger.info("[%s] Help artifacts collected: %d items", rid, len(artifacts))
    except Exception as e:
        logger.error("[%s] Failed to collect help artifacts: %s", rid, e)
        response["artifact_error"] = str(e)
    
    logger.info("[%s] Help request completed successfully", rid)
    return response


@app.post("/run_upload")
def run_upload(
    request: Request,
    # Files
    videos: List[UploadFile] = File(..., description="Video files (.mp4, .mov, etc.)"),
    # Core options (form fields)
    generated_suffixes: str = Form("synthetic,generated"),
    stage_dataset: Optional[str] = Form(None),
    link: bool = Form(False),
    # Metric execution
    compute: bool = Form(True),
    metrics: str = Form(""),
    categories: str = Form("distribution_based"),
    # Length control
    max_len: int = Form(64),
    max_seconds: Optional[float] = Form(None),
    fps: float = Form(25.0),
    pad: bool = Form(False),
    # Device
    use_cpu: bool = Form(False),
    # CD-FVD options
    use_cdfvd: bool = Form(False),
    cdfvd_model: str = Form("videomae"),
    cdfvd_resolution: int = Form(128),
    cdfvd_sequence_length: int = Form(16),
):
    """
    Accept uploaded video files and run the same pipeline as /run using a
    temporary session directory. Returns stdout/stderr plus artifacts.
    """
    rid = getattr(getattr(request, "state", object()), "rid", "-")
    t0 = time.perf_counter()
    logger.info(
        "[%s] /run_upload received %d file(s); generated_suffixes=%s categories=%s metrics=%s compute=%s",
        rid,
        len(videos or []),
        generated_suffixes,
        categories,
        metrics,
        bool(compute),
    )
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=500, detail=f"Script not found at {SCRIPT_PATH}")

    # Create a unique session directory under the app root
    session_id = str(uuid.uuid4())
    upload_dir = os.path.join(APP_ROOT, "uploads", session_id)
    os.makedirs(upload_dir, exist_ok=True)

    allowed_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    saved_files: List[str] = []
    for uf in videos:
        try:
            name = os.path.basename(uf.filename or "video")
            ext = os.path.splitext(name)[1].lower()
            if ext and ext not in allowed_exts:
                # Skip unknown extensions; do not fail entire request
                logger.warning("[%s] Skipping upload with unsupported extension: %s", rid, name)
                continue
            dest_path = os.path.join(upload_dir, name)
            # Ensure parent exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as out_f:
                shutil.copyfileobj(uf.file, out_f)
            saved_files.append(name)
            logger.info("[%s] Saved upload -> %s", rid, dest_path)
        except Exception as e:
            # Continue saving others; report error later
            saved_files.append(f"ERROR:{getattr(uf, 'filename', 'unknown')}: {e}")
            logger.exception("[%s] Error saving uploaded file %s: %s", rid, getattr(uf, 'filename', 'unknown'), e)

    # Determine stage dataset dir
    if stage_dataset:
        stage_dir = stage_dataset if os.path.isabs(stage_dataset) else os.path.join(APP_ROOT, stage_dataset)
    else:
        stage_dir = os.path.join(upload_dir, "staged")
    logger.info("[%s] Session=%s upload_dir=%s stage_dir=%s saved=%d", rid, session_id, upload_dir, stage_dir, len(saved_files))

    # Build args via existing request model helper
    # If use_cdfvd is True, only stage the dataset, don't compute legacy metrics
    req = PrepareAnnotationsRequest(
        input_dir=upload_dir,
        generated_suffixes=generated_suffixes,
        stage_dataset=stage_dir,
        link=link,
        compute=compute if not use_cdfvd else False,  # Skip legacy metrics when using CD-FVD
        metrics=(metrics or None),
        categories=(categories or None),
        max_len=max_len,
        max_seconds=max_seconds,
        fps=fps,
        pad=pad,
        use_cpu=use_cpu,
    )
    args = _build_cli_args(req)
    cmd = [sys.executable, SCRIPT_PATH] + args
    logger.info("[%s] Exec: %s", rid, " ".join(shlex.quote(c) for c in cmd))

    try:
        proc = subprocess.run(
            cmd,
            cwd=APP_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute: {e}")
    dur = (time.perf_counter() - t0) * 1000.0
    logger.info("[%s] /run_upload rc=%s in %.1f ms (stdout=%dB, stderr=%dB)", rid, proc.returncode, dur, len(proc.stdout or ""), len(proc.stderr or ""))

    response = {
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "session": {"id": session_id, "upload_dir": upload_dir, "stage_dir": stage_dir, "files": saved_files},
    }
    
    # Compute CD-FVD by default with all available models
    try:
        logger.info("[%s] Computing FVD using cd-fvd package with all models...", rid)
        # Check both possible locations for staged videos
        video_dir = os.path.join(stage_dir, "evaluate")
        if not os.path.exists(video_dir):
            print(f"[CD-FVD] evaluate dir not found, checking stage_dir directly: {stage_dir}")
            video_dir = stage_dir
        
        # List contents to debug
        print(f"[CD-FVD] Looking for videos in: {video_dir}")
        if os.path.exists(video_dir):
            files = os.listdir(video_dir)
            print(f"[CD-FVD] Files found: {files}")
            # Check subdirectories
            for item in files:
                item_path = os.path.join(video_dir, item)
                if os.path.isdir(item_path):
                    subfiles = os.listdir(item_path)
                    print(f"[CD-FVD] Subdirectory {item}: {subfiles}")
        
        # Compute CD-FVD with all available models
        cdfvd_results = {}
        models = ["videomae", "i3d"]
        
        for model in models:
            try:
                logger.info("[%s] Computing CD-FVD with model: %s", rid, model)
                cdfvd_result = _compute_cdfvd(
                    upload_dir=video_dir,
                    generated_suffixes=generated_suffixes,
                    model=model,
                    resolution=cdfvd_resolution or 128,
                    sequence_length=cdfvd_sequence_length or 16,
                    max_seconds=max_seconds,
                    fps=fps,
                )
                cdfvd_results[model] = cdfvd_result
                logger.info("[%s] CD-FVD %s score: %.4f", rid, model, cdfvd_result.get("fvd_score", 0))
            except Exception as e:
                logger.error("[%s] CD-FVD %s computation error: %s", rid, model, e)
                cdfvd_results[model] = {"error": str(e)}
        
        response["cdfvd_results"] = cdfvd_results
        
        # Save CD-FVD results
        cdfvd_json_path = os.path.join(stage_dir, "cdfvd_results.json")
        with open(cdfvd_json_path, "w") as f:
            json.dump(cdfvd_results, f, indent=2)
        
        # Return CD-FVD artifacts along with any legacy artifacts
        cdfvd_artifacts = [
            {
                "name": "cdfvd_results.json",
                "path": cdfvd_json_path,
                "json": cdfvd_results
            }
        ]
        
        # Also collect legacy artifacts
        try:
            legacy_arts = _collect_artifacts(APP_ROOT, proc.stdout or "")
            response["artifacts"] = cdfvd_artifacts + legacy_arts
            logger.info("[%s] CD-FVD + legacy artifacts: %d total", rid, len(response["artifacts"]))
        except Exception as e:
            response["artifacts"] = cdfvd_artifacts
            response["artifact_error"] = str(e)
            logger.warning("[%s] Legacy artifact collection error: %s", rid, e)
        
    except Exception as e:
        response["cdfvd_error"] = str(e)
        logger.error("[%s] CD-FVD computation error: %s", rid, e)
        # Still collect legacy artifacts on CD-FVD failure
        try:
            arts = _collect_artifacts(APP_ROOT, proc.stdout or "")
            response["artifacts"] = arts
            if arts:
                logger.info("[%s] Legacy artifacts: %s", rid, ", ".join(a.get("name", "?") for a in arts))
        except Exception as e2:
            response["artifact_error"] = str(e2)
            logger.warning("[%s] Legacy artifact collection error: %s", rid, e2)
    return response


@app.post("/run")
def run_prepare(req: PrepareAnnotationsRequest, request: Request):
    if not req.list_metrics and not req.input_dir:
        raise HTTPException(status_code=422, detail="input_dir is required unless list_metrics is true")

    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=500, detail=f"Script not found at {SCRIPT_PATH}")

    rid = getattr(getattr(request, "state", object()), "rid", "-")
    
    # Ensure input_dir exists (with comprehensive permission fallback)
    logger.info("[%s] Processing input_dir: %s", rid, req.input_dir)
    
    if req.input_dir:
        if not os.path.exists(req.input_dir):
            logger.info("[%s] Input directory does not exist, creating: %s", rid, req.input_dir)
            try:
                os.makedirs(req.input_dir, exist_ok=True)
                logger.info("[%s] Successfully created input_dir: %s", rid, req.input_dir)
            except PermissionError as e:
                # Fall back to writable container location
                import tempfile
                try:
                    fallback_dir = tempfile.mkdtemp(prefix="aigve_input_", dir="/tmp")
                    logger.warning("[%s] Permission denied creating %s (%s), using fallback: %s", rid, req.input_dir, e, fallback_dir)
                    req.input_dir = fallback_dir
                    logger.info("[%s] Fallback directory created successfully: %s", rid, fallback_dir)
                except Exception as fallback_e:
                    logger.error("[%s] Critical: Failed to create fallback directory: %s", rid, fallback_e)
                    # Continue anyway with original path
            except Exception as e:
                logger.error("[%s] Unexpected error creating input_dir %s: %s", rid, req.input_dir, e)
                # Continue anyway
        else:
            logger.info("[%s] Input directory exists, checking write permissions: %s", rid, req.input_dir)
            # Check if we can write to the existing directory
            test_file = os.path.join(req.input_dir, ".write_test")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                logger.info("[%s] Input directory is writable: %s", rid, req.input_dir)
            except (PermissionError, OSError) as e:
                logger.warning("[%s] Input directory not writable (%s), initiating file copy to fallback", rid, e)
                # Directory exists but not writable - copy files to fallback
                import tempfile
                import shutil
                
                try:
                    fallback_dir = tempfile.mkdtemp(prefix="aigve_input_", dir="/tmp")
                    logger.info("[%s] Created fallback directory: %s", rid, fallback_dir)
                    
                    # Copy all files from original to fallback
                    copied_files = 0
                    failed_files = 0
                    
                    try:
                        items = os.listdir(req.input_dir)
                        logger.info("[%s] Found %d items to copy from %s", rid, len(items), req.input_dir)
                        
                        for item in items:
                            src = os.path.join(req.input_dir, item)
                            dst = os.path.join(fallback_dir, item)
                            
                            try:
                                if os.path.isfile(src):
                                    shutil.copy2(src, dst)
                                    logger.info("[%s] Successfully copied file: %s -> %s", rid, item, dst)
                                    copied_files += 1
                                elif os.path.isdir(src):
                                    shutil.copytree(src, dst)
                                    logger.info("[%s] Successfully copied directory: %s -> %s", rid, item, dst)
                                    copied_files += 1
                                else:
                                    logger.warning("[%s] Skipping unknown item type: %s", rid, src)
                            except Exception as item_e:
                                logger.error("[%s] Failed to copy %s: %s", rid, src, item_e)
                                failed_files += 1
                                # Continue copying other files
                        
                        logger.info("[%s] Copy summary: %d successful, %d failed", rid, copied_files, failed_files)
                        req.input_dir = fallback_dir
                        logger.info("[%s] Updated input_dir to fallback: %s", rid, fallback_dir)
                        
                    except Exception as list_e:
                        logger.error("[%s] Failed to list directory contents %s: %s", rid, req.input_dir, list_e)
                        # Continue with fallback directory anyway
                        req.input_dir = fallback_dir
                        
                except Exception as fallback_e:
                    logger.error("[%s] Critical: Failed to create input fallback directory: %s", rid, fallback_e)
                    # Continue with original directory anyway
            except Exception as test_e:
                logger.error("[%s] Unexpected error testing write permissions for %s: %s", rid, req.input_dir, test_e)
                # Continue anyway
    
    # Ensure stage_dataset exists if specified (with comprehensive permission fallback)
    logger.info("[%s] Processing stage_dataset: %s", rid, req.stage_dataset)
    
    if req.stage_dataset:
        if not os.path.exists(req.stage_dataset):
            logger.info("[%s] Stage dataset directory does not exist, creating: %s", rid, req.stage_dataset)
            try:
                os.makedirs(req.stage_dataset, exist_ok=True)
                logger.info("[%s] Successfully created stage_dataset: %s", rid, req.stage_dataset)
            except PermissionError as e:
                # Fall back to writable container location
                import tempfile
                try:
                    fallback_dir = tempfile.mkdtemp(prefix="aigve_stage_", dir="/tmp")
                    logger.warning("[%s] Permission denied creating %s (%s), using fallback: %s", rid, req.stage_dataset, e, fallback_dir)
                    req.stage_dataset = fallback_dir
                    logger.info("[%s] Stage fallback directory created successfully: %s", rid, fallback_dir)
                except Exception as fallback_e:
                    logger.error("[%s] Critical: Failed to create stage fallback directory: %s", rid, fallback_e)
                    # Continue anyway with original path
            except Exception as e:
                logger.error("[%s] Unexpected error creating stage_dataset %s: %s", rid, req.stage_dataset, e)
                # Continue anyway
        else:
            logger.info("[%s] Stage dataset directory exists, checking write permissions: %s", rid, req.stage_dataset)
            # Check if we can write to the existing directory
            test_file = os.path.join(req.stage_dataset, ".write_test")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                logger.info("[%s] Stage dataset directory is writable: %s", rid, req.stage_dataset)
            except (PermissionError, OSError) as e:
                logger.warning("[%s] Stage dataset directory not writable (%s), initiating file copy to fallback", rid, e)
                # Directory exists but not writable - copy files to fallback
                import tempfile
                import shutil
                
                try:
                    fallback_dir = tempfile.mkdtemp(prefix="aigve_stage_", dir="/tmp")
                    logger.info("[%s] Created stage fallback directory: %s", rid, fallback_dir)
                    
                    # Copy all files from original to fallback
                    copied_files = 0
                    failed_files = 0
                    
                    try:
                        items = os.listdir(req.stage_dataset)
                        logger.info("[%s] Found %d items to copy from stage dataset %s", rid, len(items), req.stage_dataset)
                        
                        for item in items:
                            src = os.path.join(req.stage_dataset, item)
                            dst = os.path.join(fallback_dir, item)
                            
                            try:
                                if os.path.isfile(src):
                                    shutil.copy2(src, dst)
                                    logger.info("[%s] Successfully copied stage file: %s -> %s", rid, item, dst)
                                    copied_files += 1
                                elif os.path.isdir(src):
                                    shutil.copytree(src, dst)
                                    logger.info("[%s] Successfully copied stage directory: %s -> %s", rid, item, dst)
                                    copied_files += 1
                                else:
                                    logger.warning("[%s] Skipping unknown stage item type: %s", rid, src)
                            except Exception as item_e:
                                logger.error("[%s] Failed to copy stage item %s: %s", rid, src, item_e)
                                failed_files += 1
                                # Continue copying other files
                        
                        logger.info("[%s] Stage copy summary: %d successful, %d failed", rid, copied_files, failed_files)
                        req.stage_dataset = fallback_dir
                        logger.info("[%s] Updated stage_dataset to fallback: %s", rid, fallback_dir)
                        
                    except Exception as list_e:
                        logger.error("[%s] Failed to list stage directory contents %s: %s", rid, req.stage_dataset, list_e)
                        # Continue with fallback directory anyway
                        req.stage_dataset = fallback_dir
                        
                except Exception as fallback_e:
                    logger.error("[%s] Critical: Failed to create stage fallback directory: %s", rid, fallback_e)
                    # Continue with original directory anyway
            except Exception as test_e:
                logger.error("[%s] Unexpected error testing stage write permissions for %s: %s", rid, req.stage_dataset, test_e)
                # Continue anyway

    # Debug: Show final directories and their contents with comprehensive error handling
    try:
        if req.input_dir:
            if os.path.exists(req.input_dir):
                try:
                    files = os.listdir(req.input_dir)
                    logger.info("[%s] Final input_dir contents: %s -> %d items: %s", rid, req.input_dir, len(files), files)
                except Exception as e:
                    logger.error("[%s] Failed to list final input_dir contents %s: %s", rid, req.input_dir, e)
            else:
                logger.warning("[%s] Final input_dir does not exist: %s", rid, req.input_dir)
        else:
            logger.info("[%s] No input_dir specified", rid)
            
        if req.stage_dataset:
            if os.path.exists(req.stage_dataset):
                try:
                    files = os.listdir(req.stage_dataset)
                    logger.info("[%s] Final stage_dataset contents: %s -> %d items: %s", rid, req.stage_dataset, len(files), files)
                except Exception as e:
                    logger.error("[%s] Failed to list final stage_dataset contents %s: %s", rid, req.stage_dataset, e)
            else:
                logger.warning("[%s] Final stage_dataset does not exist: %s", rid, req.stage_dataset)
        else:
            logger.info("[%s] No stage_dataset specified", rid)
    except Exception as e:
        logger.error("[%s] Unexpected error in final directory debug: %s", rid, e)

    t0 = time.perf_counter()
    logger.info("[%s] /run input_dir=%s stage_dataset=%s compute=%s categories=%s metrics=%s", rid, req.input_dir, req.stage_dataset, bool(req.compute), req.categories, req.metrics)
    args = _build_cli_args(req)
    cmd = [sys.executable, SCRIPT_PATH] + args
    logger.info("[%s] Exec: %s", rid, " ".join(shlex.quote(c) for c in cmd))

    # Retry mechanism for prepare_annotations.py with extensive logging
    max_retries = 3
    proc = None
    logger.info("[%s] Starting script execution with %d max retries", rid, max_retries)
    
    for attempt in range(max_retries):
        attempt_start = time.perf_counter()
        logger.info("[%s] Attempt %d/%d: Executing script", rid, attempt + 1, max_retries)
        logger.debug("[%s] Attempt %d/%d: Command: %s", rid, attempt + 1, max_retries, " ".join(shlex.quote(c) for c in cmd))
        logger.debug("[%s] Attempt %d/%d: Working directory: %s", rid, attempt + 1, max_retries, APP_ROOT)
        
        try:
            proc = subprocess.run(
                cmd,
                cwd=APP_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            
            attempt_duration = (time.perf_counter() - attempt_start) * 1000.0
            logger.info("[%s] Attempt %d/%d: Completed in %.1f ms with return code %s", rid, attempt + 1, max_retries, attempt_duration, proc.returncode)
            logger.debug("[%s] Attempt %d/%d: STDOUT length: %d bytes", rid, attempt + 1, max_retries, len(proc.stdout or ""))
            logger.debug("[%s] Attempt %d/%d: STDERR length: %d bytes", rid, attempt + 1, max_retries, len(proc.stderr or ""))
            
            if proc.stdout:
                stdout_preview = proc.stdout[:200] + ("..." if len(proc.stdout) > 200 else "")
                logger.debug("[%s] Attempt %d/%d: STDOUT preview: %s", rid, attempt + 1, max_retries, stdout_preview)
            
            if proc.stderr:
                stderr_preview = proc.stderr[:200] + ("..." if len(proc.stderr) > 200 else "")
                logger.debug("[%s] Attempt %d/%d: STDERR preview: %s", rid, attempt + 1, max_retries, stderr_preview)
            
            if proc.returncode == 0:
                logger.info("[%s] Attempt %d/%d: SUCCESS - Script completed successfully", rid, attempt + 1, max_retries)
                break
            else:
                logger.warning("[%s] Attempt %d/%d: FAILURE - Return code %s", rid, attempt + 1, max_retries, proc.returncode)
                if proc.stderr:
                    logger.warning("[%s] Attempt %d/%d: Error details: %s", rid, attempt + 1, max_retries, proc.stderr.strip())
                
                if attempt < max_retries - 1:
                    logger.info("[%s] Attempt %d/%d: Retrying in 1 second...", rid, attempt + 1, max_retries)
                    time.sleep(1)
                else:
                    logger.error("[%s] All %d attempts failed - script execution unsuccessful", rid, max_retries)
                    
        except Exception as e:
            attempt_duration = (time.perf_counter() - attempt_start) * 1000.0
            logger.error("[%s] Attempt %d/%d: EXCEPTION after %.1f ms: %s", rid, attempt + 1, max_retries, attempt_duration, e)
            logger.debug("[%s] Attempt %d/%d: Exception type: %s", rid, attempt + 1, max_retries, type(e).__name__)
            
            if attempt == max_retries - 1:
                logger.critical("[%s] Critical failure: All attempts failed with exceptions", rid)
                raise HTTPException(status_code=500, detail=f"Failed to execute after {max_retries} attempts: {e}")
            else:
                logger.info("[%s] Attempt %d/%d: Retrying after exception in 1 second...", rid, attempt + 1, max_retries)
                time.sleep(1)
    
    dur = (time.perf_counter() - t0) * 1000.0
    logger.info("[%s] /run rc=%s in %.1f ms (stdout=%dB, stderr=%dB)", rid, proc.returncode, dur, len(proc.stdout or ""), len(proc.stderr or ""))

    # Build comprehensive response object with extensive logging
    logger.info("[%s] Building response object", rid)
    response = {
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    logger.debug("[%s] Response object created with keys: %s", rid, list(response.keys()))
    
    # Analyze script execution results with detailed logging
    if proc.returncode == 0:
        logger.info("[%s] Script execution completed successfully - proceeding to CD-FVD", rid)
    else:
        logger.warning("[%s] Script execution failed after %d attempts (returncode=%s) - continuing with CD-FVD computation", rid, max_retries, proc.returncode)
        if proc.stderr:
            logger.error("[%s] Script failure details: %s", rid, proc.stderr.strip())
        if proc.stdout:
            logger.debug("[%s] Script stdout despite failure: %s", rid, proc.stdout.strip())
    
    # Compute CD-FVD by default with all available models with extensive logging
    if req.input_dir:
        logger.info("[%s] Starting CD-FVD computation phase", rid)
        logger.debug("[%s] CD-FVD input_dir: %s", rid, req.input_dir)
        logger.debug("[%s] CD-FVD stage_dataset: %s", rid, req.stage_dataset)
        
        cdfvd_start_time = time.perf_counter()
        try:
            logger.info("[%s] Initializing CD-FVD computation with all available models", rid)
            
            # Determine video directory - prioritize staged locations, create if needed
            logger.info("[%s] Determining optimal video directory for CD-FVD", rid)
            video_dir = None
            if req.stage_dataset:
                logger.debug("[%s] Checking staged dataset path: %s", rid, req.stage_dataset)
                # Try staged evaluate directory first
                staged_evaluate_dir = os.path.join(req.stage_dataset, "evaluate")
                logger.debug("[%s] Checking staged evaluate directory: %s", rid, staged_evaluate_dir)
                
                if os.path.exists(staged_evaluate_dir):
                    video_dir = staged_evaluate_dir
                    logger.info("[%s] Found existing staged evaluate directory: %s", rid, video_dir)
                    print(f"[CD-FVD] Using staged evaluate directory: {video_dir}")
                else:
                    logger.info("[%s] Staged evaluate directory missing, attempting to create: %s", rid, staged_evaluate_dir)
                    # Create staged evaluate directory if it doesn't exist
                    try:
                        os.makedirs(staged_evaluate_dir, exist_ok=True)
                        video_dir = staged_evaluate_dir
                        logger.info("[%s] Successfully created staged evaluate directory: %s", rid, video_dir)
                        print(f"[CD-FVD] Created and using staged evaluate directory: {video_dir}")
                    except PermissionError as e:
                        logger.warning("[%s] Permission denied creating staged evaluate directory: %s", rid, e)
                        # Fall back to writable temporary directory
                        import tempfile
                        try:
                            fallback_dir = tempfile.mkdtemp(prefix="aigve_cdfvd_staged_", dir="/tmp")
                            logger.warning("[%s] Using CD-FVD staged fallback: %s", rid, fallback_dir)
                            video_dir = fallback_dir
                            print(f"[CD-FVD] Using permission fallback directory: {video_dir}")
                        except Exception as fallback_e:
                            logger.error("[%s] Failed to create CD-FVD staged fallback: %s", rid, fallback_e)
                    except Exception as e:
                        logger.error("[%s] Unexpected error creating staged evaluate directory: %s", rid, e)
                        # Fall back to stage_dataset root
                        if os.path.exists(req.stage_dataset):
                            video_dir = req.stage_dataset
                            logger.info("[%s] Falling back to stage_dataset root: %s", rid, video_dir)
                            print(f"[CD-FVD] Using staged dataset directory: {video_dir}")
                        else:
                            logger.warning("[%s] Stage_dataset root does not exist: %s", rid, req.stage_dataset)
                            print(f"[CD-FVD] Staged dataset directory does not exist: {req.stage_dataset}")
            
            # Fall back to input_dir if no staging was requested or staging failed
            if video_dir is None:
                logger.info("[%s] No staged directory available, checking input_dir: %s", rid, req.input_dir)
                if os.path.exists(req.input_dir):
                    video_dir = req.input_dir
                    logger.info("[%s] Using existing input directory for CD-FVD: %s", rid, video_dir)
                    print(f"[CD-FVD] Using input directory: {video_dir}")
                else:
                    logger.warning("[%s] Input directory does not exist, attempting to create: %s", rid, req.input_dir)
                    # Create input directory if it doesn't exist
                    try:
                        os.makedirs(req.input_dir, exist_ok=True)
                        video_dir = req.input_dir
                        logger.info("[%s] Successfully created input directory for CD-FVD: %s", rid, video_dir)
                        print(f"[CD-FVD] Created and using input directory: {video_dir}")
                    except PermissionError as e:
                        logger.error("[%s] Permission denied creating input_dir, using fallback: %s", rid, e)
                        # Fall back to writable temporary directory due to Docker permissions
                        import tempfile
                        try:
                            video_dir = tempfile.mkdtemp(prefix="aigve_cdfvd_input_", dir="/tmp")
                            logger.warning("[%s] Using CD-FVD input fallback directory: %s", rid, video_dir)
                            print(f"[CD-FVD] Using permission fallback directory: {video_dir}")
                        except Exception as fallback_e:
                            logger.critical("[%s] Failed to create CD-FVD input fallback: %s", rid, fallback_e)
                    except Exception as e:
                        logger.error("[%s] Unexpected error creating input_dir, using temporary: %s", rid, e)
                        # As last resort, create a temporary directory
                        import tempfile
                        try:
                            video_dir = tempfile.mkdtemp(prefix="aigve_cdfvd_", dir="/tmp")
                            logger.warning("[%s] Using CD-FVD temporary directory: %s", rid, video_dir)
                            print(f"[CD-FVD] Using temporary directory: {video_dir}")
                        except Exception as temp_e:
                            logger.critical("[%s] Failed to create CD-FVD temporary directory: %s", rid, temp_e)
            
            # Ensure video directory exists (final safety check with permission handling)
            logger.debug("[%s] Performing final video directory safety check: %s", rid, video_dir)
            if not os.path.exists(video_dir):
                logger.warning("[%s] Video directory missing, attempting final creation: %s", rid, video_dir)
                try:
                    os.makedirs(video_dir, exist_ok=True)
                    logger.info("[%s] Successfully created video directory in final check: %s", rid, video_dir)
                except PermissionError as e:
                    logger.error("[%s] Permission denied in final safety check, creating final fallback: %s", rid, e)
                    # Final fallback to guaranteed writable location
                    import tempfile
                    try:
                        video_dir = tempfile.mkdtemp(prefix="aigve_cdfvd_final_", dir="/tmp")
                        logger.warning("[%s] Using final fallback directory: %s", rid, video_dir)
                    except Exception as final_e:
                        logger.critical("[%s] Critical: Failed to create final fallback directory: %s", rid, final_e)
                except Exception as e:
                    logger.error("[%s] Unexpected error in final directory creation: %s", rid, e)
            else:
                logger.info("[%s] Video directory exists and ready for CD-FVD: %s", rid, video_dir)
            
            # List contents to debug with comprehensive error handling
            logger.info("[%s] Analyzing video directory contents for CD-FVD", rid)
            print(f"[CD-FVD] Looking for videos in: {video_dir}")
            
            try:
                files = os.listdir(video_dir)
                logger.info("[%s] Found %d items in video directory", rid, len(files))
                print(f"[CD-FVD] Files found: {files}")
                
                # Check subdirectories with detailed logging
                video_files = []
                subdirs = []
                for item in files:
                    item_path = os.path.join(video_dir, item)
                    try:
                        if os.path.isdir(item_path):
                            subfiles = os.listdir(item_path)
                            subdirs.append(item)
                            logger.debug("[%s] Subdirectory %s contains %d items: %s", rid, item, len(subfiles), subfiles)
                            print(f"[CD-FVD] Subdirectory {item}: {subfiles}")
                        elif os.path.isfile(item_path) and item.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                            video_files.append(item)
                            logger.debug("[%s] Found video file: %s", rid, item)
                    except Exception as item_e:
                        logger.error("[%s] Error checking item %s: %s", rid, item, item_e)
                
                logger.info("[%s] Video analysis complete: %d video files, %d subdirectories", rid, len(video_files), len(subdirs))
            except Exception as list_e:
                logger.error("[%s] Failed to list video directory contents: %s", rid, list_e)
                files = []
            
            # Compute CD-FVD with all available models with extensive logging
            logger.info("[%s] Starting CD-FVD model computation phase", rid)
            cdfvd_results = {}
            models = ["videomae", "i3d"]
            logger.info("[%s] CD-FVD models to compute: %s", rid, models)
            
            for model_idx, model in enumerate(models):
                model_start_time = time.perf_counter()
                logger.info("[%s] Computing CD-FVD with model %d/%d: %s", rid, model_idx + 1, len(models), model)
                logger.debug("[%s] CD-FVD %s parameters: dir=%s, suffixes=%s, resolution=%s, seq_len=%s", 
                           rid, model, video_dir, req.generated_suffixes or "synthetic,generated", 
                           req.cdfvd_resolution or 128, req.cdfvd_sequence_length or 16)
                
                try:
                    cdfvd_result = _compute_cdfvd(
                        upload_dir=video_dir,
                        generated_suffixes=req.generated_suffixes or "synthetic,generated",
                        model=model,
                        resolution=req.cdfvd_resolution or 128,
                        sequence_length=req.cdfvd_sequence_length or 16,
                        max_seconds=req.max_seconds,
                        fps=req.fps,
                    )
                    
                    model_duration = (time.perf_counter() - model_start_time) * 1000.0
                    cdfvd_results[model] = cdfvd_result
                    
                    if "fvd_score" in cdfvd_result:
                        logger.info("[%s] CD-FVD %s completed in %.1f ms - Score: %.4f", 
                                  rid, model, model_duration, cdfvd_result["fvd_score"])
                    else:
                        logger.warning("[%s] CD-FVD %s completed in %.1f ms - No score returned", 
                                     rid, model, model_duration)
                    
                    logger.debug("[%s] CD-FVD %s result keys: %s", rid, model, list(cdfvd_result.keys()))
                    
                except Exception as model_e:
                    model_duration = (time.perf_counter() - model_start_time) * 1000.0
                    logger.error("[%s] CD-FVD %s failed after %.1f ms: %s", rid, model, model_duration, model_e)
                    logger.debug("[%s] CD-FVD %s exception type: %s", rid, model, type(model_e).__name__)
                    # Re-raise the exception since we removed individual model error handling
                    raise
            
            response["cdfvd_results"] = cdfvd_results
            
            # Save CD-FVD result to a JSON file
            cdfvd_json_path = os.path.join(APP_ROOT, "cdfvd_results.json")
            with open(cdfvd_json_path, "w") as f:
                json.dump(cdfvd_results, f, indent=2)
            logger.info("[%s] CD-FVD results saved to %s", rid, cdfvd_json_path)
        except Exception as e:
            response["cdfvd_error"] = str(e)
            logger.error("[%s] CD-FVD computation error: %s", rid, e)
    
    # Attach artifacts (result JSON files)
    try:
        arts = _collect_artifacts(APP_ROOT, proc.stdout or "")
        response["artifacts"] = arts
        if arts:
            logger.info("[%s] Artifacts: %s", rid, ", ".join(a.get("name", "?") for a in arts))
    except Exception as e:
        response["artifact_error"] = str(e)
        logger.warning("[%s] Artifact collection error: %s", rid, e)
    return response
