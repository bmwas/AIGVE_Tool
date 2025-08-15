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
    logger.info("[%s] %s %s", rid, request.method, request.url.path)
    # attach request id for handlers
    try:
        request.state.rid = rid
    except Exception:
        pass
    try:
        response = await call_next(request)
    except Exception as e:
        dur_ms = (time.perf_counter() - start) * 1000.0
        logger.exception("[%s] Error handling %s %s after %.1f ms: %s", rid, request.method, request.url.path, dur_ms, e)
        raise
    dur_ms = (time.perf_counter() - start) * 1000.0
    logger.info("[%s] %s %s -> %s in %.1f ms", rid, request.method, request.url.path, getattr(response, "status_code", "?"), dur_ms)
    return response


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
                   resolution: int = 128, sequence_length: int = 16) -> Dict[str, float]:
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
    if cdfvd is None:
        raise RuntimeError("cd-fvd package is not installed. Run: pip install cd-fvd")
    
    # Parse suffixes
    suffixes = [s.strip() for s in generated_suffixes.split(',') if s.strip()]
    
    # Organize videos into real and fake
    real_videos = []
    fake_videos = []
    
    for video_file in Path(upload_dir).glob("*.mp4"):
        video_name = video_file.stem
        is_synthetic = any(video_name.endswith(f"_{suffix}") for suffix in suffixes)
        
        if is_synthetic:
            fake_videos.append(str(video_file))
        else:
            # Check if this is a real video (has a corresponding synthetic version)
            has_synthetic = False
            for suffix in suffixes:
                synthetic_path = video_file.parent / f"{video_name}_{suffix}.mp4"
                if synthetic_path.exists():
                    has_synthetic = True
                    break
            if has_synthetic:
                real_videos.append(str(video_file))
    
    # Also check other video formats
    for ext in [".mov", ".avi", ".mkv", ".webm", ".m4v"]:
        for video_file in Path(upload_dir).glob(f"*{ext}"):
            video_name = video_file.stem
            is_synthetic = any(video_name.endswith(f"_{suffix}") for suffix in suffixes)
            
            if is_synthetic:
                fake_videos.append(str(video_file))
            else:
                has_synthetic = False
                for suffix in suffixes:
                    synthetic_path = video_file.parent / f"{video_name}_{suffix}{ext}"
                    if synthetic_path.exists():
                        has_synthetic = True
                        break
                if has_synthetic:
                    real_videos.append(str(video_file))
    
    if not real_videos or not fake_videos:
        raise ValueError(f"Insufficient videos for FVD computation. Found {len(real_videos)} real and {len(fake_videos)} fake videos")
    
    logger.info("[CD-FVD] Found %d real videos and %d fake videos", len(real_videos), len(fake_videos))
    
    # Create temporary directories for organized videos
    with tempfile.TemporaryDirectory() as temp_dir:
        real_dir = Path(temp_dir) / "real"
        fake_dir = Path(temp_dir) / "fake"
        real_dir.mkdir(exist_ok=True)
        fake_dir.mkdir(exist_ok=True)
        
        # Copy/link videos to temporary directories
        for i, video_path in enumerate(real_videos):
            dest = real_dir / f"video_{i:04d}{Path(video_path).suffix}"
            shutil.copy2(video_path, dest)
        
        for i, video_path in enumerate(fake_videos):
            dest = fake_dir / f"video_{i:04d}{Path(video_path).suffix}"
            shutil.copy2(video_path, dest)
        
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
        
        logger.info("[CD-FVD] FVD Score: %.4f", score)
        return result


def _collect_artifacts(base_dir: str, stdout: str) -> List[dict]:
    """
    Collect known result JSON files produced by the metrics scripts and include their contents.
    """
    candidate_names = [
        "fid_results.json",
        "is_results.json",
        "fvd_results.json",
        "gstvqa_results.json",
        "simplevqa_results.json",
        "lightvqa_plus_results.json",
        "cdfvd_results.json",  # Add CD-FVD results to artifacts
    ]
    artifacts: List[dict] = []
    for name in candidate_names:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            item: dict = {"name": name, "path": path}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                try:
                    item["json"] = json.loads(text)
                except Exception:
                    item["text"] = text
            except Exception as e:
                item["error"] = str(e)
            artifacts.append(item)
    # Best-effort: detect staged dataset path from stdout
    try:
        m = re.search(r"Staged dataset at:\s*(.+)", stdout)
        if m:
            artifacts.append({"name": "stage_info.txt", "path": None, "text": f"staged_dataset: {m.group(1).strip()}"})
    except Exception:
        pass
    return artifacts


@app.get("/healthz")
def healthz(request: Request):
    cuda_available = False
    cuda_version: Optional[str] = None
    device_count = 0
    torch_version: Optional[str] = None
    torch_error: Optional[str] = None

    try:
        if torch is not None:
            torch_version = getattr(torch, "__version__", None)
            cuda_available = bool(torch.cuda.is_available())
            cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
            device_count = torch.cuda.device_count() if cuda_available else 0
        else:
            torch_error = "torch not importable"
    except Exception as e:  # be resilient: health should not fail
        torch_error = str(e)

    info = {
        "status": "ok",
        "python": sys.version,
        "cwd": os.getcwd(),
        "script_exists": os.path.exists(SCRIPT_PATH),
        "torch": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_count": device_count,
        "torch_error": torch_error,
    }
    rid = getattr(getattr(request, "state", object()), "rid", "-")
    logger.info("[%s] /healthz torch=%s cuda=%s devices=%s script_exists=%s", rid, torch_version, cuda_available, device_count, info["script_exists"])
    return info


@app.get("/help")
def cli_help(request: Request):
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=500, detail=f"Script not found at {SCRIPT_PATH}")
    cmd = [sys.executable, SCRIPT_PATH, "--help"]
    rid = getattr(getattr(request, "state", object()), "rid", "-")
    t0 = time.perf_counter()
    logger.info("[%s] /help running: %s", rid, " ".join(shlex.quote(c) for c in cmd))
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
    response = {
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    logger.info("[%s] /help rc=%s in %.1f ms (stdout=%dB, stderr=%dB)", rid, proc.returncode, dur, len(proc.stdout or ""), len(proc.stderr or ""))
    # Attach artifacts (result JSON files)
    try:
        response["artifacts"] = _collect_artifacts(APP_ROOT, proc.stdout or "")
    except Exception as e:
        response["artifact_error"] = str(e)
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
    req = PrepareAnnotationsRequest(
        input_dir=upload_dir,
        generated_suffixes=generated_suffixes,
        stage_dataset=stage_dir,
        link=link,
        compute=compute,
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
    
    # Compute CD-FVD if requested
    cdfvd_error = None
    if use_cdfvd:
        try:
            logger.info("[%s] Computing FVD using cd-fvd package...", rid)
            cdfvd_result = _compute_cdfvd(
                staged_dir=stage_dir,
                model=cdfvd_model or "i3d",
                resolution=cdfvd_resolution or 224,
                sequence_length=cdfvd_sequence_length or 16,
                suffix=generated_suffixes
            )
            
            # Save CD-FVD results
            cdfvd_json_path = os.path.join(stage_dir, "cdfvd_results.json")
            with open(cdfvd_json_path, "w") as f:
                json.dump(cdfvd_result, f, indent=2)
            
            response["artifacts"] = {"cdfvd_results.json": cdfvd_json_path}
            
        except Exception as e:
            cdfvd_error = str(e)
            print(f"[CD-FVD Error] {cdfvd_error}")
            # Fail immediately when CD-FVD is requested but cannot be computed
            raise HTTPException(
                status_code=500,
                detail=f"CD-FVD computation failed: {cdfvd_error}"
            )
    
    try:
        arts = _collect_artifacts(APP_ROOT, proc.stdout or "")
        response["artifacts"] = arts
        if arts:
            logger.info("[%s] Artifacts: %s", rid, ", ".join(a.get("name", "?") for a in arts))
    except Exception as e:
        response["artifact_error"] = str(e)
        logger.warning("[%s] Artifact collection error: %s", rid, e)
    return response


@app.post("/run")
def run_prepare(req: PrepareAnnotationsRequest, request: Request):
    if not req.list_metrics and not req.input_dir:
        raise HTTPException(status_code=422, detail="input_dir is required unless list_metrics is true")

    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=500, detail=f"Script not found at {SCRIPT_PATH}")

    rid = getattr(getattr(request, "state", object()), "rid", "-")
    t0 = time.perf_counter()
    logger.info("[%s] /run input_dir=%s stage_dataset=%s compute=%s categories=%s metrics=%s", rid, req.input_dir, req.stage_dataset, bool(req.compute), req.categories, req.metrics)
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
    logger.info("[%s] /run rc=%s in %.1f ms (stdout=%dB, stderr=%dB)", rid, proc.returncode, dur, len(proc.stdout or ""), len(proc.stderr or ""))

    response = {
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    
    # Compute CD-FVD if requested
    if req.use_cdfvd and req.input_dir:
        try:
            logger.info("[%s] Computing FVD using cd-fvd package...", rid)
            cdfvd_result = _compute_cdfvd(
                upload_dir=req.input_dir,
                generated_suffixes=req.generated_suffixes or "synthetic,generated",
                model=req.cdfvd_model or "videomae",
                resolution=req.cdfvd_resolution or 128,
                sequence_length=req.cdfvd_sequence_length or 16
            )
            response["cdfvd_result"] = cdfvd_result
            
            # Save CD-FVD result to a JSON file
            cdfvd_json_path = os.path.join(APP_ROOT, "cdfvd_results.json")
            with open(cdfvd_json_path, "w") as f:
                json.dump(cdfvd_result, f, indent=2)
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
