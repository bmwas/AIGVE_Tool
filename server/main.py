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
from typing import List, Optional, Dict, Tuple, Any
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
    from cdfvd import fvd  # type: ignore
    cdfvd_available = True
except ImportError:
    fvd = None  # type: ignore
    cdfvd_available = False

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
    cdfvd_all_flavors: Optional[bool] = Field(True, description="Compute all FVD flavors (both models, multiple resolutions/sequence lengths)")


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


def _compute_aigve_metrics(video_dir: str, annotation_file: str, max_len: int = 64, 
                          use_cpu: bool = False, fvd_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute AIGVE FID, IS, and FVD metrics directly - NO TRY-EXCEPT BLOCKS.
    This function MUST compute all metrics and print results to console.
    """
    import time
    start_time = time.time()
    
    print(f"\n" + "="*80, flush=True)
    print(f"🚀 MANDATORY AIGVE METRICS COMPUTATION STARTING", flush=True)
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"="*80, flush=True)
    
    print(f"📂 FUNCTION INPUTS:", flush=True)
    print(f"   🎬 Video directory: {video_dir}", flush=True)
    print(f"   📋 Annotation file: {annotation_file}", flush=True)
    print(f"   🎞️  Max frames: {max_len}", flush=True)
    print(f"   🖥️  Use CPU: {use_cpu}", flush=True)
    print(f"   🤖 FVD model: {fvd_model or 'default'}", flush=True)
    
    # Import AIGVE components - must succeed
    print(f"\n📦 IMPORTING AIGVE COMPONENTS...", flush=True)
    import_start = time.time()
    
    print(f"   ⏳ Importing FidDataset...", flush=True)
    from aigve.datasets.fid_dataset import FidDataset
    print(f"   ✅ FidDataset imported successfully", flush=True)
    
    print(f"   ⏳ Importing FIDScore...", flush=True)
    from aigve.metrics.video_quality_assessment.distribution_based.fid_metric import FIDScore
    print(f"   ✅ FIDScore imported successfully", flush=True)
    
    print(f"   ⏳ Importing ISScore...", flush=True)
    from aigve.metrics.video_quality_assessment.distribution_based.is_score_metric import ISScore
    print(f"   ✅ ISScore imported successfully", flush=True)
    
    print(f"   ⏳ Importing FVDScore...", flush=True)
    from aigve.metrics.video_quality_assessment.distribution_based.fvd.fvd_metric import FVDScore
    print(f"   ✅ FVDScore imported successfully", flush=True)
    
    import_time = time.time() - import_start
    print(f"   🎉 All imports completed in {import_time:.2f}s", flush=True)
    
    # Determine device
    print(f"\n🖥️ DEVICE DETECTION:", flush=True)
    print(f"   ⏳ Checking CUDA availability...", flush=True)
    device_available = torch.cuda.is_available()
    print(f"   🔍 CUDA available: {device_available}", flush=True)
    
    if device_available:
        print(f"   🔍 CUDA device count: {torch.cuda.device_count()}", flush=True)
        print(f"   🔍 Current CUDA device: {torch.cuda.current_device()}", flush=True)
        print(f"   🔍 CUDA device name: {torch.cuda.get_device_name()}", flush=True)
    
    use_gpu = not use_cpu and device_available
    final_device = 'cuda' if use_gpu else 'cpu'
    print(f"   🎯 Final device selection: {final_device}", flush=True)
    
    print(f"\n📊 CONFIGURATION SUMMARY:", flush=True)
    print(f"   🎬 Video directory: {video_dir}", flush=True)
    print(f"   📋 Annotation file: {annotation_file}", flush=True)
    print(f"   🖥️  Device: {final_device}", flush=True)
    print(f"   🎞️  Max frames: {max_len}", flush=True)
    
    # Build dataset - must succeed
    print(f"\n📊 BUILDING DATASET...", flush=True)
    dataset_start = time.time()
    print(f"   ⏳ Creating FidDataset with parameters:", flush=True)
    print(f"      🎬 video_dir: {video_dir}", flush=True)
    print(f"      📋 prompt_dir: {annotation_file}", flush=True)
    print(f"      🎞️  max_len: {max_len}", flush=True)
    print(f"      📏 if_pad: False", flush=True)
    
    # Check if files exist before creating dataset
    print(f"   🔍 Validating input files...", flush=True)
    if not os.path.exists(video_dir):
        print(f"   ❌ Video directory does not exist: {video_dir}", flush=True)
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    else:
        print(f"   ✅ Video directory exists: {video_dir}", flush=True)
        
    if not os.path.exists(annotation_file):
        print(f"   ❌ Annotation file does not exist: {annotation_file}", flush=True)
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    else:
        print(f"   ✅ Annotation file exists: {annotation_file}", flush=True)
    
    # List video files in directory
    print(f"   📂 Scanning video directory contents...", flush=True)
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']
    all_files = os.listdir(video_dir)
    video_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in video_extensions)]
    print(f"   📁 Total files in directory: {len(all_files)}", flush=True)
    print(f"   🎬 Video files found: {len(video_files)}", flush=True)
    for i, vf in enumerate(video_files):
        print(f"      {i+1}. {vf}", flush=True)
    
    # Check annotation file content
    print(f"   📋 Reading annotation file...", flush=True)
    with open(annotation_file, 'r') as f:
        annotation_data = json.load(f)
        
    print(f"   📊 Annotation structure:", flush=True)
    print(f"      📏 Length: {annotation_data.get('metainfo', {}).get('length', 'unknown')}", flush=True)
    print(f"      📋 Data list size: {len(annotation_data.get('data_list', []))}", flush=True)
    
    for i, item in enumerate(annotation_data.get('data_list', [])[:3]):  # Show first 3 items
        print(f"      📑 Item {i+1}:", flush=True)
        print(f"         🎬 GT video: {item.get('video_path_gt', 'unknown')}", flush=True)
        print(f"         🤖 Generated video: {item.get('video_path_pd', 'unknown')}", flush=True)
    
    print(f"   ⏳ Instantiating FidDataset...", flush=True)
    dataset = FidDataset(
        video_dir=str(video_dir),
        prompt_dir=str(annotation_file),
        max_len=max_len,
        if_pad=False
    )
    
    dataset_time = time.time() - dataset_start
    print(f"   ✅ FidDataset created successfully in {dataset_time:.2f}s", flush=True)
    print(f"   📊 Dataset contains {len(dataset)} samples", flush=True)
    
    # Test dataset access
    print(f"   🧪 Testing dataset access...", flush=True)
    if len(dataset) > 0:
        print(f"   ⏳ Loading first sample for validation...", flush=True)
        try:
            sample_start = time.time()
            gt_tensor, gen_tensor, gt_name, gen_name = dataset[0]
            sample_time = time.time() - sample_start
            print(f"   ✅ First sample loaded successfully in {sample_time:.2f}s", flush=True)
            print(f"      🎬 GT tensor shape: {gt_tensor.shape if hasattr(gt_tensor, 'shape') else type(gt_tensor)}", flush=True)
            print(f"      🤖 Gen tensor shape: {gen_tensor.shape if hasattr(gen_tensor, 'shape') else type(gen_tensor)}", flush=True)
            print(f"      📝 GT name: {gt_name}", flush=True)
            print(f"      📝 Gen name: {gen_name}", flush=True)
        except Exception as e:
            print(f"   ❌ Failed to load first sample: {e}", flush=True)
            raise
    else:
        print(f"   ⚠️  Dataset is empty!", flush=True)
        raise ValueError("Dataset contains no samples")
    
    results = {}
    
    # Helper to iterate dataset samples with detailed logging
    def iter_samples():
        print(f"   🔄 Starting dataset iteration...", flush=True)
        for idx in range(len(dataset)):
            sample_start = time.time()
            print(f"      ⏳ Loading sample {idx+1}/{len(dataset)}...", flush=True)
            gt_tensor, gen_tensor, gt_name, gen_name = dataset[idx]
            sample_time = time.time() - sample_start
            print(f"      ✅ Sample {idx+1} loaded in {sample_time:.3f}s: {gt_name} -> {gen_name}", flush=True)
            yield gt_tensor, gen_tensor, gt_name, gen_name
        print(f"   🏁 Dataset iteration completed", flush=True)
    
    # COMPUTE FID - NO TRY-EXCEPT
    print(f"\n🧮 COMPUTING FID METRIC...", flush=True)
    fid_start = time.time()
    print(f"   ⏳ Initializing FIDScore metric...", flush=True)
    print(f"      🖥️  Using GPU: {use_gpu}", flush=True)
    print(f"      🎯 Device: {final_device}", flush=True)
    
    fid_metric = FIDScore(is_gpu=use_gpu)
    print(f"   ✅ FIDScore initialized successfully", flush=True)
    
    print(f"   🔄 Processing samples for FID computation...", flush=True)
    sample_count = 0
    for gt_tensor, gen_tensor, gt_name, gen_name in iter_samples():
        sample_count += 1
        print(f"      📊 Processing FID sample {sample_count}: {gt_name} vs {gen_name}", flush=True)
        data_samples = ((gt_tensor,), (gen_tensor,), (gt_name,), (gen_name,))
        print(f"         🔧 Calling fid_metric.process()...", flush=True)
        process_start = time.time()
        fid_metric.process(data_batch={}, data_samples=data_samples)
        process_time = time.time() - process_start
        print(f"         ✅ Sample processed in {process_time:.3f}s", flush=True)
    
    print(f"   🎯 Computing final FID metrics from {sample_count} samples...", flush=True)
    compute_start = time.time()
    fid_summary = fid_metric.compute_metrics([])
    compute_time = time.time() - compute_start
    print(f"   🎉 FID computation completed in {compute_time:.3f}s", flush=True)
    
    fid_score = fid_summary.get('FID_Mean_Score', fid_summary.get('FID', fid_summary.get('fid_score', 'UNKNOWN')))
    results['fid'] = {'score': fid_score, 'summary': fid_summary}
    
    fid_total_time = time.time() - fid_start
    print(f"   📊 FID RESULTS:", flush=True)
    print(f"      🏆 FID Score: {fid_score}", flush=True)
    print(f"      ⏱️  Total time: {fid_total_time:.2f}s", flush=True)
    print(f"      📋 Full summary: {fid_summary}", flush=True)
    print(f"[CONSOLE OUTPUT] ✅ FID COMPUTED: {fid_score}", flush=True)
    
    # COMPUTE IS - NO TRY-EXCEPT
    print(f"\n🧮 COMPUTING IS METRIC...", flush=True)
    is_start = time.time()
    print(f"   ⏳ Initializing ISScore metric...", flush=True)
    print(f"      🖥️  Using GPU: {use_gpu}", flush=True)
    print(f"      🎯 Device: {final_device}", flush=True)
    
    is_metric = ISScore(is_gpu=use_gpu)
    print(f"   ✅ ISScore initialized successfully", flush=True)
    
    print(f"   🔄 Processing samples for IS computation...", flush=True)
    sample_count = 0
    for _, gen_tensor, _, gen_name in iter_samples():
        sample_count += 1
        print(f"      📊 Processing IS sample {sample_count}: {gen_name} (generated only)", flush=True)
        data_samples = ((), (gen_tensor,), (), (gen_name,))
        print(f"         🔧 Calling is_metric.process()...", flush=True)
        process_start = time.time()
        is_metric.process(data_batch={}, data_samples=data_samples)
        process_time = time.time() - process_start
        print(f"         ✅ Sample processed in {process_time:.3f}s", flush=True)
    
    print(f"   🎯 Computing final IS metrics from {sample_count} samples...", flush=True)
    compute_start = time.time()
    is_summary = is_metric.compute_metrics([])
    compute_time = time.time() - compute_start
    print(f"   🎉 IS computation completed in {compute_time:.3f}s", flush=True)
    
    is_score = is_summary.get('IS_Mean_Score', is_summary.get('IS', is_summary.get('is_score', 'UNKNOWN')))
    results['is'] = {'score': is_score, 'summary': is_summary}
    
    is_total_time = time.time() - is_start
    print(f"   📊 IS RESULTS:", flush=True)
    print(f"      🏆 IS Score: {is_score}", flush=True)
    print(f"      ⏱️  Total time: {is_total_time:.2f}s", flush=True)
    print(f"      📋 Full summary: {is_summary}", flush=True)
    print(f"[CONSOLE OUTPUT] ✅ IS COMPUTED: {is_score}", flush=True)
    
    # COMPUTE FVD - NO TRY-EXCEPT
    print(f"\n🧮 COMPUTING FVD METRIC...", flush=True)
    fvd_start = time.time()
    print(f"   ⏳ Determining FVD model path...", flush=True)
    
    if fvd_model:
        model_path = Path(fvd_model).expanduser().resolve()
        print(f"      📁 Using custom FVD model: {fvd_model}", flush=True)
    else:
        model_path = Path(APP_ROOT) / 'aigve/metrics/video_quality_assessment/distribution_based/fvd/model_rgb.pth'
        print(f"      📁 Using default FVD model: {model_path}", flush=True)
    
    print(f"   🔍 Checking FVD model file...", flush=True)
    if model_path.exists():
        model_size = os.path.getsize(model_path)
        print(f"   ✅ FVD model file exists: {model_path} ({model_size:,} bytes)", flush=True)
    else:
        print(f"   ⚠️  FVD model file not found: {model_path} (will try to proceed)", flush=True)
    
    print(f"   ⏳ Initializing FVDScore metric...", flush=True)
    print(f"      🖥️  Using GPU: {use_gpu}", flush=True)
    print(f"      🎯 Device: {final_device}", flush=True)
    print(f"      📁 Model path: {model_path}", flush=True)
    
    fvd_metric = FVDScore(model_path=str(model_path), is_gpu=use_gpu)
    print(f"   ✅ FVDScore initialized successfully", flush=True)
    
    print(f"   🔄 Processing samples for FVD computation...", flush=True)
    sample_count = 0
    for gt_tensor, gen_tensor, gt_name, gen_name in iter_samples():
        sample_count += 1
        print(f"      📊 Processing FVD sample {sample_count}: {gt_name} vs {gen_name}", flush=True)
        data_samples = ((gt_tensor,), (gen_tensor,), (gt_name,), (gen_name,))
        print(f"         🔧 Calling fvd_metric.process()...", flush=True)
        process_start = time.time()
        fvd_metric.process(data_batch={}, data_samples=data_samples)
        process_time = time.time() - process_start
        print(f"         ✅ Sample processed in {process_time:.3f}s", flush=True)
    
    print(f"   🎯 Computing final FVD metrics from {sample_count} samples...", flush=True)
    compute_start = time.time()
    fvd_summary = fvd_metric.compute_metrics([])
    compute_time = time.time() - compute_start
    print(f"   🎉 FVD computation completed in {compute_time:.3f}s", flush=True)
    
    fvd_score = fvd_summary.get('FVD_Mean_Score', fvd_summary.get('FVD', fvd_summary.get('fvd_score', 'UNKNOWN')))
    results['fvd'] = {'score': fvd_score, 'summary': fvd_summary}
    
    fvd_total_time = time.time() - fvd_start
    print(f"   📊 FVD RESULTS:", flush=True)
    print(f"      🏆 FVD Score: {fvd_score}", flush=True)
    print(f"      ⏱️  Total time: {fvd_total_time:.2f}s", flush=True)
    print(f"      📋 Full summary: {fvd_summary}", flush=True)
    print(f"[CONSOLE OUTPUT] ✅ FVD COMPUTED: {fvd_score}", flush=True)
    
    # FINAL RESULTS DISPLAY
    total_time = time.time() - start_time
    print(f"\n" + "="*80, flush=True)
    print(f"🏆 MANDATORY AIGVE METRICS RESULTS SUMMARY", flush=True)
    print(f"⏰ Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"="*80, flush=True)
    print(f"📊 METRIC SCORES:", flush=True)
    print(f"   🎯 FID Score:  {results['fid']['score']}", flush=True)
    print(f"   🎯 IS Score:   {results['is']['score']}", flush=True)  
    print(f"   🎯 FVD Score:  {results['fvd']['score']}", flush=True)
    print(f"📈 TIMING BREAKDOWN:", flush=True)
    print(f"   ⏱️  FID time:   {fid_total_time:.2f}s", flush=True)
    print(f"   ⏱️  IS time:    {is_total_time:.2f}s", flush=True)
    print(f"   ⏱️  FVD time:   {fvd_total_time:.2f}s", flush=True)
    print(f"   ⏱️  Total time: {total_time:.2f}s", flush=True)
    print(f"🖥️ SYSTEM INFO:", flush=True)
    print(f"   🎯 Device used: {final_device}", flush=True)
    print(f"   📊 Samples processed: {len(dataset)}", flush=True)
    print(f"="*80, flush=True)
    
    return results


def _compute_cdfvd(upload_dir: str, generated_suffixes: str, model: str = "videomae", 
                   resolution: int = 128, sequence_length: int = 16,
                   max_seconds: Optional[float] = None, fps: Optional[float] = 25.0,
                   compute_all_flavors: bool = True) -> Dict[str, Any]:
    """
    Compute FVD using cd-fvd package. Can compute single flavor or all flavors.
    
    Args:
        upload_dir: Directory containing videos
        generated_suffixes: Comma-separated suffixes for synthetic videos
        model: CD-FVD model type ('videomae' or 'i3d') - used only if compute_all_flavors=False
        resolution: Video resolution for processing - used only if compute_all_flavors=False
        sequence_length: Number of frames to process - used only if compute_all_flavors=False
        compute_all_flavors: If True, compute all model/resolution/sequence combinations
    
    Returns:
        Dict with FVD scores and metadata. If compute_all_flavors=True, includes 'flavors' dict.
    """
    import time
    start_time = time.time()
    
    print(f"\n" + "="*80, flush=True)
    print(f"🚀 MANDATORY CD-FVD METRICS COMPUTATION STARTING", flush=True)
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"="*80, flush=True)
    print(f"📂 FUNCTION INPUTS:", flush=True)
    print(f"   🎬 Upload directory: {upload_dir}", flush=True)
    print(f"   🤖 Generated suffixes: {generated_suffixes}", flush=True)
    print(f"   🔧 Model: {model}", flush=True)
    print(f"   📐 Resolution: {resolution}", flush=True)
    print(f"   🎞️  Sequence length: {sequence_length}", flush=True)
    print(f"   ⏱️  Max seconds: {max_seconds}", flush=True)
    print(f"   🎭 FPS: {fps}", flush=True)
    print(f"   🔄 Compute all flavors: {compute_all_flavors}", flush=True)
    
    print(f"\n📦 IMPORTING CD-FVD COMPONENTS...", flush=True)
    import_start = time.time()
    
    logger.info("[CD-FVD] Starting FVD computation with model=%s, resolution=%d, seq_len=%d", 
                model, resolution, sequence_length)
    logger.debug("[CD-FVD] Parameters: upload_dir=%s, suffixes=%s, max_seconds=%s, fps=%s", 
                upload_dir, generated_suffixes, max_seconds, fps)
    
    print(f"   ⏳ Checking CD-FVD package availability...", flush=True)
    if not cdfvd_available:
        print(f"   ❌ CD-FVD package not available!", flush=True)
        logger.error("[CD-FVD] cd-fvd package is not installed")
        raise RuntimeError("cd-fvd package is not installed. Run: pip install cd-fvd")
    else:
        print(f"   ✅ CD-FVD package imported successfully", flush=True)
    
    import_time = time.time() - import_start
    print(f"   🎉 All imports completed in {import_time:.2f}s", flush=True)
    
    # Parse suffixes and build a robust checker for synthetic naming
    print(f"\n🔍 PARSING GENERATED SUFFIXES...", flush=True)
    suffixes = [s.strip() for s in generated_suffixes.split(',') if s.strip()]
    print(f"   📝 Raw suffixes input: '{generated_suffixes}'", flush=True)
    print(f"   🔧 Parsed suffixes: {suffixes}", flush=True)
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
    
    print(f"\n📂 ANALYZING VIDEO DIRECTORY...", flush=True)
    logger.info("[CD-FVD] Analyzing video directory: %s", upload_dir)
    print(f"   🔍 Looking for videos in: {upload_dir}", flush=True)
    print(f"   🤖 Suffixes to identify synthetic videos: {suffixes}", flush=True)
    
    # Check if directory exists and list contents
    print(f"   ⏳ Validating directory existence...", flush=True)
    upload_path = Path(upload_dir)
    if not upload_path.exists():
        print(f"   ❌ Upload directory does not exist: {upload_dir}", flush=True)
        logger.error("[CD-FVD] Upload directory does not exist: %s", upload_dir)
        raise RuntimeError(f"Upload directory does not exist: {upload_dir}")
    else:
        print(f"   ✅ Upload directory exists: {upload_dir}", flush=True)
    
    print(f"   📋 Scanning directory contents...", flush=True)
    try:
        scan_start = time.time()
        all_files = list(upload_path.glob("*"))
        scan_time = time.time() - scan_start
        print(f"   📁 Found {len(all_files)} total files in {scan_time:.3f}s", flush=True)
        logger.info("[CD-FVD] Found %d total files in directory", len(all_files))
        logger.debug("[CD-FVD] Directory contents: %s", [f.name for f in all_files])
        print(f"   📄 Files in directory: {[f.name for f in all_files]}", flush=True)
    except Exception as e:
        print(f"   ❌ Failed to list directory contents: {e}", flush=True)
        logger.error("[CD-FVD] Failed to list directory contents: %s", e)
        raise RuntimeError(f"Failed to access upload directory: {e}")
    
    # Organize videos into real and fake
    print(f"\n📹 ORGANIZING VIDEOS INTO REAL AND FAKE...", flush=True)
    real_videos = []
    fake_videos = []
    
    # Check all video formats
    print(f"   🔍 Searching for video files...", flush=True)
    video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.m4v"]
    all_videos = []
    for ext in video_extensions:
        ext_start = time.time()
        videos = list(Path(upload_dir).glob(ext))
        ext_time = time.time() - ext_start
        all_videos.extend(videos)
        if videos:
            print(f"   📁 Found {len(videos)} {ext} files in {ext_time:.3f}s", flush=True)
    
    print(f"   🎬 Total videos found: {len(all_videos)}", flush=True)
    
    print(f"   🔄 Classifying videos as real or synthetic...", flush=True)
    for i, video_file in enumerate(all_videos):
        video_name = video_file.stem
        is_synthetic = _is_synthetic_name(video_name)
        file_size = os.path.getsize(video_file) if video_file.exists() else 0

        print(f"   📊 Processing video {i+1}/{len(all_videos)}: {video_file.name}", flush=True)
        print(f"      📝 Stem: {video_name}", flush=True)
        print(f"      🤖 Is synthetic: {is_synthetic}", flush=True)
        print(f"      📏 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)", flush=True)

        if is_synthetic:
            fake_videos.append(str(video_file))
            print(f"      ➡️  Added to fake_videos: {video_file.name}", flush=True)
        else:
            # For real videos, we add them regardless of whether they have synthetic counterpart
            # The pairing will be handled by CD-FVD itself
            real_videos.append(str(video_file))
            print(f"      ➡️  Added to real_videos: {video_file.name}", flush=True)
    
    print(f"\n📊 VIDEO CLASSIFICATION SUMMARY:", flush=True)
    print(f"   🎬 Real videos: {len(real_videos)}", flush=True)
    print(f"   🤖 Synthetic videos: {len(fake_videos)}", flush=True)
    
    if not real_videos or not fake_videos:
        print(f"   ❌ Insufficient videos for FVD computation!", flush=True)
        print(f"      Need at least 1 real and 1 synthetic video", flush=True)
        raise ValueError(f"Insufficient videos for FVD computation. Found {len(real_videos)} real and {len(fake_videos)} fake videos")
    else:
        print(f"   ✅ Sufficient videos for computation", flush=True)
    
    logger.info("[CD-FVD] Found %d real videos and %d fake videos", len(real_videos), len(fake_videos))
    # Create temporary directories for organized videos
    print(f"\n📁 CREATING TEMPORARY DIRECTORIES...", flush=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"   📂 Temporary directory: {temp_dir}", flush=True)
        real_dir = Path(temp_dir) / "real_videos"
        fake_dir = Path(temp_dir) / "fake_videos"
        real_dir.mkdir(exist_ok=True)
        fake_dir.mkdir(exist_ok=True)
        print(f"   ✅ Created real videos directory: {real_dir}", flush=True)
        print(f"   ✅ Created fake videos directory: {fake_dir}", flush=True)
        
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
        print(f"\n🎬 PROCESSING REAL VIDEOS...", flush=True)
        for i, video_path in enumerate(real_videos):
            dest = real_dir / f"video_{i:04d}{Path(video_path).suffix}"
            print(f"   ⏳ Processing real video {i+1}/{len(real_videos)}: {Path(video_path).name}", flush=True)
            process_start = time.time()
            _trim_or_copy(video_path, dest)
            process_time = time.time() - process_start
            print(f"   ✅ Real video {i+1} processed in {process_time:.2f}s -> {dest.name}", flush=True)

        print(f"\n🤖 PROCESSING SYNTHETIC VIDEOS...", flush=True)
        for i, video_path in enumerate(fake_videos):
            dest = fake_dir / f"video_{i:04d}{Path(video_path).suffix}"
            print(f"   ⏳ Processing synthetic video {i+1}/{len(fake_videos)}: {Path(video_path).name}", flush=True)
            process_start = time.time()
            _trim_or_copy(video_path, dest)
            process_time = time.time() - process_start
            print(f"   ✅ Synthetic video {i+1} processed in {process_time:.2f}s -> {dest.name}", flush=True)
        
        if compute_all_flavors:
            print(f"\n🧮 COMPUTING ALL CD-FVD FLAVORS...", flush=True)
            # Minimal working configurations - balanced for testing without errors
            fast_configs = [
                ('i3d', 224, 16),       # Minimal: 224x224 resolution, 16 frames - balance between speed and kernel requirements
                # ('videomae', 224, 16), # Reliable fallback model - temporarily commented out
            ]
            
            print(f"   📋 Available model configurations:", flush=True)
            for i, (model_name, res, seq_len) in enumerate(fast_configs):
                print(f"      {i+1}. {model_name} (res={res}, seq_len={seq_len})", flush=True)
            
            logger.info("[CD-FVD] Computing %d fast FVD flavors (optimized for speed)", len(fast_configs))
            
            flavors = {}
            total_combinations = len(fast_configs)
            print(f"   🎯 Total combinations to compute: {total_combinations}", flush=True)
            logger.info("[CD-FVD] Computing all %d FVD flavors", total_combinations)
            
            for config_idx, (model_name, res, seq_len) in enumerate(fast_configs):
                flavor_key = f"{model_name}_res{res}_len{seq_len}"
                print(f"\n🔧 COMPUTING FLAVOR {config_idx+1}/{total_combinations}: {flavor_key}", flush=True)
                flavor_start = time.time()
                
                print(f"   📊 Configuration details:", flush=True)
                print(f"      🤖 Model: {model_name}", flush=True)
                print(f"      📐 Resolution: {res}x{res}", flush=True)
                print(f"      🎞️  Sequence length: {seq_len} frames", flush=True)
                
                logger.info("[CD-FVD] Computing flavor: %s", flavor_key)
                
                try:
                    # Initialize evaluator for this configuration
                    print(f"   ⏳ Initializing {model_name} evaluator...", flush=True)
                    init_start = time.time()
                    evaluator = fvd.cdfvd(model_name, ckpt_path=None, device='cuda')
                    init_time = time.time() - init_start
                    print(f"   ✅ {model_name} evaluator initialized in {init_time:.2f}s", flush=True)
                    
                    # Load and compute real video statistics using directory path
                    print(f"   🎬 Loading real videos from {real_dir}...", flush=True)
                    real_load_start = time.time()
                    real_videos = evaluator.load_videos(
                        str(real_dir), 
                        data_type='video_folder',
                        resolution=res, 
                        sequence_length=seq_len,
                        sample_every_n_frames=1
                    )
                    real_load_time = time.time() - real_load_start
                    print(f"   ✅ Loaded {len(real_videos)} real videos in {real_load_time:.2f}s", flush=True)
                    
                    print(f"   🧮 Computing real video statistics...", flush=True)
                    real_stats_start = time.time()
                    evaluator.compute_real_stats(real_videos)
                    real_stats_time = time.time() - real_stats_start
                    print(f"   ✅ Real video statistics computed in {real_stats_time:.2f}s", flush=True)
                    
                    # Load and compute fake video statistics using directory path
                    print(f"   🤖 Loading synthetic videos from {fake_dir}...", flush=True)
                    fake_load_start = time.time()
                    fake_videos = evaluator.load_videos(
                        str(fake_dir), 
                        data_type='video_folder',
                        resolution=res, 
                        sequence_length=seq_len,
                        sample_every_n_frames=1
                    )
                    fake_load_time = time.time() - fake_load_start
                    print(f"   ✅ Loaded {len(fake_videos)} synthetic videos in {fake_load_time:.2f}s", flush=True)
                    
                    print(f"   🧮 Computing synthetic video statistics...", flush=True)
                    fake_stats_start = time.time()
                    evaluator.compute_fake_stats(fake_videos)
                    fake_stats_time = time.time() - fake_stats_start
                    print(f"   ✅ Synthetic video statistics computed in {fake_stats_time:.2f}s", flush=True)
                    
                    # Compute FVD score from statistics
                    print(f"   🎯 Computing FVD score from statistics...", flush=True)
                    fvd_compute_start = time.time()
                    fvd_score = evaluator.compute_fvd_from_stats()
                    fvd_compute_time = time.time() - fvd_compute_start
                    
                    flavor_total_time = time.time() - flavor_start
                    
                    flavors[flavor_key] = {
                        "fvd_score": float(fvd_score),
                        "model": model_name,
                        "resolution": res,
                        "sequence_length": seq_len,
                        "num_real_videos": len(real_videos),
                        "num_fake_videos": len(fake_videos),
                        "computation_time": flavor_total_time
                    }
                    
                    print(f"   🎉 FVD computation completed in {fvd_compute_time:.2f}s", flush=True)
                    print(f"   📊 FLAVOR RESULTS:", flush=True)
                    print(f"      🏆 FVD Score: {fvd_score:.6f}", flush=True)
                    print(f"      ⏱️  Total time: {flavor_total_time:.2f}s", flush=True)
                    print(f"      🎬 Real videos processed: {len(real_videos)}", flush=True)
                    print(f"      🤖 Synthetic videos processed: {len(fake_videos)}", flush=True)
                    
                    logger.info("[CD-FVD] %s: %.4f", flavor_key, fvd_score)
                    print(f"[CONSOLE OUTPUT] ✅ {model_name.upper()} CD-FVD COMPUTED: {fvd_score:.6f}", flush=True)
                    logger.info("[CD-FVD] ✅ COMPLETED: %s model finished successfully!", model_name.upper())
                    
                    # Clear stats for next iteration
                    print(f"   🧹 Clearing statistics for next iteration...", flush=True)
                    evaluator.empty_real_stats()
                    evaluator.empty_fake_stats()
                    
                except Exception as e:
                    flavor_error_time = time.time() - flavor_start
                    print(f"   ❌ FLAVOR COMPUTATION FAILED after {flavor_error_time:.2f}s", flush=True)
                    print(f"      🚨 Error: {str(e)}", flush=True)
                    logger.error("[CD-FVD] Failed to compute %s: %s", flavor_key, e)
                    print(f"[CONSOLE OUTPUT] ❌ {model_name.upper()} CD-FVD FAILED: {str(e)}", flush=True)
                    flavors[flavor_key] = {"error": str(e), "computation_time": flavor_error_time}
            
            # FINAL RESULTS DISPLAY FOR ALL FLAVORS
            total_time = time.time() - start_time
            print(f"\n" + "="*80, flush=True)
            print(f"🏆 MANDATORY CD-FVD METRICS RESULTS SUMMARY", flush=True)
            print(f"⏰ Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            print(f"="*80, flush=True)
            print(f"📊 FLAVOR SCORES:", flush=True)
            
            successful_flavors = 0
            failed_flavors = 0
            
            for flavor_key, flavor_data in flavors.items():
                if "error" in flavor_data:
                    failed_flavors += 1
                    print(f"   ❌ {flavor_key}: FAILED - {flavor_data['error']}", flush=True)
                else:
                    successful_flavors += 1
                    print(f"   🎯 {flavor_key}: {flavor_data['fvd_score']:.6f}", flush=True)
            
            print(f"📊 COMPUTATION SUMMARY:", flush=True)
            print(f"   ✅ Successful flavors: {successful_flavors}", flush=True)
            print(f"   ❌ Failed flavors: {failed_flavors}", flush=True)
            print(f"   📊 Total flavors attempted: {len(flavors)}", flush=True)
            print(f"   ⏱️  Total computation time: {total_time:.2f}s", flush=True)
            print(f"🔧 SYSTEM INFO:", flush=True)
            print(f"   🎬 Videos processed: {len(real_videos)} real, {len(fake_videos)} synthetic", flush=True)
            print(f"   📂 Temporary directory: {temp_dir}", flush=True)
            print(f"="*80, flush=True)
            
            result = {
                "flavors": flavors,
                "total_flavors": len(flavors),
                "successful_flavors": successful_flavors,
                "failed_flavors": failed_flavors,
                "compute_all_flavors": True,
                "total_computation_time": total_time
            }
            
        else:
            # Single flavor computation (legacy behavior)
            print(f"\n🔧 COMPUTING SINGLE CD-FVD FLAVOR...", flush=True)
            print(f"   🤖 Model: {model}", flush=True)
            print(f"   📐 Resolution: {resolution}x{resolution}", flush=True)
            print(f"   🎞️  Sequence length: {sequence_length} frames", flush=True)
            
            logger.info("[CD-FVD] Computing single flavor with model='%s'", model)
            
            print(f"   ⏳ Initializing {model} evaluator...", flush=True)
            init_start = time.time()
            evaluator = fvd.cdfvd(model, ckpt_path=None, device='cuda')
            init_time = time.time() - init_start
            print(f"   ✅ {model} evaluator initialized in {init_time:.2f}s", flush=True)
            
            # Load and compute real video statistics using directory path
            print(f"   🎬 Loading real videos from {real_dir}...", flush=True)
            logger.info("[CD-FVD] Loading real video statistics from directory: %s", real_dir)
            real_load_start = time.time()
            real_videos = evaluator.load_videos(
                str(real_dir), 
                data_type='video_folder',
                resolution=resolution, 
                sequence_length=sequence_length,
                sample_every_n_frames=1
            )
            real_load_time = time.time() - real_load_start
            print(f"   ✅ Loaded {len(real_videos)} real videos in {real_load_time:.2f}s", flush=True)
            
            print(f"   🧮 Computing real video statistics...", flush=True)
            real_stats_start = time.time()
            evaluator.compute_real_stats(real_videos)
            real_stats_time = time.time() - real_stats_start
            print(f"   ✅ Real video statistics computed in {real_stats_time:.2f}s", flush=True)
            
            # Load and compute fake video statistics using directory path
            print(f"   🤖 Loading synthetic videos from {fake_dir}...", flush=True)
            logger.info("[CD-FVD] Loading fake video statistics from directory: %s", fake_dir)
            fake_load_start = time.time()
            fake_videos = evaluator.load_videos(
                str(fake_dir), 
                data_type='video_folder',
                resolution=resolution, 
                sequence_length=sequence_length,
                sample_every_n_frames=1
            )
            fake_load_time = time.time() - fake_load_start
            print(f"   ✅ Loaded {len(fake_videos)} synthetic videos in {fake_load_time:.2f}s", flush=True)
            
            print(f"   🧮 Computing synthetic video statistics...", flush=True)
            fake_stats_start = time.time()
            evaluator.compute_fake_stats(fake_videos)
            fake_stats_time = time.time() - fake_stats_start
            print(f"   ✅ Synthetic video statistics computed in {fake_stats_time:.2f}s", flush=True)
            
            # Compute FVD score from statistics
            print(f"   🎯 Computing FVD score from statistics...", flush=True)
            logger.info("[CD-FVD] Computing FVD score...")
            fvd_compute_start = time.time()
            fvd_score = evaluator.compute_fvd_from_stats()
            fvd_compute_time = time.time() - fvd_compute_start
            
            total_time = time.time() - start_time
            
            print(f"   🎉 FVD computation completed in {fvd_compute_time:.2f}s", flush=True)
            print(f"\n" + "="*80, flush=True)
            print(f"🏆 MANDATORY CD-FVD SINGLE FLAVOR RESULTS", flush=True)
            print(f"⏰ Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            print(f"="*80, flush=True)
            print(f"📊 METRIC SCORE:", flush=True)
            print(f"   🎯 {model.upper()} FVD Score: {fvd_score:.6f}", flush=True)
            print(f"📊 TIMING BREAKDOWN:", flush=True)
            print(f"   ⏱️  Initialization: {init_time:.2f}s", flush=True)
            print(f"   ⏱️  Real video loading: {real_load_time:.2f}s", flush=True)
            print(f"   ⏱️  Real stats computation: {real_stats_time:.2f}s", flush=True)
            print(f"   ⏱️  Synthetic video loading: {fake_load_time:.2f}s", flush=True)
            print(f"   ⏱️  Synthetic stats computation: {fake_stats_time:.2f}s", flush=True)
            print(f"   ⏱️  FVD score computation: {fvd_compute_time:.2f}s", flush=True)
            print(f"   ⏱️  Total time: {total_time:.2f}s", flush=True)
            print(f"🔧 SYSTEM INFO:", flush=True)
            print(f"   🎬 Real videos processed: {len(real_videos)}", flush=True)
            print(f"   🤖 Synthetic videos processed: {len(fake_videos)}", flush=True)
            print(f"   📐 Resolution used: {resolution}x{resolution}", flush=True)
            print(f"   🎞️  Sequence length: {sequence_length} frames", flush=True)
            print(f"="*80, flush=True)
            
            print(f"[CONSOLE OUTPUT] ✅ {model.upper()} CD-FVD COMPUTED: {fvd_score:.6f}", flush=True)
            
            result = {
                "fvd_score": float(fvd_score),
                "num_real_videos": len(real_videos),
                "num_fake_videos": len(fake_videos),
                "model": model,
                "resolution": resolution,
                "sequence_length": sequence_length,
                "compute_all_flavors": False,
                "total_computation_time": total_time
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
        
        # Add final summary logging
        if compute_all_flavors:
            successful = result.get("successful_flavors", 0)
            total = result.get("total_flavors", 0)
            logger.info("[CD-FVD] Computed %d/%d FVD flavors successfully", successful, total)
        else:
            logger.info("[CD-FVD] Single flavor FVD Score: %.6f", result["fvd_score"])
        
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
        "annotations.json",
        "evaluate.json",
        "cdfvd_results.json",  # Add CD-FVD results to artifacts
    ]
    
    logger.debug("[Artifacts] Searching for %d candidate artifact files: %s", 
                len(candidate_names), candidate_names)
    
    artifacts = []
    files_found = 0
    total_size = 0
    for name in candidate_names:
        full_path = os.path.join(base_dir, name)
        if os.path.exists(full_path):
            files_found += 1
            file_size = os.path.getsize(full_path)
            total_size += file_size
            logger.debug("[Artifacts] Found result file: %s", name)
            try:
                with open(full_path, 'r') as f:
                    content = json.load(f)
                artifacts.append({
                    "name": name,
                    "path": full_path,
                    "content": content,
                    "source": "file_system"
                })
                logger.debug("[Artifacts] Successfully loaded: %s", name)
                
                # Print results to console for key metrics - MANDATORY OUTPUT
                if name == "fid_results.json" and isinstance(content, dict):
                    fid_score = content.get('FID_Mean_Score', content.get('fid_score', content.get('FID', 'N/A')))
                    print(f"[METRICS RESULT] ✅ FID COMPLETED: Score = {fid_score}", flush=True)
                    print(f"[CONSOLE OUTPUT] FID = {fid_score}", flush=True)
                elif name == "is_results.json" and isinstance(content, dict):
                    is_score = content.get('IS_Mean_Score', content.get('is_score', content.get('IS', 'N/A')))
                    print(f"[METRICS RESULT] ✅ IS COMPLETED: Score = {is_score}", flush=True)
                    print(f"[CONSOLE OUTPUT] IS = {is_score}", flush=True)
                elif name == "fvd_results.json" and isinstance(content, dict):
                    fvd_score = content.get('FVD_Mean_Score', content.get('fvd_score', content.get('FVD', 'N/A')))
                    print(f"[METRICS RESULT] ✅ FVD COMPLETED: Score = {fvd_score}", flush=True)
                    print(f"[CONSOLE OUTPUT] FVD = {fvd_score}", flush=True)
                    
            except Exception as e:
                logger.warning("[Artifacts] Failed to load JSON from %s: %s", name, e)
                artifacts.append({
                    "name": name,
                    "path": full_path,
                    "error": str(e),
                    "source": "file_system"
                })
        else:
            logger.debug("[Artifacts] Not found: %s", name)
    
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
    cdfvd_all_flavors: bool = Form(True),
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

    # ENFORCE EXACTLY 2 VIDEOS REQUIREMENT
    if not videos or len(videos) != 2:
        logger.error("[%s] Invalid video count: received %d videos, exactly 2 required", rid, len(videos or []))
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Exactly 2 videos required",
                "received_count": len(videos or []),
                "required_count": 2,
                "expected": "One real video and one generated/synthetic video",
                "naming_convention": "Generated video should contain 'synthetic' or 'generated' in filename"
            }
        )

    allowed_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    saved_files: List[str] = []
    upload_errors: List[str] = []
    
    for uf in videos:
        try:
            name = os.path.basename(uf.filename or "video")
            ext = os.path.splitext(name)[1].lower()
            if ext and ext not in allowed_exts:
                upload_errors.append(f"Unsupported extension: {name}")
                logger.error("[%s] Unsupported video extension: %s", rid, name)
                continue
            dest_path = os.path.join(upload_dir, name)
            # Ensure parent exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as out_f:
                shutil.copyfileobj(uf.file, out_f)
            saved_files.append(name)
            logger.info("[%s] Saved upload -> %s", rid, dest_path)
        except Exception as e:
            upload_errors.append(f"Failed to save {getattr(uf, 'filename', 'unknown')}: {e}")
            logger.exception("[%s] Error saving uploaded file %s: %s", rid, getattr(uf, 'filename', 'unknown'), e)

    # Validate that we successfully saved exactly 2 videos
    valid_videos = [f for f in saved_files if not f.startswith("ERROR:")]
    
    if len(valid_videos) != 2:
        logger.error("[%s] Failed to save exactly 2 valid videos: saved %d valid, %d errors", 
                    rid, len(valid_videos), len(upload_errors))
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Failed to process exactly 2 valid videos",
                "valid_videos_saved": len(valid_videos),
                "required_count": 2,
                "saved_files": valid_videos,
                "upload_errors": upload_errors,
                "supported_extensions": list(allowed_exts)
            }
        )

    # Validate naming convention for real vs generated videos
    suffixes = [s.strip().lower() for s in generated_suffixes.split(',') if s.strip()]
    logger.debug("[%s] Generated video suffixes for validation: %s", rid, suffixes)
    
    def _is_generated_video(filename: str) -> bool:
        base = filename.lower()
        for suffix in suffixes:
            if suffix in base:
                return True
        return False
    
    real_videos = [f for f in valid_videos if not _is_generated_video(f)]
    generated_videos = [f for f in valid_videos if _is_generated_video(f)]
    
    logger.info("[%s] Video classification: %d real, %d generated", rid, len(real_videos), len(generated_videos))
    logger.debug("[%s] Real videos: %s", rid, real_videos)
    logger.debug("[%s] Generated videos: %s", rid, generated_videos)
    
    if len(real_videos) != 1 or len(generated_videos) != 1:
        logger.error("[%s] Invalid video pair: need exactly 1 real and 1 generated video", rid)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Invalid video pair: need exactly 1 real and 1 generated video",
                "real_videos_found": len(real_videos),
                "generated_videos_found": len(generated_videos),
                "real_videos": real_videos,
                "generated_videos": generated_videos,
                "generated_suffixes": suffixes,
                "naming_requirement": "Generated video filename must contain one of the suffixes: " + ", ".join(suffixes)
            }
        )

    # Determine stage dataset dir
    if stage_dataset:
        stage_dir = stage_dataset if os.path.isabs(stage_dataset) else os.path.join(APP_ROOT, stage_dataset)
    else:
        stage_dir = os.path.join(upload_dir, "staged")
    logger.info("[%s] Session=%s upload_dir=%s stage_dir=%s saved=%d", rid, session_id, upload_dir, stage_dir, len(saved_files))

    # Build args via existing request model helper
    # ALWAYS compute ALL metrics: FID, IS, FVD from legacy script + CD-FVD variants
    req = PrepareAnnotationsRequest(
        input_dir=upload_dir,
        generated_suffixes=generated_suffixes,
        stage_dataset=stage_dir,
        link=link,
        compute=True,  # ALWAYS compute legacy metrics (FID, IS, FVD)
        metrics="fid,is,fvd",  # Explicitly request all distribution-based metrics
        categories="distribution_based",  # Ensure we get FID, IS, FVD
        max_len=max_len,
        max_seconds=max_seconds,
        fps=fps,
        pad=pad,
        use_cpu=use_cpu,
    )
    args = _build_cli_args(req)
    cmd = [sys.executable, SCRIPT_PATH] + args
    logger.info("[%s] Exec: %s", rid, " ".join(shlex.quote(c) for c in cmd))

    # Execute script with retry mechanism for robustness
    max_retries = 3
    proc = None
    script_success = False
    
    logger.info("[%s] Starting script execution with %d max retries for ALL metrics", rid, max_retries)
    
    for attempt in range(1, max_retries + 1):
        logger.info("[%s] Attempt %d/%d: Executing script for FID/IS/FVD computation", rid, attempt, max_retries)
        
        try:
            proc = subprocess.run(
                cmd,
                cwd=APP_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=300  # 5 minute timeout per attempt
            )
            
            attempt_duration = (time.perf_counter() - t0) * 1000.0
            logger.info("[%s] Attempt %d/%d: Completed in %.1f ms with return code %d", 
                       rid, attempt, max_retries, attempt_duration, proc.returncode)
            
            # Check for successful execution
            if proc.returncode == 0:
                logger.info("[%s] Attempt %d/%d: SUCCESS - Script executed successfully", rid, attempt, max_retries)
                script_success = True
                break
            else:
                logger.warning("[%s] Attempt %d/%d: FAILURE - Return code %d", rid, attempt, max_retries, proc.returncode)
                if proc.stderr:
                    stderr_preview = proc.stderr.strip()[:200] + ("..." if len(proc.stderr.strip()) > 200 else "")
                    logger.warning("[%s] Attempt %d/%d: Error details: %s", rid, attempt, max_retries, stderr_preview)
                
                if attempt < max_retries:
                    logger.info("[%s] Attempt %d/%d: Retrying in 2 seconds...", rid, attempt, max_retries)
                    time.sleep(2)
                
        except subprocess.TimeoutExpired:
            logger.error("[%s] Attempt %d/%d: Script execution timed out after 5 minutes", rid, attempt, max_retries)
            if attempt < max_retries:
                logger.info("[%s] Attempt %d/%d: Retrying after timeout...", rid, attempt, max_retries)
            else:
                raise HTTPException(status_code=500, detail="Script execution timed out after all retry attempts")
        except Exception as e:
            logger.error("[%s] Attempt %d/%d: Script execution failed: %s", rid, attempt, max_retries, e)
            if attempt < max_retries:
                logger.info("[%s] Attempt %d/%d: Retrying after exception...", rid, attempt, max_retries)
            else:
                raise HTTPException(status_code=500, detail=f"Failed to execute script after all attempts: {e}")
    
    if not script_success:
        logger.error("[%s] All %d attempts failed - script execution unsuccessful", rid, max_retries)
        # MANDATORY: Compute AIGVE metrics directly if script failed
        logger.warning("[%s] Script failed - executing MANDATORY direct AIGVE metrics computation", rid)
    
    dur = (time.perf_counter() - t0) * 1000.0
    logger.info("[%s] /run_upload script phase complete: rc=%s in %.1f ms (stdout=%dB, stderr=%dB)", 
                rid, proc.returncode if proc else -1, dur, len(proc.stdout or "") if proc else 0, len(proc.stderr or "") if proc else 0)

    # MANDATORY: Compute AIGVE metrics directly - ALWAYS EXECUTE
    logger.info("[%s] Starting MANDATORY direct AIGVE metrics computation", rid)
    
    # Find annotation file for metrics computation
    annotation_file = None
    possible_annotations = [
        os.path.join(stage_dir, "annotations", "evaluate.json"),
        os.path.join(stage_dir, "evaluate.json"),
        os.path.join(upload_dir, "annotations.json"),
        os.path.join(stage_dir, "annotations.json")
    ]
    
    for ann_path in possible_annotations:
        if os.path.exists(ann_path):
            annotation_file = ann_path
            logger.info("[%s] Found annotation file: %s", rid, annotation_file)
            break
    
    if not annotation_file:
        # Create a minimal annotation file from uploaded videos
        annotation_file = os.path.join(stage_dir, "annotations.json")
        os.makedirs(os.path.dirname(annotation_file), exist_ok=True)
        
        # Use the validated real and generated video names
        annotation_data = {
            "metainfo": {"source": "uploaded_videos", "length": 1},
            "data_list": [{
                "prompt_gt": "",
                "video_path_pd": generated_videos[0],
                "video_path_gt": real_videos[0]
            }]
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        logger.info("[%s] Created annotation file: %s", rid, annotation_file)
    
    # Determine video directory for metrics
    metrics_video_dir = None
    video_search_dirs = [
        os.path.join(stage_dir, "evaluate"),
        stage_dir,
        upload_dir
    ]
    
    for vdir in video_search_dirs:
        if os.path.exists(vdir):
            video_files = [f for f in os.listdir(vdir) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'))]
            if len(video_files) >= 2:
                metrics_video_dir = vdir
                logger.info("[%s] Using video directory for metrics: %s (%d videos)", rid, vdir, len(video_files))
                break
    
    if not metrics_video_dir:
        metrics_video_dir = upload_dir
        logger.warning("[%s] Using fallback video directory: %s", rid, metrics_video_dir)
    
    # MANDATORY AIGVE METRICS COMPUTATION - NO TRY-EXCEPT
    logger.info("[%s] Computing AIGVE metrics: video_dir=%s, annotation=%s", rid, metrics_video_dir, annotation_file)
    aigve_results = _compute_aigve_metrics(
        video_dir=metrics_video_dir,
        annotation_file=annotation_file,
        max_len=max_len,
        use_cpu=use_cpu
    )
    
    # Save AIGVE results
    aigve_results_file = os.path.join(stage_dir, "aigve_direct_results.json")
    os.makedirs(os.path.dirname(aigve_results_file), exist_ok=True)
    with open(aigve_results_file, 'w') as f:
        json.dump(aigve_results, f, indent=2, default=str)
    
    logger.info("[%s] AIGVE results saved to: %s", rid, aigve_results_file)

    # Print standard metrics results to console immediately after script execution
    if script_success and proc:
        logger.info("[%s] Collecting and printing standard metrics results", rid)
        # Search in the staged directory where results are actually saved - NO TRY-EXCEPT
        _collect_artifacts(stage_dir, proc.stdout or "")

    response = {
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "session": {"id": session_id, "upload_dir": upload_dir, "stage_dir": stage_dir, "files": saved_files},
    }
    
    # OPTIONAL CD-FVD computation - only when explicitly requested
    cdfvd_results = {}
    if use_cdfvd:
        logger.info("[%s] Starting CD-FVD computation (explicitly requested)", rid)
        models = ["i3d"] if not cdfvd_all_flavors else ["i3d", "videomae"]
        logger.info("[%s] CD-FVD models to compute: %s", rid, models)
    else:
        logger.info("[%s] Skipping CD-FVD computation (use_cdfvd=False) - computing AIGVE native metrics only", rid)
        models = []
    
    # Only determine video directory if CD-FVD is requested
    if use_cdfvd:
        # Determine video directory with multiple fallback strategies
        video_locations = [
            os.path.join(stage_dir, "evaluate"),  # Staged location
            stage_dir,                           # Stage directory directly
            upload_dir                           # Original upload directory
        ]
    
        video_dir = None
        for location in video_locations:
            if os.path.exists(location):
                # Check if it contains video files
                try:
                    files = os.listdir(location)
                    video_files = [f for f in files if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'))]
                    if video_files:
                        video_dir = location
                        logger.info("[%s] Found videos in directory: %s (%d video files)", rid, location, len(video_files))
                        break
                    else:
                        logger.debug("[%s] Directory %s exists but contains no video files", rid, location)
                except Exception as e:
                    logger.warning("[%s] Failed to check directory %s: %s", rid, location, e)
                    
        if not video_dir:
            logger.error("[%s] CRITICAL: No directory with video files found for CD-FVD", rid)
            # As last resort, use upload directory and ensure videos are there
            video_dir = upload_dir
            logger.warning("[%s] Using upload directory as fallback: %s", rid, video_dir)
        
        # Ensure videos are accessible in the chosen directory
        logger.info("[%s] CD-FVD using video directory: %s", rid, video_dir)
    
        try:
            files = os.listdir(video_dir)
            video_files = [f for f in files if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'))]
            logger.info("[%s] CD-FVD directory analysis: %d total files, %d video files", rid, len(files), len(video_files))
            logger.debug("[%s] Video files found: %s", rid, video_files)
            
            if len(video_files) < 2:
                logger.error("[%s] CRITICAL: Insufficient video files for CD-FVD (%d found, 2+ required)", rid, len(video_files))
                # Copy videos from upload directory if needed
                if video_dir != upload_dir:
                    logger.info("[%s] Attempting to copy videos from upload directory: %s", rid, upload_dir)
                    try:
                        upload_files = os.listdir(upload_dir)
                        upload_videos = [f for f in upload_files if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'))]
                        
                        for video_file in upload_videos:
                            src = os.path.join(upload_dir, video_file)
                            dst = os.path.join(video_dir, video_file)
                            if not os.path.exists(dst):
                                shutil.copy2(src, dst)
                                logger.info("[%s] Copied video for CD-FVD: %s -> %s", rid, src, dst)
                        
                        # Re-check video count
                        files = os.listdir(video_dir)
                        video_files = [f for f in files if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'))]
                        logger.info("[%s] After copying: %d video files available", rid, len(video_files))
                        
                    except Exception as copy_e:
                        logger.error("[%s] Failed to copy videos for CD-FVD: %s", rid, copy_e)
                
        except Exception as e:
            logger.error("[%s] Failed to analyze video directory for CD-FVD: %s", rid, e)
    
    # CD-FVD computation for each model (when requested)
    for model_idx, model in enumerate(models):
        model_start_time = time.perf_counter()
        logger.info("[%s] CD-FVD computation %d/%d: %s", rid, model_idx + 1, len(models), model)
        
        max_model_retries = 3
        model_success = False
        
        for model_attempt in range(1, max_model_retries + 1):
            try:
                logger.info("[%s] CD-FVD %s attempt %d/%d", rid, model, model_attempt, max_model_retries)
                
                cdfvd_result = _compute_cdfvd(
                    upload_dir=video_dir,
                    generated_suffixes=generated_suffixes,
                    model=model,
                    resolution=cdfvd_resolution or 128,
                    sequence_length=cdfvd_sequence_length or 16,
                    max_seconds=max_seconds,
                    fps=fps,
                    compute_all_flavors=cdfvd_all_flavors,
                )
                
                model_duration = (time.perf_counter() - model_start_time) * 1000.0
                cdfvd_results[model] = cdfvd_result
                model_success = True
                
                fvd_score = cdfvd_result.get("fvd_score", 0)
                logger.info("[%s] CD-FVD %s SUCCESS in %.1f ms: score=%.4f", rid, model, model_duration, fvd_score)
                # Print result immediately to console
                print(f"[UPLOAD CD-FVD RESULT] ✅ {model.upper()} COMPLETED: FVD Score = {fvd_score:.4f}")
                break
                
            except Exception as e:
                logger.error("[%s] CD-FVD %s attempt %d/%d failed: %s", rid, model, model_attempt, max_model_retries, e)
                if model_attempt < max_model_retries:
                    logger.info("[%s] CD-FVD %s retrying in 1 second...", rid, model)
                    time.sleep(1)
                else:
                    # Final attempt failed - record error but continue with next model
                    model_duration = (time.perf_counter() - model_start_time) * 1000.0
                    cdfvd_results[model] = {
                        "error": str(e), 
                        "attempts": max_model_retries,
                        "duration_ms": model_duration
                    }
                    logger.error("[%s] CD-FVD %s FAILED after %d attempts in %.1f ms", 
                               rid, model, max_model_retries, model_duration)
                    # Print failure immediately to console
                    print(f"[UPLOAD CD-FVD RESULT] ❌ {model.upper()} FAILED: {str(e)}")
        
        if not model_success:
            logger.warning("[%s] CD-FVD %s computation unsuccessful - continuing with next model", rid, model)
    
    # CD-FVD computation phase complete
    if use_cdfvd:
        logger.info("[%s] CD-FVD computation phase complete: %d models processed", rid, len(models))
        response["cdfvd_results"] = cdfvd_results
        
        # Save CD-FVD results to file
        try:
            cdfvd_json_path = os.path.join(stage_dir, "cdfvd_results.json")
            os.makedirs(os.path.dirname(cdfvd_json_path), exist_ok=True)
            with open(cdfvd_json_path, "w") as f:
                json.dump(cdfvd_results, f, indent=2)
            logger.info("[%s] CD-FVD results saved to: %s", rid, cdfvd_json_path)
            
            # Return CD-FVD artifacts along with any legacy artifacts
            cdfvd_artifacts = [
                {
                    "name": "cdfvd_results.json",
                    "path": cdfvd_json_path,
                    "json": cdfvd_results
                }
            ]
            
            # Also collect legacy artifacts (FID, IS, FVD results)
            try:
                legacy_arts = _collect_artifacts(APP_ROOT, proc.stdout or "")
                response["artifacts"] = cdfvd_artifacts + legacy_arts
                logger.info("[%s] ALL artifacts collected: %d total (CD-FVD + legacy)", rid, len(response["artifacts"]))
            except Exception as e:
                response["artifacts"] = cdfvd_artifacts
                response["artifact_error"] = str(e)
                logger.warning("[%s] Legacy artifact collection error: %s", rid, e)
                
        except Exception as cdfvd_save_e:
            logger.error("[%s] Failed to save CD-FVD results: %s", rid, cdfvd_save_e)
            response["cdfvd_results"] = cdfvd_results
            response["cdfvd_save_error"] = str(cdfvd_save_e)
            
            # Still collect legacy artifacts on CD-FVD save failure
            try:
                arts = _collect_artifacts(APP_ROOT, proc.stdout or "")
                response["artifacts"] = arts
                logger.info("[%s] Legacy artifacts collected: %d", rid, len(arts))
            except Exception as e:
                response["artifact_error"] = str(e)
                logger.warning("[%s] Artifact collection error: %s", rid, e)
    else:
        # No CD-FVD computation - collect only legacy artifacts (AIGVE native metrics)
        logger.info("[%s] Collecting AIGVE native artifacts only (FID, IS, FVD)", rid)
        try:
            arts = _collect_artifacts(APP_ROOT, proc.stdout or "")
            response["artifacts"] = arts
            logger.info("[%s] AIGVE native artifacts collected: %d", rid, len(arts))
        except Exception as e:
            response["artifact_error"] = str(e)
            logger.warning("[%s] Artifact collection error: %s", rid, e)
    
    # Final validation: ensure ALL required metrics were computed
    logger.info("[%s] Final processing validation", rid)
    total_duration = (time.perf_counter() - t0) * 1000.0
    
{{ ... }}
    # Check legacy metrics (from script)
    legacy_metrics = ["fid", "is", "fvd"]
    legacy_success = script_success and proc and proc.returncode == 0
    
    # Check CD-FVD metrics (only if CD-FVD was requested)
    cdfvd_success_count = 0
    if use_cdfvd:
        cdfvd_success_count = sum(1 for model in models if "error" not in cdfvd_results.get(model, {}))
    
    processing_summary = {
        "total_duration_ms": total_duration,
        "script_success": legacy_success,
        "legacy_metrics_attempted": legacy_metrics,
        "cdfvd_models_successful": cdfvd_success_count,
        "cdfvd_models_total": len(models),
        "videos_processed": len(valid_videos)
    }
    
    response["processing_summary"] = processing_summary
    
    logger.info("[%s] Processing complete: script=%s, cd-fvd=%d/%d models, duration=%.1f ms", 
                rid, legacy_success, cdfvd_success_count, len(models), total_duration)
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

    # Log final directory contents and validate input
    try:
        input_files = os.listdir(req.input_dir) if os.path.exists(req.input_dir) else []
        logger.info("[%s] Final input_dir contents: %s -> %d items: %s", rid, req.input_dir, len(input_files), input_files)
        
        # Check for video files in input directory
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
        video_files = [f for f in input_files if any(f.lower().endswith(ext) for ext in video_extensions)]
        logger.info("[%s] Found %d video files in input directory: %s", rid, len(video_files), video_files)
        
        if len(video_files) == 0:
            logger.error("[%s] No video files found in input directory", rid)
            raise HTTPException(
                status_code=422, 
                detail={
                    "error": "No video files found in input directory",
                    "input_dir": req.input_dir,
                    "files_found": input_files,
                    "supported_extensions": list(video_extensions),
                    "suggestion": "Use /run_upload endpoint to upload video files, or ensure your input_dir contains videos with supported extensions"
                }
            )
            
    except HTTPException:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error("[%s] Failed to list final input_dir contents: %s", rid, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to access input directory: {e}"
        )
    
    try:
        stage_files = os.listdir(req.stage_dataset) if req.stage_dataset and os.path.exists(req.stage_dataset) else []
        logger.info("[%s] Final stage_dataset contents: %s -> %d items: %s", rid, req.stage_dataset, len(stage_files), stage_files)
    except Exception as e:
        logger.error("[%s] Failed to list final stage_dataset contents: %s", rid, e)

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
                        compute_all_flavors=req.cdfvd_all_flavors if req.cdfvd_all_flavors is not None else True,
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
