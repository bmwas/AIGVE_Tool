from __future__ import annotations

import os
import sys
import shlex
import subprocess
import json
import re
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Optional torch import (server runs inside conda env where torch is present)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - tolerate missing/failed import
    torch = None  # type: ignore

APP_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
SCRIPT_PATH = os.path.join(APP_ROOT, "scripts", "prepare_annotations.py")

app = FastAPI(title="AIGVE Prepare Annotations API", version="1.0.0")


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
def healthz():
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

    return {
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


@app.get("/help")
def cli_help():
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=500, detail=f"Script not found at {SCRIPT_PATH}")
    cmd = [sys.executable, SCRIPT_PATH, "--help"]
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
    response = {
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    # Attach artifacts (result JSON files)
    try:
        response["artifacts"] = _collect_artifacts(APP_ROOT, proc.stdout or "")
    except Exception as e:
        response["artifact_error"] = str(e)
    return response


@app.post("/run")
def run_prepare(req: PrepareAnnotationsRequest):
    if not req.list_metrics and not req.input_dir:
        raise HTTPException(status_code=422, detail="input_dir is required unless list_metrics is true")

    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=500, detail=f"Script not found at {SCRIPT_PATH}")

    args = _build_cli_args(req)
    cmd = [sys.executable, SCRIPT_PATH] + args

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

    response = {
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    # Attach artifacts (result JSON files)
    try:
        response["artifacts"] = _collect_artifacts(APP_ROOT, proc.stdout or "")
    except Exception as e:
        response["artifact_error"] = str(e)
    return response
