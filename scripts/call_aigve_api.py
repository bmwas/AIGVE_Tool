#!/usr/bin/env python3
"""
A simple client to call the AIGVE API (server/main.py) to compute
distribution-based metrics (FID/IS/FVD) on a folder of mixed videos.

Assumptions
- The AIGVE Docker container is already running and exposes the API at
  http://localhost:2200 (override via env AIGVE_API_URL or --base-url).
- Your host folder ./data is mounted into the container at /app/data.
  Example container run (GPU):
    docker run -d --name aigve --restart unless-stopped \
      --gpus all -p 2200:2200 \
      -v "$PWD/data":/app/data -v "$PWD/out":/app/out \
      ghcr.io/bmwas/aigve:latest

What this script does
1) GET /healthz to verify the API is up.
2) POST /run with fields mapped to scripts/prepare_annotations.py flags:
   - input_dir: "/app/data"
   - stage_dataset: "/app/out/staged"  (dataset will be created here)
   - compute: true
   - categories: "distribution_based"  (i.e., fid,is,fvd)
   - max_seconds, fps: control duration

Notes
- Inside Docker, paths must be container paths (e.g., /app/data, /app/out).
- Generated files (e.g., fid_results.json) are written to the container's CWD
  (normally /app). When running with the example docker command above, you'll
  find staged data under ./out/staged on the host.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

import requests


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
) -> Dict[str, Any]:
    """
    Calls POST /run with the minimal JSON body to stage and compute
    distribution-based metrics. See server/main.py and
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
) -> Dict[str, Any]:
    """
    Uploads local video files to the server and calls POST /run_upload.
    When using this mode, server-side paths are not required; the server
    computes on the uploaded files only.
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

        print(f"[upload] Sending {len(files_param)} files:")
        for _, (fname, _, _) in files_param:
            print(" -", fname)

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
        if isinstance(art.get("json"), (dict, list)):
            content = json.dumps(art["json"], indent=2)
        elif isinstance(art.get("text"), str):
            content = art["text"]
        # Skip if no readable content
        if content is None:
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
    ap.add_argument("--max-seconds", type=float, default=8.0,
                    help="Clip duration in seconds (overrides max_len). Default: 8.0")
    ap.add_argument("--fps", type=float, default=25.0,
                    help="FPS used with --max-seconds. Default: 25.0")
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
        )
    else:
        print(f"\n[3/3] Running distribution metrics via {base_url}/run ...", flush=True)
        result = run_distribution_metrics(
            base_url=base_url,
            input_dir=args.input_dir,
            stage_dataset=args.stage_dataset,
            max_seconds=args.max_seconds,
            fps=args.fps,
            use_cpu=args.cpu,
            generated_suffixes=args.generated_suffixes,
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

    # Save any returned artifacts locally
    try:
        save_artifacts_locally(result, args.save_dir)
    except Exception as e:
        print(f"[artifacts] Error while saving artifacts: {e}", flush=True)

    rc = int(result.get("returncode", 0) or 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
