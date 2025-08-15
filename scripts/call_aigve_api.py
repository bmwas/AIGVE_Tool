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
from typing import Any, Dict

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Call AIGVE API to run distribution-based metrics.")
    ap.add_argument("--base-url", default=os.getenv("AIGVE_API_URL", "http://localhost:2200"),
                    help="Base URL for the AIGVE API (default: http://localhost:2200 or env AIGVE_API_URL)")
    ap.add_argument("--input-dir", default="/app/data",
                    help="Container path to mixed videos (mounted). Default: /app/data")
    ap.add_argument("--stage-dataset", default="/app/out/staged",
                    help="Container path where dataset will be staged. Default: /app/out/staged")
    ap.add_argument("--max-seconds", type=float, default=8.0,
                    help="Clip duration in seconds (overrides max_len). Default: 8.0")
    ap.add_argument("--fps", type=float, default=25.0,
                    help="FPS used with --max-seconds. Default: 25.0")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--no-help", action="store_true", help="Skip calling /help before /run")

    args = ap.parse_args(argv)

    base_url = args.base_url

    # 1) Health
    print(f"[1/3] Checking health at {base_url}/healthz ...", flush=True)
    health = check_health(base_url)
    print(json.dumps(health, indent=2))

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

    # 3) Run distribution metrics
    print(f"\n[3/3] Running distribution metrics via {base_url}/run ...", flush=True)
    result = run_distribution_metrics(
        base_url=base_url,
        input_dir=args.input_dir,
        stage_dataset=args.stage_dataset,
        max_seconds=args.max_seconds,
        fps=args.fps,
        use_cpu=args.cpu,
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

    rc = int(result.get("returncode", 0) or 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
