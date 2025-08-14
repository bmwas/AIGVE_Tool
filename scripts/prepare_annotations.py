#!/usr/bin/env python3
"""
Prepare AIGVE evaluation annotations for reference-based metrics (FID/IS/FVD).

Given a folder that contains both ground-truth and generated videos, where
generated videos are named by appending a suffix to the ground-truth basename
(e.g., 4056.mp4 -> 4056_synthetic.mp4 or 4056_generated.mp4), this script:
  - finds GT/GEN pairs,
  - optionally stages them into a dataset folder structure,
  - writes an AIGVE-compatible JSON annotation file.

JSON schema (per AIGVE toy example):
{
  "metainfo": {"source": "your_dataset", "length": N},
  "data_list": [
    {
      "prompt_gt": "",
      "video_path_pd": "generated_filename.mp4",
      "video_path_gt": "ground_truth_filename.mp4"
    }
  ]
}

Usage examples:
  # Minimal: scan IN_DIR, write JSON next to it
  python scripts/prepare_annotations.py \
      --input-dir /path/to/mixed_videos \
      --out-json /path/to/mixed_videos/annotations.json

  # Use default suffixes "synthetic,generated"; set a single suffix explicitly
  python scripts/prepare_annotations.py --input-dir IN --out-json OUT.json \
      --generated-suffixes synthetic

  # Stage into a dataset folder like AIGVE toy layout
  python scripts/prepare_annotations.py --input-dir IN \
      --stage-dataset /path/to/my_dataset   # creates my_dataset/evaluate/ and my_dataset/annotations/evaluate.json

  # Prefer symlinks instead of copying when staging
  python scripts/prepare_annotations.py --input-dir IN --stage-dataset DST --link
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Ensure project root (parent of this script) is on sys.path so 'aigve' package imports when run directly
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# Also add the 'aigve' package directory so imports like 'from core ...' work
_AIGVE_DIR = _PROJECT_ROOT / 'aigve'
if _AIGVE_DIR.is_dir() and str(_AIGVE_DIR) not in sys.path:
    sys.path.insert(0, str(_AIGVE_DIR))

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# Metric categories
# - distribution_based: reference-based distribution comparison metrics
# - nn_based_video: video-only neural network-based metrics
METRIC_CATEGORIES = {
    "distribution_based": ["fid", "is", "fvd"],
    "nn_based_video": ["gstvqa", "simplevqa", "lightvqa+"],
}


def _tokenize_suffixes(suffix_csv: str) -> List[str]:
    # Accept comma-separated tokens; e.g., "synthetic,generated,_syn"
    raw = [s.strip() for s in suffix_csv.split(",") if s.strip()]
    tokens: List[str] = []
    for s in raw:
        # Prefer separator forms first so base stems are clean (no trailing '_'/'-')
        if s.startswith("_") or s.startswith("-"):
            tokens.append(s)
        else:
            tokens.append("_" + s)
            tokens.append("-" + s)
            tokens.append(s)
    # Preserve order while removing duplicates
    seen = set()
    out = []
    for t in tokens:
        tl = t.lower()
        if tl not in seen:
            out.append(t)
            seen.add(tl)
    return out


def _find_pair(stem: str, suffix_tokens: List[str]) -> Optional[Tuple[str, str]]:
    """
    If stem ends with any token, return (token, base_stem_without_token).
    E.g., stem='4056_synthetic' with tokens ['_synthetic','-synthetic'] -> ('_synthetic','4056')
    """
    stem_l = stem.lower()
    for tok in suffix_tokens:
        tok_l = tok.lower()
        if stem_l.endswith(tok_l):
            return tok, stem[: -len(tok)]
    return None


def discover_pairs(input_dir: Path, generated_suffixes: str) -> Tuple[List[Tuple[Path, Path]], Dict[str, List[str]]]:
    suffix_tokens = _tokenize_suffixes(generated_suffixes)

    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    gt_set = {p.name for p in files}  # candidate GT names (basenames)

    pairs: List[Tuple[Path, Path]] = []  # (gen_path, gt_path)
    issues = {"no_gt": [], "no_gen": [], "duplicates": []}

    # Index by basename for quick lookup
    by_name: Dict[str, Path] = {p.name: p for p in files}

    # Track seen GT basenames to detect multiple generated versions
    seen_gen_for_gt: Dict[str, List[str]] = {}

    for p in files:
        stem = p.stem  # without extension
        match = _find_pair(stem, suffix_tokens)
        if not match:
            continue  # looks like a ground-truth (no gen suffix)
        tok, base_stem = match
        gt_name = f"{base_stem}{p.suffix}"
        if gt_name not in gt_set:
            issues["no_gt"].append(p.name)
            continue
        gen_path = p
        gt_path = by_name[gt_name]
        pairs.append((gen_path, gt_path))
        seen_gen_for_gt.setdefault(gt_name, []).append(gen_path.name)

    # Any GT without generated counterpart?
    for gt_name in gt_set:
        stem = Path(gt_name).stem
        if _find_pair(stem, suffix_tokens):  # if GT name itself ends with token, skip
            continue
        if gt_name not in seen_gen_for_gt:
            issues["no_gen"].append(gt_name)

    # Duplicates: more than one gen per GT
    for gt_name, gens in seen_gen_for_gt.items():
        if len(gens) > 1:
            issues["duplicates"].append(f"{gt_name} <- {gens}")

    return pairs, issues


def write_json(pairs: List[Tuple[Path, Path]], out_json: Path, relative_to: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data_list = []
    for gen_path, gt_path in pairs:
        # Paths in JSON should be relative to 'video_dir' (relative_to)
        gen_rel = os.path.relpath(gen_path, relative_to)
        gt_rel = os.path.relpath(gt_path, relative_to)
        data_list.append({
            "prompt_gt": "",
            "video_path_pd": gen_rel.replace("\\", "/"),
            "video_path_gt": gt_rel.replace("\\", "/"),
        })
    payload = {
        "metainfo": {"source": str(relative_to), "length": len(data_list)},
        "data_list": data_list,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def stage_dataset(pairs: List[Tuple[Path, Path]], dest_root: Path, link: bool = False) -> Path:
    """
    Stage files into AIGVE-like dataset layout under dest_root:
      dest_root/
        evaluate/               # contains both GT and GEN files
        annotations/evaluate.json
    Returns the path to the created evaluate/ directory.
    """
    eval_dir = dest_root / "evaluate"
    ann_dir = dest_root / "annotations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Copy/symlink both GT and GEN (deduplicate by name)
    staged: Dict[str, Path] = {}
    for gen_path, gt_path in pairs:
        for src in (gen_path, gt_path):
            dst = eval_dir / src.name
            if dst.exists():
                staged[src.name] = dst
                continue
            if link:
                try:
                    os.symlink(src.resolve(), dst)
                except FileExistsError:
                    pass
            else:
                shutil.copy2(src, dst)
            staged[src.name] = dst

    return eval_dir


def _parse_metrics_list(s: str) -> List[str]:
    s = (s or '').strip().lower()
    if not s or s == 'none':
        return []
    parts = [p.strip() for p in s.split(',') if p.strip()]
    out: List[str] = []
    for p in parts:
        if p == 'all':
            # Backward compatibility: 'all' means distribution metrics only
            out.extend(METRIC_CATEGORIES["distribution_based"])
        elif p in METRIC_CATEGORIES:
            out.extend(METRIC_CATEGORIES[p])
        elif p in {"fid", "is", "fvd", "gstvqa", "simplevqa", "lightvqa+"}:
            out.append(p)
        else:
            print(f"[WARN] Unknown metric/category '{p}' (skipped).")
    # preserve order and uniqueness
    seen = set()
    uniq: List[str] = []
    for m in out:
        if m not in seen:
            uniq.append(m)
            seen.add(m)
    return uniq


def run_reference_metrics(video_dir: Path,
                          prompt_json: Path,
                          metrics: List[str],
                          use_gpu: bool,
                          max_len: int,
                          if_pad: bool,
                          fvd_model: Optional[Path],
                          gstvqa_model: Optional[Path],
                          simplevqa_model: Optional[Path],
                          lightvqa_model: Optional[Path],
                          lightvqa_swin: Optional[Path]) -> None:
    """
    Run selected metrics on the prepared dataset.

    This uses batch_size=1 to avoid variable-length frame collation issues.
    Results are written by each metric to the current working directory, e.g.
    fid_results.json, is_results.json, fvd_results.json, gstvqa_results.json,
    simplevqa_results.json, lightvqaplus_results.json.
    """
    # Ensure legacy absolute imports like 'from core.registry import ...' resolve
    # by aliasing 'core' -> 'aigve.core' before importing aigve subpackages.
    try:
        import core as _core  # type: ignore
    except ModuleNotFoundError:
        import importlib
        _core = importlib.import_module("aigve.core")
        sys.modules.setdefault("core", _core)
    # Lazy imports to avoid importing heavy modules unless needed
    from aigve.datasets.fid_dataset import FidDataset
    from aigve.metrics.video_quality_assessment.distribution_based.fid_metric import FIDScore
    from aigve.metrics.video_quality_assessment.distribution_based.is_score_metric import ISScore
    from aigve.metrics.video_quality_assessment.distribution_based.fvd.fvd_metric import FVDScore
    import torch

    device_ok = torch.cuda.is_available()
    gpu_flag = bool(use_gpu and device_ok)
    if use_gpu and not device_ok:
        print("[WARN] --use-cpu not set but CUDA not available; falling back to CPU.")

    # Build dataset for distribution metrics
    fid_dataset = FidDataset(video_dir=str(video_dir),
                             prompt_dir=str(prompt_json),
                             max_len=max_len,
                             if_pad=if_pad)

    print(f"\n[Metrics] Using video_dir={video_dir}")
    print(f"[Metrics] Using prompt_json={prompt_json}")
    print(f"[Metrics] max_len={max_len} if_pad={if_pad} device={'cuda' if gpu_flag else 'cpu'}")

    # Helper to iterate FidDataset items safely
    def iter_fid_samples():
        for idx in range(len(fid_dataset)):
            gt_tensor, gen_tensor, gt_name, gen_name = fid_dataset[idx]
            yield gt_tensor, gen_tensor, gt_name, gen_name

    if 'fid' in metrics:
        print("\n[Metrics] Computing FID...")
        fid_metric = FIDScore(is_gpu=gpu_flag)
        for gt_tensor, gen_tensor, gt_name, gen_name in iter_fid_samples():
            data_samples = ((gt_tensor,), (gen_tensor,), (gt_name,), (gen_name,))
            fid_metric.process(data_batch={}, data_samples=data_samples)
        fid_summary = fid_metric.compute_metrics([])
        print(f"[Metrics] FID summary: {fid_summary}")

    if 'is' in metrics:
        print("\n[Metrics] Computing IS...")
        is_metric = ISScore(is_gpu=gpu_flag)
        for _, gen_tensor, _, gen_name in iter_fid_samples():
            data_samples = ((), (gen_tensor,), (), (gen_name,))
            is_metric.process(data_batch={}, data_samples=data_samples)
        is_summary = is_metric.compute_metrics([])
        print(f"[Metrics] IS summary: {is_summary}")

    if 'fvd' in metrics:
        print("\n[Metrics] Computing FVD...")
        default_model = (Path(__file__).resolve().parent.parent /
                         'aigve/metrics/video_quality_assessment/distribution_based/fvd/model_rgb.pth')
        model_path = Path(fvd_model).expanduser().resolve() if fvd_model else default_model
        print(f"[Metrics] FVD model checkpoint: {model_path} (missing allowed)")
        fvd_metric = FVDScore(model_path=str(model_path), is_gpu=gpu_flag)
        for gt_tensor, gen_tensor, gt_name, gen_name in iter_fid_samples():
            data_samples = ((gt_tensor,), (gen_tensor,), (gt_name,), (gen_name,))
            fvd_metric.process(data_batch={}, data_samples=data_samples)
        fvd_summary = fvd_metric.compute_metrics([])
        print(f"[Metrics] FVD summary: {fvd_summary}")

    # Neural network-based video-only metrics
    root_dir = Path(__file__).resolve().parent.parent

    if 'gstvqa' in metrics:
        print("\n[Metrics] Computing GSTVQA...")
        from aigve.datasets.gstvqa_dataset import GSTVQADataset
        from aigve.metrics.video_quality_assessment.nn_based.gstvqa.gstvqa_metric import GstVqa

        gstvqa_default = (root_dir / 'aigve/metrics/video_quality_assessment/nn_based/gstvqa/'
                          'GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/models/'
                          'training-all-data-GSTVQA-konvid-EXP0-best')
        gstvqa_path = Path(gstvqa_model).expanduser().resolve() if gstvqa_model else gstvqa_default
        if not gstvqa_path.exists():
            print(f"[WARN] GSTVQA model not found at {gstvqa_path}; skipping GSTVQA.")
        else:
            dataset = GSTVQADataset(video_dir=str(video_dir),
                                    prompt_dir=str(prompt_json),
                                    model_name='vgg16',
                                    max_len=max_len)
            metric = GstVqa(model_path=str(gstvqa_path))
            for idx in range(len(dataset)):
                deep_features, num_frames, video_name = dataset[idx]
                data_samples = ((deep_features,), (num_frames,), (video_name,))
                metric.process(data_batch={}, data_samples=data_samples)
            summary = metric.compute_metrics([])
            print(f"[Metrics] GSTVQA summary: {summary}")

    if 'simplevqa' in metrics:
        print("\n[Metrics] Computing SimpleVQA...")
        from aigve.datasets.simplevqa_dataset import SimpleVQADataset
        from aigve.metrics.video_quality_assessment.nn_based.simplevqa.simplevqa_metric import SimpleVqa

        simplevqa_default = (root_dir / 'aigve/metrics/video_quality_assessment/nn_based/simplevqa/'
                             'SimpleVQA/ckpts/UGC_BVQA_model.pth')
        simplevqa_path = Path(simplevqa_model).expanduser().resolve() if simplevqa_model else simplevqa_default
        if not simplevqa_path.exists():
            print(f"[WARN] SimpleVQA model not found at {simplevqa_path}; skipping SimpleVQA.")
        else:
            dataset = SimpleVQADataset(video_dir=str(video_dir),
                                       prompt_dir=str(prompt_json),
                                       min_video_seconds=8)
            metric = SimpleVqa(model_path=str(simplevqa_path), is_gpu=gpu_flag)
            for idx in range(len(dataset)):
                spatial_features, motion_features, video_name = dataset[idx]
                # Convert list of motion tensors to a list of single-item tuples to emulate batch dim
                motion_features_batched = [(mf,) for mf in motion_features]
                data_samples = ((spatial_features,), motion_features_batched, (video_name,))
                metric.process(data_batch={}, data_samples=data_samples)
            summary = metric.compute_metrics([])
            print(f"[Metrics] SimpleVQA summary: {summary}")

    if 'lightvqa+' in metrics:
        print("\n[Metrics] Computing LightVQA+...")
        from aigve.datasets.lightvqa_plus_dataset import LightVQAPlusDataset
        from aigve.metrics.video_quality_assessment.nn_based.lightvqa_plus.lightvqa_plus_metric import LightVQAPlus

        lightvqa_model_default = (root_dir / 'aigve/metrics/video_quality_assessment/nn_based/lightvqa_plus/'
                                  'Light_VQA_plus/ckpts/last2_SI+TI_epoch_19_SRCC_0.925264.pth')
        lightvqa_swin_default = (root_dir / 'aigve/metrics/video_quality_assessment/nn_based/lightvqa_plus/'
                                 'Light_VQA_plus/swin_small_patch4_window7_224.pth')
        lvqa_model_path = Path(lightvqa_model).expanduser().resolve() if lightvqa_model else lightvqa_model_default
        lvqa_swin_path = Path(lightvqa_swin).expanduser().resolve() if lightvqa_swin else lightvqa_swin_default
        if not lvqa_model_path.exists():
            print(f"[WARN] LightVQA+ model not found at {lvqa_model_path}; skipping LightVQA+.")
        else:
            dataset = LightVQAPlusDataset(video_dir=str(video_dir),
                                          prompt_dir=str(prompt_json),
                                          min_video_seconds=8)
            metric = LightVQAPlus(model_path=str(lvqa_model_path), swin_weights=str(lvqa_swin_path), is_gpu=gpu_flag)
            for idx in range(len(dataset)):
                spatial_features, temporal_features, bns_features, bc_features, video_name = dataset[idx]
                data_samples = ((spatial_features,), (temporal_features,), (bns_features,), (bc_features,), (video_name,))
                metric.process(data_batch={}, data_samples=data_samples)
            summary = metric.compute_metrics([])
            print(f"[Metrics] LightVQA+ summary: {summary}")


def main():
    ap = argparse.ArgumentParser(description="Prepare AIGVE annotations for reference-based metrics.")
    ap.add_argument("--input-dir", required=True, help="Directory containing mixed GT and generated videos")
    ap.add_argument("--out-json", default=None, help="Where to write the JSON. If --stage-dataset is set, this is ignored.")
    ap.add_argument("--generated-suffixes", default="synthetic,generated", help="Comma-separated list of suffix names appended to GT basenames (e.g., 'synthetic,generated'). The script tries both '_suffix' and '-suffix' variants.")
    ap.add_argument("--stage-dataset", default=None, help="Optional destination root to create an AIGVE-like dataset layout (evaluate/ and annotations/).")
    ap.add_argument("--link", action="store_true", help="When staging, symlink instead of copy.")
    # Metric execution options
    ap.add_argument("--compute", action="store_true", help="If set, compute metrics after preparing annotations.")
    ap.add_argument(
        "--metrics",
        default="all",
        help=(
            "CSV of metrics or categories to run: "
            "fid,is,fvd,gstvqa,simplevqa,lightvqa+ or categories distribution_based,nn_based_video. "
            "'all' maps to distribution_based (fid,is,fvd) for backward compatibility."
        ),
    )
    ap.add_argument("--max-len", type=int, default=64, help="Max frames to read per video for evaluation.")
    ap.add_argument("--pad", action="store_true", help="Pad videos to exactly --max-len frames.")
    ap.add_argument("--use-cpu", action="store_true", help="Force CPU even if CUDA is available.")
    ap.add_argument("--fvd-model", default=None, help="Optional path to I3D/R3D checkpoint for FVD. If missing, default weights are used.")
    # NN-based metric model arguments
    ap.add_argument("--gstvqa-model", default=None, help="Optional path to GSTVQA checkpoint. If missing, tries default bundled path.")
    ap.add_argument("--simplevqa-model", default=None, help="Optional path to SimpleVQA checkpoint (UGC_BVQA_model.pth). If missing, tries default bundled path.")
    ap.add_argument("--lightvqa-plus-model", default=None, help="Optional path to LightVQA+ checkpoint (.pth). If missing, tries default bundled path.")
    ap.add_argument("--lightvqa-plus-swin", default=None, help="Optional path to Swin weights for LightVQA+. If missing, tries default bundled path.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {input_dir}")

    pairs, issues = discover_pairs(input_dir, args.generated_suffixes)
    print(f"Discovered pairs: {len(pairs)}")
    if issues["no_gt"]:
        print(f"[WARN] Generated without GT: {len(issues['no_gt'])} examples (e.g., {issues['no_gt'][:5]})")
    if issues["no_gen"]:
        print(f"[WARN] GT without generated counterpart: {len(issues['no_gen'])} examples (e.g., {issues['no_gen'][:5]})")
    if issues["duplicates"]:
        print(f"[WARN] Multiple generated per GT: {len(issues['duplicates'])} examples (first: {issues['duplicates'][0]})")

    if not pairs:
        raise SystemExit("No GT/GEN pairs found. Check --generated-suffixes and file naming.")

    if args.stage_dataset:
        dest_root = Path(args.stage_dataset).expanduser().resolve()
        eval_dir = stage_dataset(pairs, dest_root, link=args.link)
        out_json = dest_root / "annotations" / "evaluate.json"
        write_json([(eval_dir / gen.name, eval_dir / gt.name) for gen, gt in pairs], out_json, relative_to=eval_dir)
        print(f"Staged dataset at: {dest_root}")
        print(f"  video_dir: {eval_dir}")
        print(f"  prompt_dir: {out_json}")
        video_dir_path = eval_dir
        prompt_json_path = out_json
    else:
        # Write JSON relative to the input_dir
        out_json = Path(args.out_json).expanduser().resolve() if args.out_json else (input_dir / "annotations.json")
        write_json(pairs, out_json, relative_to=input_dir)
        print(f"Wrote annotations: {out_json}")
        print(f"  video_dir: {input_dir}")
        print(f"  prompt_dir: {out_json}")
        video_dir_path = input_dir
        prompt_json_path = out_json

    # Optionally run metrics
    if args.compute:
        metrics_list = _parse_metrics_list(args.metrics)
        if not metrics_list:
            print("[INFO] --compute set but no metrics selected; skipping evaluation.")
        else:
            run_reference_metrics(video_dir=video_dir_path,
                                  prompt_json=prompt_json_path,
                                  metrics=metrics_list,
                                  use_gpu=(not args.use_cpu),
                                  max_len=args.max_len,
                                  if_pad=args.pad,
                                  fvd_model=Path(args.fvd_model) if args.fvd_model else None,
                                  gstvqa_model=Path(args.gstvqa_model) if args.gstvqa_model else None,
                                  simplevqa_model=Path(args.simplevqa_model) if args.simplevqa_model else None,
                                  lightvqa_model=Path(args.lightvqa_plus_model) if args.lightvqa_plus_model else None,
                                  lightvqa_swin=Path(args.lightvqa_plus_swin) if args.lightvqa_plus_swin else None)
    else:
        print("\nUse in config (fid.py example):")
        print("  val_dataloader = dict(")
        print("    dataset=dict(")
        print("      type='datasets.FidDataset',")
        print("      video_dir='<video_dir shown above>',")
        print("      prompt_dir='<prompt_dir shown above>',")
        print("      max_len=64, if_pad=False))")


if __name__ == "__main__":
    main()
