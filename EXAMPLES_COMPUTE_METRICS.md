# Examples: Computing Metrics with AIGVE

This guide shows you how to compute metrics using AIGVE. There are multiple ways depending on your setup.

## 🎯 Quick Start: Upload Two Videos Directly (Easiest Method!)

**If you just have a reference video and a generated video, this is the simplest way!**

### Using Python Client (Recommended)

```bash
# 1. Start the API server first
docker run -d --name aigve --restart unless-stopped \
    --gpus '"device=1"' -p 2200:2200 \
    -v "$PWD/data":/app/data -v "$PWD/out":/app/out \
    ghcr.io/bmwas/aigve:latest

# 2. Compute metrics with explicit video paths (RECOMMENDED!)
python scripts/call_aigve_api.py \
    --reference-video ./my_real_video.mp4 \
    --generated-video ./my_ai_generated_video.mp4 \
    --categories distribution_based
```

**That's it!** This will automatically:
- ✅ Detect FPS from the reference video
- ✅ Use the full duration of the reference video  
- ✅ Handle naming/pairing automatically
- ✅ Compute all metrics (FID, IS, FVD, CD-FVD)

**Output example:**
```
📹 Video pair specified explicitly:
   Reference:  my_real_video.mp4
   Generated:  my_ai_generated_video.mp4
   Auto-suffix: 'generated' (identifies generated video)
   Auto-detect: enabled (FPS and duration from reference)

[auto-detect] ✅ Detected properties:
              FPS: 30.00
              Duration: 12.50 seconds
              Total frames: 375
```

### Short form (using aliases):

```bash
python scripts/call_aigve_api.py \
    --ref ./reference.mp4 \
    --gen ./generated.mp4 \
    --categories distribution_based
```

### Limit duration (optional):

```bash
# Use only first 5 seconds of the video
python scripts/call_aigve_api.py \
    --reference-video ./reference.mp4 \
    --generated-video ./generated.mp4 \
    --max-seconds 5 \
    --categories distribution_based
```

### Using curl (Direct API Call)

```bash
# 1. Start the API server first (see above)

# 2. Upload videos directly via curl
curl -X POST http://localhost:2200/run_upload \
    -F "videos=@./reference_video.mp4" \
    -F "videos=@./generated_video.mp4" \
    -F "categories=distribution_based" \
    -F "max_seconds=8" \
    -F "fps=25" \
    -F "compute=true"
```

**What this computes:**
- ✅ FID (Fréchet Inception Distance)
- ✅ IS (Inception Score)
- ✅ FVD (Fréchet Video Distance)
- ✅ CD-FVD (8 flavors automatically)

**No directories needed!** Just provide the two video files directly.

---

## Available Metrics

### Distribution-Based Metrics (Reference-based)
- **FID** (Fréchet Inception Distance)
- **IS** (Inception Score)
- **FVD** (Fréchet Video Distance)

### Neural Network-Based Video Metrics (No reference needed)
- **GSTVQA** - Video quality assessment
- **SimpleVQA** - Video quality assessment
- **LightVQA+** - Video quality assessment

### CD-FVD Metrics (via API)
- **CD-FVD** - 8 flavors (2 models × 2 resolutions × 2 sequence lengths)

---

## Method 1: Using `scripts/prepare_annotations.py` (RECOMMENDED)

**This is the main script that can compute ALL metrics at once!**

### Prerequisites
Your video directory should contain:
- Ground-truth videos (e.g., `video1.mp4`, `video2.mp4`)
- Generated videos with suffixes (e.g., `video1_synthetic.mp4`, `video2_generated.mp4`)

The script automatically pairs them based on filename suffixes.

### Example 1: Compute ALL Distribution-Based Metrics (FID, IS, FVD)

```bash
# Basic usage - computes FID, IS, and FVD
python scripts/prepare_annotations.py \
    --input-dir ./data/videos \
    --stage-dataset ./out/staged \
    --compute \
    --categories distribution_based \
    --max-seconds 8 \
    --fps 25
```

**Output files:**
- `fid_results.json`
- `is_results.json`
- `fvd_results.json`

### Example 2: Compute ALL Metrics (Distribution + NN-based)

```bash
# Compute ALL available metrics (FID, IS, FVD, GSTVQA, SimpleVQA, LightVQA+)
python scripts/prepare_annotations.py \
    --input-dir ./data/videos \
    --stage-dataset ./out/staged \
    --compute \
    --categories distribution_based,nn_based_video \
    --max-seconds 8 \
    --fps 25 \
    --gstvqa-model /path/to/GSTVQA.ckpt \
    --simplevqa-model /path/to/UGC_BVQA_model.pth \
    --lightvqa-plus-model /path/to/lightvqa_plus.pth \
    --lightvqa-plus-swin /path/to/swin_small_patch4_window7_224.pth
```

**Output files:**
- `fid_results.json`
- `is_results.json`
- `fvd_results.json`
- `gstvqa_results.json`
- `simplevqa_results.json`
- `lightvqaplus_results.json`

### Example 3: Compute Specific Metrics Only

```bash
# Compute only FID and IS (skip FVD)
python scripts/prepare_annotations.py \
    --input-dir ./data/videos \
    --stage-dataset ./out/staged \
    --compute \
    --metrics fid,is \
    --max-seconds 8 \
    --fps 25
```

### Example 4: Mix Categories and Specific Metrics

```bash
# Compute all distribution-based metrics + SimpleVQA
python scripts/prepare_annotations.py \
    --input-dir ./data/videos \
    --stage-dataset ./out/staged \
    --compute \
    --categories distribution_based \
    --metrics simplevqa \
    --simplevqa-model /path/to/UGC_BVQA_model.pth \
    --max-seconds 8 \
    --fps 25
```

### Example 5: List Available Metrics

```bash
# See what metrics are available
python scripts/prepare_annotations.py --list-metrics
```

### Example 6: Custom Suffixes for Generated Videos

```bash
# If your generated videos use different suffixes
python scripts/prepare_annotations.py \
    --input-dir ./data/videos \
    --stage-dataset ./out/staged \
    --compute \
    --categories distribution_based \
    --generated-suffixes synthetic,generated,ai \
    --max-seconds 8 \
    --fps 25
```

### Example 7: Using Docker Container

```bash
# Run inside Docker container (using GPU device=1)
docker run --rm --gpus '"device=1"' \
    -v "$PWD/data":/app/data \
    -v "$PWD/out":/app/out \
    ghcr.io/bmwas/aigve:latest \
    python scripts/prepare_annotations.py \
        --input-dir /app/data/videos \
        --stage-dataset /app/out/staged \
        --compute \
        --categories distribution_based \
        --max-seconds 8 \
        --fps 25
```

---

## Method 2: Using `compute_metrics_standalone.py` (Simple - FID, IS, FVD only)

This script computes only the 3 distribution-based metrics (FID, IS, FVD).

### Example: Compute FID, IS, FVD

```bash
python compute_metrics_standalone.py \
    --video-dir ./data/videos \
    --annotation-file ./data/annotations.json \
    --max-len 200 \
    --output-json ./results/metrics.json
```

**Note:** This requires you to have the annotation JSON file already prepared. Use `prepare_annotations.py` first to generate it, or use Method 1 which does both.

---

## Method 3: Using API Client (`scripts/call_aigve_api.py`)

This method uses the REST API server. **This is the best method if you have individual video files!**

### Step 1: Start API Server

```bash
# Start the API server (using GPU device=1)
docker run -d --name aigve --restart unless-stopped \
    --gpus '"device=1"' -p 2200:2200 \
    -v "$PWD/data":/app/data \
    -v "$PWD/out":/app/out \
    ghcr.io/bmwas/aigve:latest
```

### Step 2: Compute Metrics via API

#### Option A: Explicit Video Pair (⭐ RECOMMENDED)

**This is the simplest and most intuitive method!**

```bash
# Specify reference and generated videos explicitly
python scripts/call_aigve_api.py \
    --reference-video ./my_real_video.mp4 \
    --generated-video ./my_ai_video.mp4 \
    --categories distribution_based
```

This automatically:
- ✅ Detects FPS from the reference video
- ✅ Uses the full duration of the reference video
- ✅ Handles pairing without needing special filenames
- ✅ No manual specification needed!

**Short form using aliases:**
```bash
python scripts/call_aigve_api.py \
    --ref ./real.mp4 \
    --gen ./output.mp4 \
    --categories distribution_based
```

**Limit duration:**
```bash
# Use only first 10 seconds
python scripts/call_aigve_api.py \
    --ref ./real.mp4 \
    --gen ./output.mp4 \
    --max-seconds 10 \
    --categories distribution_based
```

**Requirements:**
- Exactly 2 videos (1 reference + 1 generated)
- Generated video filename should contain "synthetic" or "generated" (or use `--generated-suffixes custom_suffix`)

**Example with custom suffix:**
```bash
# If your generated video is named "output.mp4" instead of "output_synthetic.mp4"
python scripts/call_aigve_api.py \
    --upload-files ./reference.mp4 ./output.mp4 \
    --generated-suffixes output \
    --max-seconds 8 \
    --fps 25 \
    --categories distribution_based
```

#### Option B: Use Server-Side Paths (for directories)

```bash
# Compute metrics on videos already in the container
python scripts/call_aigve_api.py \
    --input-dir /app/data \
    --stage-dataset /app/out/staged \
    --max-seconds 8 \
    --fps 25 \
    --categories distribution_based
```

**Note:** The API also supports CD-FVD metrics (8 flavors) automatically when using `distribution_based` category.

---

## Method 4: Direct API Calls (curl)

### Example 1: Upload Videos Directly (⭐ Best for individual videos)

```bash
# First, ensure API server is running (see Method 3, Step 1)

# Upload two videos directly and compute metrics
curl -X POST http://localhost:2200/run_upload \
  -F "videos=@./reference_video.mp4" \
  -F "videos=@./generated_video.mp4" \
  -F "categories=distribution_based" \
  -F "max_seconds=8" \
  -F "fps=25" \
  -F "compute=true"
```

### Example 2: Use Server-Side Paths (for directories)

```bash
# Compute distribution-based metrics on videos in container
curl -X POST http://localhost:2200/run \
  -H 'Content-Type: application/json' \
  -d '{
    "input_dir": "/app/data",
    "stage_dataset": "/app/out/staged",
    "compute": true,
    "categories": "distribution_based",
    "max_seconds": 8,
    "fps": 25
  }'
```

---

## Quick Reference: Common Use Cases

### ✅ I have just 2 videos (reference + generated) - EASIEST!
```bash
# Start server
docker run -d --name aigve --restart unless-stopped \
    --gpus '"device=1"' -p 2200:2200 \
    -v "$PWD/data":/app/data -v "$PWD/out":/app/out \
    ghcr.io/bmwas/aigve:latest

# Compute metrics (auto-detects FPS and duration from reference video)
python scripts/call_aigve_api.py \
    --reference-video ./my_real_video.mp4 \
    --generated-video ./my_ai_video.mp4 \
    --categories distribution_based
```

**Or with short aliases:**
```bash
python scripts/call_aigve_api.py \
    --ref ./real.mp4 \
    --gen ./generated.mp4 \
    --categories distribution_based
```

### ✅ I want to compute ALL metrics at once
```bash
python scripts/prepare_annotations.py \
    --input-dir ./data/videos \
    --stage-dataset ./out/staged \
    --compute \
    --categories distribution_based,nn_based_video \
    --max-seconds 8 --fps 25 \
    --gstvqa-model /path/to/GSTVQA.ckpt \
    --simplevqa-model /path/to/UGC_BVQA_model.pth \
    --lightvqa-plus-model /path/to/lightvqa_plus.pth \
    --lightvqa-plus-swin /path/to/swin_small_patch4_window7_224.pth
```

### ✅ I want to compute only FID, IS, FVD (most common)
```bash
python scripts/prepare_annotations.py \
    --input-dir ./data/videos \
    --stage-dataset ./out/staged \
    --compute \
    --categories distribution_based \
    --max-seconds 8 --fps 25
```

### ✅ I want to compute metrics via Docker
```bash
docker run --rm --gpus '"device=1"' \
    -v "$PWD/data":/app/data \
    -v "$PWD/out":/app/out \
    ghcr.io/bmwas/aigve:latest \
    python scripts/prepare_annotations.py \
        --input-dir /app/data \
        --stage-dataset /app/out/staged \
        --compute \
        --categories distribution_based \
        --max-seconds 8 --fps 25
```

### ✅ I want to use the API server
```bash
# 1. Start server
docker run -d --name aigve --restart unless-stopped \
    --gpus '"device=1"' -p 2200:2200 \
    -v "$PWD/data":/app/data -v "$PWD/out":/app/out \
    ghcr.io/bmwas/aigve:latest

# 2. Compute metrics
python scripts/call_aigve_api.py \
    --input-dir /app/data \
    --stage-dataset /app/out/staged \
    --max-seconds 8 --fps 25 \
    --categories distribution_based
```

---

## Understanding the Output

### Distribution-Based Metrics Output
- **FID**: Lower is better (measures distribution similarity)
- **IS**: Higher is better (measures quality and diversity)
- **FVD**: Lower is better (measures video distribution similarity)

### NN-Based Metrics Output
- **GSTVQA, SimpleVQA, LightVQA+**: Quality scores (interpretation depends on the metric)

### Result Files
All results are saved as JSON files in your current working directory:
- `fid_results.json`
- `is_results.json`
- `fvd_results.json`
- `gstvqa_results.json`
- `simplevqa_results.json`
- `lightvqaplus_results.json`

---

## Tips

1. **Start with distribution_based**: These metrics (FID, IS, FVD) are the most commonly used and don't require additional model files.

2. **Use --max-seconds**: More intuitive than frame counts. Default FPS is 25.0.

3. **Check your video naming**: Generated videos must contain suffixes like `_synthetic` or `_generated` in their filenames.

4. **GPU vs CPU**: By default, the script uses GPU if available. Use `--use-cpu` to force CPU (slower).

5. **Docker paths**: When using Docker, use container paths like `/app/data` and `/app/out`, not host paths.

---

## Troubleshooting

### "No metrics selected"
- Make sure you use `--compute` flag
- Specify `--categories` or `--metrics`

### "No video pairs found"
- Check your video filenames match the suffix pattern
- Use `--generated-suffixes` to specify custom suffixes

### "CUDA not available"
- Ensure Docker has GPU access: `docker run --rm --gpus '"device=1"' nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
- Check NVIDIA drivers are installed

