# Quickstart Tutorial

## 🚀 What is AIGVE?

AIGVE (**AI Generated Video Evaluation Toolkit**) provides a **comprehensive** and **structured** evaluation framework for assessing AI-generated video quality. It integrates multiple evaluation metrics, covering diverse aspects of video evaluation, including neural-network-based assessment, distribution comparison, vision-language alignment, and multi-faceted analysis.

---

### 🧠 **Neural Network-Based Evaluation Metrics**
These metrics leverage deep learning models to assess AI-generated video quality based on learned representations.

- ✅ **[GSTVQA](./configs/gstvqa.py)**: Video Quality Assessment using spatiotemporal deep learning models.
- ✅ **[ModularBVQA]()**: A modular framework for Blind Video Quality Assessment (BVQA).

---

### 📊 **Distribution-Based Evaluation Metrics**
These metrics assess the quality of generated videos by comparing the distribution of real and generated samples.

- ✅ **[FID]()**: Frechet Inception Distance (FID) measures the visual fidelity of generated samples.
- ✅ **[FVD]()**: Frechet Video Distance (FVD) extends FID for temporal coherence in videos.
- ✅ **[IS]()**: Inception Score (IS) evaluates the diversity and realism of generated content.

---

### 🔍 **Vision-Language Similarity-Based Evaluation Metrics**
These metrics evaluate **alignment, similarity, and coherence** between visual and textual representations, often using embeddings from models like CLIP and BLIP.

- ✅ **[CLIPSim](./configs/clipsim.py)**: Measures image-text similarity using CLIP embeddings.
- ✅ **[CLIPTemp](./configs/cliptemp.py)**: Assesses temporal consistency in video-text alignment.
- ✅ **[BLIP](./configs/blipsim.py)**: Evaluates cross-modal similarity and retrieval-based alignment.
- ✅ **[Pickscore](./configs/pickscore.py)**: Ranks text-image pairs based on alignment quality.

---

### 🧠 **Vision-Language Understanding-Based Evaluation Metrics**
These metrics assess **higher-level understanding, reasoning, and factual consistency** in vision-language models.

- ✅ **[VIEScore](./configs/viescore.py)**: Evaluates video grounding and entity-based alignment.
- ✅ **[TIFA](./configs/tifa.py)**: Measures textual integrity and factual accuracy in video descriptions.
- ✅ **[DSG](./configs/dsg.py)**: A deep structured grounding metric for assessing cross-modal comprehension.

---

### 🔄 **Multi-Faceted Evaluation Metrics**
These metrics integrate **structured, multi-dimensional assessments** to provide a **holistic benchmarking framework** for AI-generated videos.

- ✅ **[VideoPhy](./configs/videophy.py)**: Evaluates physics-based video understanding and reasoning.
- ✅ **[VBench]()**: Benchmarking framework covering diverse video evaluation dimensions.
- ✅ **[EvalCrafter]()**: Customizable and modular framework for structured AI evaluation.

---

### 🔥 Key Features
- **Multi-Dimensional Evaluation**: Covers video coherence, physics, and benchmarking.
- **Open-Source & Customizable**: Designed for easy integration.
- **Cutting-Edge AI Assessment**: Supports various AI-generated video tasks.

---