# About us

{{toolkit}} is a website hosting the documentations, tutorials, examples and the latest updates about the `AIGVE` library.

## 🚀 What is `AIGVE`?

`AIGVE` (**AI Generated Video Evaluation Toolkit**) provides a **comprehensive** and **structured** evaluation framework for assessing AI-generated video quality developed by the [IFM Lab](https://www.ifmlab.org/). It integrates multiple evaluation metrics, covering diverse aspects of video evaluation, including neural-network-based assessment, distribution comparison, vision-language alignment, and multi-faceted analysis.

* Official Website: [https://www.aigve.org/](https://www.aigve.org/)
* Github Repository: [https://github.com/ShaneXiangH/VQA_Toolkit](https://github.com/ShaneXiangH/VQA_Toolkit)
<!-- * PyPI Package: [https://pypi.org/project/tinybig/](https://pypi.org/project/tinybig/) -->
* IFM Lab [https://www.ifmlab.org/](https://www.ifmlab.org/)

## Citing Us

If you find `AIGVE` library and `...` papers useful in your work, please cite the papers as follows:
```

```

## Library Organization


### 📊 **Distribution Comparison-Based Evaluation Metrics**
These metrics assess the quality of generated videos by comparing the distribution of real and generated samples.

- ✅ **[FID](../documentations/metrics/fid.md)**: Frechet Inception Distance (FID) quantifies the similarity between real and generated video feature distributions by measuring the Wasserstein-2 distance.
- ✅ **[FVD](../documentations/metrics/fvd.md)**: Frechet Video Distance (FVD) extends the FID approach to video domain by leveraging spatio-temporal features extracted from action recognition networks.
- ✅ **[IS](../documentations/metrics/is_score.md)**: Inception Score (IS) evaluates both the quality and diversity of generated content by analyzing conditional label distributions.

---

### 🧠 **Video-only Neural Network-Based Evaluation Metrics**
These metrics leverage deep learning models to assess AI-generated video quality based on learned representations.

- ✅ **[GSTVQA](../documentations/metrics/gstvqa.md)**: Generalized Spatio-Temporal VQA (GSTVQA) employs graph-based spatio-temporal analysis to assess video quality.
- ✅ **[SimpleVQA](../documentations/metrics/simplevqa.md)**: Simple Video Quality Assessment (Simple-VQA) utilizes deep learning features for no-reference video quality assessment.
- ✅ **[LightVQA+](../documentations/metrics/lightvqaplus.md)**: Light Video Quality Assessment Plus (Light-VQA+) incorporates exposure quality guidance to evaluate video quality.

---

### 🔍 **Vision-Language Similarity-Based Evaluation Metrics**
These metrics evaluate **alignment, similarity, and coherence** between visual and textual representations, often using embeddings from models like CLIP and BLIP.

- ✅ **[CLIPSim](../documentations/metrics/clipsim.md)**: CLIP Similarity (CLIPSim) leverages CLIP embeddings to measure semantic similarity between videos and text.
- ✅ **[CLIPTemp](../documentations/metrics/cliptemp.md)**: CLIP Temporal (CLIPTemp) extends CLIPSim by incorporating temporal consistency assessment.
- ✅ **[BLIPSim](../documentations/metrics/blipsim.md)**: Bootstrapped Language-Image Pre-training Similarity (BLIPSim) uses advanced pre-training techniques to improve video-text alignment evaluation.
- ✅ **[Pickscore](../documentations/metrics/pickscore.md)**: PickScore incorporates human preference data to provide more perceptually aligned measurement of video-text matching.

---

### 🧠 **Vision-Language Understanding-Based Evaluation Metrics**
These metrics assess **higher-level understanding, reasoning, and factual consistency** in vision-language models.

- ✅ **[VIEScore](../documentations/metrics/viescore.md)**: Video Information Evaluation Score (VIEScore) provides explainable assessments of conditional image synthesis.
- ✅ **[TIFA](../documentations/metrics/tifa.md)**: Text-Image Faithfulness Assessment (TIFA) employs question-answering techniques to evaluate text-to-image alignment.
- ✅ **[DSG](../documentations/metrics/dsg.md)**: Davidsonian Scene Graph (DSG) improves fine-grained evaluation reliability through advanced scene graph representations.

---

### 🔄 **Multi-Faceted Evaluation Metrics**
These metrics integrate **structured, multi-dimensional assessments** to provide a **holistic benchmarking framework** for AI-generated videos.

- ✅ **[VideoPhy](../documentations/metrics/videophy.md)**: Video Physics Evaluation (VideoPhy) specifically assesses the physical plausibility of generated videos.
- ✅ **[VideoScore](../documentations/metrics/viescore.md)**: Video Score (VideoScore) simulates fine-grained human feedback across multiple evaluation dimensions.
---

## Key Features
- **Multi-Dimensional Evaluation**: Covers video coherence, physics, and benchmarking.
- **Open-Source & Customizable**: Designed for easy integration.
- **Cutting-Edge AI Assessment**: Supports various AI-generated video tasks.

---

<!-- | Components                                                                            | Descriptions                                                                                     |
|:--------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| [`tinybig`]()                          | a deep function learning library like torch.nn, deeply integrated with autograd                  |
| [`tinybig.model`]()                      | a library providing the RPN models for addressing various deep function learning tasks           | -->
                                  


## License & Copyright

Copyright © 2025 [IFM Lab](https://www.ifmlab.org/). All rights reserved.

* `AIGVE` source code is published under the terms of the MIT License. 
* `AIGVE` documentation and the `...` papers are licensed under a Creative Commons Attribution-Share Alike 4.0 Unported License ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)). 