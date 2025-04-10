# What is `AIGVE`?

`AIGVE` (**AI Generated Video Evaluation Toolkit**) provides a **comprehensive** and **structured** evaluation framework for assessing AI-generated video quality developed by the [IFM Lab](https://www.ifmlab.org/). It integrates multiple evaluation metrics, covering diverse aspects of video evaluation, including neural-network-based assessment, distribution comparison, vision-language alignment, and multi-faceted analysis.

* **Official Website**: [https://www.aigve.org/](https://www.aigve.org/)
* **Github Repository**: [https://github.com/ShaneXiangH/AIGVE_Tool](https://github.com/ShaneXiangH/AIGVE_Tool)
* **PyPI Package**: [https://pypi.org/project/aigve/](https://pypi.org/project/aigve/)
* **IFM Lab** [https://www.ifmlab.org/](https://www.ifmlab.org/)

![AIGVE Toolkit Structure](docs/assets/img/toolkit_structure.png)


## Library Organization

| Components                                                                  | Descriptions                                                          |
|:----------------------------------------------------------------------------|:----------------------------------------------------------------------|
| [`aigve`](https://www.aigve.org/documentations/aigve/)                      | The library for assessing AI-generated video quality                  |
| [`aigve.configs`](https://www.aigve.org/documentations/configs/)              | a library for parameter configuration and management                  |
| [`aigve.core`](https://www.aigve.org/documentations/core/)                  | a library for video evaluation process design                         |
| [`aigve.datasets`](https://www.aigve.org/documentations/datasets/)          | a library for dataset loading design                                  |
| [`aigve.metrics`](https://www.aigve.org/documentations/metrics/)            | a library for video evaluation metrics design and building            |
| [`aigve.utils`](https://www.aigve.org/documentations/utils/)                | a library for utility function definition                             |



## Evaluation Metrics Zoo

###  **Distribution Comparison-Based Evaluation Metrics**

<!-- <h2><img src="../../assets/icons/dis_based.png" alt="chart icon" style="height: 1.2em; vertical-align: middle;"> <strong>Distribution Comparison-Based Evaluation Metrics</strong></h2> -->

These metrics assess the quality of generated videos by comparing the distribution of real and generated samples.

- ✅ **[FID](aigve/configs/fid.py)**: Frechet Inception Distance (FID) quantifies the similarity between real and generated video feature distributions by measuring the Wasserstein-2 distance.
- ✅ **[FVD](aigve/configs/fvd.p)**: Frechet Video Distance (FVD) extends the FID approach to video domain by leveraging spatio-temporal features extracted from action recognition networks.
- ✅ **[IS](aigve/configs/is_score.py)**: Inception Score (IS) evaluates both the quality and diversity of generated content by analyzing conditional label distributions.

---

### **Video-only Neural Network-Based Evaluation Metrics**
These metrics leverage deep learning models to assess AI-generated video quality based on learned representations.

- ✅ **[GSTVQA](aigve/configs/gstvqa.py)**: Generalized Spatio-Temporal VQA (GSTVQA) employs graph-based spatio-temporal analysis to assess video quality.
- ✅ **[SimpleVQA](aigve/configs/simplevqa.py)**: Simple Video Quality Assessment (Simple-VQA) utilizes deep learning features for no-reference video quality assessment.
- ✅ **[LightVQA+](aigve/configs/lightvqa_plus.py)**: Light Video Quality Assessment Plus (Light-VQA+) incorporates exposure quality guidance to evaluate video quality.

---

### **Vision-Language Similarity-Based Evaluation Metrics**
These metrics evaluate **alignment, similarity, and coherence** between visual and textual representations, often using embeddings from models like CLIP and BLIP.

- ✅ **[CLIPSim](aigve/configs/clipsim.py)**: CLIP Similarity (CLIPSim) leverages CLIP embeddings to measure semantic similarity between videos and text.
- ✅ **[CLIPTemp](aigve/configs/cliptemp.py)**: CLIP Temporal (CLIPTemp) extends CLIPSim by incorporating temporal consistency assessment.
- ✅ **[BLIPSim](aigve/configs/blipsim.py)**: Bootstrapped Language-Image Pre-training Similarity (BLIPSim) uses advanced pre-training techniques to improve video-text alignment evaluation.
- ✅ **[Pickscore](aigve/configs/pickscore.py)**: PickScore incorporates human preference data to provide more perceptually aligned measurement of video-text matching.

---

### **Vision-Language Understanding-Based Evaluation Metrics**
These metrics assess **higher-level understanding, reasoning, and factual consistency** in vision-language models.

- ✅ **[VIEScore](aigve/configs/viescore.py)**: Video Information Evaluation Score (VIEScore) provides explainable assessments of conditional image synthesis.
- ✅ **[TIFA](aigve/configs/tifa.py)**: Text-Image Faithfulness Assessment (TIFA) employs question-answering techniques to evaluate text-to-image alignment.
- ✅ **[DSG](aigve/configs/dsg.py)**: Davidsonian Scene Graph (DSG) improves fine-grained evaluation reliability through advanced scene graph representations.

---

### **Multi-Faceted Evaluation Metrics**
These metrics integrate **structured, multi-dimensional assessments** to provide a **holistic benchmarking framework** for AI-generated videos.

- ✅ **[VideoPhy](aigve/configs/videophy.py)**: Video Physics Evaluation (VideoPhy) specifically assesses the physical plausibility of generated videos.
- ✅ **[VideoScore](aigve/configs/viescore.py)**: Video Score (VideoScore) simulates fine-grained human feedback across multiple evaluation dimensions.
---




## Key Features
- **Multi-Dimensional Evaluation**: Covers video coherence, physics, and benchmarking.
- **Open-Source & Customizable**: Designed for easy integration.
- **Cutting-Edge AI Assessment**: Supports various AI-generated video tasks.


## Built-in Dataset:
1. [Toy dataset](aigve/data/toy) 
2. [AIGVE-Bench toy](aigve/data/AIGVE_Bench_toy)

## Installation

The `aigve` library has been published at both [PyPI](https://pypi.org/project/aigve/) and the [project github repository](https://github.com/ShaneXiangH/AIGVE_Tool).


### Install from PyPI

To install `aigve` from [PyPI](https://pypi.org/project/aigve/), use the following command:

```shell
pip install aigve
```


### Install from Source Code

You can also install `aigve` from the source code, which has been released at the 
[project github repository](https://github.com/ShaneXiangH/AIGVE_Tool). 

You can download the public repository either from the project github webpage or via the following command:
```shell
git clone https://github.com/ShaneXiangH/AIGVE_Tool.git
```

Please check the [installation page](https://www.aigve.org/guides/installation/#dependency-packages) for dependency packages.

## Environment

```
conda env remove --name aigve
conda env create -f environment.yml
conda activate aigve
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Run:
``
python main.py {metric_config_file}.py
``

Take Examples:

```
rm -rf ~/.cache
cd AIGVE_Tool/aigve
```

For GSTVQA:
``
python main_aigve.py AIGVE_Tool/aigve/configs/gstvqa.py --work-dir ./output
``

For SimpleVQA:
``
python main_aigve.py AIGVE_Tool/aigve/configs/simplevqa.py --work-dir ./output
``

For LightVQAPlus:
``
python main_aigve.py AIGVE_Tool/aigve/configs/lightvqa_plus.py --work-dir ./output
``

For CLIPSim:
``
python main_aigve.py AIGVE_Tool/aigve/configs/clipsim.py --work-dir ./output
``

For VideoPhy:
``
python main_aigve.py AIGVE_Tool/aigve/configs/clipsim.py --work-dir ./output
``

## Citing Us

`aigve` is developed based on the AIGVE-Tool paper from IFM Lab, which can be downloaded via the following links:

* AIGVE-Tool Paper (2025): [https://arxiv.org/abs/2503.14064](https://arxiv.org/abs/2503.14064)

If you find `AIGVE` library and the AIGVE-Tool papers useful in your work, please cite the papers as follows:
```
@article{xiang2025aigvetoolaigeneratedvideoevaluation,
      title={AIGVE-Tool: AI-Generated Video Evaluation Toolkit with Multifaceted Benchmark}, 
      author={Xinhao Xiang and Xiao Liu and Zizhong Li and Zhuosheng Liu and Jiawei Zhang},
      year={2025},
      journal={arXiv preprint arXiv:2503.14064},
      eprint={2503.14064},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.14064}, 
}
```


## Acknowledge

The Toolkit is build top the top of [MMEngine](https://github.com/open-mmlab/mmengine)

We acknowledge original repositories of various AIGVE methods:
[GSTVQA](https://github.com/Baoliang93/GSTVQA),
[CLIPSim](https://github.com/zhengxu-1997/),
<!-- [ModularBVQA](https://github.com/winwinwenwen77/ModularBVQA), -->
<!-- [StarVQA](https://github.com/GZHU-DVL/StarVQA) -->