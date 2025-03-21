# AIGVE-Tool
AI Generated Video Evaluation toolkit


## Implemented:

### Models:
#### Video-Only Neural Network-Based evaluation metrics:
1. [GSTVQA](.aigve/configs/gstvqa.py) 
2. [SimpleVQA](.aigve/configs/simplevqa.py) 
3. [LightVQA_Plus](.aigve/configs/lightvqa_plus.py)


#### Distribuition Comparison-Based evaluation metrics:
These metrics primarily assess the quality of generated samples by comparing distributions of real and generated data:
1. [FID](.aigve/configs/fid.py)
2. [FVD](.aigve/configs/fvd.py)
3. [IS](.aigve/configs/is_score.py)

#### Vision-Language Similarity-Based evaluation metrics:
These metrics primarily evaluate alignment, similarity, and coherence between visual and textual representations. They focus on how well images and text match, often using embeddings from models like CLIP and BLIP:
1. [CLIPSim](.aigve/configs/clipsim.py) 
2. [CLIPTemp](.aigve/configs/cliptemp.py) 
3. [BLIP](.aigve/configs/blipsim.py)
4. [Pickscore](.aigve/configs/pickscore.py)

#### Vision-Language Understanding-Based evaluation metrics:
These metrics assess higher-level understanding, reasoning, and factual consistency in vision-language models. They go beyond similarity, evaluating semantic correctness, factual alignment, and structured comprehension:
1. [VIEScore](.aigve/configs/viescore.py) 
2. [TIFA](.aigve/configs/tifa.py)
3. [DSG](.aigve/configs/dsg.py)


#### Multi-Faceted evaluation metrics
These metrics are structured, multi-dimensional evaluation metrics designed to assess AI models across diverse sub-evaluation dimensions. They provide a comprehensive benchmarking framework that integrates aspects like video understanding, physics-based reasoning, and modular evaluation, enabling a more holistic assessment of model performance.
1. [VideoPhy](.aigve/configs/videophy.py)
2. [VideoScore](.aigve/configs/viescore.py)

### Dataset:
1. [Toy dataset](.aigve/data/toy) 
2. [AIGVE-Bench](.aigve/data/AIGVE_Bench_Toy)



## Environment

conda env remove --name aigve
```
conda env create -f environment.yml
conda activate aigve
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
(MMCV from v1.7.2 support PyTorch 2.1.0 and 2.0.0)

## Run:
``
python main.py {metric_config_file}.py
``

Take Examples:

```
rm -rf ~/.cache
```

For GSTVQA:
``
cd VQA_Toolkit/aigve
python main_aigve.py AIGVE_Tool/aigve/configs/gstvqa.py --work-dir ./output
``

For SimpleVQA:
``
cd VQA_Toolkit/aigve
python main_aigve.py AIGVE_Tool/aigve/configs/simplevqa.py --work-dir ./output
``

For LightVQAPlus:
``
cd VQA_Toolkit/aigve
python main_aigve.py AIGVE_Tool/aigve/configs/lightvqa_plus.py --work-dir ./output

``

For GSTVQACrossData:
``
cd VQA_Toolkit/aigve
python main_aigve.py AIGVE_Tool/aigve/configs/gstvqa_crossdata.py --work-dir ./output
``

For CLIPSim:
``
cd VQA_Toolkit/aigve
python main_aigve.py AIGVE_Tool/aigve/configs/clipsim.py --work-dir ./output
``

For VideoPhy:
``
cd VQA_Toolkit/aigve
python main_aigve.py AIGVE_Tool/aigve/configs/clipsim.py --work-dir ./output
``


## Acknowledge

The Toolkit is build top the top of [MMEngine](https://github.com/open-mmlab/mmengine)

We acknowledge original repositories of various VQA methods:
[GSTVQA](https://github.com/Baoliang93/GSTVQA),
[CLIPSim](https://github.com/zhengxu-1997/),
<!-- [ModularBVQA](https://github.com/winwinwenwen77/ModularBVQA), -->
<!-- [StarVQA](https://github.com/GZHU-DVL/StarVQA) -->