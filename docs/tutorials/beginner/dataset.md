# Tutorial on Dataset Preparation

AIGVE supports flexible and modular dataset loading, designed for a wide range of evaluation scenarios. To ensure compatibility with the evaluation loop and metric modules, datasets in AIGVE follow a unified format based on [MMFormat annotation schema](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html). This tutorial introduces how to prepare and organize datasets for use with AIGVE. We will use [AIGVE-Bench toy dataset](https://github.com/ShaneXiangH/AIGVE_Tool/tree/main/aigve/data/AIGVE_Bench_toy) as an example to introduce the complete dataset preparation process, including directory structure, annotation format, and data organization following the AIGVE specification.

---

## Dataset Structure

AIGVE expects datasets to be organized into two major components:

1. **Raw Media Files (e.g., videos or images)**  
   These files contain the visual content to be evaluated. `AIGVE` supoorts both raw videos (.mp4, .avi, etc.) and frame sequences (pre-extracted frames as .mp4, .avi, etc.).

2. **Annotation File**  
   A JSON file containing the prompts, ground truth annotations and metadata necessary for evaluation. Depending on the specific metric, additional fields like reference videos or multi-modal inputs may be required.

---

## MMFormat-Style JSON Annotation

AIGVE standardizes how datasets are defined and used across different evaluation tasks, allowing for flexible training or evaluation within the same framework. The annotation file must be in `.json` format and must contain two fields:

* `metainfo`: a dictionary describing meta information about the dataset.  

* `data_list`: a list of dictionaries where each item defines a sample and its associated groud truth annotations.

Here is a part of annotation file used in [AIGVE-Bench toy dataset](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/data/AIGVE_Bench_toy/annotations/train.json):

```json
{
  "metainfo": {
    "length": 10,
    "split": "train",
    "generators": [
      "cogvideox"
    ]
  },
  "data_list": [
    {
      "id": 0,
      "prompt_gt": "A vast desert expanse, with golden sands swirling under the shimmering heat of midday sun...",
      "technical_quality": 4,
      "dynamic": 1,
      "consistency": 5,
      "physics": 5,
      "element_presentence": 5,
      "element_quality": 5,
      "action_presentence": 0,
      "action_quality": 0,
      "overall": 2,
      "model": "cogvideox",
      "subject": ["desert"],
      "dynamic_type": ["daylight transition"],
      "detailed_dynamic": ["daylight transition"],
      "category": "global",
      "video_path_pd": "cogvideox_0.mp4"
    },
    {
      "id": 2,
      "prompt_gt": "An aerial view of a modern city with sleek skyscrapers...",
      "technical_quality": 4,
      "dynamic": 1,
      "consistency": 5,
      "physics": 3,
      "element_presentence": 5,
      "element_quality": 2,
      "action_presentence": 0,
      "action_quality": 0,
      "overall": 1,
      "model": "cogvideox",
      "subject": ["city", "architecture"],
      "dynamic_type": ["season transition", "camera movement"],
      "detailed_dynamic": ["season transition", "camera movement"],
      "category": "global",
      "video_path_pd": "cogvideox_2.mp4"
    }
  ]
}
```

* `metainfo`: general information about the dataset (e.g., total length, split type, video generators used).

* `data_list`: a list of video samples and their corresponding ground truth annotations.

* The keys in each entry of `data_list` could be different for different dataset. `AIGVE-Bench` is a large-scale, multifaceted benchmark dataset designed to systematically evaluate different video generation models across nine quality dimensions. Its `data_list` keys entry includes:

    * `id`: unique identification number of this sample.

    * `prompt_gt`: ground-truth prompt used to generate the video.

    * Multifaceted human annotated performance scores. Such as `technical_quality`, `dynamic`, `consistency`, `physics`, `overall`, etc.

    * Classification fields of the video. such as `subject`, `dynamic_type`, `detailed_dynamic`, `category`, etc.

    * `video_path_pd`: relative path to the generated video file from video generator.

For more information about `AIGVE-Bench` dataset, please refer to Section 4 & 5 of the [AIGVE-Tool paper](https://arxiv.org/pdf/2503.14064).

---

## Directory Layout

Your dataset directory should be organized as follows:
```
AIGVE_Tool/
├── aigve/
│   └── data/
│       └── DATASET_NAME/
│           ├── DATASET_SPLIT/  # Folder of raw video files or extracted frames
│           │   ├── video_001.mp4
│           │   └── video_002.mp4
│           └── annotations/
│               └── ANNOTATION_NAME.json   # JSON annotation file
```

You could custmoize these folder names:

* `DATASET_NAME`, the name of the dataset you are evaluating (e.g., `AIGVE_Bench`, `toy`, etc.)

* `DATASET_SPLIT`, the name of the subfolder containing video clips or frame sequences used in evaluation. You could create multiple splits of the dataset and put them in different subfolders.  

* `ANNOTATION_NAME`, the name of the annotation file. You could either create multiple annotation file corresponding to different data splits, or a single file including all annotations for various data splits.


Taking an example, [AIGVE-Bench toy dataset](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/data/AIGVE_Bench_toy/annotations/train.json) is orginaized as:
```
AIGVE_Tool/
├── aigve/
│   └── data/
│       └── AIGVE_Bench_toy/
|           ├── annotations/
│           │   └── train.json   # JSON annotation file
|           ├── videos/  # Folder of raw video files 
│           │   ├── cogvideox_0.mp4
│           │   ├── cogvideox_2.mp4
│           │   ├── cogvideox_3.mp4
│           │   └── ...  
│           ├── videos_3frame/  # Folder of raw video files extracted in 3 frames for each
│           │   ├── cogvideox_0.mp4
│           │   ├── cogvideox_2.mp4
│           │   ├── cogvideox_3.mp4
│           │   └── ...
│           └── ...   
```
---


## Customizing Annotations for Your Dataset

In summary, when you are preparing a new datset, you can consider customizing the following fields:

* Prompts. (question or task-specific instruction)

* Model-specific inputs. (e.g., prompt-based generated videos, precomputed features or conditions)

* Metadata. (resolution, duration, FPS, modality infomation, etc.)

* Ground truth scores. (if applicable, for supervised evaluation)

* The video and annotation folder path. As long as the paths match what is configured (such as `video_dir` and `prompt_dir` parameters) in your dataloader configuration. 

Your custom dataloader can then parse and use these fields accordingly. For more information about customizing your dataloaders based on your prepared dataset, please refer to [Tutorial on Customizable Dataloaders](./dataloader.md).

---

## Tips for Preparing Datasets

* Ensure video names match those in the annotation file.

* Validate your JSON file.

* Keep all paths relative to your project root for portability.

* Use smaller frame samples or lower resolution data for debugging.

---

## What's Next?

- [Customize dataloaders](./dataloader.md)

- [Customize evaluation metrics](./evaluator.md)