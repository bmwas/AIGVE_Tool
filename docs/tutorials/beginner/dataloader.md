# Tutorial on Customizable Dataloaders

AIGVE supports flexible dataloader design to handle diverse datasets, video formats, and evaluation settings. Each dataloader inherits from [PyTorch's Dataset class](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), and can be easily customized to load videos, extract features, and return evaluation-ready inputs. This tutorial introduces how to implement and customize dataloaders in AIGVE. Taking [`GSTVQADataset`](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/datasets/gstvqa_dataset.py#L62) as an example, we will introduce how to customize a dataloader in AIGVE to support various data-related tasks such as reads video inputs, parses prompts, extracts features, and feeds standardized tensors to the evaluator.

## Design Overview

Each dataloader in AIGVE follows a modular structure and is designed to support evaluation-only workflows. The core responsibilities of a custom dataloader include:

* Loading raw videos or frame sequences

* Parsing annotation files (in MMFormat style)

* Returning each sample as a Python `dict` containing all necessary fields for downstream evaluation (e.g., prompt, video tensor, metadata)

AIGVE decouples dataloaders from models and metrics, allowing seamless plug-and-play usage with different evaluation modules.


## Dataset Base Class

All custom datasets in AIGVE inherit from BaseDataset (from MMEngine or implemented similarly), and follow the MMFormat-style annotation loading introduced in Tutorial on Dataset Preparation.

A minimal custom dataset looks like this:

```python
from mmengine.dataset import BaseDataset

@DATASETS.register_module()
class CustomVideoDataset(BaseDataset):
    def __init__(self, video_dir, prompt_dir, **kwargs):
        super().__init__(ann_file=prompt_dir, **kwargs)
        self.video_dir = video_dir

    def load_data_list(self):
        # Load JSON file from self.ann_file
        # Build self.data_list: list of dicts with fields like 'video_path', 'prompt', etc.
        pass

    def get_data_info(self, idx):
        # Returns one sample in the expected dict format
        pass
```
For details on the format of ann_file, see Tutorial on Dataset Preparation.


### Example: GSTVQADataset

In AIGVE, a concrete example is provided in GSTVQADataset. It supports dynamic frame selection, flexible video backend loading, and annotation-based control.

```python
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=GSTVQADataset,
        video_dir='aigve/data/AIGVE_Bench/videos_3frame/',
        prompt_dir='aigve/data/AIGVE_Bench/annotations/test.json',
        model_name='vgg16',
        max_len=3,
    )
)
```

### Key Features of `GSTVQADataset`:

* Loads videos using torchvision backend (read_video)

* Selects a fixed number of frames (e.g., `max_len`=3)

* Converts frame sequences into tensors

* Aligns videos with their corresponding prompts and attributes from the JSON


### Supported Fields per Sample

Each sample (returned by `get_data_info`) should be a dictionary that includes:

* `video`: a video tensor or path

* `prompt`: the prompt text (`prompt_gt` or similar)

* `metadata`: optional fields like model name, subject, category, etc.

Any other task-specific fields needed by the evaluator


## Tips for Customizing Datasets

* Use MMFormat-style annotations for consistency

* Keep video_dir and prompt_dir flexible via config

* Support frame sampling (max_len) or resizing if applicable

* Ensure the return format of each sample matches what the evaluator expects

* Test your dataloader with a toy dataset before large-scale use


## What's Next?

After customizing the dataloader under a dataset, you can proceed to:

- [Customize evaluation metrics](./evaluator.md)

- [Run the AIGVE loop on your own metrics or datasets](./running.md)



## Dataset Integration in Configuration Files

In the config file (e.g., `gstvqa.py`), the dataset is referenced as follows:

```python
val_dataloader = dict(
    dataset=dict(
        type=GSTVQADataset,
        video_dir='aigve/data/AIGVE_Bench/videos_3frame/',
        prompt_dir='aigve/data/AIGVE_Bench/annotations/test.json',
        ...
    )
)
```

Please make sure your `video_dir` contains the visual data, and `prompt_dir` points to your JSON annotations. For more information about customizing your dataloaders based on your prepared dataset, please refer to [Tutorial on Customizable Dataloaders](./dataloader.md).

---

This `GSTVQADataset` defined the unique logics about video data management, including handling diverse video sources reading, prompt retrieval, resolution normalization, frame sampling and padding, format conversion, feature extraction and pre-processing.