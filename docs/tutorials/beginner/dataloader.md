# Tutorial on Customizable Dataloaders

AIGVE supports flexible dataloader design to handle diverse datasets, video formats, and evaluation settings. Each dataloader inherits from [PyTorch's Dataset class](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), and can be easily customized to load videos, extract features, and return evaluation-ready inputs. 
This tutorial introduces how to implement and customize dataloaders in AIGVE. Taking [`GSTVQADataset`](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/datasets/gstvqa_dataset.py#L62) as an example, we will introduce how to customize a dataloader in AIGVE to support various data-related tasks such as reads video inputs, parses prompts, extracts features, and feeds standardized tensors to the evaluator.

## Design Overview

Each dataloader in AIGVE follows a modular structure and is designed to support evaluation-only workflows. The core responsibilities of a custom dataloader include:

* Loading raw videos or frame sequences

* Parsing annotations

* Returning each sample as a Python `dict` containing all necessary fields for downstream evaluation (e.g., prompt, video tensor, metadata)

AIGVE decouples dataloaders from models and metrics, allowing seamless plug-and-play usage with different evaluation modules.


## Dataset Base Class

All custom datasets in AIGVE inherit from [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) and need to implement two essential methods:

* `__len__(self)`: returns the number of samples.

* `__getitem__(self, index)`: returns one sample at the given index.

Each dataset class defines its own logic for reading videos, parsing annotations, feature extraction, and returning evaluation-ready outputs. 
While AIGVE natively supports datasets formatted using MMFormat-style JSON annotations (see [Tutorial on Dataset Preparation](./dataset.md)), it is compatible with any custom format as long as the dataloader returns the expected sample format for evaluation.

A minimal dataloader example that loading from standard AIGVE JSON annotations looks like this:

```python
from torch.utils.data import Dataset
import torch
import os
import cv2
import json

@DATASETS.register_module()
class CustomVideoDataset(Dataset):
    def __init__(self, video_dir, prompt_dir, max_len=30):
        super().__init__()
        self.video_dir = video_dir
        self.prompt_dir = prompt_dir
        self.max_len = max_len

        # Load annotations
        with open(self.prompt_dir, 'r') as reader:
            read_data = json.load(reader)
        self.video_names = [item['video_path_pd'] for item in read_data['data_list']]
        self.prompts = [item['prompt_gt'] for item in read_data['data_list']]

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        video_path = os.path.join(self.video_dir, video_name)

        # Load video frames as tensor
        cap = cv2.VideoCapture(video_path)
        input_frames = []
        while cap.isOpened() and len(input_frames) < self.max_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_frames.append(torch.tensor(frame).float())
        cap.release()

        # Pad to fixed length
        if len(input_frames) < self.max_len:
            pad_frames = torch.zeros((self.max_len - len(input_frames), *input_frames[0].shape))
            input_frames_tensor = torch.cat((torch.stack(input_frames), pad_frames), dim=0)
        else:
            input_frames_tensor = torch.stack(input_frames[:self.max_len])

        # Permute shape to [T, C, H, W]
        input_frames_tensor = input_frames_tensor.permute(0, 3, 1, 2)

        return input_frames_tensor, self.prompts[index], video_name
```

This structure is highly adaptable and can be extended to support a wide range of dataset types and evaluation scenarios. You can build additional logic into your custom dataset class to support:

* Feature extraction using pre-trained backbones

* Multimodal inputs, such as language prompts, audio tracks, reference videos, or scene metadata

* Sample-wise metadata returns, including model name, subject, dynamic type, quality tags, etc.

* Flexible temporal control, such as dynamic frame sampling or resolution normalization

* Input padding and format conversion, ensuring consistent tensors for evaluators

By modifying only the `__getitem__()` method and how the annotations are parsed, developers can customize new data modalities and processing pipelines.

#### Returned Outputs

Each sample returned by the `__getitem__()` method should typically include:

* `video` (Tensor): a video tensor (e.g., shape `[T, C, H, W]`)

* `prompt` (str): text field, often `prompt_gt` from the annotation file

* Some metadata: additional information fields such as model_name, subject, dynamic_type, category, etc.

* Other additional fields added depending on the needs of the downstream evaluator.

With AIGVE's modular design, all outputs returned from the dataloader will be passed into the `data_samples` argument of the `process()` function in the metric evaluator.
This ensures seamless integration between your dataloader and the evaluation pipeline.
For more details on implementing or customizing the `process()` method, please refer to, please refer to [Tutorial on customize evaluation metrics](./evaluator.md)


### Example: GSTVQADataset

`GSTVQADataset` supports dynamic frame selection, flexible video backend loading, and annotation-based control. You can check the implementation from [here](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/datasets/gstvqa_dataset.py#L62).

##### Key Features of `GSTVQADataset`:

The `GSTVQADataset` showcases how to build a robust dataloader with integrated feature extraction and dynamic preprocessing. Key capabilities include:

* Video loading via OpenCV backend, with optional support for frame sampling

* Frame preprocessing and conversion to PyTorch tensor format

* Parsing annotations and mapping prompts and video paths accordingly

* Mean and standard deviation feature extraction using either VGG16 or ResNet18

* Temporal alignment and zero-padding to a fixed length for batch consistency

After implemented the `GSTVQADataset`, you could configure it in the configuration file: 
```python
from datasets import GSTVQADataset

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
Please make sure your `video_dir` contains the visual data, and `prompt_dir` points to your JSON annotations. 

## Tips for Customizing Datasets

* Ensure `__getitem__()` returns all required fields for the evaluator.

* Normalize tensor shapes using padding or format conversion.

* Add support for frame sampling or resizing if needed.

* Test your dataloader with a toy-version dataset before large-scale use


## What's Next?

After customizing the dataloader under a dataset, you can proceed to:

- [Customize evaluation metrics](./evaluator.md)

- [Run the AIGVE loop on your own metrics or datasets](./running.md)


