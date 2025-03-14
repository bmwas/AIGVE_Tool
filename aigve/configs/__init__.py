r"""
In `AIGVE`, configuration management is handled using MMEngine's configuration 
system, which provides a modular, hierarchical, and flexible approach to defining 
experiment settings. The config system allows users to efficiently configure 
**video evaluation metrics**, **datasets**, **dataloaders, etc.**, making 
benchmarking and experimentation more streamlined in a structured manner.

---

# Key Features of AIGVE Config System
- **Modular Design**: Uses `_base_` configurations to reduce redundancy.
- **Customizable Pipelines**: Define different evaluation metrics and datasets easily.
- **Flexible Overriding**: Modify parameters dynamically via command-line arguments.
- **Scalability**: Supports large-scale video evaluation with efficient data loading.

---

## AIGVE Configuration Example

AIGVE uses **structured configuration files** to define evaluation settings. 
Below is an example of a **[CLIPSim metric](https://github.com/ShaneXiangH/VQA_Toolkit/blob/main/aigve/configs/clipsim.py)** configuration file:

```python
# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from metrics.text_video_alignment.similarity_based import CLIPSimScore

with read_base():
    from ._base_.datasets.clipsim_dataset import *
    from ._base_.default import *

val_evaluator = dict(
    type=CLIPSimScore,
    model_name='openai/clip-vit-base-patch32',
    logit_scale=False,
)

val_dataloader = dict(
    batch_size=2, 
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=CLIPSimDataset,
        processor_name='openai/clip-vit-base-patch32',
        video_dir='AIGVE_Tool/data/toy/evaluate/',
        prompt_dir='AIGVE_Tool/data/toy/annotations/evaluate.json',
    )
)
```


"""