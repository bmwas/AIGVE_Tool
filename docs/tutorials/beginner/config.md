# Tutorial on Configuration Files

AIGVE use [MMEngine's Python style config system](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html). It has a modular and inheritance design, which is convenient to conduct various experiments. 
It allows you to define all parameters in one centralized location, enabling easy access data just like getting values from `dict`.
AIGVE's config system provides an inheritance mechanism, which could help you better organize and manage the configuration files.

## Config file content

AIGVE uses a modular design, all modules with different functions can be configured through the config. Taking [`gstqva`](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/configs/gstvqa.py) as an example, we will introduce each field in the config according to different function modules.

### Default Base configuration

These configuration defines default configuration of the [MMEngine runner system](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py). This includes specifying the evaluation loop class, the model wrapper, logging levels, and hook behavior. Since these configuration are not dependent on specific evaluation metrics, they are defined in the base configuration to ensure reusability and maintain a clean structure.

```python
from mmengine.config import read_base
with read_base():
    from ._base_.default import *
```

where the `default` config is defined as: 

```python
from core import AIGVELoop, AIGVEModel

default_scope = None

log_level = 'INFO'

model = dict(type=AIGVEModel)

default_hooks = None # Execute default hook actions as https://github.com/open-mmlab/mmengine/blob/85c83ba61689907fb1775713622b1b146d82277b/mmengine/runner/runner.py#L1896

val_cfg = dict(type=AIGVELoop)
```

* `default_scope` and `log_level` control default scoping and logging behaviors.

* `model` is a required component of MMEngine's runner pipeline. In AIGVE, it is by default set to [`AIGVEModel`](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/core/models.py), allowing loaded data batch to flow directly into the evaluator without model-specific logic.

* `default_hooks` is by default set to `None` to disable default behaviors.

* `val_cfg` defines the whole evaluation pipeline behaviors. It is set to [`AIGVELoop`](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/core/loops.py), the core loop class in AIGVE for running evaluations from datasets loading to metric evaluation.


### Dataset & Loader configuration

Following [`AIGVELoop`](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/core/loops.py), `val_dataloader` is required for batch data loading. For different AIGVE metrics, they may use different data sources, data loading manners, data sampling strategies, or data preprocessing methods. As a result, they may use different dataloader classes or set different parameters to build it. All our customizable dataLoaders are inherited from [PyTorch's `Dataset` class](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). In `gstvqa`, its `val_dataloader` is configured as: 

```python
from mmengine.dataset import DefaultSampler
from datasets import GSTVQADataset

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=GSTVQADataset,
        # video_dir='AIGVE_Tool/aigve/data/toy/evaluate/', # it has 16 frames for each video, each frame is [512, 512, 3]
        # prompt_dir='AIGVE_Tool/aigve/data/toy/annotations/evaluate.json',
        video_dir='AIGVE_Tool/aigve/data/AIGVE_Bench/videos_3frame/', # it has 81 frames for each video, each frame is [768, 1360, 3]
        prompt_dir='AIGVE_Tool/aigve/data/AIGVE_Bench/annotations/test.json',
        model_name='vgg16',  # User can choose 'vgg16' or 'resnet18'
        max_len=3,
    )
)
```

* `batch_size`, `num_workers`, `persistent_workers`, `drop_last`, `sampler` and `dataset` are parameters for [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). Specifically for `dataset` parameter, the `gstvqa` is configured to use [`GSTVQADataset`](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/datasets/gstvqa_dataset.py#L62). 

* `video_dir`, `prompt_dir`, `model_name` and `max_len` are four parameters defined in the `GSTVQADataset`. 

* For more details about customizing such dataloader, please refer to [Tutorial on Customizable Dataloaders](./dataloader.md). Our `AIGVE` support loading any dataset formatted using [MMFormat JSON annotation file](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html). For more details about preparing dataset into this format, please refer to [Tutorial on Dataset Preparation](./dataset.md). 


### Evaluator configuration

Following [`AIGVELoop`](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/core/loops.py), `val_evaluator` is required for batch data processing, evaluation, and final metric score computing. Different AIGVE metrics generally will define their own evaluation class, to perform their unique evaluation-related operations, such as model loading, dynamic feature extraction, per-sample evaluation, score aggregation, or flexible computational resource management. All our modular evaluator metrics are inherited from [MMEngine's `BaseMetric` class](https://github.com/open-mmlab/mmengine/blob/main/mmengine/evaluator/metric.py#L16). In `gstvqa`, its `val_evaluator` is configured as: 

```python
from metrics.video_quality_assessment.nn_based.gstvqa.gstvqa_metric import GstVqa

val_evaluator = dict(
    type=GstVqa,
    model_path="metrics/video_quality_assessment/nn_based/gstvqa/GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/models/training-all-data-GSTVQA-konvid-EXP0-best",
)
```

* the `gstvqa` is configured to use [`GstVqa`](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/metrics/video_quality_assessment/nn_based/gstvqa/gstvqa_metric.py#L15). `model_path` is the parameters defined in the `GstVqa`. 

* For more details about customizing such metric evaluator, please refer to [Tutorial on Modular Metrics](./evaluator.md). 