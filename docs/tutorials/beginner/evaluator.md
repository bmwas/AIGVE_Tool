# Tutorial on Modular Metrics

AIGVE provides a modular and extensible metric design to support evaluation across diverse video quality dimensions and tasks. Each evaluation metric inherits from MMEngine's [BaseMetric](https://github.com/open-mmlab/mmengine/blob/main/mmengine/evaluator/metric.py), and is automatically called in the evaluation loop.

This tutorial introduces how to implement and integrate custom evaluation metrics into the AIGVE framework. Taking [`GstVqa`](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/metrics/video_quality_assessment/nn_based/gstvqa/gstvqa_metric.py#L15) as an example, we show how metrics can be plugged into the pipeline and reused across datasets.

## Design Overview

All metric classes in AIGVE inherit from `BaseMetric` and implement the following methods:

* `process(self, data_batch, data_samples)`: called during evaluation, processes each mini-batch.

* `compute_metrics(self, results)`: computes final aggregated metrics after all samples are processed.

The `AIGVELoop` will calls these methods automatically in the evaluation stage.

A minimal example of a custom metric class looks like this:

```python
from mmengine.evaluator import BaseMetric
from core.registry import METRICS

@METRICS.register_module()
class MyMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.results = []

    def process(self, data_batch, data_samples):
        # Extract needed fields from the sample
        for sample in data_samples:
            video = sample['video']
            prompt = sample['prompt']
            # Do custom processing and scoring
            score = 1.0  # Placeholder score
            self.results.append(dict(video_name=sample['video_name'], score=score))

    def compute_metrics(self, results):
        # Aggregate final scores
        avg_score = sum([item['score'] for item in self.results]) / len(self.results)
        return dict(MyScore=avg_score)
```

**Note**: The format of `data_samples` passed to `process()` must match the output structure of the dataset's [`__getitem__()`](https://www.aigve.org/tutorials/beginner/dataloader/#dataset-base-class) method. For example, if your dataset returns: 

```python
def __getitem__(self, index) -> tuple[torch.Tensor, int, str]:
    return deep_features, num_frames, video_name
```
For the help of our AIGVE Loop, the `data_samples` in `process()` will be a list of three tuples (i.e. List[[torch.Tensor], Tuple[int], Tuple[str]]):
```python
deep_features_tuple, num_frames_tuple, video_name_tuple = data_samples
```
Each tuple has length equal to the batch size.

### Example: GstVqa

`GstVqa` is a video-only neural network-based metric which uses a pretrained model to compute quality scores.
In its `process()`, features are loaded and passed through the model to produce scores. 
In its `compute_metrics()`, average scores are reported.
You can find full implementation [here](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/aigve/metrics/video_quality_assessment/nn_based/gstvqa/gstvqa_metric.py#L15).

After implemented the `GstVqa`, you could configure it in the configuration file: 
```python
from metrics.video_quality_assessment.nn_based.gstvqa.gstvqa_metric import GstVqa

val_evaluator = dict(
    type=GstVqa,
    model_path='path/to/gst_vqa_model.pth'
)
```
**Note**: Some metrics require downloading pretrained models manually. Make sure they are downloaded correctly and placed in the correct paths as specified in the configuration files. For example here, please make sure your `model_path` contains the path of pretrained model you downloaded.

## Tips for Customizing Evaluation Metrics

* Register your metric using `@METRICS.register_module()`.

* Implement `process()` to process each batch and store per-sample results in `self.results`.

* Make sure `data_samples` format corresponds to your dataset's `__getitem__()` output.

* You may log or save results in `compute_metrics()` if needed.

* Some metrics require downloading pretrained models manually. Make sure they are downloaded correctly and placed in the correct paths as specified in the configuration files.

## What's Next?

After customizing the modular metrics, you can proceed to:

- [Run the AIGVE loop on your own metrics or datasets](./running.md)