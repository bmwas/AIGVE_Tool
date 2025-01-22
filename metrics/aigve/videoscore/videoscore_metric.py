
from typing import Dict, List, Optional, Sequence, Union, Any

from mantis.models.idefics2 import Idefics2ForSequenceClassification
from mmengine.evaluator import BaseMetric
from sympy.logic.inference import entails
from transformers import LlamaTokenizer

from core.registry import METRICS
from mmengine.logging import MMLogger
import torch
from .videoscore_utils import _read_video_pyav
import torch.nn as nn
import numpy as np

@METRICS.register_module()
class VideoScore(BaseMetric):
    def __init__(self,
                collect_device: Optional[Union[str, torch.device]] = None,
                prefix: Optional[str] = None,
                metric_path: str = None,
                model_path: str = 'TIGER-Lab/VideoScore-v1.1',
                datainfo_path: str = None,
                test_index: int = None,
                 **kwargs):
        """
        Args:
            collect_device (Optional[Union[str, torch.device]]): The device to collect the data on.
            prefix (Optional[str]): The prefix to use for the metric.
            metric_path (str): The path to the metric file.
            model_path (str): The path to the model file.
            datainfo_path (str): The path to the datainfo file.
            test_index (int): The index of the test data.
        """
        super().__init__(collect_device=collect_device, prefix=prefix)
        # self.train_index = train_index
        # TODO: ARE THERE PARAMETERS REQUIRED FOR THIS METRIC?
        self.metric_path = metric_path
        self.model_path = model_path
        self.datainfo_path = datainfo_path
        self.test_index = test_index


        self.model = Idefics2ForSequenceClassification.from_pretrained(self.model_path, torch_dtype=torch.bfloat16).eval()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        self.results = []

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """
        Args:
            data_batch (Any): The data batch to process.
            data_samples (Sequence[dict]): The data samples to process.
        """


        data_batch = {k: v[0].to(self.model.device) for k, v in data_batch.items()}

        with torch.no_grad():
            outputs = self.model(**data_batch)

        logits = outputs.logits.cpu().detach().to(torch.float32).numpy()
        num_aspects = logits.shape[-1]

        aspect_scores = []
        for i in range(num_aspects):
            aspect_scores.append(round(logits[0, i].item(), 3))

        self.results.append(aspect_scores)

    def compute_metrics(self, results: list) -> dict:
        """
        Args:
            results (list): The results to compute the metrics from.
        """
        results = np.array(results)
        mean_scores = np.mean(results, axis=1)

        return {'visual_quailty': results[:, 0].tolist(),
                'temporal_consistency': results[:, 1].tolist(),
                'dynamic_degree': results[:, 2].tolist(),
                'text-to-video_alignment': results[:, 3].tolist(),
                'factual_consistency': results[:, 4].tolist(),
                'summary': {'visual_quality': mean_scores[0], 'temporal_consistency': mean_scores[1],
                            'dynamic_degree': mean_scores[2], 'text-to-video_alignment': mean_scores[3],
                            'factual_consistency': mean_scores[4]}}