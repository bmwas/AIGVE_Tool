# Copyright (c) IFM Lab. All rights reserved.

from typing import Dict, List, Optional, Sequence, Union
from mmengine.evaluator import BaseMetric
from core.registry import METRICS

from mmengine.logging import MMLogger

import os, time
# from .global_descriptor import GlobalDescriptor
# from .local_descriptor import LocalDescriptor
# from utils import read_image_detectron2
# from grit.predictor import VisualizationDemo



@METRICS.register_module()
class LLMScore(BaseMetric):
    """The LLMScore evaluation metric. It is a text-image alignemnet framework.
    
    Args:
        collect_device (str): Device used for collecting results from workers.
            Options: 'cpu' and 'gpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            Default: None.
    """

    default_prefix: Optional[str] = 'llm_score'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 openai_key: str = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.openai_key = openai_key

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """LLMScore process
        Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            prompt_gt = data_sample['prompt_gt']
            video_pd = data_sample['video_pd']
            img_pd = video_pd[data_sample['img_frame']] # torch.uint8(C, H, W)

            openai_key = os.environ['OPENAI_KEY']
            global_descriptor = GlobalDescriptor()
            local_descriptor = LocalDescriptor()

            img = read_image_detectron2(img_pd, format="BGR")
            start_time = time.time()


            result['scores'] = llmscore

            self.results.append(result)




    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()


