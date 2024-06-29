# Copyright (c) IFM Lab. All rights reserved.

from typing import Dict, List, Optional, Sequence, Union
from mmengine.evaluator import BaseMetric
from core.registry import METRICS
from mmengine.logging import MMLogger

from utils import add_git_submodule, submodule_exists

metric_path = '/metrics/video_quality_assessment/nn_based/gstvqa'

@METRICS.register_module()
class GSTVQA(BaseMetric):
    """The GSTVQA evaluation metric. https://arxiv.org/pdf/2012.13936
    
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
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not submodule_exists(metric_path):
            add_git_submodule(repo_url='https://github.com/Baoliang93/GSTVQA.git', submodule_path=metric_path)


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """GSTVQA process
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
            # prompt_gt = data_sample['prompt_gt'] # str
            video_pd = data_sample['video_pd'] # torch.uint8(F, C, H, W)

            
            


            result['scores'] = gstvqa

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


