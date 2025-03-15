# encoding = utf-8
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAST_SCRIPT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname(LAST_SCRIPT_DIR))

import torch
import numpy as np
from core.registry import METRICS
from PIL import Image
from typing import Sequence, Dict

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from utils import add_git_submodule, submodule_exists


@METRICS.register_module()
class VIEEvalScore(BaseMetric):
    """ Initialize the ``VIEEvalScore`` evaluator.
    
    Args:
        llm_backbone (str): The name of the LLM model used in the VIEEvalScore evaluator. Defaults to ``got4o``.
        api_key_path (str): The user's api key path to initialize LLM models provides by openai.
        task (str): The task the VIEEvalScore evaluator conducts. Defaults to ''t2v''.
    """
    def __init__(self,
                 llm_backbone: str = "gpt4o",
                 api_key_path: str = 'AIGVE_Tool/metrics/text_video_alignment/gpt_based/VIE/api_key.txt',
                 task: str = 't2v',
                 ):
        super().__init__()
        
        self.api_key_path = api_key_path
        self.llm_backbone = llm_backbone
        self.task = task

        self.submodel_path = 'metrics/text_video_alignment/gpt_based/VIE'
        if not submodule_exists(self.submodel_path):
            add_git_submodule(
                repo_url='https://github.com/TIGER-AI-Lab/VIEScore.git', 
                submodule_path=self.submodel_path
            )  
        self.submodel_path = 'metrics/text_video_alignment/gpt_based/dsg'
        if not submodule_exists(self.submodel_path):
            add_git_submodule(
                repo_url='https://github.com/j-min/DSG.git', 
                submodule_path=self.submodel_path
            )  
        from .VIEScore.viescore import VIEScore 
        from .DSG.dsg.vqa_utils import MPLUG, InstructBLIP

        
        self.vie_score = VIEScore(backbone=self.llm_backbone, task=self.task, key_path=self.api_key_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def process(self, data_batch: Sequence, data_samples: Sequence) -> None:
        """VIEScore process
        Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence): A batch of data from the dataloader.
            data_samples (Sequence): A batch of data samples that
                contain annotations and predictions.
        """

        result = dict()

        input_prompts, input_videos = data_samples
        bsz = len(input_prompts)

        # Ensure prompt_input is a tensor
        if isinstance(input_prompts, tuple):
            input_prompts = list(input_prompts)
        
        if isinstance(input_videos, tuple):
            input_videos = list(input_videos)
        
        average_vie_score_list = []
        for input_prompt, input_video in zip(input_prompts, input_videos):
            vie_score_list = []
            for index, frame_path in enumerate(input_video):
                pil_image = Image.open(frame_path)
                score_list = self.vie_score.evaluate(pil_image, input_prompt)
                sementics_score, quality_score, overall_score = score_list
                vie_score_list.append(overall_score)
            average_vie_score = sum(vie_score_list)/len(vie_score_list)
            average_vie_score_list.append(average_vie_score)
    
        result['vie_score'] = sum(average_vie_score_list)/len(average_vie_score_list)

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

        vie_score_np = np.zeros(len(results))
        for i, result in enumerate(results):
            vie_score_np[i] = result['vie_score']
        
        vie_score_np_mean = np.mean(vie_score_np) 

        print("Test results: vie score with dependency={:.4f}"
              .format(vie_score_np_mean))

        return result
