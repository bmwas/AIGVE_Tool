# encoding = utf-8

import os
import torch
import cv2
import time
import logging
import numpy as np

from core.registry import METRICS
from typing import Dict, Optional, Sequence, Union
from transformers import AutoProcessor, AutoModel

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from tqdm import tqdm


@METRICS.register_module()
class PickScore(BaseMetric):
    """ Initialize the ``PickScore`` evaluator.
    
    Args:
        model_name (str): The name of the PickScore model. Defaults to ``yuvalkirstain/PickScore_v1``.
        logit_scale (bool): Whether to calcualte the cosine similarity as logits. Defaults to False.
    """
    def __init__(self, 
                 model_name: str = "yuvalkirstain/PickScore_v1", 
                 logit_scale: bool = False) -> None:
        super().__init__()
        self.model_name = model_name
        self.logit_scale = logit_scale

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model =AutoModel.from_pretrained(self.model_name).eval().to(self.device)
        self.model.eval()


    # def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
    def process(self, data_batch: Sequence, data_samples: Sequence) -> None:
        """PickScore process
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

        pickscore_sum, pickscore_cnt = 0, 0
        logit_scale = self.model.logit_scale.exp() if self.logit_scale else 1
        with torch.no_grad():
            for input_prompt, input_frames in zip(input_prompts, input_videos):

                input_prompt = input_prompt.to(self.device)
                text_feature = self.model.get_text_features(input_prompt)
                text_feature = text_feature / torch.norm(text_feature, dim=-1, keepdim=True)

                input_frames = input_frames.to(self.device)  # Add batch dimension and move the frame to the device
                frame_features = self.model.get_image_features(input_frames)
                frame_features = frame_features / torch.norm(frame_features, dim=-1, keepdim=True)

                pick_score = logit_scale *  (frame_features @ text_feature.T).mean().item()
                print('current pickscore', pick_score)
                pickscore_sum += pick_score
                pickscore_cnt += 1

        # get probabilities if you have multiple images to choose from
        # probs = torch.softmax(scores, dim=-1)
        pickscore_total_avg = pickscore_sum/pickscore_cnt
        result['pick_score'] = pickscore_total_avg

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

        pickscore_np = np.zeros(len(results))
        for i, result in enumerate(results):
            pickscore_np[i] = result['pick_score']
        
        pickscore_sim_mean = np.mean(pickscore_np) 
        
        print("Test results: PickScore={:.4f}"
              .format(pickscore_sim_mean))

        return result



