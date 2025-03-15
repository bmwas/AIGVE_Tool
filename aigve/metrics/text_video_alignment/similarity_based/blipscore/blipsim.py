# encoding = utf-8

import os
import torch
import cv2
import time
import logging
import numpy as np

from core.registry import METRICS
from typing import Dict, Optional, Sequence, Union
from transformers import BlipForImageTextRetrieval

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from tqdm import tqdm


@METRICS.register_module()
class BlipSimScore(BaseMetric):
    """ Initialize the ``BLIPSimScore`` evaluator.
    
    Args:
        model_name (str): The name of the BLIP model. Defaults to ``Salesforce/blip-itm-base-coco``.
        logit_scale (bool): Whether to calcualte the cosine similarity as logits. Defaults to False.
    """
    def __init__(self,
                 model_name: str = "Salesforce/blip-itm-base-coco",
                 logit_scale: bool = False,
                 ) -> None:
        super().__init__()
        self.model_name = model_name
        self.logit_scale = logit_scale

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BlipForImageTextRetrieval.from_pretrained(self.model_name).to(self.device)
        self.model.eval()


# def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
    def process(self, data_batch: Sequence, data_samples: Sequence) -> None:
        """BLIPSimScore process
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


        # Initialize an empty tensor to store the concatenated features
        blip_score_sum, blip_score_cnt = 0, 0
        logit_scale = self.model.logit_scale.exp() if self.logit_scale else 1
        with torch.no_grad():
            for input_prompt, input_frames in zip(input_prompts, input_videos):
                # If frame is a tuple, extract the tensor. Assume tensor is the first element.
                # if isinstance(input_prompt_frame_pair, tuple):
                #     input_prompt_frame_pair = input_prompt_frame_pair[0]
                
                # for key, value in input_prompt_frame_pair.items():
                #     if isinstance(value, list):
                #         input_prompt_frame_pair[key] = value[0]

                # input_prompt_frame_pair = input_prompt_frame_pair.to("cuda")  # Add batch dimension and move the frame to the device
                # blip_cosine_sim_score = self.model(**input_prompt_frame_pair, use_itm_head=False)[0].item()
                # blip_scores.append(blip_cosine_sim_score)
                input_prompt = input_prompt.to(self.device)
                input_frames = input_frames.to(self.device)
                blip_cosine_sim_score = self.model(input_ids=input_prompt, pixel_values=input_frames, use_itm_head=False)[0].mean().item()
                blip_cosine_sim_score *= logit_scale
                print('current blip cosine similarity score', blip_cosine_sim_score)
                blip_score_sum += blip_cosine_sim_score
                blip_score_cnt += 1
                
        # Calculate the average BLIP score across all frames
        blip_score_frames_avg = blip_score_sum/blip_score_cnt
        
        result['blip_sim_score'] = blip_score_frames_avg

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

        blip_score_np = np.zeros(len(results))
        for i, result in enumerate(results):
            blip_score_np[i] = result['blip_sim_score']
        
        blip_sim_mean = np.mean(blip_score_np) 
        
        print("Test results: blip similarity score={:.4f}"
              .format(blip_sim_mean))
        
        return result
