# encoding = utf-8

import os
import torch
import cv2
import time
import logging
import numpy as np

from core.registry import METRICS
from typing import Dict, Optional, Sequence, Union
from transformers import AutoProcessor, CLIPModel

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from tqdm import tqdm


@METRICS.register_module()
class CLIPSimScore(BaseMetric):
    """ Initialize the ``CLIPSimScore`` evaluator.
    
    Args:
        processor_name (str): The name of the CLIP processor, which wraps a CLIP feature extractor and a CLIP tokenizer into this single procesor. 
                                Defaults to ``openai/clip-vit-base-patch32``.
        model_name (str): The name of the CLIP model. Defaults to ``openai/clip-vit-base-patch32``.
        logit_scale (bool): Whether to calcualte the cosine similarity as logits. Defaults to False.
    """
    def __init__(self,
                 processor_name: str = "openai/clip-vit-base-patch32",
                 model_name: str = "openai/clip-vit-base-patch32",
                 logit_scale: bool = False,
                #  train_index: int = 4
                 ) -> None:
        super().__init__()
        self.processor_name = processor_name
        self.model_name = model_name
        self.logit_scale = logit_scale

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(self.processor_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def process(self, data_batch: Sequence, data_samples: Sequence) -> None:
        """CLIPSimScore process
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
        
        # Initialize an empty list to store each similarity score
        clip_score_sum, clip_score_cnt = 0, 0
        logit_scale = self.model.logit_scale.exp() if self.logit_scale else 1
        with torch.no_grad():
            for input_prompt, input_frames in zip(input_prompts, input_videos):
                input_prompt = input_prompt.to(self.device)
                text_feature = self.model.get_text_features(input_prompt) # [bsz, hid_dim]
                text_feature = text_feature / torch.norm(text_feature, dim=-1, keepdim=True)

                input_frames = input_frames.to(self.device)  # Add batch dimension and move the frame to the device
                frame_feature = self.model.get_image_features(input_frames)
                frame_feature = frame_feature / torch.norm(frame_feature, dim=-1, keepdim=True)

                clip_score = logit_scale * (frame_feature @ text_feature.T).mean().item()
                print('current clip similarity score', clip_score)
                clip_score_sum += clip_score
                clip_score_cnt += 1

        # Calculate the average CLIP score across all frames
        clip_score_videos_avg = clip_score_sum/clip_score_cnt
       
        result['clip_sim_score'] = clip_score_videos_avg

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

        clip_score_np = np.zeros(len(results))
        for i, result in enumerate(results):
            clip_score_np[i] = result['clip_sim_score']
        
        clip_sim_mean = np.mean(clip_score_np) 
        
        print("Test results: clip similarity score={:.4f}"
              .format(clip_sim_mean))

        return result
