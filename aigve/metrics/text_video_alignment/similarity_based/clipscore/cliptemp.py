# encoding = utf-8

import os
import torch
import cv2
import time
import logging
import numpy as np

from core.registry import METRICS
from typing import Dict, Optional, Sequence, Union
from transformers import CLIPProcessor, CLIPModel

import torchvision.transforms as transforms
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from tqdm import tqdm


@METRICS.register_module()
class CLIPTempScore(BaseMetric):
    """ Initialize the ``CLIPTempScore`` evaluator.
    
    Args:
        model_name (str): The name of the CLIP encoder model. Defaults to ``openai/clip-vit-base-patch32``.
        logit_scale (bool): Whether to calcualte the cosine similarity as logits. Defaults to False.

    """
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 logit_scale: bool = False,
                #  train_index: int = 4
                 ) -> None:
        super().__init__()
        self.model_name = model_name
        self.logit_scale = logit_scale

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def process(self, data_batch: Sequence, data_samples: Sequence) -> None:
        """CLIPTempScore process
        Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence): A batch of data from the dataloader.
            data_samples (Sequence): A batch of data samples that
                contain annotations and predictions.
        """

        result = dict()

        input_videos = data_samples
        # bsz = len(input_videos)
        

        # Ensure prompt_input is a tensor        
        if isinstance(input_videos, tuple):
            input_videos = list(input_videos)

        # Generate embeddings for each frame and concatenate the features
        clip_temp_score_sum, clip_temp_score_cnt = 0, 0
        logit_scale = self.model.logit_scale.exp() if self.logit_scale else 1
        with torch.no_grad():  
            for input_frames in input_videos: # Too many frames in a video, must split before CLIP embedding, limited by the memory
                input_frames = input_frames.to(self.device)
                frame_feature = self.model.get_image_features(input_frames)
                frame_feature = frame_feature / torch.norm(frame_feature, dim=-1, keepdim=True)
                # print(frame_feature.shape)

                clip_temp_score_list = []
                for i in range(frame_feature.shape[0]-1):
                    clip_temp_score = logit_scale * frame_feature[i].unsqueeze(0) @ frame_feature[i+1].unsqueeze(0).T
                    clip_temp_score = clip_temp_score.item()
                    # print(clip_temp_score)
                    clip_temp_score_list.append(clip_temp_score)
                clip_temp_cur_avg_score = sum(clip_temp_score_list)/len(clip_temp_score_list)
                clip_temp_score_sum += clip_temp_cur_avg_score
                clip_temp_score_cnt += 1
                print('current clip temp similarity score', clip_temp_cur_avg_score)
        
        clip_temp_score_avg = clip_temp_score_sum/clip_temp_score_cnt
        
        result['clip_temp_score'] = clip_temp_score_avg

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
            clip_score_np[i] = result['clip_temp_score']
        
        clip_temp_mean = np.mean(clip_score_np) 
        
        print("Test results: clip temporal consistency score={:.4f}"
              .format(clip_temp_mean))
        
        return result
