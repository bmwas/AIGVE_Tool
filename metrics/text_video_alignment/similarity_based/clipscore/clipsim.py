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

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from tqdm import tqdm


@METRICS.register_module()
class CLIPSimScore(BaseMetric):
    """
    """
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 test_index: int = None,
                #  train_index: int = 4
                 ) -> None:
        super().__init__()
        self.model_name = model_name
        self.test_index = test_index

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()


# def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
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

        prompt_input, tensor_frames = data_samples  

        # Ensure prompt_input is a tensor
        if isinstance(prompt_input, tuple):
            prompt_input = prompt_input[0]

        text_input = prompt_input.to(self.device)

        # Initialize an empty tensor to store the concatenated features
        concatenated_features = torch.tensor([], device=self.device)
        with torch.no_grad():
            for frame in tensor_frames:

                # If frame is a tuple, extract the tensor. Assume tensor is the first element.
                if isinstance(frame, tuple):
                    frame = frame[0]

                frame_input = frame.unsqueeze(0).to(self.device)  # Add batch dimension and move the frame to the device
                frame_features = self.model.get_image_features(frame_input)
                concatenated_features = torch.cat((concatenated_features, frame_features), dim=0)

        with torch.no_grad():
            text_features = self.model.get_text_features(text_input)

        concatenated_features = concatenated_features / concatenated_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        clip_score_frames = concatenated_features @ text_features.T
        # Calculate the average CLIP score across all frames, reflects temporal consistency 
        clip_score_frames_avg = clip_score_frames.mean().item()
       
        result['clip_sim_score'] = clip_score_frames_avg

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




