# Copyright (c) IFM Lab. All rights reserved.

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Sequence, Tuple
# from scipy import stats
from mmengine.evaluator import BaseMetric
from core.registry import METRICS
from utils import add_git_submodule, submodule_exists

@METRICS.register_module()
class GstVqa(BaseMetric):
    """GstVQA metric modified for the toy dataset. (Supporting 2944-dim features)."""

    def __init__(self, model_path: str):
        super(GstVqa, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.submodel_path = os.path.join(os.getcwd(), 'metrics/video_quality_assessment/nn_based/gstvqa')
        if not submodule_exists(self.submodel_path):
            add_git_submodule(
                repo_url='https://github.com/Baoliang93/GSTVQA.git', 
                submodule_path=self.submodel_path
            )
        from .GSTVQA.TCSVT_Release.GVQA_Release.GVQA_Cross.cross_test import GSTVQA as GSTVQA_model
        self.model = GSTVQA_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        # self.criterion = nn.L1Loss().to(self.device)

    def compute_stat_features(self, features: torch.Tensor, num_valid_frames: int) -> Tuple[torch.Tensor]:
        """Compute statistical features mean_var, std_var, mean_mean, std_mean from extracted deep features.

        Args:
            features (torch.Tensor): Tensor of shape [T, 2944].
            num_valid_frames (int): Number of valid frames before padding.

        Returns:
            Tuple[torch.Tensor]: (mean_var, std_var, mean_mean, std_mean), each of shape [1472].
        """
        # Ignore padded frames
        features = features[:num_valid_frames]  # Shape: [num_valid_frames, feature_dim]: [10, 1472]

        if num_valid_frames == 0:  # Edge case: all frames were padded
            return (
                torch.zeros(1472, device=self.device),
                torch.zeros(1472, device=self.device),
                torch.zeros(1472, device=self.device),
                torch.zeros(1472, device=self.device),
            )

        # Split into mean and std components
        mean_features = features[:, :1472]  # First 1472 features are mean-based
        std_features = features[:, 1472:]   # Last 1472 features are std-based

        # Compute per-feature statistics over frames
        mean_mean = mean_features.mean(dim=0)  # Shape: [1472]
        std_mean = std_features.mean(dim=0)    # Shape: [1472]
        mean_var = mean_features.var(dim=0, unbiased=False)  # Shape: [1472]
        std_var = std_features.var(dim=0, unbiased=False)    # Shape: [1472]

        return mean_var, std_var, mean_mean, std_mean

    def process(self, data_batch: Sequence, data_samples: Sequence) -> None:
        """
        Process a batch of extracted deep features for GSTVQA evaluation and store results in a JSON file.

        Args:
            data_batch (SequencTuplee): A batch of data from the dataloader (not used here).
            data_samples (List[ [torch.Tensor], Tuple[int], Tuple[str] ]): 
                A list containing three tuples:
                - A tuple of `deep_features`: Each item is a Tensor of shape [T, 2944]. 
                - A tuple of `num_frames`: Each item is an integer representing the number of valid frames.
                - A tuple of `video_name`: Each item is a string representing the file name for the video.
                The len of each three tuples are the batch size.
        """
        # data_samples an example: [
        #     (tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        #              [0., 0., 0.,  ..., 0., 0., 0.],
        #              ...
        #              [0., 0., 0.,  ..., 0., 0., 0.]]), 
        #      tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        #              [0., 0., 0.,  ..., 0., 0., 0.],
        #              ...
        #              [0., 0., 0.,  ..., 0., 0., 0.]])), 
        #     (10, 10)
        # ]
        results = []
        deep_features_tuple, num_frames_tuple, video_name_tuple = data_samples
        with torch.no_grad():
            for deep_features, num_valid_frames, video_name in zip(deep_features_tuple, num_frames_tuple, video_name_tuple):
                if not isinstance(deep_features, torch.Tensor) or not isinstance(num_valid_frames, int):
                    raise TypeError("Expected deep_features to be a torch.Tensor and num_valid_frames to be an int.")

                if num_valid_frames == 0:  # Edge case: No valid frames
                    results.append({"video_name": 'N/A', "GSTVQA_Score": 0.0})
                    continue

                # Remove padded features
                features = deep_features[:num_valid_frames].to(self.device)

                # Compute statistical features only on valid frames
                mean_var, std_var, mean_mean, std_mean = self.compute_stat_features(features, num_valid_frames)
                mean_var, std_var, mean_mean, std_mean = (
                    mean_var.to(self.device),
                    std_var.to(self.device),
                    mean_mean.to(self.device),
                    std_mean.to(self.device),
                )

                # Length tensor indicating the number of valid frames
                length = torch.tensor([num_valid_frames]).to(self.device)
                # print('features(input) shape', features.unsqueeze(1).shape) # torch.Size([10, 1, 1472])
                # print('input_length shape', length.shape) # torch.Size([1])
                # print('input_length', length) # torch.Size([1])
                # print('mean_mean shape', mean_mean.shape) # torch.Size([1472])
                # print('std_mean shape', std_mean.shape) # torch.Size([1472])
                # print('mean_var shape', mean_var.shape) # torch.Size([1472])
                # print('std_var shape', std_var.shape) # torch.Size([1472])
                
                # Run GSTVQA model
                outputs = self.model(features.unsqueeze(1), length, mean_var, std_var, mean_mean, std_mean)
                score = outputs.item()
                results.append({"video_name": video_name, "GSTVQA_Score": score})
                # print(f"Processed score {score:.4f} for {video_name}")

        self.results.extend(results)


    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute final GSTVQA-based metrics."""
        scores = np.array([res['GSTVQA_Score'] for res in self.results])
        mean_score = np.mean(scores)
        print(f"GSTVQA mean score: {mean_score:.4f}")

        json_file_path = os.path.join(os.getcwd(), "gstvqa_results.json")
        final_results = {"video_results": self.results, "GSTVQA_Mean_Score": mean_score}
        with open(json_file_path, "w") as json_file:
            json.dump(final_results, json_file, indent=4)
        print(f"GSTVQA mean score saved to {json_file_path}")

        return {'GSTVQA_Mean_Score': mean_score}