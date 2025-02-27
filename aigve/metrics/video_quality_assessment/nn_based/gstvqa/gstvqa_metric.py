# Copyright (c) IFM Lab. All rights reserved.

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Sequence
# from scipy import stats
from mmengine.evaluator import BaseMetric
from core.registry import METRICS
from utils import add_git_submodule, submodule_exists

@METRICS.register_module()
class GSTVQA(BaseMetric):
    """GSTVQA metric modified for the toy dataset."""

    def __init__(self, model_path: str):
        super(GSTVQA, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.submodel_path = 'metrics/video_quality_assessment/nn_based/gstvqa'
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

    def compute_stat_features(self, features: torch.Tensor, num_valid_frames: int):
        """Compute mean_var, std_var, mean_mean, std_mean from extracted deep features.

        Args:
            features (torch.Tensor): Tensor of shape [T, feature_dim] (deep features).
            num_valid_frames (int): Number of valid frames before padding.

        Returns:
            Tuple[torch.Tensor]: (mean_var, std_var, mean_mean, std_mean), each of shape [1472].
        """
        # Ignore padded frames
        features = features[:num_valid_frames]  # Shape: [num_valid_frames, feature_dim]

        if num_valid_frames == 0:  # Edge case: all frames were padded
            return (
                torch.zeros(1472).to(self.device),
                torch.zeros(1472).to(self.device),
                torch.zeros(1472).to(self.device),
                torch.zeros(1472).to(self.device),
            )

        # Compute per-frame mean and std
        per_frame_mean = features.mean(dim=1)  # Shape: [num_valid_frames]
        per_frame_std = features.std(dim=1)    # Shape: [num_valid_frames]

        # Compute mean and std over time
        mean_var = per_frame_mean.var(dim=0)  # Shape: []
        std_var = per_frame_std.var(dim=0)    # Shape: []
        mean_mean = per_frame_mean.mean(dim=0)  # Shape: []
        std_mean = per_frame_std.mean(dim=0)    # Shape: []

        # Ensure 1472-dimensional output (repeat features to match GSTVQA requirements)
        repeat_factor = 1472 // features.shape[1]  # Adjust to match GSTVQA expected size
        mean_var = mean_var.repeat(repeat_factor)
        std_var = std_var.repeat(repeat_factor)
        mean_mean = mean_mean.repeat(repeat_factor)
        std_mean = std_mean.repeat(repeat_factor)

        return mean_var, std_var, mean_mean, std_mean

    def process(self, data_batch: Sequence, data_samples: Sequence) -> None:
        """Process a batch of extracted deep features for GSTVQA evaluation.

        Args:
            data_batch (Sequence): A batch of data from the dataloader (not used here).
            data_samples (Sequence[Tuple[torch.Tensor, int]]): 
                A sequence where each item is a tuple containing:
                - `deep_features`: Tensor of shape [T, feature_dim] (extracted features).
                - `num_frames`: Integer representing the number of valid frames.
        """
        results = []

        with torch.no_grad():
            for deep_features, num_valid_frames in data_samples:
                if not isinstance(deep_features, torch.Tensor) or not isinstance(num_valid_frames, int):
                    raise TypeError("Expected deep_features to be a torch.Tensor and num_valid_frames to be an int.")

                if num_valid_frames == 0:  # Edge case: No valid frames
                    results.append({'score': 0.0})
                    continue

                features = features[:num_valid_frames].to(self.device)  # Remove padded features

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

                # Run GSTVQA model
                outputs = self.model(features.unsqueeze(1), length.float(), mean_var, std_var, mean_mean, std_mean)
                results.append({'score': outputs.item()})

        self.results.extend(results)


    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute final GSTVQA-based metrics."""
        scores = np.array([res['score'] for res in results])

        mean_score = np.mean(scores)
        print(f"GSTVQA mean score: {mean_score:.4f}")

        return {'GSTVQA_Score': mean_score}
