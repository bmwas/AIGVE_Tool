# Copyright (c) IFM Lab. All rights reserved.

import os
import json
import torch
import torch.nn as nn
import numpy as np
from mmengine.evaluator import BaseMetric
from core.registry import METRICS
from tqdm import tqdm
from typing import Dict
from utils import add_git_submodule, submodule_exists

@METRICS.register_module()
class SimpleVQA(BaseMetric):
    """SimpleVQA metric for evaluating video quality."""
    def __init__(self, model_spatial_path: str, model_motion_path: str, is_gpu: bool = True):
        super(SimpleVQA, self).__init__()
        self.device = torch.device("cuda" if is_gpu else "cpu")
        self.submodel_path = os.path.join(os.getcwd(), 'metrics/video_quality_assessment/nn_based/simplevqa')
        if not submodule_exists(self.submodel_path):
            add_git_submodule(
                repo_url='https://github.com/sunwei925/SimpleVQA.git', 
                submodule_path=self.submodel_path
            )
        from .SimpleVQA.UGC_BVQA_model import resnet50
        from .SimpleVQA.slowfast import slowfast
        # Load the spatial quality model (ResNet50)
        self.model_spatial = resnet50(pretrained=False)
        self.model_spatial = torch.nn.DataParallel(self.model_spatial).to(self.device)
        self.model_spatial.load_state_dict(torch.load(model_spatial_path, map_location=self.device))
        self.model_spatial.eval()

        # Load the motion quality model (SlowFast)
        self.model_motion = slowfast().to(self.device).eval()

    def process(self, data_batch: list, data_samples: list) -> None:
        """
        Process a batch of extracted deep features for SimpleVQA evaluation.
        Args:
            data_batch (list): A batch of data from the dataloader (not used here).
            data_samples (list): A list containing three tuples:
                - spatial_features: Extracted ResNet50 spatial features
                - motion_features: Extracted SlowFast motion features
                - video_name: Video filename.
        """
        results = []
        json_file_path = os.path.join(os.getcwd(), "simplevqa_results.json")

        for spatial_features, motion_features, video_name in tqdm(data_samples, desc="Processing videos"):
            if spatial_features is None or motion_features is None:
                results.append({video_name: {"SimpleVQA_Score": 0.0}})
                continue

            # Reshape inputs
            spatial_features = spatial_features.unsqueeze(dim=0).to(self.device)
            motion_features = motion_features.unsqueeze(dim=0).to(self.device)

            with torch.no_grad():
                score = self.model_spatial(spatial_features, motion_features).item()

            results.append({video_name: {"SimpleVQA_Score": score}})
            print(f"Processed {video_name}: SimpleVQA_Score = {score:.4f}")

        # Save results to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)
        print(f"SimpleVQA results saved to {json_file_path}")

        self.results.extend(results)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute final SimpleVQA-based metrics."""
        scores = np.array([list(res.values())[0]["SimpleVQA_Score"] for res in results])
        mean_score = np.mean(scores)
        print(f"SimpleVQA mean score: {mean_score:.4f}")

        # Save final mean score
        json_file_path = os.path.join(os.getcwd(), "simplevqa_results.json")
        with open(json_file_path, "r") as json_file:
            results_json = json.load(json_file)

        final_results = {"video_results": results_json, "SimpleVQA_Mean_Score": mean_score}
        with open(json_file_path, "w") as json_file:
            json.dump(final_results, json_file, indent=4)
        print(f"Final SimpleVQA results saved to {json_file_path}")

        return {"SimpleVQA_Mean_Score": mean_score}
