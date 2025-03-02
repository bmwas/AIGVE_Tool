# Copyright (c) IFM Lab. All rights reserved.

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from mmengine.evaluator import BaseMetric
from core.registry import METRICS
from typing import Dict
from utils import add_git_submodule, submodule_exists

@METRICS.register_module()
class LightVQAPlus(BaseMetric):
    """LightVQA+ metric for evaluating video quality."""
    
    def __init__(self, model_path: str, swin_weights: str, is_gpu: bool = True):
        super(LightVQAPlus, self).__init__()
        self.model_path = model_path
        self.swin_weights = swin_weights
        self.device = torch.device("cuda" if is_gpu else "cpu")

        self.submodel_path = os.path.join(os.getcwd(), 'metrics/video_quality_assessment/nn_based/lightvqa_plus')
        if not submodule_exists(self.submodel_path):
            add_git_submodule(
                repo_url='https://github.com/SaMMyCHoo/Light-VQA-plus.git', 
                submodule_path=self.submodel_path
            )

        lightvqa_path = os.path.join(self.submodel_path, "Light-VQA-plus")
        if lightvqa_path not in sys.path:
            sys.path.insert(0, lightvqa_path)

        from .Light_VQA_plus.final_fusion_model import swin_small_patch4_window7_224 as create_model
        self.model = create_model().to(self.device)

        weights_dict = torch.load(os.path.join(os.getcwd(), self.model_path), map_location=self.device)
        print(self.model.load_state_dict(weights_dict))

        self.model.eval()

    def process(self, data_batch: list, data_samples: list) -> None:
        """
        Process a batch of extracted deep features for LightVQA+ evaluation.
        Args:
            data_batch (Sequence): A batch of data from the dataloader (not used here).
            data_samples (List[Tuple[torch.Tensor], Tuple[str]]):
                - spatial_features (torch.Tensor): Shape [v_len_second, 3, 672, 1120].
                - video_name (str): Video filename.
        """
        results = []
        spatial_features_tuple, video_name_tuple = data_samples

        batch_size = len(spatial_features_tuple)
        with torch.no_grad():
            for i in range(batch_size):
                video_name = video_name_tuple[i]
                spatial_features = spatial_features_tuple[i].to(self.device).unsqueeze(0)  # Add batch dim.

                outputs = self.model(spatial_features)
                score = outputs.item()

                results.append({"video_name": video_name, "LightVQA_Score": score})
                print(f"Processed score {score:.4f} for {video_name}")

        self.results.extend(results)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute final LightVQA+ metrics."""
        scores = np.array([res["LightVQA_Score"] for res in self.results])
        mean_score = np.mean(scores) if scores.size > 0 else 0.0
        print(f"LightVQA+ mean score: {mean_score:.4f}")

        json_file_path = os.path.join(os.getcwd(), "lightvqa_results.json")
        final_results = {"video_results": self.results, "LightVQA_Mean_Score": mean_score}
        with open(json_file_path, "w") as json_file:
            json.dump(final_results, json_file, indent=4)
        print(f"LightVQA+ mean score saved to {json_file_path}")

        return {"LightVQA_Mean_Score": mean_score}
