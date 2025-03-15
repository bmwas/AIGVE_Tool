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
class SimpleVqa(BaseMetric):
    """SimpleVQA metric for evaluating video quality."""
    def __init__(self, model_path: str, is_gpu: bool = True):
        super(SimpleVqa, self).__init__()
        self.model_path = model_path
        self.device = torch.device("cuda" if is_gpu else "cpu")
        self.submodel_path = os.path.join(os.getcwd(), 'metrics/video_quality_assessment/nn_based/simplevqa')
        if not submodule_exists(self.submodel_path):
            add_git_submodule(
                repo_url='https://github.com/sunwei925/SimpleVQA.git', 
                submodule_path=self.submodel_path
            )
        simplevqa_path = os.path.join(self.submodel_path, "SimpleVQA")
        if simplevqa_path not in sys.path:
            sys.path.insert(0, simplevqa_path)
        from .SimpleVQA.model import UGC_BVQA_model
        from .SimpleVQA.test_demo import slowfast
        self.model_motion = slowfast().to(self.device)
        self.model = UGC_BVQA_model.resnet50(pretrained=False)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(os.getcwd(), self.model_path), map_location=self.device))
        self.model.eval()

    def process(self, data_batch: list, data_samples: list) -> None:
        """
        Process a batch of extracted deep features for SimpleVQA evaluation.
        Args:
            data_batch (Sequence): A batch of data from the dataloader (not used here).
            data_samples (List[ Tuple[torch.Tensor], List[Tuple[torch.Tensor]], Tuple[str] ]):
                A list containing three tuples:
                - A tuple of `spatial_features` (torch.Tensor): Shape [v_len_second, 3, 448, 448]. 
                    `v_len_second` is total seconds of the video (though 2 for toy dataset) with minium 8 (i.e. min_video_seconds). 
                    The len of the tuple is the batch size. 
                - A list of `motion_features` (Tuple[torch.Tensor]): 
                    len(List) is total seconds of the video, with minium 8 (i.e. min_video_seconds).
                    Each item of the list is a Tuple of motion feature tensors. Each has shape [32, 3, 224, 224].
                    The len of the tuple is the batch size.
                - A tuple of `video_name` (str): Video filename. The len of the tuple is the batch size.
        """
        from .SimpleVQA.test_demo import pack_pathway_output

        results = []
        # print(type(data_samples)) # list
        spatial_features_tuple, motion_features_list, video_name_tuple = data_samples
        # print(len(spatial_features_tuple)) # 1
        # print(spatial_features_tuple[0].shape) # torch.Size([8, 3, 448, 448])
        
        # print(type(motion_features_list)) # List
        # print(len(motion_features_list)) # 8
        # print(type(motion_features_list[0])) # tuple
        # print(len(motion_features_list[0])) # 1
        # print(type(motion_features_list[0][0])) # Tensor
        # print(motion_features_list[0][0].shape) # torch.Size([32, 3, 224, 224])
        
        batch_size = len(spatial_features_tuple)
        with torch.no_grad():
            for i in range(batch_size):
                video_name = video_name_tuple[i]
                spatial_features = spatial_features_tuple[i].to(self.device).unsqueeze(0)  # Add batch dim. Shape: tensor with Size([1, v_len_second, 3, 448, 448])

                # Take the i-th element from each tuple in motion_features_list
                motion_features = [motion_features_list[j][i] for j in range(len(motion_features_list))] # Shape: List[tensor with Size([32, 3, 224, 224])], len of it is total seconds of the video, with minium 8.

                if not all(isinstance(mf, torch.Tensor) for mf in motion_features):
                    raise TypeError("Expected motion_features to be a list of tensors.")

                if len(motion_features) == 0:  # Edge case: No valid motion features
                    results.append({"video_name": video_name, "SimpleVQA_Score": 0.0})
                    continue

                n_clip = len(motion_features)  # 8
                feature_motion = torch.zeros([n_clip, 2048 + 256], device=self.device) 
                # Process each motion clip
                for idx, clip in enumerate(motion_features):
                    clip = clip.unsqueeze(dim=0).permute(0, 2, 1, 3, 4)  # Reshape to [1, C(3), T(32), H(224), W(224)]
                    clip = pack_pathway_output(clip, self.device)  # Convert to SlowFast format
                    slow_feature, fast_feature = self.model_motion(clip)
                    slow_feature = slow_feature.squeeze()
                    fast_feature = fast_feature.squeeze()

                    motion_feature = torch.cat([slow_feature, fast_feature]).unsqueeze(0)  # Shape: [1, 2304]
                    feature_motion[idx] = motion_feature 

                feature_motion = feature_motion.unsqueeze(0)  # Shape: [1, n_clip, 2304]

                outputs = self.model(spatial_features, feature_motion)
                score = outputs.item()

                results.append({"video_name": video_name, "SimpleVQA_Score": score})
                print(f"Processed score {score:.4f} for {video_name}")

        self.results.extend(results)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute final SimpleVQA-based metrics."""
        scores = np.array([res["SimpleVQA_Score"] for res in self.results])
        mean_score = np.mean(scores) if scores.size > 0 else 0.0
        print(f"SimpleVQA mean score: {mean_score:.4f}")

        json_file_path = os.path.join(os.getcwd(), "simplevqa_results.json")
        final_results = {"video_results": self.results, "SimpleVQA_Mean_Score": mean_score}
        with open(json_file_path, "w") as json_file:
            json.dump(final_results, json_file, indent=4)
        print(f"SimpleVQA mean score saved to {json_file_path}")

        return {"SimpleVQA_Mean_Score": mean_score}