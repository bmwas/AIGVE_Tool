# Copyright (c) IFM Lab. All rights reserved.

import os, json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from scipy.linalg import sqrtm
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from typing import Dict, Sequence


@METRICS.register_module()
class FVDScore(BaseMetric):
    """
    Fréchet Video Distance (FVD) computation using I3D model.
    Users can first download the pretrained I3D model from: 
    https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth
    Then put in the folder: 
    AIGVE_Tool/aigve/metrics/video_quality_assessment/distribution_based/fvd/

    Args:
        model_path (str): Path to pre-trained I3D model.
        feature_layer (int): Layer to extract features from. Default is -2 (penultimate layer).
        is_gpu (bool): Whether to use GPU. Default is True.
    """
    def __init__(self, 
                 model_path: str, 
                 feature_layer: int = -2, 
                 is_gpu: bool = True):
        super(FVDScore, self).__init__()
        self.device = torch.device("cuda" if is_gpu and torch.cuda.is_available() else "cpu")
        self.model = self.load_i3d_model(model_path, feature_layer)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # I3D input size
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    def load_i3d_model(self, model_path: str, feature_layer: int) -> torch.nn.Module:
        """
        Load a pre-trained I3D model and modify it to extract features.

        Args:
            model_path (str): Path to the I3D model checkpoint.
            feature_layer (int): The layer index from which to extract features.

        Returns:
            torch.nn.Module: I3D feature extraction model.
        """
        model = models.video.r3d_18(pretrained=True)  # Using ResNet3D as an I3D alternative
        model.fc = nn.Identity()  # Remove classification head

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: Model checkpoint not found at {model_path}, using default weights.")

        return model

    def preprocess_tensor(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize and normalize a video tensor.

        Args:
            video_tensor (torch.Tensor): Tensor of shape [T, C, H, W].

        Returns:
            torch.Tensor: Preprocessed tensor of shape [T, C, H, W].
        """
        return self.transform(video_tensor / 255.0)

    def calculate_statistics(self, video_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract activation statistics from video frames.

        Args:
            video_tensor (torch.Tensor): Video tensor [T, C, H, W].

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and covariance of extracted features.
        """
        video_tensor = self.preprocess_tensor(video_tensor).to(self.device)
        self.model.to(self.device)
        # Permute to match I3D input format [B, C, T, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # Shape: [1, 3, T, H, W]
        with torch.no_grad():
            features = self.model(video_tensor).cpu().numpy()

        # print('features: ', features.shape)
        mu = features.mean(axis=0)
        # Ensure at least 2 samples to compute covariance
        if features.shape[0] > 1:
            sigma = np.cov(features, rowvar=False)
        else:
            sigma = np.zeros((features.shape[1], features.shape[1])) # Identity fallback
        return mu, sigma

    def calculate_fvd(self, real: torch.Tensor, fake: torch.Tensor) -> float:
        """
        Compute FVD score between real and generated videos.

        Args:
            real (torch.Tensor): Real video tensor [T, C, H, W].
            fake (torch.Tensor): Generated video tensor [T, C, H, W].

        Returns:
            float: FVD score.
        """
        mu1, sigma1 = self.calculate_statistics(real) # Shape[512], Shape[512, 512]
        mu2, sigma2 = self.calculate_statistics(fake)
        # print(f"mu1 shape: {mu1.shape}, sigma1 shape: {sigma1.shape}")
        # print(f"mu2 shape: {mu2.shape}, sigma2 shape: {sigma2.shape}")

        # Ensure sigma matrices are at least 2D
        if sigma1.ndim < 2:
            sigma1 = np.expand_dims(sigma1, axis=0)
        if sigma2.ndim < 2:
            sigma2 = np.expand_dims(sigma2, axis=0)

        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1 @ sigma2)

        # Check and correct for imaginary numbers
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process a batch of videos and compute FVD.

        Args:
            data_batch (dict): Not used here.
            data_samples (List[Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[str], Tuple[str]]):
                A list containing two tuples:
                - A tuple of `real_tensor` (torch.Tensor): Real video tensor [T, C, H, W].
                - A tuple of `gen_tensor` (torch.Tensor): Generated video tensor [T, C, H, W].
                - A tuple of `real_video_name` (str): Ground-truth video filename.
                - A tuple of `gen_video_name` (str): Generated video filename.
                The len of each tuples are the batch size.
        """
        results = []
        real_tensor_tuple, gen_tensor_tuple, real_video_name_tuple, gen_video_name_tuple = data_samples

        batch_size = len(real_tensor_tuple)
        with torch.no_grad():
            for i in range(batch_size):
                real_video_name = real_video_name_tuple[i]
                gen_video_name = gen_video_name_tuple[i]
                real_tensor = real_tensor_tuple[i]
                gen_tensor = gen_tensor_tuple[i]

                fvd_score = self.calculate_fvd(real_tensor, gen_tensor)

                results.append({
                    "Real video_name": real_video_name, 
                    "Generated video_name": gen_video_name, 
                    "FVD_Score": fvd_score
                })
                print(f"Processed FVD score {fvd_score:.4f} between {real_video_name} and {gen_video_name}")

        self.results.extend(results)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        Compute the final FVD score.

        Args:
            results (list): List of FVD scores for each batch.

        Returns:
            Dict[str, float]: Dictionary containing mean FVD score.
        """
        scores = np.array([res["FVD_Score"] for res in self.results])
        mean_score = np.mean(scores) if scores.size > 0 else 0.0
        print(f"FVD mean score: {mean_score:.4f}")

        json_file_path = os.path.join(os.getcwd(), "fvd_results.json")
        final_results = {
            "video_results": self.results, 
            "FVD_Mean_Score": mean_score
        }
        with open(json_file_path, "w") as json_file:
            json.dump(final_results, json_file, indent=4)
        print(f"FVD mean score saved to {json_file_path}")

        return {"FVD_Mean_Score": mean_score}
