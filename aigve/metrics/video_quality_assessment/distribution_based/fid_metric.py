# Copyright (c) IFM Lab. All rights reserved.
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
import numpy as np
from scipy.linalg import sqrtm
from typing import Dict, Sequence
import json, os


@METRICS.register_module()
class FIDScore(BaseMetric):

    def __init__(self, 
                 model_name: str = 'inception_v3', 
                 input_shape: tuple = (299, 299, 3), 
                 is_gpu: str = True):
        super(FIDScore, self).__init__()
        self.device = torch.device("cuda" if is_gpu else "cpu")
        self.model_name = model_name
        self.input_shape = input_shape
        if self.model_name == "inception_v3":
            self.model = models.inception_v3(pretrained=True, transform_input=False)
            self.model.fc = nn.Identity()  # Remove classification head
            self.model.eval().to(self.device)
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported for FID computation.")

        # Define preprocessing for InceptionV3
        self.transform = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[1])),  # InceptionV3 input size
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
    def preprocess_tensor(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize and normalize a video tensor.

        Args:
            video_tensor (torch.Tensor): Tensor of shape [T, C, H, W].

        Returns:
            torch.Tensor: Preprocessed tensor of shape [T, C, H, W].
        """
        video_tensor = self.transform(video_tensor / 255.0)
        return video_tensor

    def calculate_statistics(self, video_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate activation statistics (mean and covariance) from video frames.

        Args:
            video_tensor (torch.Tensor): Video tensor [T, C, H, W].

        Returns:
            Tuple of mean and covariance matrix.
        """
        video_tensor = self.preprocess_tensor(video_tensor).to(self.device)
        with torch.no_grad():
            features = self.model(video_tensor).cpu().numpy()  # Extract 2048-d feature vectors

        mu = features.mean(axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fid(self, real: torch.Tensor, fake: torch.Tensor) -> float:
        """
        Calculate FID score between real and generated videos.

        Args:
            real (torch.Tensor): Real video tensor [T, C, H, W].
            fake (torch.Tensor): Generated video tensor [T, C, H, W].

        Returns:
            float: FID score.
        """
        mu1, sigma1 = self.calculate_statistics(real) # Shape[2048], Shape[2048, 2048]
        mu2, sigma2 = self.calculate_statistics(fake)

        # Compute FID score
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1 @ sigma2)

        # Check and correct for imaginary numbers
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data samples and compute FID.

        Args:
            data_batch (dict): A batch of data from the dataloader (not used here).
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
                fid_score = self.calculate_fid(real_tensor, gen_tensor)

                results.append({
                    "Real video_name": real_video_name, 
                    "Generated video_name": gen_video_name, 
                    "FID_Score": fid_score
                })
                print(f"Processed score {fid_score:.4f} between {real_video_name} and {gen_video_name}")

        self.results.extend(results)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the final FID score."""
        scores = np.array([res["FID_Score"] for res in self.results])
        mean_score = np.mean(scores) if scores.size > 0 else 0.0
        print(f"FID mean score: {mean_score:.4f}")

        json_file_path = os.path.join(os.getcwd(), "fid_results.json")
        final_results = {
            "video_results": self.results, 
            "FID_Mean_Score": mean_score
        }
        with open(json_file_path, "w") as json_file:
            json.dump(final_results, json_file, indent=4)
        print(f"FID mean score saved to {json_file_path}")
        
        return {'FID_Mean_Score': mean_score}