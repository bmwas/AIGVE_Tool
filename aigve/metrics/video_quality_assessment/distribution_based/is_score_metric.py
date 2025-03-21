# Copyright (c) IFM Lab. All rights reserved.

import os, json
from typing import Dict, Sequence
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS



@METRICS.register_module()
class ISScore(BaseMetric):
    """
    Inception Score (IS) implementation.
    
    The Inception Score measures the quality and diversity of generated images
    by evaluating the KL divergence between the conditional class distribution
    and the marginal class distribution.
    
    Args:
        model_name (str): Name of the model to use. Currently only 'inception_v3' is supported.
        input_shape (tuple): Input shape for the model (height, width, channels).
        splits (int): Number of splits to use when calculating the score.
        is_gpu (bool): Whether to use GPU. Defaults to True.
    """

    def __init__(
            self, 
            model_name: str = 'inception_v3', 
            input_shape: tuple = (299, 299, 3), 
            splits: int = 10,
            is_gpu: bool = True):
        super(ISScore, self).__init__()
        self.device = torch.device("cuda" if is_gpu and torch.cuda.is_available() else "cpu")
        self.splits = splits

        if model_name == 'inception_v3':
            self.model = models.inception_v3(pretrained=True, transform_input=False)
            self.model.fc = nn.Identity()  # Remove classification head
            self.model.eval().to(self.device)
        else:
            raise ValueError(f"Model '{model_name}' is not supported for Inception Score computation.")
        
        self.transform = transforms.Compose([
            transforms.Resize((input_shape[0], input_shape[1])),  # InceptionV3 input size
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    def preprocess_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """
        Resize and normalize images.

        Args:
            images (torch.Tensor): Tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Preprocessed images.
        """
        return self.transform(images / 255.0)

    def compute_inception_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute Inception features for a batch of images.

        Args:
            images (torch.Tensor): Preprocessed image tensor.

        Returns:
            torch.Tensor: Feature activations from InceptionV3.
        """
        images = self.preprocess_tensor(images).to(self.device)
        with torch.no_grad():
            features = self.model(images).cpu()
        return features

    def calculate_is(self, images: torch.Tensor) -> tuple[float, float]:
        """
        Compute Inception Score.

        Args:
            images (torch.Tensor): Image tensor [B, C, H, W].

        Returns:
            Tuple[float, float]: Mean and standard deviation of Inception Score.
        """
        features = self.compute_inception_features(images)
        preds = torch.nn.functional.softmax(features, dim=1).numpy()

        # Split into chunks
        scores = []
        batch_size = preds.shape[0]
        split_size = batch_size // self.splits

        for i in range(self.splits):
            part = preds[i * split_size:(i + 1) * split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))

        return float(np.mean(scores)), float(np.std(scores))
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data samples and compute IS.

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

        batch_size = len(gen_tensor_tuple)
        with torch.no_grad():
            for i in range(batch_size):
                real_video_name = real_video_name_tuple[i]
                gen_video_name = gen_video_name_tuple[i]
                real_tensor = real_tensor_tuple[i]
                gen_tensor = gen_tensor_tuple[i]
                is_score, is_std = self.calculate_is(gen_tensor)

                results.append({
                    "Real video_name": real_video_name, 
                    "Generated video_name": gen_video_name, 
                    "IS_Score": is_score,
                    "IS_Std": is_std
                })
                print(f"Processed IS score {is_score:.4f} ± {is_std:.4f} for {gen_video_name}")

        self.results.extend(results)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        Compute the final IS score.

        Args:
            results (list): List of IS scores for each batch.

        Returns:
            Dict[str, float]: Dictionary containing mean IS score and standard deviation.
        """
        scores = np.array([res["IS_Score"] for res in self.results])
        stds = np.array([res["IS_Std"] for res in self.results])

        mean_score = np.mean(scores) if scores.size > 0 else 0.0
        mean_std = np.mean(stds) if stds.size > 0 else 0.0

        print(f"IS mean score: {mean_score:.4f} ± {mean_std:.4f}")

        json_file_path = os.path.join(os.getcwd(), "is_results.json")
        final_results = {
            "video_results": self.results, 
            "IS_Mean_Score": mean_score, 
            "IS_Mean_Std": mean_std
        }
        with open(json_file_path, "w") as json_file:
            json.dump(final_results, json_file, indent=4)
        print(f"IS mean score saved to {json_file_path}")

        return {"IS_Mean_Score": mean_score, "IS_Mean_Std": mean_std}

    