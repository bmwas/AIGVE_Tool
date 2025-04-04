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
            self.model = models.inception_v3(pretrained=True, transform_input=False, aux_logits=True)
            self.model.eval().to(self.device)
        else:
            raise ValueError(f"Model '{model_name}' is not supported for Inception Score computation.")

    def preprocess_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """
        Resize and normalize images.

        Args:
            images (torch.Tensor): Tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Preprocessed images.
        """
        images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, -1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, -1, 1, 1)
        images = (images - mean) / std
        return images

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
            output = self.model(images)
            if isinstance(output, tuple):
                output = output[0]
        return output.cpu()

    def calculate_is(self, preds: np.ndarray) -> float:
        """
        Calculate the Inception Score (IS) for a set of predicted class probabilities.

        Args:
            preds (np.ndarray): Array of predicted softmax probabilities with shape [N, num_classes].

        Returns:
            (float): Inception Score.
        """
        kl = preds * (np.log(preds + 1e-10) - np.log(np.expand_dims(np.mean(preds, axis=0), 0) + 1e-10))
        kl_mean = np.mean(np.sum(kl, axis=1))
        return float(np.exp(kl_mean))
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data samples and compute IS.

        Args:
            data_batch (dict): A batch of data from the dataloader (not used here).
            data_samples (List[Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[str], Tuple[str]]):
                A list containing four tuples:
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
                gen_video_name = gen_video_name_tuple[i]
                gen_tensor = gen_tensor_tuple[i]

                logits = self.compute_inception_features(gen_tensor)
                preds = torch.nn.functional.softmax(logits, dim=1).numpy()
                is_score = self.calculate_is(preds)

                results.append({
                    "Generated video_name": gen_video_name, 
                    "IS_Score": is_score,
                })
                print(f"Processed IS score {is_score:.4f} for {gen_video_name}")

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

        mean_score = np.mean(scores) if scores.size > 0 else 0.0

        print(f"IS mean score: {mean_score:.4f}")

        json_file_path = os.path.join(os.getcwd(), "is_results.json")
        final_results = {
            "video_results": self.results, 
            "IS_Mean_Score": mean_score, 
        }
        with open(json_file_path, "w") as json_file:
            json.dump(final_results, json_file, indent=4)
        print(f"IS mean score saved to {json_file_path}")

        return {"IS_Mean_Score": mean_score}

    