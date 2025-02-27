# Copyright (c) IFM Lab. All rights reserved.
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from mmengine.model import BaseModel
from mmengine.registry import MODELS, METRICS
import numpy as np
from typing import Dict, Sequence
from mmengine.logging import MMLogger
import tensorflow as tf

@MODELS.register_module()
@METRICS.register_module()
class ISScore(BaseModel):
    """
    Inception Score (IS) implementation.
    
    The Inception Score measures the quality and diversity of generated images
    by evaluating the KL divergence between the conditional class distribution
    and the marginal class distribution.
    
    Args:
        model_name (str): Name of the model to use. Currently only 'inception_v3' is supported.
        input_shape (tuple): Input shape for the model (height, width, channels).
        splits (int): Number of splits to use when calculating the score.
    """

    def __init__(self, model_name='inception_v3', input_shape=(299, 299, 3), splits=10):
        super().__init__()
        if model_name == 'inception_v3':
            self.model = InceptionV3(include_top=True, input_shape=input_shape)
        else:
            raise ValueError("Only 'inception_v3' is currently supported.")
        
        self.splits = splits
        self.results = []

    def preprocess_images(self, images):
        """
        Preprocess the images for the InceptionV3 model.

        Parameters:
        images (numpy array): Input images.

        Returns:
        numpy array: Preprocessed images.
        """
        return preprocess_input(images)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data samples and predictions.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        result = dict()
        images = data_samples

        # Calculate IS score
        is_score, is_std = self.calculate_is(images)
        result['is_score'] = is_score
        result['is_std'] = is_std

        self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        is_scores = np.zeros(len(results))
        is_stds = np.zeros(len(results))
        
        for i, result in enumerate(results):
            is_scores[i] = result['is_score']
            is_stds[i] = result['is_std']
        
        mean_is = np.mean(is_scores)
        mean_std = np.mean(is_stds)
        
        print("Test results: IS Score={:.4f} ± {:.4f}".format(mean_is, mean_std))

        return {'is': mean_is, 'is_std': mean_std}

    def calculate_is(self, images):
        """
        Calculate the Inception Score for a set of images.

        Parameters:
        images (numpy array): Input images.

        Returns:
        tuple: The mean and standard deviation of the inception score.
        """
        # Preprocess images
        images = self.preprocess_images(images)
        
        # Get predictions
        preds = self.model.predict(images)
        
        # Calculate scores for each split
        scores = []
        n = preds.shape[0]
        split_size = n // self.splits
        
        for i in range(self.splits):
            part = preds[i * split_size:(i + 1) * split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))
            
        return np.mean(scores), np.std(scores)