# Copyright (c) IFM Lab. All rights reserved.
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from mmengine.model import BaseModel
from mmengine.registry import MODELS, METRICS
import numpy as np
from scipy.linalg import sqrtm
from typing import Dict, Sequence
from mmengine.logging import MMLogger

@MODELS.register_module()
@METRICS.register_module()
class FIDScore(BaseModel):

    def __init__(self, model_name='inception_v3', input_shape=(299, 299, 3), pooling='avg'):
        super().__init__()
        if model_name == 'inception_v3':
            self.model = InceptionV3(include_top=False, pooling=pooling, input_shape=input_shape)
        else:
            raise ValueError("Only 'inception_v3' is currently supported.")

    def preprocess_images(self, images):
        """
        Preprocess the images for the InceptionV3 model.

        Parameters:
        images (numpy array): Input images.

        Returns:
        numpy array: Preprocessed images.
        """
        return preprocess_input(images)

    def calculate_statistics(self, images):
        """
        Calculate the feature statistics (mean and covariance) for a set of images.

        Parameters:
        images (numpy array): Preprocessed images.

        Returns:
        tuple: Mean and covariance of the features.
        """
        features = self.model.predict(images)
        mu = features.mean(axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data samples and predictions.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        result = dict()
        images1, images2 = data_samples

        # Calculate FID score
        fid_score = self.calculate_fid(images1, images2)
        result['fid_score'] = fid_score

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

        fid_scores = np.zeros(len(results))
        for i, result in enumerate(results):
            fid_scores[i] = result['fid_score']
        
        mean_fid = np.mean(fid_scores)
        
        print("Test results: FID Score={:.4f}".format(mean_fid))

        return {'fid': mean_fid}

    def calculate_fid(self, images1, images2):
        """
        Calculate the FID score between two sets of images.

        Parameters:
        images1 (numpy array): First set of images.
        images2 (numpy array): Second set of images.

        Returns:
        float: The FID score.
        """
        # Preprocess images
        images1 = self.preprocess_images(images1)
        images2 = self.preprocess_images(images2)

        # Calculate statistics
        mu1, sigma1 = self.calculate_statistics(images1)
        mu2, sigma2 = self.calculate_statistics(images2)

        # Compute FID score
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))

        # Check and correct for imaginary numbers
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
