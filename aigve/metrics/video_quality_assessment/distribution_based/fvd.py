import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
from scipy.linalg import sqrtm
# from tensorflow.keras.applications.inception_v3 import preprocess_input
from mmengine.model import BaseModel
from mmengine.registry import MODELS, METRICS
from typing import Dict, Sequence
from mmengine.logging import MMLogger

@MODELS.register_module()
@METRICS.register_module()
class FVDScore(BaseModel):
    def __init__(self, model_path, feature_layer=-2):
        super().__init__()
        self.i3d_model = self.load_i3d_model(model_path, feature_layer)
        self.results = []

    @staticmethod
    def load_i3d_model(model_path, feature_layer):
        i3d_model = tf.keras.models.load_model(model_path)
        feature_model = Model(inputs=i3d_model.input, outputs=i3d_model.layers[feature_layer].output)
        return feature_model

    def preprocess_videos(self, videos):
        return preprocess_input(videos)

    def calculate_statistics(self, videos):
        features = self.i3d_model.predict(videos)
        mu = features.mean(axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        result = dict()
        videos1, videos2 = data_samples

        # Calculate FVD score
        fvd_score = self.calculate_fvd(videos1, videos2)
        result['fvd_score'] = fvd_score

        self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        fvd_scores = np.zeros(len(results))
        for i, result in enumerate(results):
            fvd_scores[i] = result['fvd_score']
        
        mean_fvd = np.mean(fvd_scores)
        
        print("Test results: FVD Score={:.4f}".format(mean_fvd))

        return {'fvd': mean_fvd}

    def calculate_fvd(self, videos1, videos2):
        # Preprocess videos
        videos1 = self.preprocess_videos(videos1)
        videos2 = self.preprocess_videos(videos2)

        # Calculate statistics
        mu1, sigma1 = self.calculate_statistics(videos1)
        mu2, sigma2 = self.calculate_statistics(videos2)

        # Compute FVD score
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))

        # Check and correct for imaginary numbers
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fvd = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fvd
