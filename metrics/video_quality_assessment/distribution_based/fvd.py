import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import preprocess_input


class FVDScore:
    def __init__(self, model_path, feature_layer=-2):
        """
        Initialize the FVDScore evaluator.

        Parameters:
        model_path (str): Path to the pre-trained I3D model.
        feature_layer (int or str): The layer of the I3D model to use for feature extraction.
        """
        self.i3d_model = self.load_i3d_model(model_path, feature_layer)

    @staticmethod
    def load_i3d_model(model_path, feature_layer):
        """
        Load a pre-trained I3D model for feature extraction.

        Parameters:
        model_path (str): Path to the pre-trained I3D model.
        feature_layer (int or str): The layer of the I3D model to use for feature extraction.

        Returns:
        Model: The I3D model for feature extraction.
        """
        i3d_model = tf.keras.models.load_model(model_path)
        feature_model = Model(inputs=i3d_model.input, outputs=i3d_model.layers[feature_layer].output)
        return feature_model

    def preprocess_videos(self, videos):
        """
        Preprocess videos for the I3D model.

        Parameters:
        videos (numpy array): Input videos as a numpy array of shape (num_videos, num_frames, height, width, channels).

        Returns:
        numpy array: Preprocessed videos.
        """
        return preprocess_input(videos)

    def calculate_statistics(self, videos):
        """
        Calculate the feature statistics (mean and covariance) for a set of videos.

        Parameters:
        videos (numpy array): Preprocessed videos.

        Returns:
        tuple: Mean and covariance of the features.
        """
        features = self.i3d_model.predict(videos)
        mu = features.mean(axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fvd(self, videos1, videos2):
        """
        Calculate the FVD score between two sets of videos.

        Parameters:
        videos1 (numpy array): First set of videos of shape (num_videos, num_frames, height, width, channels).
        videos2 (numpy array): Second set of videos of shape (num_videos, num_frames, height, width, channels).

        Returns:
        float: The FVD score.
        """
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
