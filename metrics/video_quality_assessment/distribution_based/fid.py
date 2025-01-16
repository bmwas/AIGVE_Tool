import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm


class FIDScore:
    def __init__(self, model_name='inception_v3', input_shape=(299, 299, 3), pooling='avg'):
        """
        Initialize the FIDScore evaluator.

        Parameters:
        model_name (str): The model to use for feature extraction (default: 'inception_v3').
        input_shape (tuple): Input shape of the images (default: (299, 299, 3)).
        pooling (str): Pooling type to use ('avg' or 'max') (default: 'avg').
        """
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
