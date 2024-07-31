import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm

def calculate_fid_score(images1, images2):
    """
    Calculate the Frechet Inception Distance (FID) score between two sets of images.
    
    Parameters:
    images1 (numpy array): Numpy array of the first set of images.
    images2 (numpy array): Numpy array of the second set of images.
    
    Returns:
    float: The FID score.
    """
    
    # Load the InceptionV3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    
    # Preprocess the images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    
    # Get the feature representations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    
    # Calculate the mean and covariance of the features
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    # Calculate the FID score
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check and correct for imaginary numbers in the result
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
