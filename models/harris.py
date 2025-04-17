'''
This class implements the Harris corner detection algorithm.
It is used to detect corners in an image by analyzing the intensity gradients in the image.
'''
from image_model import Image_Model
import numpy as np
import cv2

class Corner_Extractor:

    def __init__(self):
        '''
        Initializes the Harris corner detector with default parameters.
        '''
        self.k = 0.04
        self.threshold = 0.01
        
        self.sigma = 1.0
        self.smoothing_window = 3
        
    def corner_detection_harris(self, image : Image_Model, sigma=1.0, smoothing_window=3, k=0.04, threshold=0.01):
        '''
        Detects corners in the given image using the Harris corner detection algorithm.
        
        Parameters:
            image (numpy.ndarray): Input image in grayscale.
            sigma (float): Standard deviation for Gaussian filter.
            smoothing_window (int): Size of the window used for corner detection.
            k (float): Harris detector free parameter.
            threshold (float): Threshold for corner response function.
        
        Returns:
            corners (numpy.ndarray): Coordinates of detected corners.
        '''
        sigma, smoothing_window, k, threshold = self.get_parameters()
        
        # Compute gradients
        
        # Compute products of gradients
        Ixx = image.Ix * image.Ix
        Ixy = image.Ix * image.Iy
        Iyy = image.Iy * image.Iy
        
        # Apply Gaussian filter to products of gradients
        Sxx = cv2.GaussianBlur(Ixx, (smoothing_window, smoothing_window), sigma)
        Sxy = cv2.GaussianBlur(Ixy, (smoothing_window, smoothing_window), sigma)
        Syy = cv2.GaussianBlur(Iyy, (smoothing_window, smoothing_window), sigma)
        
        # Compute determinant and trace of the matrix M
        trace = (Sxx + Syy) 
        diff = (Sxx - Syy) 
        det = (Sxx * Syy) - (Ixy ** 2)
        
        
        
        # Compute Harris response
        R = det - k * (trace ** 2)
        corners_mask = np.argwhere(R > threshold * np.max(R))

        
        sqrt_term = np.sqrt((diff/2)**2 + Sxy**2)
        min_eigenval = (trace/2) - sqrt_term  # Shi-Tomasi
        # Thresholding
        
        return corners_mask
    
    def corner_detection_shi(self, image : Image_Model, max_corners=100, quality_level=0.01, minDistance=10):
        sigma, smoothing_window, k, threshold = self.get_parameters()
        
        # Compute gradients
        
        # Compute products of gradients
        Ixx = image.Ix * image.Ix
        Ixy = image.Ix * image.Iy
        Iyy = image.Iy * image.Iy
        
        Sxx = cv2.GaussianBlur(Ixx, (smoothing_window, smoothing_window), sigmaX=1)
        Syy = cv2.GaussianBlur(Iyy, (smoothing_window, smoothing_window), sigmaX=1)
        Sxy = cv2.GaussianBlur(Ixy, (smoothing_window, smoothing_window), sigmaX=1)
        

        # Step 4: Compute min eigenvalue using vectorized formula
        trace = (Sxx + Syy) / 2
        diff = (Sxx - Syy) / 2
        sqrt_term = np.sqrt(diff**2 + Sxy**2)
        min_eigenval = trace - sqrt_term  # Shi-Tomasi)
        threshold = quality_level * min_eigenval.max()
        corner_mask = min_eigenval > threshold
        
        corners = np.argwhere(corner_mask)
        corner_vals = min_eigenval[corner_mask]
        sorted_indices = np.argsort(-corner_vals)
        sorted_corners= corners[sorted_indices]
        
        selected_corners = sorted_corners[:max_corners]
        
        # Return as list of (x, y) tuples
        return [(int(x), int(y)) for y, x in selected_corners]
        
     
     
    def get_parameters(self):
        
        '''
        Returns the parameters for the Harris corner detection algorithm.
        
        Returns:
            sigma (float): Standard deviation for Gaussian filter.
            smoothing_window (int): Size of the window used for corner detection.
            k (float): Harris detector free parameter.
            threshold (float): Threshold for corner response function.
        '''
        return self.sigma, self.smoothing_window, self.k, self.threshold
    
    def set_parameters(self, sigma=1.0, smoothing_window=3, k=0.04, threshold=0.01):
        '''
        Sets the parameters for the Harris corner detection algorithm.
        
        Parameters:
            sigma (float): Standard deviation for Gaussian filter.
            smoothing_window (int): Size of the window used for corner detection.
            k (float): Harris detector free parameter.
            threshold (float): Threshold for corner response function.
        '''
        self.sigma = sigma
        self.smoothing_window = smoothing_window
        self.k = k
        self.threshold = threshold

