
import numpy as np
import cv2
from math import pi, sqrt, exp, log
from typing import List
from numpy import float32, array, zeros, stack, dot, trace
from cv2 import KeyPoint, INTER_LINEAR, BORDER_CONSTANT
from scipy.linalg import lstsq, det

class ImageModel:
    
    def __init__(self):
        '''
        Initializes the image model with the given image.
        
        Parameters:
            image (numpy.ndarray): Input image.
        '''
        self.image = None
        self.gray = None
        self.base_image = None # Assuming sigma=1.0 and assumed_blur=0.5 for base image generation
        self.Ix= None
        self.Iy = None
        self.Ixx = None
        self.Ixy = None
        self.gradient_magnitude = None
        self.gradient_direction =  None
        self.corners = None # Placeholder for corner detection results
        self.octaves = None
        self.pyramid = None
        self.keypoints = None
        self.descriptors = None
    def set_image(self, image):
        '''
        Sets the image for the model.
        
        Parameters:
            image (numpy.ndarray): Input image.
        '''
        self.image = image
        if len(image.shape) == 3 and image.shape[2] == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image
        print("Image shape:", self.gray.shape)
        self.Ix, self.Iy = self.compute_gradients()
        self.Ixx = self.Ix**2
        self.Ixy = self.Ix * self.Iy
        self.Iyy = self.Iy**2
        print("Gradient shape:", self.Ix.shape, self.Iy.shape)
        self.gradient_magnitude = np.sqrt(self.Ix**2 + self.Iy**2)
        self.gradient_direction = np.arctan2(self.Iy, self.Ix)
        
        self.base_image = self.generateBaseImage(1.0, 0.5) # Assuming sigma=1.0 and assumed_blur=0.5 for base image generation
        print
        self.octaves = self.computeNumberOfOctaves(self.gray.shape)
        print("Number of octaves:", self.octaves)
        # Build the image pyramid
        self.pyramid = self.build_image_pyramid(self.gray, self.octaves)
        print("Pyramid shape:", [p.shape for p in self.pyramid])
        
    
    def generateBaseImage(self, sigma, assumed_blur):
        """Generate base image from input image by upsampling by 2 in both directions and blurring
        """
        image = cv2.resize(self.gray, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
        sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur

    def compute_gradients(self, kerna="sobel"):
        '''
        Computes the gradients of the image using Sobel operator.
        
        Returns:
            Ix (numpy.ndarray): Gradient in x direction.
            Iy (numpy.ndarray): Gradient in y direction.
        '''
        Ix = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=5)
        
        return Ix, Iy
    
    def build_image_pyramid(self, image, num_octaves):
        """
        Builds an image pyramid by downsampling the image for each octave.
        
        Args:
            image: Input grayscale image (H x W).
            num_octaves: Number of octaves (pyramid levels).
        
        Returns:
            pyramid: List of images, each half the size of the previous.
        """
        pyramid = [image]
        for _ in range(num_octaves - 1):
            # Downsample by 2x using nearest-neighbor interpolation (for precision)
            downsampled = cv2.resize(
                pyramid[-1], 
                (int(pyramid[-1].shape[1] / 2), int(pyramid[-1].shape[0] / 2)), 
                interpolation=cv2.INTER_NEAREST
            )
            pyramid.append(downsampled)
        return pyramid
    def detect_corners(self, method='harris'):
        if method == 'harris':
            dst = cv2.cornerHarris(self.gray, blockSize=2, ksize=3, k=0.04)
            self.corners = dst
        elif method == 'shi-tomasi':
            corners = cv2.goodFeaturesToTrack(self.gray, 100, 0.01, 10)
            self.corners = np.int0(corners)
        else:
            raise ValueError("Unsupported method")
    
    def computeNumberOfOctaves(self, image_shape):
        return min(int(round(log(min(image_shape)) / log(2) - 1)), 6)# This is the number of octaves in the image whith upper limmit of 6 
    
    def feature_matching(self, other_image_model):
        '''
        Matches features between this image model and another image model.
        
        Parameters:
            other_image_model (image_model): Another image model to match features with.
        
        Returns:
            matches (list): List of matched keypoints.
        '''
        # Placeholder for feature matching logic
        matches = []
        
        return matches
    def export_keypoints_to_file(self, filepath):
        """
        Exports keypoints and descriptors to a text file.
        Format: x y size angle response octave descriptor[128...]
        """
        if self.keypoints is None or self.descriptors is None:
            raise ValueError("Keypoints and descriptors must be computed before exporting.")
        
        with open(filepath, 'w') as f:
            for kp, desc in zip(self.keypoints, self.descriptors):
                x, y = kp.pt
                size = kp.size
                angle = kp.angle if kp.angle is not None else -1
                response = kp.response
                octave = kp.octave
                desc_str = ' '.join([f'{v:.6f}' for v in desc])
                f.write(f"{x:.6f} {y:.6f} {size:.6f} {angle:.6f} {response:.6f} {octave} {desc_str}\n")

    def load_keypoints_from_file(self, filepath):
        """
        Loads keypoints and descriptors from a text file.
        Returns: keypoints (list), descriptors (list)
        """
        keypoints = []
        descriptors = []

        with open(filepath, 'r') as f:
            for line in f:
                values = line.strip().split()
                x, y = float(values[0]), float(values[1])
                size = float(values[2])
                angle = float(values[3]) if float(values[3]) != -1 else None
                response = float(values[4])
                octave = int(values[5])
                descriptor = np.array([float(v) for v in values[6:]], dtype=np.float32)

                kp = KeyPoint()
                kp.pt = (x, y)
                kp.size = size
                kp.angle = angle
                kp.response = response
                kp.octave = octave
                kp.descriptor = descriptor

                keypoints.append(kp)
                descriptors.append(descriptor)

        return keypoints, descriptors
    
    
if __name__ == '__main__':
    # Example usage
    image = cv2.imread('box.png')
    model = ImageModel()
    model.set_image(image)
    
    print("Image set successfully.")