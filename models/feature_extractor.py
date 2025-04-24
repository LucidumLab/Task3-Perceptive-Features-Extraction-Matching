import cv2

class FeatureExtractor:
    def __init__(self, method='DoG'):
        """
        Initialize the keypoint extractor with the desired method.
        Supported methods: 'DoG', 'Harris', 
        """
        if method == 'SIFT':
            self.extractor = cv2.SIFT_create()
        elif method == 'ORB':
            self.extractor = cv2.ORB_create()
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'SIFT' or 'ORB'.")

    def extract_keypoints_and_descriptors(self, image):
        """
        Extract keypoints and descriptors from the given image.
        
        Args:
            image (numpy.ndarray): Input image in grayscale format.
        
        Returns:
            keypoints (list): List of detected keypoints.
            descriptors (numpy.ndarray): Corresponding descriptors for the keypoints.
        """
        if image is None:
            raise ValueError("Input image is None. Please provide a valid image.")
        
        keypoints, descriptors = self.extractor.detectAndCompute(image, None)
        return keypoints, descriptors