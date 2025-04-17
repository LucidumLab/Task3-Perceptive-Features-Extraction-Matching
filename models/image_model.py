'''
Custom class for image model save the image and its metadata, features.

'''

class Image_Model:
    
    def __init__(self, image):
        '''
        Initializes the image model with the given image.
        
        Parameters:
            image (numpy.ndarray): Input image.
        '''
        self.image = image
        if len(image.shape) == 3 and image.shape[2] == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image
        self.Ix, self.Iy = self.compute_gradients()
        self.Ixx = None
        self.Ixy = None
        self.gradient_magnitude = np.sqrt(self.Ix**2 + self.Iy**2)
        self.gradient_direction = np.arctan2(self.Iy, self.Ix)
        self.corners = None # Placeholder for corner detection results
        
        
        self.keypoints = None
        self.descriptors = None
        
    
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
    
    def build_gaussian_pyramid(self, levels=3):
        '''
        Builds a Gaussian pyramid of the image.
        
        Parameters:
            levels (int): Number of levels in the pyramid.
        
        Returns:
            pyramid (list): List of images in the pyramid.
        '''
        pyramid = [self.image]
        for i in range(levels):
            self.image = cv2.pyrDown(self.image)
            pyramid.append(self.image)
        
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