'''
This class implements the Harris corner detection algorithm.
It is used to detect corners in an image by analyzing the intensity gradients in the image.
'''
from models.ImageModel import ImageModel
import numpy as np
import cv2
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
from io import BytesIO
import numpy as np
from PIL import Image

class CornerDetection:

    def __init__(self):
        '''
        Initializes the Harris corner detector with default parameters.
        '''
        self.k = 0.04
        self.threshold = 0.01
        
        self.sigma = 1.0
        self.smoothing_window = 3
        
    def corner_detection_harris(self, image : ImageModel, sigma=1.0, smoothing_window=3, k=0.04, threshold=0.01):
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
        # Apply Gaussian filter to products of gradients
        Sxx = cv2.GaussianBlur(image.Ixx, (smoothing_window, smoothing_window), sigma)
        Sxy = cv2.GaussianBlur(image.Ixy, (smoothing_window, smoothing_window), sigma)
        Syy = cv2.GaussianBlur(image.Iyy, (smoothing_window, smoothing_window), sigma)
        
        # Compute determinant and trace of the matrix M
        trace = (Sxx + Syy) 
        diff = (Sxx - Syy) 
        det = (Sxx * Syy) - (image.Ixy ** 2)
        
        
        
        # Compute Harris response
        R = det - k * (trace ** 2)
        print("R shape:", R.shape)
        corners_mask = np.argwhere(R > threshold * np.max(R))

        
        sqrt_term = np.sqrt((diff/2)**2 + Sxy**2)
        min_eigenval = (trace/2) - sqrt_term  # Shi-Tomasi
        
        return corners_mask
    
    def corner_detection_shi(self, image : ImageModel, max_corners=100, quality_level=0.01, minDistance=10):
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
    def visualize_corners(self, image, corners, title='Corners'):
        # Convert grayscale image to BGR for visualization
        img_color = cv2.cvtColor( image, cv2.COLOR_GRAY2BGR)
        
        # Create a new figure and axes for Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))  # Show image in RGB format
        
        # Plot corners as red circles
        for y, x in corners:
            ax.plot(x, y, 'ro', markersize=3)

        # Render the figure to a canvas
        canvas = FigureCanvas(fig)
        canvas.draw()

        # Save the canvas to a buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Convert the buffer to a PIL Image and then to a NumPy array
        image_pil = Image.open(buf)
        image_np = np.array(image_pil)

        # Close the figure to prevent it from showing
        plt.close(fig)
        
        # Return the image as a NumPy array
        return image_np
    
     
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

class BlobsDetection:
    def __init__(self,
                 sigma: float = 1.6,
                 num_intervals: int = 3,
                 contrast_threshold: float = 0.04,
                 image_border_width: int = 5,
                 eigenvalue_ratio: float = 10,
                 max_interp_steps: int = 5):
        """
        Blob detector for DoG and DoH methods, integrated with Image_Model.
        """
        self.sigma = sigma
        self.num_intervals = num_intervals
        self.contrast_threshold = contrast_threshold
        self.image_border_width = image_border_width
        self.eigenvalue_ratio = eigenvalue_ratio
        self.max_interp_steps = max_interp_steps

    def dog_blob_detector(self, image_model):
        """
        Detect blobs via Difference of Gaussians (DoG).
        Returns a list of cv2.KeyPoint.
        """
        # 1. Build pyramid if needed
        if image_model.pyramid is None:
            image_model.pyramid = image_model.build_image_pyramid(image_model.gray, image_model.octaves)
        # 2. Generate Gaussian images
        kernels = self.generate_gaussian_kernels(self.sigma, self.num_intervals)
        print("Smoothing Kernels:", kernels)
        gaussians = self.apply_gaussian_to_octaves(image_model.pyramid, kernels)
        # 3. Generate DoG images
        dog_imgs = self.generate_dog_images(gaussians)
        # 4. Find scale-space extrema
        keypoints = self.find_scale_space_extrema(gaussians, dog_imgs)
        image_model.keypoints = keypoints
        return keypoints

    def doh_blob_detector(self, image_model):
        """
        Detect blobs via Determinant of Hessian (DoH).
        Returns a list of cv2.KeyPoint.
        """
        # 1. Ensure pyramid
        if image_model.pyramid is None:
            image_model.pyramid = image_model.build_image_pyramid(image_model.gray, image_model.octaves)
        # 2. Generate Gaussian images
        kernels = self.generate_gaussian_kernels(self.sigma, self.num_intervals)
        gaussians = self.apply_gaussian_to_octaves(image_model.pyramid, kernels)
        # 3. Compute DoH responses at each scale
        doh_pyramid = []
        for octave in gaussians:
            octave_doh = []
            for img in octave:
                hxx = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=3)
                hyy = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=3)
                hxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
                det_h = (hxx * hyy) - (hxy ** 2)
                octave_doh.append(det_h)
            doh_pyramid.append(octave_doh)
        # 4. Find 3D maxima in DoH responses
        keypoints = []
        thr = self.contrast_threshold
        for o_idx, octave in enumerate(doh_pyramid):
            for s in range(1, len(octave) - 1):
                prev_img, cur_img, next_img = octave[s-1], octave[s], octave[s+1]
                h, w = cur_img.shape
                for i in range(self.image_border_width, h - self.image_border_width):
                    for j in range(self.image_border_width, w - self.image_border_width):
                        v = cur_img[i, j]
                        if v < thr:
                            continue
                        # extract 3x3x3 neighborhood
                        cube = np.stack([
                            prev_img[i-1:i+2, j-1:j+2],
                            cur_img[i-1:i+2, j-1:j+2],
                            next_img[i-1:i+2, j-1:j+2]
                        ])
                        # center must be strictly largest
                        if v >= cube.max():
                            x = j * (2 ** o_idx)
                            y = i * (2 ** o_idx)
                            size = self.sigma * (2 ** (o_idx + 1))
                            kp = KeyPoint(x, y, _size=size)
                            keypoints.append(kp)
        image_model.keypoints = keypoints
        return keypoints

    @staticmethod
    def generate_gaussian_kernels(sigma, num_intervals):
        """
        Compute sigma differences for Gaussian blurs in an octave.
        """
        num_images = num_intervals + 3
        k = 2 ** (1.0 / num_intervals)
        sigmas = np.zeros(num_images, dtype=np.float32)
        sigmas[0] = sigma
        for idx in range(1, num_images):
            prev = (k ** (idx - 1)) * sigma
            total = k * prev
            sigmas[idx] = np.sqrt(max(total**2 - prev**2, 1e-10))
        return sigmas

    @staticmethod
    def apply_gaussian_to_octaves(pyramid, kernels):
        """
        Blur each octave image with successive Gaussian kernels.
        """
        gaussians = []
        for base in pyramid:
            octave = [base]
            for sig in kernels[1:]:
                octave.append(cv2.GaussianBlur(octave[-1], (0, 0), sigmaX=sig, sigmaY=sig))
            print(f"Octave of Gaussians Shape {octave.shape}")
            gaussians.append(octave)
        print("Gaussian Pyramid Shape:", [octave.shape for octave in gaussians])
        return gaussians

    @staticmethod
    def generate_dog_images(gaussians):
        """
        Subtract blurred images to form DoG images.
        """
        dog_pyramid = []
        for octave in gaussians:
            dog_oct = []
            for a, b in zip(octave, octave[1:]):
                dog_oct.append(b.astype(np.float32) - a.astype(np.float32))
            dog_pyramid.append(dog_oct)
            
        print("DoG Pyramid Shape:", [octave.shape for octave in dog_pyramid])
        return dog_pyramid

    def find_scale_space_extrema(self, gaussians, dogs):
        """
        Detect local extrema in DoG scale-space (SIFT-like blob detection).
        """
        thr = npfloor(0.5 * self.contrast_threshold / self.num_intervals * 255)
        keypoints = []
        for o_idx, dog_oct in enumerate(dogs):
            for s in range(len(dog_oct) - 2):
                f_img, m_img, n_img = dog_oct[s], dog_oct[s+1], dog_oct[s+2]
                h, w = m_img.shape
                for i in range(self.image_border_width, h - self.image_border_width):
                    for j in range(self.image_border_width, w - self.image_border_width):
                        if BlobsDetection.is_pixel_extremum(
                            f_img[i-1:i+2, j-1:j+2],
                            m_img[i-1:i+2, j-1:j+2],
                            n_img[i-1:i+2, j-1:j+2],
                            thr):
                            # for simplicity, no subpixel refinement here
                            x = j * (2 ** o_idx)
                            y = i * (2 ** o_idx)
                            size = self.sigma * (2 ** (o_idx + 1))
                            kp = KeyPoint(x, y, _size=size)
                            keypoints.append(kp)
        return keypoints

    @staticmethod
    def is_pixel_extremum(prev_patch, mid_patch, next_patch, threshold):
        """
        Check if center of mid_patch is an extremum in 3x3x3 neighborhood.
        """
        cube = np.stack((prev_patch, mid_patch, next_patch))
        val = cube[1, 1, 1]
        if abs(val) < threshold:
            return False
        neigh = np.delete(cube.flatten(), 13)
        if val > 0:
            return val > neigh.max()
        else:
            return val < neigh.min()


class DoGDetector:
    def __init__(self, num_intervals, sigma, contrast_threshold=0.04, image_border_width=5, eigenvalue_ratio=10, max_iters=5):
        self.num_intervals = num_intervals
        self.sigma = sigma
        self.contrast_threshold = contrast_threshold
        self.image_border_width = image_border_width
        self.eigenvalue_ratio = eigenvalue_ratio
        self.max_iters = max_iters

    def findScaleSpaceExtrema(self, gaussian_images, dog_images):
        logger.debug('Finding scale-space extrema...')
        threshold = floor(0.5 * self.contrast_threshold / self.num_intervals * 255)
        keypoints = []

        for octave_index, dog_images_in_octave in enumerate(dog_images):
            for image_index, (first, second, third) in enumerate(zip(
                dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):

                for i in range(self.image_border_width, first.shape[0] - self.image_border_width):
                    for j in range(self.image_border_width, first.shape[1] - self.image_border_width):
                        if self.isPixelAnExtremum(first[i-1:i+2, j-1:j+2],
                                                  second[i-1:i+2, j-1:j+2],
                                                  third[i-1:i+2, j-1:j+2],
                                                  threshold):
                            result = self.localizeExtremumViaQuadraticFit(
                                i, j, image_index + 1, octave_index, dog_images_in_octave
                            )
                            if result is not None:
                                keypoint, localized_image_index = result
                                orientations = self.computeKeypointsWithOrientations(
                                    keypoint, octave_index, gaussian_images[octave_index][localized_image_index]
                                )
                                keypoints.extend(orientations)
        return keypoints

    def isPixelAnExtremum(self, first, second, third, threshold):
        cube = np.stack((first, second, third), axis=0)
        center_value = cube[1, 1, 1]

        if abs(center_value) < threshold:
            return False

        neighbors = np.delete(cube.flatten(), 13)
        return center_value > np.max(neighbors) if center_value > 0 else center_value < np.min(neighbors)

    def localizeExtremumViaQuadraticFit(self, i, j, image_index, octave_index, dog_images_in_octave):
        image_shape = dog_images_in_octave[0].shape

        for attempt in range(self.max_iters):
            pixel_cube = self.extract_pixel_cube(dog_images_in_octave, i, j, image_index)
            update, gradient, hessian = self.compute_extremum_update(pixel_cube)

            if all(abs(x) < 0.5 for x in update):
                break

            j += int(round(update[0]))
            i += int(round(update[1]))
            image_index += int(round(update[2]))

            if not self.is_within_bounds(i, j, image_index, image_shape):
                return None

        if attempt >= self.max_iters - 1:
            return None

        if self.passes_contrast_and_edge_tests(gradient, hessian, pixel_cube, update):
            kp = KeyPoint()
            kp.pt = ((j + update[0]) * (2 ** octave_index), (i + update[1]) * (2 ** octave_index))
            kp.octave = octave_index + image_index * (1 << 8) + int(round((update[2] + 0.5) * 255)) * (1 << 16)
            kp.size = self.sigma * (2 ** ((image_index + update[2]) / float32(self.num_intervals))) * (2 ** (octave_index + 1))
            kp.response = abs(pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, update))
            return kp, image_index

        return None

    def extract_pixel_cube(self, dog_images_in_octave, i, j, image_index):
        first = dog_images_in_octave[image_index - 1][i-1:i+2, j-1:j+2]
        second = dog_images_in_octave[image_index][i-1:i+2, j-1:j+2]
        third = dog_images_in_octave[image_index + 1][i-1:i+2, j-1:j+2]
        return np.stack([first, second, third]).astype('float32') / 255.

    def compute_extremum_update(self, pixel_cube):
        gradient = self.computeGradientAtCenterPixel(pixel_cube)
        hessian = self.computeHessianAtCenterPixel(pixel_cube)
        update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        return update, gradient, hessian

    def computeGradientAtCenterPixel(self, pixel_array):
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return array([dx, dy, ds])

    def computeHessianAtCenterPixel(self, pixel_array):
        center = pixel_array[1, 1, 1]
        dxx = pixel_array[1, 1, 2] - 2 * center + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * center + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * center + pixel_array[0, 1, 1]
        dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
        dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
        dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        return array([[dxx, dxy, dxs],
                      [dxy, dyy, dys],
                      [dxs, dys, dss]])

    def is_within_bounds(self, i, j, image_index, image_shape):
        return (
            self.image_border_width <= i < image_shape[0] - self.image_border_width and
            self.image_border_width <= j < image_shape[1] - self.image_border_width and
            1 <= image_index <= self.num_intervals
        )

    def passes_contrast_and_edge_tests(self, gradient, hessian, pixel_cube, update):
        value_at_extremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, update)
        if abs(value_at_extremum) * self.num_intervals < self.contrast_threshold:
            return False
        h_xy = hessian[:2, :2]
        trace = np.trace(h_xy)
        det = np.linalg.det(h_xy)
        if det <= 0:
            return False
        return self.eigenvalue_ratio * (trace ** 2) < ((self.eigenvalue_ratio + 1) ** 2) * det

    def computeKeypointsWithOrientations(self, keypoint, octave_index, gaussian_image):
        # Placeholder: your orientation assignment code should go here
        return [keypoint]

class KeyPoint:
    def __init__(self):
        self.pt = None                # (x, y)
        self.octave = None        # Encodes octave
        self.layer = None
        self.size = None            # Scale
        self.response = None    # Strength of the keypoint
        self.descriptor = None
        self.angle = None
        
    def __str__(self):
        return f"KeyPoint(pt={self.pt}, octave={self.octave}, size={self.size}, response={self.response}, angle={self.angle})"

class UnifiedBlobDetector:
    def __init__(self,
                 sigma: float = 1.6,
                 num_intervals: int = 3,
                 contrast_threshold: float = 0.04,
                 image_border_width: int = 5,
                 eigenvalue_ratio: float = 10,
                 max_interp_steps: int = 5,
                 max_iters: int = 5):
        """
        Unified class for blob detection using DoG and DoH methods.
        """
        self.sigma = sigma
        self.num_intervals = num_intervals
        self.contrast_threshold = contrast_threshold
        self.image_border_width = image_border_width
        self.eigenvalue_ratio = eigenvalue_ratio
        self.max_interp_steps = max_interp_steps
        self.max_iters = max_iters

    def dog_blob_detector(self, image_model):
        """
        Detect blobs via Difference of Gaussians (DoG).
        """
        if image_model.pyramid is None:
            image_model.pyramid = image_model.build_image_pyramid(image_model.gray, image_model.octaves)

        kernels = self.generate_gaussian_kernels(self.sigma, self.num_intervals)
        
        print("Smoothing Kernels:", kernels)

        
        gaussians = self.apply_gaussian_to_octaves(image_model.pyramid, kernels)
        
        dog_imgs = self.generate_dog_images(gaussians)
        keypoints = self.find_scale_space_extrema(gaussians, dog_imgs)
        
        self.add_orientations_to_keypoints(keypoints, image_model.gray)

        image_model.keypoints = keypoints
        return keypoints
    
    def doh_blob_detector(self, image_model):
        """
        Detect blobs via Determinant of Hessian (DoH).
        """
        if image_model.pyramid is None:
            image_model.pyramid = image_model.build_image_pyramid(image_model.gray, image_model.octaves)

        kernels = self.generate_gaussian_kernels(self.sigma, self.num_intervals)
        gaussians = self.apply_gaussian_to_octaves(image_model.pyramid, kernels)

        doh_pyramid = []
        for octave in gaussians:
            octave_doh = []
            for img in octave:
                hxx = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=3)
                hyy = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=3)
                hxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
                det_h = (hxx * hyy) - (hxy ** 2)
                octave_doh.append(det_h)
            doh_pyramid.append(octave_doh)

        keypoints = []
        thr = self.contrast_threshold
        for o_idx, octave in enumerate(doh_pyramid):
            for s in range(1, len(octave) - 1):
                prev_img, cur_img, next_img = octave[s-1], octave[s], octave[s+1]
                h, w = cur_img.shape
                for i in range(self.image_border_width, h - self.image_border_width):
                    for j in range(self.image_border_width, w - self.image_border_width):
                        v = cur_img[i, j]
                        if v < thr:
                            continue
                        cube = np.stack([
                            prev_img[i-1:i+2, j-1:j+2],
                            cur_img[i-1:i+2, j-1:j+2],
                            next_img[i-1:i+2, j-1:j+2]
                        ])
                        if v >= cube.max():
                            x = j * (2 ** o_idx)
                            y = i * (2 ** o_idx)
                            size = self.sigma * (2 ** (o_idx + 1))
                            kp = KeyPoint(x, y, _size=size)
                            keypoints.append(kp)
        self.add_orientations_to_keypoints(keypoints, image_model.gray)
        image_model.keypoints = keypoints
        return keypoints


    def add_orientations_to_keypoints(self, keypoints, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8):
        """
        Add orientations to keypoints by analyzing gradient distribution in their neighborhood.
        This creates multiple keypoints when multiple dominant orientations exist.
        
        Args:
            keypoints: List of keypoints (will be modified in place)
            gaussian_image: Blurred image at the keypoint's scale level
            radius_factor: Determines neighborhood size relative to keypoint scale
            num_bins: Number of orientation histogram bins
            peak_ratio: Minimum peak magnitude relative to maximum peak
        
        Returns:
            List of keypoints with orientations (may be longer than input list)
        """
        keypoints_with_orientations = []
        
        for keypoint in keypoints:
            # Calculate neighborhood parameters based on keypoint scale
            scale = 1.5 * keypoint.size / (2 ** (keypoint.octave + 1))
            radius = int(round(radius_factor * scale))
            weight_factor = -0.5 / (scale ** 2)
            
            # Initialize orientation histogram
            raw_histogram = np.zeros(num_bins)
            smooth_histogram = np.zeros(num_bins)
            
            # Get keypoint coordinates in this octave's image
            x = int(round(keypoint.pt[0] / (2 ** keypoint.octave)))
            y = int(round(keypoint.pt[1] / (2 ** keypoint.octave)))
            
            # Analyze gradient distribution in the neighborhood
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if (0 < y+i < gaussian_image.shape[0]-1 and 
                        0 < x+j < gaussian_image.shape[1]-1):
                        # Compute gradients
                        dx = gaussian_image[y+i, x+j+1] - gaussian_image[y+i, x+j-1]
                        dy = gaussian_image[y+i-1, x+j] - gaussian_image[y+i+1, x+j]
                        
                        # Compute magnitude and orientation
                        magnitude = np.sqrt(dx*dx + dy*dy)
                        orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        
                        # Apply Gaussian weighting and add to histogram
                        weight = np.exp(weight_factor * (i**2 + j**2))
                        bin_idx = int(round(orientation * num_bins / 360))
                        raw_histogram[bin_idx % num_bins] += weight * magnitude
            
            # Smooth the histogram
            for n in range(num_bins):
                smooth_histogram[n] = (6 * raw_histogram[n] + 
                                    4 * (raw_histogram[n-1] + raw_histogram[(n+1)%num_bins]) + 
                                    raw_histogram[n-2] + raw_histogram[(n+2)%num_bins]) / 16.0
            
            # Find peaks in the histogram
            orientation_max = np.max(smooth_histogram)
            peak_indices = np.where((smooth_histogram > np.roll(smooth_histogram, 1)) & 
                                (smooth_histogram > np.roll(smooth_histogram, -1)))[0]
            
            # Create keypoints for each significant peak
            for peak_idx in peak_indices:
                peak_value = smooth_histogram[peak_idx]
                if peak_value >= peak_ratio * orientation_max:
                    # Quadratic interpolation for peak position
                    left = smooth_histogram[peak_idx-1]
                    right = smooth_histogram[(peak_idx+1)%num_bins]
                    interpolated_peak = (peak_idx + 0.5 * (left - right) / 
                                        (left - 2*peak_value + right)) % num_bins
                    
                    # Convert to angle in degrees
                    orientation = 360. - interpolated_peak * 360./num_bins
                    if abs(orientation - 360.) < 1e-6:
                        orientation = 0
                    
                    # Create new keypoint with this orientation
                    keypoint.angle = orientation
                    keypoints_with_orientations.append(keypoint)
                    print(keypoint.__str__())
        
        return keypoints_with_orientations
        
    # def add_orientations_to_keypoints(self, keypoints, image):
    #     """
    #     Add orientations to keypoints by calculating the gradient direction.
    #     """
    #     for kp in keypoints:
    #         x, y = int(kp.pt[0]), int(kp.pt[1])
    #         # Define the region around the keypoint to compute the gradient
    #         window_size = 16  # Example size for the window
    #         half_window = window_size // 2
    #         x1, y1 = max(0, x - half_window), max(0, y - half_window)
    #         x2, y2 = min(image.shape[1], x + half_window), min(image.shape[0], y + half_window)

    #         # Extract the region around the keypoint
    #         region = image[y1:y2, x1:x2]
            
    #         # Compute the gradients in X and Y direction
    #         grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    #         grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)

    #         # Compute the gradient magnitude and orientation
    #         magnitude, angle = cv2.cartToPolar(grad_x, grad_y)
            
    #         # Compute the dominant orientation (the orientation with the largest magnitude)
    #         hist = np.histogram(angle, bins=36, range=(0, 2 * np.pi), weights=magnitude)
    #         max_bin = np.argmax(hist[0])
    #         orientation = (max_bin + 0.5) * (2 * np.pi / 36)  # Dominant orientation

    #         # Assign orientation to the keypoint
    #         kp.angle = orientation
    # def find_scale_space_extrema(self, gaussians, dogs):
    #     """
    #     Detect local extrema in DoG scale-space.
    #     """
    #     thr = np.floor(0.5 * self.contrast_threshold / self.num_intervals * 255)
    #     keypoints = []
    #     for o_idx, dog_oct in enumerate(dogs):
    #         for s in range(len(dog_oct) - 2):
    #             f_img, m_img, n_img = dog_oct[s], dog_oct[s+1], dog_oct[s+2]
    #             h, w = m_img.shape
    #             for i in range(self.image_border_width, h - self.image_border_width):
    #                 for j in range(self.image_border_width, w - self.image_border_width):
    #                     if self.is_pixel_extremum(
    #                         f_img[i-1:i+2, j-1:j+2],
    #                         m_img[i-1:i+2, j-1:j+2],
    #                         n_img[i-1:i+2, j-1:j+2],
    #                         thr):
    #                         print(f"pixel of ({i}, {j}) and octave {s} is an extremum")
    #                         result = self.localize_extremum_via_quadratic_fit(
    #                             i, j, s + 1, o_idx, dog_oct
    #                         )
    #                         if result is not None:
    #                             keypoint, localized_image_index = result
    #                             keypoints.append(keypoint)
    #     return keypoints

    # withour quadratic estimation
    def find_scale_space_extrema(self, gaussians, dogs):
        """
        Detect local extrema in DoG scale-space and create KeyPoint objects.
        """
        thr = np.floor(0.5 * self.contrast_threshold / self.num_intervals * 255)
        keypoints = []

        for o_idx, dog_oct in enumerate(dogs):
            for s in range(len(dog_oct) - 2):
                f_img, m_img, n_img = dog_oct[s], dog_oct[s + 1], dog_oct[s + 2]
                h, w = m_img.shape

                for i in range(self.image_border_width, h - self.image_border_width):
                    for j in range(self.image_border_width, w - self.image_border_width):
                        if self.is_pixel_extremum(
                            f_img[i - 1:i + 2, j - 1:j + 2],
                            m_img[i - 1:i + 2, j - 1:j + 2],
                            n_img[i - 1:i + 2, j - 1:j + 2],
                            thr
                        ):
                            keypoint = KeyPoint()
                            # Keypoint location
                            keypoint.pt = (j, i)
                            # Store octave and layer
                            keypoint.octave = o_idx
                            keypoint.layer = s + 1
                            self.k = 2 ** (1.0 / self.num_intervals)

                            # Size from scale (sigma * 2 for full Gaussian window)
                            sigma = self.sigma * (self.k ** (s + 1))
                            keypoint.size = sigma * 2
                            # DoG response
                            keypoint.response = abs(m_img[i, j]) 
                            print(keypoint.__str__())
                            keypoints.append(keypoint)

        return keypoints


    def localize_extremum_via_quadratic_fit(self, i, j, image_index, octave_index, dog_images_in_octave):
        """
        Refine the location of an extremum using quadratic fitting.
        """
        image_shape = dog_images_in_octave[0].shape

        for attempt in range(self.max_iters):
            pixel_cube = self.extract_pixel_cube(dog_images_in_octave, i, j, image_index)
            update, gradient, hessian = self.compute_extremum_update(pixel_cube)

            if all(abs(x) < 0.5 for x in update):
                break

            j += int(round(update[0]))
            i += int(round(update[1]))
            image_index += int(round(update[2]))

            if not self.is_within_bounds(i, j, image_index, image_shape):
                return None

        if attempt >= self.max_iters - 1:
            return None

        if self.passes_contrast_and_edge_tests(gradient, hessian, pixel_cube, update):
            kp = KeyPoint()
            kp.pt = ((j + update[0]) * (2 ** octave_index), (i + update[1]) * (2 ** octave_index))
            kp.octave = octave_index + image_index * (1 << 8) + int(round((update[2] + 0.5) * 255)) * (1 << 16)
            kp.size = self.sigma * (2 ** ((image_index + update[2]) / np.float32(self.num_intervals))) * (2 ** (octave_index + 1))
            kp.response = abs(pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, update))
            print(kp.__str__())
            return kp, image_index

        return None

    @staticmethod
    def is_pixel_extremum(prev_patch, mid_patch, next_patch, threshold):
        """
        Check if center of mid_patch is an extremum in 3x3x3 neighborhood.
        """
        cube = np.stack((prev_patch, mid_patch, next_patch))
        val = cube[1, 1, 1]
        if abs(val) < threshold:
            return False
        neigh = np.delete(cube.flatten(), 13)
        if val > 0:
            return val > neigh.max()
        else:
            return val < neigh.min()

    @staticmethod
    def generate_gaussian_kernels(sigma, num_intervals):
        """
        Compute sigma differences for Gaussian blurs in an octave.
        """
        num_images = num_intervals + 3
        k = 2 ** (1.0 / num_intervals)
        sigmas = np.zeros(num_images, dtype=np.float32)
        sigmas[0] = sigma
        for idx in range(1, num_images):
            prev = (k ** (idx - 1)) * sigma
            total = k * prev
            sigmas[idx] = np.sqrt(max(total**2 - prev**2, 1e-10))
        return sigmas

    @staticmethod
    def apply_gaussian_to_octaves(pyramid, kernels):
        """
        Blur each octave image with successive Gaussian kernels.
        """
        gaussians = []
        for base in pyramid:
            octave = [base]
            for sig in kernels[1:]:
                octave.append(cv2.GaussianBlur(octave[-1], (0, 0), sigmaX=sig, sigmaY=sig))
            gaussians.append(octave)
        print("Gaussian Pyramid Shape:", [len(octave) for octave in gaussians])
        return gaussians

    @staticmethod
    def generate_dog_images(gaussians):
        """
        Subtract blurred images to form DoG images.
        """
        dog_pyramid = []
        for octave in gaussians:
            dog_oct = []
            for a, b in zip(octave, octave[1:]):
                dog_oct.append(b.astype(np.float32) - a.astype(np.float32))
            dog_pyramid.append(dog_oct)
        print("DoG Pyramid Shape:", [len(octave) for octave in dog_pyramid])
        return dog_pyramid

    def extract_pixel_cube(self, dog_images_in_octave, i, j, image_index):
        """
        Extract a 3x3x3 cube of pixels centered at (i, j, image_index).
        """
        first = dog_images_in_octave[image_index - 1][i-1:i+2, j-1:j+2]
        second = dog_images_in_octave[image_index][i-1:i+2, j-1:j+2]
        third = dog_images_in_octave[image_index + 1][i-1:i+2, j-1:j+2]
        return np.stack([first, second, third]).astype('float32') / 255.

    def compute_extremum_update(self, pixel_cube):
        """
        Compute the gradient and Hessian to refine the extremum location.
        """
        gradient = self.compute_gradient_at_center_pixel(pixel_cube)
        hessian = self.compute_hessian_at_center_pixel(pixel_cube)
        update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        return update, gradient, hessian

    @staticmethod
    def compute_gradient_at_center_pixel(pixel_array):
        """
        Compute the gradient at the center pixel of a 3x3x3 cube.
        """
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return np.array([dx, dy, ds])

    @staticmethod
    def compute_hessian_at_center_pixel(pixel_array):
        """
        Compute the Hessian matrix at the center pixel of a 3x3x3 cube.
        """
        center = pixel_array[1, 1, 1]
        dxx = pixel_array[1, 1, 2] - 2 * center + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * center + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * center + pixel_array[0, 1, 1]
        dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
        dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
        dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        return np.array([[dxx, dxy, dxs],
                         [dxy, dyy, dys],
                         [dxs, dys, dss]])

    def is_within_bounds(self, i, j, image_index, image_shape):
        """
        Check if the pixel is within valid bounds.
        """
        return (
            self.image_border_width <= i < image_shape[0] - self.image_border_width and
            self.image_border_width <= j < image_shape[1] - self.image_border_width and
            1 <= image_index <= self.num_intervals
        )

    def passes_contrast_and_edge_tests(self, gradient, hessian, pixel_cube, update):
        """
        Check if the extremum passes contrast and edge response tests.
        """
        value_at_extremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, update)
        if abs(value_at_extremum) * self.num_intervals < self.contrast_threshold:
            return False
        h_xy = hessian[:2, :2]
        trace = np.trace(h_xy)
        det = np.linalg.det(h_xy)
        if det <= 0:
            return False
        return self.eigenvalue_ratio * (trace ** 2) < ((self.eigenvalue_ratio + 1) ** 2) * det
from functools import cmp_to_key

from functools import cmp_to_key

class DescriptorGenerator:
    def __init__(self, window_size=16, num_bins=8, scale_factor=1.0):
        self.window_size = window_size  # Size of the patch around each keypoint (16x16)
        self.num_bins = num_bins  # Number of bins in the orientation histogram (typically 8)
        self.scale_factor = scale_factor  # Scaling factor for descriptor (typically 1.0)
    
    def generate_sift_descriptors(self, keypoints, image):
        """
        Generate SIFT-like descriptors for the given keypoints and image.
        Includes preprocessing steps for keypoint deduplication and border handling.
        """
        # Preprocess keypoints - remove duplicates
        keypoints = self._preprocess_keypoints(keypoints)
        
        descriptors = []
        valid_keypoints = []
        
        for kp in keypoints:
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            
            # Skip keypoints too close to the image border
            if not self._is_keypoint_valid(image, x, y):
                continue
                
            window = self._extract_window(image, x, y)
            
            # Skip if window extraction failed (empty window)
            if window is None or window.size == 0:
                continue

            try:
                # Calculate gradients within the window
                grad_magnitude, grad_orientation = self._compute_gradients(window)

                # Compute histogram of gradients in 4x4 cells
                descriptor = self._compute_histogram(grad_magnitude, grad_orientation)
                
                # Normalize the descriptor
                normalized_descriptor = self._normalize_descriptor(descriptor)

                kp.descriptor = normalized_descriptor  # Assign the descriptor to the keypoint
                descriptors.append(normalized_descriptor)
                valid_keypoints.append(kp)
            except cv2.error:
                continue
        
        return descriptors, valid_keypoints

    def _is_keypoint_valid(self, image, x, y):
        """Check if keypoint is far enough from image borders"""
        half_window = self.window_size // 2
        return (half_window <= x < image.shape[1] - half_window and 
                half_window <= y < image.shape[0] - half_window)

    def _extract_window(self, image, x, y):
        """
        Extract a window around the keypoint (x, y).
        Returns None if window would go out of bounds.
        """
        half_window = self.window_size // 2
        try:
            window = image[y - half_window:y + half_window, 
                          x - half_window:x + half_window]
            # Ensure we got exactly the window size we wanted
            if window.shape[0] == self.window_size and window.shape[1] == self.window_size:
                return window
            return None
        except:
            return None

    def _compute_gradients(self, window):
        """
        Compute the gradient magnitude and orientation for each pixel in the window.
        """
        # Convert to float32 if needed
        if window.dtype != np.float32:
            window = window.astype(np.float32)
            
        grad_x = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)
        
        grad_magnitude, grad_orientation = cv2.cartToPolar(grad_x, grad_y)
        return grad_magnitude, grad_orientation

    # [Rest of the methods remain unchanged...]
    def _preprocess_keypoints(self, keypoints):
        """Preprocess keypoints by removing duplicates and sorting"""
        if not keypoints:
            return keypoints
            
        # Remove duplicate keypoints
        print("Before removing duplicates:", len(keypoints))
        unique_keypoints = self._remove_duplicate_keypoints(keypoints)
        print("After removing duplicates:", len(unique_keypoints))
        return unique_keypoints

    def _compare_keypoints(self, keypoint1, keypoint2):
        """Comparison function for sorting keypoints"""
        print(f"comparting {keypoint1.__str__()} and {keypoint2.__str__()}")
        if keypoint1.pt[0] != keypoint2.pt[0]:
            return keypoint1.pt[0] - keypoint2.pt[0]
        if keypoint1.pt[1] != keypoint2.pt[1]:
            return keypoint1.pt[1] - keypoint2.pt[1]
        if keypoint1.size != keypoint2.size:
            return keypoint2.size - keypoint1.size
        if keypoint1.angle != keypoint2.angle:
            return keypoint1.angle - keypoint2.angle
        if keypoint1.response != keypoint2.response:
            return keypoint2.response - keypoint1.response
        if keypoint1.octave != keypoint2.octave:
            return keypoint2.octave - keypoint1.octave
        if keypoint1.layer != keypoint2.layer:
            return keypoint2.layer - keypoint1.layer
        return 0  # All fields equal

    def _remove_duplicate_keypoints(self, keypoints):
        """Sort keypoints and remove duplicate keypoints"""
        if len(keypoints) < 2:
            return keypoints

        keypoints.sort(key=cmp_to_key(self._compare_keypoints))
        unique_keypoints = [keypoints[0]]

        for next_keypoint in keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if (last_unique_keypoint.pt[0] != next_keypoint.pt[0] or
                last_unique_keypoint.pt[1] != next_keypoint.pt[1] or
                last_unique_keypoint.size != next_keypoint.size or
                last_unique_keypoint.angle != next_keypoint.angle or
                last_unique_keypoint.octave != next_keypoint.octave or
                last_unique_keypoint.layer != next_keypoint.layer):
                unique_keypoints.append(next_keypoint)
                
        return unique_keypoints

    def _compute_histogram(self, grad_magnitude, grad_orientation):
        """
        Compute a histogram of gradient orientations in 4x4 cells.
        Each cell contributes 8 bins in the histogram.
        """
        cell_size = self.window_size // 4  # Dividing 16x16 into 4x4 cells
        descriptor = []

        for i in range(4):
            for j in range(4):
                cell_magnitude = grad_magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                cell_orientation = grad_orientation[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]

                # Compute the histogram of orientations in this cell
                hist, _ = np.histogram(cell_orientation, bins=self.num_bins, range=(0, 2*np.pi), weights=cell_magnitude)

                # Append the histogram to the descriptor
                descriptor.extend(hist)

        return np.array(descriptor)

    def _normalize_descriptor(self, descriptor):
        """
        Normalize the descriptor to unit length and apply a threshold (if necessary).
        """
        descriptor /= np.linalg.norm(descriptor)  # L2 normalization

        # Optional: Apply thresholding to avoid overflow and improve stability
        descriptor = np.clip(descriptor, 0, 0.2)  # Threshold at 0.2
        descriptor /= np.linalg.norm(descriptor)  # Re-normalize
        
        return descriptor


def keypoint_matching(desc1, kp1, desc2, kp2, threshold=0.9):
    """
    Manually match keypoints between two images using descriptor distances.
    
    Args:
        desc1: List of descriptors from first image
        kp1: List of keypoints from first image
        desc2: List of descriptors from second image
        kp2: List of keypoints from second image
        threshold: Maximum distance for a match (0-1, normalized descriptor space)
        
    Returns:
        match_percentage: Percentage of matched keypoints (0-100)
        matches: List of tuples (index1, index2, distance) for each match
    """
    # Convert descriptors to numpy arrays if they aren't already
    desc1 = np.array(desc1, dtype=np.float32)
    desc2 = np.array(desc2, dtype=np.float32)
    
    matches = []
    

    # For each descriptor in the first image
    for i, d1 in enumerate(desc1):
        min_dist = float('inf')
        best_match = None
        
        # Compare against all descriptors in second image
        for j, d2 in enumerate(desc2):
            # Calculate Euclidean distance
            dist = np.linalg.norm(d1 - d2)
            
            # Track the best match
            if dist < min_dist:
                min_dist = dist
                best_match = j
        
        # If the best match is below threshold, add it
        if min_dist < threshold:
            matches.append((i, best_match, min_dist))
            
        print(f"Match {i}: Keypoint {i} in image 1 matched with keypoint {best_match} in image 2 with distance {min_dist:.4f}")
    
    # Calculate matching percentage
    total_keypoints = min(len(kp1), len(kp2))
    print(f'Total keypoints in image 1: {len(kp1)}, image 2: {len(kp2)}')
    print(f'Total matches found: {len(matches)}')
    match_percentage = (len(matches) / total_keypoints * 100) if total_keypoints > 0 else 0
    
    return match_percentage, matches

def visualize_manual_matches(image1, kp1, image2, kp2, matches):
    """
    Visualize matches between two images.
    
    Args:
        image1: First image
        kp1: Keypoints from first image
        image2: Second image
        kp2: Keypoints from second image
        matches: List of matches from manual_keypoint_matching
    """
    # Create a new output image that concatenates the two input images
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = image1
    vis[:h2, w1:w1+w2] = image2
    
    # Draw lines between matches
    for idx1, idx2, _ in matches:
        # Get the keypoints
        kp_first = kp1[idx1]
        kp_second = kp2[idx2]
        
        # Get the points
        x1, y1 = int(kp_first.pt[0]), int(kp_first.pt[1])
        x2, y2 = int(kp_second.pt[0]) + w1, int(kp_second.pt[1])
        
        # Draw the match
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(vis, (x1, y1), (x2, y2), color, 1)
        cv2.circle(vis, (x1, y1), 3, color, -1)
        cv2.circle(vis, (x2, y2), 3, color, -1)
    
    return vis

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load grayscale image
    img = cv2.imread('book1.jpg', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "Image not found!"

    # Initialize image model and corner detector
    image_model = ImageModel()
    image_model.set_image(img)
    corner_detector = CornerDetection()
    # Harris corner detection
    harris_corners = corner_detector.corner_detection_harris(image_model)

# Shi-Tomasi corner detection
    shi_corners = corner_detector.corner_detection_shi(image_model)
    
    def visualize_corners(image, corners, title='Corners'):
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for y, x in corners:
            cv2.circle(img_color, (x, y), 3, (0, 0, 255), -1)
        
        plt.figure(figsize=(8, 6))
        plt.title(title)
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

