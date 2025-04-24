from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging


logger = logging.getLogger(__name__)
float_tolerance = 1e-7

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

class GaussianScaleSpace:
    
    def __init__(self):
        """
        Initialize the GaussianScaleSpace object with default parameters.
        """
        self.n_oct = 4 # Number of octaves
        self.n_spo = 3 # Number of scales per octave
        self.eita_min = 0.5 # the sampling distance of the first image of the scale space
        self.sigma_min = 0.8 # Initial sigma value for Gaussian blur
        self.sigma_in = 0.5 # The assumed blur level for the input image
        self.gaussian_pyramid = []
        
    def set_parameters(self, n_oct=4, n_spo=5, eita_min=0.5, sigma_min=1.0, sigma_in=1.6):
        """
        Set the parameters for the Gaussian scale space.

        Parameters:
            n_oct (int): Number of octaves.
            n_spo (int): Number of scales per octave.
            eita_min (float): Minimum sampling distance.
            sigma_min (float): Initial sigma value for Gaussian blur.
            sigma_in (float): Assumed blur level for the input image.
        """
        self.n_oct = n_oct
        self.n_spo = n_spo
        self.eita_min = eita_min
        self.sigma_min = sigma_min
        self.sigma_in = sigma_in
        self.seed_image = None
        self.o = None
        self.pyramid = None
        self.gaussians = None  
        self.dogs= None 
        
    
    def build_digital_scale_space(self, image):
        '''
        steps of the algorithm:
        - compute the first octave
        - compute the seed image
        - compute the images in the first octave
        - compute the rest of the octaves
        '''
        self.set_parameters()
        self.n_oct = int(round(log(min(image.shape)) / log(2) - 3)) # Number of octaves stops at 8 pixel
        self.eita_list = [self.eita_min*2**(o-1) for o in range(1,self.n_oct+1)]
        self.sigma_list = [
            (self.sigma_min / self.eita_min) * sqrt(
                max(2 ** (2 * (s / self.n_spo)) - 2 ** (2 * ((s - 1) / self.n_spo)), 0.01)
            )
            for s in range(1, self.n_spo + 3)
        ]   
        logger.debug(f'Generating sigma list{self.sigma_list}...')
            # Step 1: Compute the first octave
        self.seed_image = self.compute_seed_image(image, self.sigma_in, self.sigma_min)
        self.pyramid = self.generate_resized_pyramid()
        self.apply_gaussian_to_octaves()
        self.compute_dog_pyramid()
        logger.debug(f'Generating pyramid with {len(self.pyramid)} images...')
    
    def generate_resized_pyramid(self):
        resized_images = []

        for eita in self.eita_list:
            scale = self.eita_min/ eita  # smaller scale means downsampling
            print(f"Resizing image with scale factor: {scale}")
            
            # Calculate the new dimensions based on the scale factor
            height, width = self.seed_image.shape
            new_height = int(height * scale)
            new_width = int(width * scale)
            
            # Resize the image using the computed new dimensions
            resized = cv2.resize(self.seed_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            resized_images.append(resized)

        return resized_images

    def compute_seed_image(self, image, sigma_in, sigma_min):
        """
        Compute the seed image for the Gaussian scale space.

        Parameters:
            image (numpy.ndarray): Input image.
            
        """
        logger.debug('Generating base image...')
        image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
        sigma_diff = (1/self.eita_min)* sqrt(max((self.sigma_min ** 2) - ((2 * self.sigma_in) ** 2), 0.01))
        return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur
    
    def apply_gaussian_to_octaves(self):
        """
        Blur each octave image with successive Gaussian kernels.
        The sigma values for each octave are scaled by 2^octave_index.
        """
        self.gaussians = []

        for o, base in enumerate(self.pyramid):  # o: octave index
            sigma_list = [(2 ** o) * s for s in self.sigma_list]  # scale sigmas for this octave

            octave = [base]
            print(f"Octave {o} - sigma scale: 2^{o}")

            for sig in sigma_list[1:]:  # skip first since base is already at sigma_list[0]
                blurred = cv2.GaussianBlur(octave[-1], (0, 0), sigmaX=sig, sigmaY=sig)
                octave.append(blurred)
                print(f"  Applying Gaussian with sigma: {sig}")

            self.gaussians.append(octave)

        print("Gaussian Pyramid Shape:", [len(octave) for octave in self.gaussians])
    def compute_dog_pyramid(self):
        """
        Compute the Difference of Gaussians (DoG) for each octave in the Gaussian pyramid.
        The result is stored in self.dogs, where each element is a list of DoG images for that octave.
        """
        self.dogs = []  # Clear existing DoG pyramid if any

        for octave_idx, gaussian_images in enumerate(self.gaussians):
            dogs_in_octave = []
            for i in range(1, len(gaussian_images)):
                dog = cv2.subtract(gaussian_images[i], gaussian_images[i - 1])
                dogs_in_octave.append(dog)
            self.dogs.append(dogs_in_octave)

        print("DoG Pyramid Shape:", [len(dogs) for dogs in self.dogs])



class keypointsExtractor:
    
    def __init__(self, scale_space):
        self.scale_space = scale_space
        self.dog_scale_space = scale_space.dogs
        self.n_spo = len(self.dog_scale_space[0]) - 2
        self.stability_threshold = 0.8* 0.015 * (2**(1/self.n_spo)) / (2**(1/3))
        self.keypoints = []


    def extract_local_extrema(self, threshold=0.03):
        """
        Detect local extrema in the DoG pyramid using 3x3x3 neighborhood.

        Returns:
            keypoints_per_octave (list): List of keypoints per octave. Each keypoint is (s, y, x).
        """
        print(f'Number of DoG {len(self.dog_scale_space)}')
        print(f'Number of scales {len(self.dog_scale_space[0])}')   
        keypoints_per_octave = []
        
        for o, dog_images_in_octave in enumerate(self.dog_scale_space):
            octave_keypoints = []

            num_scales = len(dog_images_in_octave)
            height, width = dog_images_in_octave[0].shape

            for s in range(1, num_scales - 1):  # Skip first and last scale
                # print(f"Keypoint found at octave {o}, scale {s})")

                for i in range(1, height - 1): #Skipping the borders
                    for j in range(1, width - 1):
                        cube = self.extract_pixel_cube(dog_images_in_octave, i, j, s)
                        
                        if self.is_pixel_extremum(cube[0], cube[1], cube[2], threshold):
                            # print(f"Keypoint value: {cube[1][1][1]}") 
                            refined, _ = self.localize_keypoint(i, j, s, self.dog_scale_space[o], self.n_spo, contrast_threshold=threshold, eigenvalue_ratio=10, octave_index=o, sigma=self.scale_space.sigma_list[s], image_border_width=5)
                            if refined is not None:
                                    self.keypoints.append(refined)
       
    
    def extract_pixel_cube(self, dog_images_in_octave, i, j, image_index):
        """
        Extract a 3x3x3 cube of pixels centered at (i, j, image_index).
        """
        # print(image_index)
        first = dog_images_in_octave[image_index - 1][i-1:i+2, j-1:j+2]
        second = dog_images_in_octave[image_index][i-1:i+2, j-1:j+2]
        third = dog_images_in_octave[image_index + 1][i-1:i+2, j-1:j+2]
        return np.stack([first, second, third]).astype('float32') / 255.
    
    def is_pixel_extremum(self,prev_patch, mid_patch, next_patch, threshold):
        """
        Check if center of mid_patch is an extremum in 3x3x3 neighborhood.
        """
        
        cube = np.stack((prev_patch, mid_patch, next_patch))
        val = cube[1, 1, 1]
        if abs(val) < threshold:
            return False
        neigh = np.delete(cube.flatten(), 13)
        if val > 0.03:
            return val > neigh.max()
        else:
            return val < neigh.min()
    def localize_keypoint(self, i, j, image_index, dog_images_in_octave, num_intervals, 
                     contrast_threshold=0.03, eigenvalue_ratio=10, octave_index=0, 
                     sigma=1.6, image_border_width=5):
        """
        Refine keypoint using quadratic fit in scale-space with additional checks.

        Parameters:
            i, j, image_index: initial discrete extremum location
            dog_images_in_octave: list of DoG images for the current octave
            num_intervals: number of scale intervals
            contrast_threshold: minimum DoG value (|ω|) to accept keypoint
            eigenvalue_ratio: for edge rejection threshold
            octave_index: current octave index
            sigma: base scale
            image_border_width: border to avoid when checking bounds

        Returns:
            keypoint: OpenCV-style KeyPoint object if successful
            refined_scale_index: refined scale index
            or (None, None) if refinement fails
        """
        num_attempts_until_convergence = 5
        image_shape = dog_images_in_octave[0].shape
        extremum_is_outside_image = False

        for attempt_index in range(num_attempts_until_convergence):
            try:
                pixel_cube = self.extract_pixel_cube(dog_images_in_octave, i, j, image_index)
            except IndexError:
                return None, None  # Out of bounds

            gradient = self.compute_gradient(pixel_cube)
            hessian = self.compute_hessian(pixel_cube)

            try:
                extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None, None  # Singular matrix

            # Check if we've converged (all updates < 0.5)
            if (np.abs(extremum_update) < 0.5).all():
                break

            # Update position
            j += int(round(extremum_update[0]))
            i += int(round(extremum_update[1]))
            image_index += int(round(extremum_update[2]))

            # Check bounds
            if (i < image_border_width or i >= image_shape[0] - image_border_width or
                j < image_border_width or j >= image_shape[1] - image_border_width or
                image_index < 1 or image_index > num_intervals):
                extremum_is_outside_image = True
                break

        if extremum_is_outside_image:
            return None, None
        if attempt_index >= num_attempts_until_convergence - 1:
            return None, None

        # Compute final contrast at refined location
        function_value_at_updated_extremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)

        # Contrast threshold check (scaled by num_intervals)
        if abs(function_value_at_updated_extremum) * num_intervals < contrast_threshold:
            return None, None

        # Edge rejection check using Hessian
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)

        if xy_hessian_det <= 0:
            return None, None
        if eigenvalue_ratio * (xy_hessian_trace ** 2) >= ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            return None, None

        keypoint = KeyPoint()
        keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), 
                    (i + extremum_update[1]) * (2 ** octave_index))
        keypoint.octave = octave_index 
        keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float(num_intervals))) * (2 ** (octave_index + 1))
        keypoint.response = abs(function_value_at_updated_extremum)
        keypoint.layer = image_index
        return keypoint, image_index
    def refine_extremum_with_quadratic_fit(self, pixel_cube):
        """
        Apply quadratic Taylor expansion fit to refine extremum position within a DoG pixel cube.

        Returns:
            offset (np.array shape [3]): subpixel offset [dx, dy, ds]
            contrast (float): refined DoG value at offset
            is_valid (bool): if the keypoint is within valid offset range
        """
        try:
            grad = self.compute_gradient(pixel_cube)  # ∇D
            hessian = self.compute_hessian(pixel_cube)  # H(D)

            hessian_inv = np.linalg.inv(hessian)
            offset = -hessian_inv @ grad  # α* = -H⁻¹ ∇D

            # Compute contrast using Taylor expansion: D + 1/2 αᵀ ∇D
            contrast = pixel_cube[1, 1, 1] + 0.5 * grad @ offset

            is_valid = np.all(np.abs(offset) <= 0.5)

            return offset, contrast, is_valid
        except np.linalg.LinAlgError:
            return None, None, False

    
    def refine_keypoint(self, dog_images_in_octave, i, j, s, max_iter=5):
        for iter_count in range(max_iter):
            pixel_cube = self.extract_pixel_cube(dog_images_in_octave, i, j, s)
            gradient = self.compute_gradient(pixel_cube)
            hessian = self.compute_hessian(pixel_cube)

            try:
                hessian_inv = np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                return None  # Singular Hessian, skip this keypoint

            offset = -hessian_inv @ gradient

            if np.all(np.abs(offset) < 0.5):
                # Final refined subpixel offset
                return (i + offset[1], j + offset[0], s + offset[2])

            # Move to new location
            j += int(round(offset[0]))
            i += int(round(offset[1]))
            s += int(round(offset[2]))

            if not (1 <= s < len(dog_images) - 1 and 1 <= i < dog_images[0].shape[0] - 1 and 1 <= j < dog_images[0].shape[1] - 1):
                return None  # Out of bounds
        return None  # No convergence

                
            

        
        
    
    def compute_gradient(self, pixel_cube):
        """
        Compute the gradient of the pixel cube.
        """
        # Gradient in x, y, and scale directions
        grad_x = (pixel_cube[1, 1, 2] - pixel_cube[1, 1, 0]) / 2.0
        grad_y = (pixel_cube[1, 2, 1] - pixel_cube[1, 0, 1]) / 2.0
        grad_s = (pixel_cube[2, 1, 1] - pixel_cube[0, 1, 1]) / 2.0

        return np.array([grad_x, grad_y, grad_s])
    
    def compute_hessian(self, pixel_cube):
        
        H = np.zeros((3, 3), dtype=np.float32)

        center = pixel_cube[1, 1, 1]

        # Second-order partial derivatives (diagonal of the Hessian)
        dxx = pixel_cube[1, 1, 2] - 2 * center + pixel_cube[1, 1, 0]
        dyy = pixel_cube[1, 2, 1] - 2 * center + pixel_cube[1, 0, 1]
        dss = pixel_cube[2, 1, 1] - 2 * center + pixel_cube[0, 1, 1]

        # Cross derivatives (off-diagonal of the Hessian)
        dxy = (pixel_cube[1, 2, 2] - pixel_cube[1, 2, 0] - pixel_cube[1, 0, 2] + pixel_cube[1, 0, 0]) / 4.0
        dxs = (pixel_cube[2, 1, 2] - pixel_cube[2, 1, 0] - pixel_cube[0, 1, 2] + pixel_cube[0, 1, 0]) / 4.0
        dys = (pixel_cube[2, 2, 1] - pixel_cube[2, 0, 1] - pixel_cube[0, 2, 1] + pixel_cube[0, 0, 1]) / 4.0

        # Construct symmetric Hessian
        H[0, 0] = dxx
        H[1, 1] = dyy
        H[2, 2] = dss

        H[0, 1] = H[1, 0] = dxy
        H[0, 2] = H[2, 0] = dxs
        H[1, 2] = H[2, 1] = dys

        return H
    def plot_keypoints(self, scale_space_images):
        """
        Plot keypoints on top of the original scale-space images used to compute DoG.

        Args:
            scale_space_images (list): The Gaussian pyramid images used to generate the DoG pyramid.
                                    Should match in structure with self.keypoints (same octaves and scale levels).
        """
        num_octaves = len(self.keypoints)

        for o in range(num_octaves):
            octave_keypoints = self.keypoints[o]
            octave_images = scale_space_images[o]

            fig, axes = plt.subplots(1, len(octave_images), figsize=(3 * len(octave_images), 3))
            if len(octave_images) == 1:
                axes = [axes]

            for s, img in enumerate(octave_images):
                axes[s].imshow(img, cmap='gray')
                axes[s].axis('off')
                axes[s].set_title(f'Octave {o}, Scale {s}')

                # Filter keypoints at this scale
                kps = [(y, x) for scale, y, x in octave_keypoints if scale == s]
                if kps:
                    ys, xs = zip(*kps)
                    axes[s].scatter(xs, ys, c='r', s=5)

            plt.tight_layout()
            plt.suptitle(f"Keypoints in Octave {o}", fontsize=14)
            plt.show()



from math import atan2, degrees, exp, pi

class OrientationGenerator:
    def __init__(self, gaussian_scale_space):
        self.gss = gaussian_scale_space
        self.oriented_keypoints = []  # Will store updated cv2.KeyPoint objects with angle set

    def generate_orientations(self, keypoints):
        self.oriented_keypoints = []
        print(len(keypoints))
        for keypoint in keypoints:
            print(keypoint.__str__())
            image = self.gss.gaussians[keypoint.octave][keypoint.layer]
            sigma = self.gss.sigma_list[keypoint.layer]
            radius = int(round(3 * sigma))
            kernel = self._gaussian_kernel(radius, sigma)

            x, y = int(round(keypoint.pt[0])), int(round(keypoint.pt[1]))

            if x < radius or y < radius or x >= image.shape[1] - radius or y >= image.shape[0] - radius:
                continue

            region = image[y - radius:y + radius + 1, x - radius:x + radius + 1].astype(np.float32)
            dx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(dx**2 + dy**2)
            orientation = (np.arctan2(dy, dx) * (180 / np.pi)) % 360

            weighted_mags = magnitude * kernel
            hist = np.zeros(36)
            bin_width = 360 // 36

            for i in range(orientation.shape[0]):
                for j in range(orientation.shape[1]):
                    angle = orientation[i, j]
                    mag = weighted_mags[i, j]
                    bin_idx = int(angle // bin_width) % 36
                    hist[bin_idx] += mag

            max_val = np.max(hist)
            for i in range(36):
                if hist[i] >= 0.8 * max_val:
                    theta = i * bin_width
                    new_keypoint = keypoint
                    new_keypoint.angle = theta
                    print(f"Keypoint angle: {new_keypoint.angle}")
                    self.oriented_keypoints.append(new_keypoint)


    def _gaussian_kernel(self, radius, sigma):
        size = 2 * radius + 1
        kernel = np.zeros((size, size), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                dx, dy = i - radius, j - radius
                kernel[i, j] = exp(-(dx**2 + dy**2) / (2 * sigma**2))
        return kernel / kernel.sum()
  
  

class DescriptorGenerator:
    def __init__(self, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
        self.window_width = window_width
        self.num_bins = num_bins
        self.scale_multiplier = scale_multiplier
        self.descriptor_max_value = descriptor_max_value

    def unpack_octave(self, keypoint):
        octave = keypoint.octave 
        layer = keypoint.layer
        if octave >= 128:
            octave = octave | -128
        scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
        return octave, layer, scale

    def generate(self, keypoints, gaussian_images):
        logger.debug('Generating descriptors...')
        descriptors = []

        for keypoint in keypoints:
            octave, layer, scale = self.unpack_octave(keypoint)
            
            print(f" Octave: {octave}, Layer: {layer}, Scale: {scale}")
            gaussian_image = gaussian_images[octave + 1][layer]
            num_rows, num_cols = gaussian_image.shape
            point = round(scale * array(keypoint.pt)).astype('int')
            bins_per_degree = self.num_bins / 360.
            angle = 360. - keypoint.angle
            cos_angle = cos(deg2rad(angle))
            sin_angle = sin(deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * self.window_width) ** 2)

            histogram_tensor = zeros((self.window_width + 2, self.window_width + 2, self.num_bins))
            hist_width = self.scale_multiplier * 0.5 * scale * keypoint.size
            half_width = int(round(hist_width * sqrt(2) * (self.window_width + 1) * 0.5))
            half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))

            row_bin_list, col_bin_list, magnitude_list, orientation_bin_list = [], [], [], []

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * self.window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * self.window_width - 0.5

                    if -1 < row_bin < self.window_width and -1 < col_bin < self.window_width:
                        window_row = int(round(point[1] + row))
                        window_col = int(round(point[0] + col))

                        if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                            dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                            gradient_magnitude = sqrt(dx * dx + dy * dy)
                            gradient_orientation = rad2deg(arctan2(dy, dx)) % 360
                            weight = exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))

                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
                row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor

                if orientation_bin_floor < 0:
                    orientation_bin_floor += self.num_bins
                if orientation_bin_floor >= self.num_bins:
                    orientation_bin_floor -= self.num_bins

                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)

                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % self.num_bins] += c001
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % self.num_bins] += c011
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % self.num_bins] += c101
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % self.num_bins] += c111

            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
            threshold = norm(descriptor_vector) * self.descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(norm(descriptor_vector), float_tolerance)
            descriptor_vector = round(512 * descriptor_vector)
            descriptor_vector = array([min(max(v, 0), 255) for v in descriptor_vector], dtype='float32')
            keypoint.descriptor = descriptor_vector
            descriptors.append(descriptor_vector)

        return array(descriptors, dtype='float32')


def keypoint_matching(desc1, kp1, desc2, kp2, ratio_thresh=0.75):
    """
    Perform descriptor matching using a ratio test and return the match percentage and list of good matches.
    """
    good_matches = []

    for i, d1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d1, axis=1)
        if len(distances) < 2:
            continue
        nearest = np.argsort(distances)[:2]
        if distances[nearest[0]] < ratio_thresh * distances[nearest[1]]:
            match = cv2.DMatch(_queryIdx=i, _trainIdx=nearest[0], _distance=distances[nearest[0]])
            good_matches.append(match)

    match_percentage = (len(good_matches) / len(kp1)) * 100 if kp1 else 0
    return match_percentage, good_matches


def visualize_manual_matches(image1, kp1, image2, kp2, matches):
    """
    Visualize matches between two sets of keypoints.
    """
    # Convert keypoints from custom format to cv2.KeyPoint for visualization
    kp1_cv2 = [cv2.KeyPoint(float(k.pt[0]), float(k.pt[1]), float(k.size), float(k.angle), float(k.response), int(k.octave), int(k.class_id)) for k in kp1]
    kp2_cv2 = [cv2.KeyPoint(float(k.pt[0]), float(k.pt[1]), float(k.size), float(k.angle), float(k.response), int(k.octave), int(k.class_id)) for k in kp2]

    matched_image = cv2.drawMatches(image1, kp1_cv2, image2, kp2_cv2, matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image
  
if __name__ == "__main__":
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load grayscale images
    image1 = cv2.imread("box.png", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("box_in_scene.png", cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        raise FileNotFoundError("One or both images not found. Please check the paths.")

    # ---------- IMAGE 1 PROCESSING ----------
    gss1 = GaussianScaleSpace()
    gss1.build_digital_scale_space(image1)

    extractor1 = keypointsExtractor(gss1)
    extractor1.extract_local_extrema(threshold=0)

    orientation_gen1 = OrientationGenerator(gss1)
    orientation_gen1.generate_orientations(extractor1.keypoints)

    descriptor_gen1 = DescriptorGenerator()
    descriptor_gen1.generate(orientation_gen1.oriented_keypoints, gss1.gaussians)

    # ---------- IMAGE 2 PROCESSING ----------
    gss2 = GaussianScaleSpace()
    gss2.build_digital_scale_space(image2)

    extractor2 = keypointsExtractor(gss2)
    extractor2.extract_local_extrema(threshold=0)

    orientation_gen2 = OrientationGenerator(gss2)
    orientation_gen2.generate_orientations(extractor2.keypoints)

    descriptor_gen2 = DescriptorGenerator()
    descriptor_gen2.generate(orientation_gen2.oriented_keypoints, gss2.gaussians)

    # ---------- MATCHING ----------
    desc1 = np.array([kp.descriptor for kp in orientation_gen1.oriented_keypoints])
    desc2 = np.array([kp.descriptor for kp in orientation_gen2.oriented_keypoints])

    kp1 = orientation_gen1.oriented_keypoints
    kp2 = orientation_gen2.oriented_keypoints


    match_percentage, good_matches = keypoint_matching(desc1, kp1, desc2, kp2)

    print(f"Matching Percentage: {match_percentage:.2f}%")

    matched_image = visualize_manual_matches(image1, kp1, image2, kp2, good_matches)

    # Show result
    matched_image_rgb = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 10))
    plt.imshow(matched_image_rgb)
    plt.axis('off')
    plt.title(f"Matching Results - {match_percentage:.2f}% Matches")
    plt.show()
