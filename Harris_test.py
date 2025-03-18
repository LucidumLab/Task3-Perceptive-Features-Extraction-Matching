import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_pad(img, pad_height, pad_width):
    """
    Pads an image by replicating edge values.

    :param img: Input image (2D NumPy array).
    :param pad_height: Padding size for the height (top and bottom).
    :param pad_width: Padding size for the width (left and right).
    :return: Padded image.
    """
    # Get the original image dimensions
    height, width = img.shape

    # Create an empty padded image
    padded_img = np.zeros((height + 2 * pad_height, width + 2 * pad_width), dtype=img.dtype)

    # Fill the center with the original image
    padded_img[pad_height:pad_height + height, pad_width:pad_width + width] = img

    # Pad the top and bottom rows
    padded_img[:pad_height, pad_width:pad_width + width] = img[0:1, :]  # Top padding (replicate first row)
    padded_img[pad_height + height:, pad_width:pad_width + width] = img[-1:, :]  # Bottom padding (replicate last row)

    # Pad the left and right columns
    padded_img[:, :pad_width] = padded_img[:, pad_width:pad_width + 1]  # Left padding (replicate leftmost column)
    padded_img[:, pad_width + width:] = padded_img[:, pad_width + width - 1:pad_width + width]  # Right padding (replicate rightmost column)

    return padded_img

def convolve(img, kernel):
    """
    Applies convolution to an image using a given kernel with zero-order interpolation.
    
    :param img: Input grayscale image (NumPy array).
    :param kernel: Filter kernel (NumPy array).
    :return: Convolved image.
    """
    # Get kernel dimensions
    kernel_height, kernel_width = kernel.shape

    # Calculate padding sizes
    pad_height = kernel_height // 2  # Padding for rows (top and bottom)
    pad_width = kernel_width // 2    # Padding for columns (left and right)

    # Pad the image using custom padding function
    img_padded = custom_pad(img, pad_height, pad_width)

    # Create an empty output image
    output = np.zeros_like(img, dtype=np.float32)
    
    # Perform convolution
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Extract region matching the kernel size
            region = img_padded[i:i + kernel_height, j:j + kernel_width]
            # Apply filter
            output[i, j] = np.sum(region * kernel)
    
    # Normalize output to valid range (0-255) and convert to uint8
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def gaussian_kernel(size, sigma=1.0):
    """
    Generates a Gaussian kernel.

    :param size: Kernel size (odd number).
    :param sigma: Standard deviation of the Gaussian distribution.
    :return: Normalized Gaussian kernel.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def apply_gaussian_filter(img, kernel_size=3, sigma=1.0):
    """
    Applies a Gaussian filter.

    :param img: Input grayscale image (NumPy array).
    :param kernel_size: Size of the Gaussian kernel (default=3).
    :param sigma: Standard deviation (default=1.0).
    :return: Filtered image.
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve(img, kernel)

def globalthresholding(image, T = 128, value = 255):
    binary_img  = (image > T).astype(np.uint8) * value
    return binary_img

def convert_to_grayscale(image):
    """
    Converts a BGR image to grayscale using the weighted sum method.

    :param image: Input color image (BGR format).
    :return: Grayscale image as NumPy array.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    
    # Apply grayscale conversion formula
    gray_image = 0.299 * r + 0.587 * g + 0.114 * b
    return gray_image.astype(np.uint8)

def compute_sobel_gradients(image):
    """
    Computes the Sobel gradients of an image.
    
    :param image: Grayscale image.
    :return: Tuple (Gx, Gy) representing horizontal and vertical gradients.
    """
    # Define Sobel kernels
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
    
    Gx = convolve(image, kernel_x)
    Gy = convolve(image, kernel_y)
    return Gx, Gy


def compute_harris_response(Ix, Iy, k=0.04):
    # Calculate products of gradients
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Apply a Gaussian filter to the products of gradients
    Ixx = apply_gaussian_filter(Ixx, 5, 1)
    Iyy = apply_gaussian_filter(Iyy, 5, 1)
    Ixy = apply_gaussian_filter(Ixy, 5, 1)

    # Compute Harris corner response
    det = (Ixx * Iyy) - (Ixy ** 2)
    trace = Ixx + Iyy
    R = det - k * (trace ** 2)
    
    return R

def normalize_response(R):
    R_normalized = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return R_normalized

def compute_threshold_value(R, threshold_ratio):
    # Calculate the threshold value based on the maximum value in R
    T = threshold_ratio * R.max()
    
    # Calculate the equivalent threshold in the normalized space (0-255)
    T_normalized = int((T - R.min()) / (R.max() - R.min()) * 255) if R.max() != R.min() else 0
    
    return T, T_normalized

def harris_corner_detector(image, k=0.04, threshold=0.01):
    # Convert the image to grayscale
    gray = convert_to_grayscale(image)

    # Calculate gradients using Sobel filter
    Ix, Iy = compute_sobel_gradients(gray)

    # Compute Harris response
    R = compute_harris_response(Ix, Iy, k)
    
    # Calculate threshold values
    _, T_normalized = compute_threshold_value(R, threshold)
    
    # Normalize response for thresholding
    R_normalized = normalize_response(R)
    
    # Apply global thresholding to get corners
    corners = globalthresholding(R_normalized, T_normalized, 255)
    
    return corners

# Read the image
image = cv2.imread('data/chess-board-brown.jpg')

# Detect corners using Harris Corner Detector
corners = harris_corner_detector(image)

# Display the original image and the detected corners using matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(corners, cmap='gray')
plt.title('Harris Corners')
plt.axis('off')

plt.show()