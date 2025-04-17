import numpy as np
import cv2

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

def magnitude(x, y):
    return np.sqrt(x**2 + y**2)

def compute_gradient(image, axis = 0):
    """
    NOT IMPLEMENTED
    Computes the gradient of an image using the Sobel operator.
    """
    """
    Computes the gradient of a 2D array along the specified axis.
    
    For interior points, uses central differences.
    For boundaries, uses forward difference at the beginning and backward difference at the end.
    
    :param image: 2D NumPy array (e.g., grayscale image).
    :param axis: Axis along which to compute the gradient (0 for rows, 1 for columns).
    :return: Gradient of the image along the specified axis.
    """
    if image.ndim != 2:
        raise ValueError("my_gradient currently supports only 2D arrays.")
    
    grad = np.zeros_like(image, dtype=np.uint8)
    
    if axis == 0:
        # Compute gradient along rows
        # Forward difference for the first row
        grad[0, :] = image[1, :] - image[0, :]
        # Central differences for interior rows
        for i in range(1, image.shape[0] - 1):
            grad[i, :] = (image[i+1, :] - image[i-1, :]) / 2.0
        # Backward difference for the last row
        grad[-1, :] = image[-1, :] - image[-2, :]
        
    elif axis == 1:
        # Compute gradient along columns
        # Forward difference for the first column
        grad[:, 0] = image[:, 1] - image[:, 0]
        # Central differences for interior columns
        for j in range(1, image.shape[1] - 1):
            grad[:, j] = (image[:, j+1] - image[:, j-1]) / 2.0
        # Backward difference for the last column
        grad[:, -1] = image[:, -1] - image[:, -2]
        
    else:
        raise ValueError("Unsupported axis. Use axis=0 for rows or axis=1 for columns.")
        
    return np.clip(grad, 0, 255).astype(np.uint8)
import numpy as np

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
# def convolve(img, kernel):
#     """
#     Applies convolution to an image using a given kernel (without using convolve2d).

#     :param img: Input grayscale image (NumPy array).
#     :param kernel: Filter kernel (NumPy array).
#     :return: Convolved image.
#     """
#     # Ensure the kernel has odd dimensions
#     kernel_height, kernel_width = kernel.shape
#     if kernel_height % 2 == 0 or kernel_width % 2 == 0:
#         raise ValueError("Kernel dimensions must be odd (e.g., 3x3, 5x5).")

#     # Calculate padding sizes
#     pad_h = kernel_height // 2  # Padding for rows (top and bottom)
#     pad_w = kernel_width // 2   # Padding for columns (left and right)

#     # Pad the image using custom padding function
#     img_padded = custom_pad(img, pad_h, pad_w)

#     # Create an empty output image
#     output = np.zeros_like(img, dtype=np.float32)
    
#     # Perform convolution
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             # Extract region matching the kernel size
#             region = img_padded[i:i + kernel_height, j:j + kernel_width]
#             # Ensure the region and kernel have the same shape
#             if region.shape == kernel.shape:
#                 output[i, j] = np.sum(region * kernel)  # Apply filter
#             else:
#                 # Handle edge cases where the region is smaller than the kernel
#                 output[i, j] = img[i, j]  # Keep the original pixel value
    
#     # Normalize output to valid range (0-255) and convert to uint8
#     return np.clip(output, 0, 255).astype(np.uint8)

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
    return kernel / np.sum(kernel)  # Normalize kernel to sum to 1


def get_dimensions(image):
    """
    Get the height and width of an image, handling both grayscale and color images.

    :param image: NumPy array representing the image.
    :return: Tuple (height, width)
    """
    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) == 2:  # Grayscale image (H, W)
        h, w = image.shape
    elif len(image.shape) == 3:  # Color image (H, W, C)
        h, w, _ = image.shape
    else:
        raise ValueError("Invalid image format. Expected 2D (grayscale) or 3D (color) NumPy array.")

    return h, w


def cumsum(arr):
    """
    Computes the cumulative sum of an array.

    :param arr: List of numbers.
    :return: List containing the cumulative sum.
    """
    cumulative_sum = [0] * len(arr)  # Initialize output array with zeros
    
    cumulative_sum[0] = arr[0]  # First element remains the same

    for i in range(1, len(arr)):  # Start from the second element
        cumulative_sum[i] = cumulative_sum[i - 1] + arr[i]  # Add previous sum

    return np.cumsum(arr)