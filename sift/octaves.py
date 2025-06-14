
import itertools
import math
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from . import const


def relative_sigma(layer_idx: int):
    """
    layer_sigma = (octave_idx / min_pixel_dist) * min_sigma * 2 ** (layer_idx / scales_per_octave)

    Args:
        layer_idx: The index of a layer in octave.
    Returns:
        sigma: The Gaussian blur filter's std required for (layer_idx - 1) -> layer_idx.
    """
    sigma = (const.min_sigma / const.min_pixel_dist) \
            * math.sqrt(2 ** (2 * layer_idx / const.scales_per_octave)
                        - 2 ** (2 * (layer_idx - 1) / const.scales_per_octave))
    return sigma


def absolute_sigma(octave_idx: int,
                   layer_idx: int):
    """
        Calculates layer's absolute sigma
    Args:
        octave_idx: Note we start with the octave of the base image
    Returns:
        sigma: The relative Sigmas.
    """
    pixel_dist = pixel_dist_in_octave(octave_idx)
    sigma = (pixel_dist / const.min_pixel_dist) * const.min_sigma * 2 ** (layer_idx / const.scales_per_octave)
    return sigma


def build_gaussian_octaves(img: np.ndarray):
    """ For each octave apply the sigma for building the scale space
    Args:
        img: The image in the pyramid used
    Returns:
        octaves: A list of octaves of Gaussian convolved images.
            Here, each octave is a [s, y, x] 3D tensor.
    """
    layers_per_octave = const.scales_per_octave + const.auxiliary_scales
    octaves = []
    previous_octave = None

    for octave_idx in range(const.nr_octaves):

        #first octave 2x upsampled input image... Base Image
        if octave_idx == 0:
            img = cv2.resize(img, None, fx=const.first_upscale, fy=const.first_upscale, interpolation=cv2.INTER_LINEAR)
            img = gaussian_filter(img, const.init_sigma)
            octave = [img]
        else:
            img = cv2.resize(previous_octave[-2], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            octave = [img]

        # Convolve layers with gaussians to generate successive layers.
        # The previous octave's[-2] upsampled image is considered layer
        # index 0, so indexing starts at 1
        for layer_idx in range(1, layers_per_octave):
            sigma = relative_sigma(layer_idx)
            img = gaussian_filter(img, sigma)
            octave.append(img)

        previous_octave = octave
        octave = np.array(octave)
        octaves.append(octave)

    return octaves


def build_dog_octave(gauss_octave: np.ndarray):
    """ Builds a Difference of Gaussian octave.

    Args:
        gauss_octave: An octave of Gaussian convolved images.
    Returns:
        dog_octave: An octave of Difference of Gaussian images.
    """
    dog_octave = []

    for layer_idx, layer in enumerate(gauss_octave):
        if layer_idx:
            previous_layer = gauss_octave[layer_idx - 1]
            dog = layer - previous_layer
            dog_octave.append(dog)

    return np.array(dog_octave)


def shift(array: np.ndarray,
          shift_spec: list or tuple):
    """ make a movement in the space.

    Args:
        array: The 3D array that is to be shifted.
        shift_spec: The shift specification for each of
            the 3 axes. E.g., [1, 0, 0] will make the
            element (x,x,x) equal element (x+1, x+1, x+1) in
            the original image, effectively shifting the
            image "to the left", along the first axis.
    Returns:
        shifted: The shifted array.
    """
    padded = np.pad(array, 1, mode='edge')
    s, y, x = shift_spec
    shifted = padded[1 + s: -1 + s if s != 1 else None,
                     1 + y: -1 + y if y != 1 else None,
                     1 + x: -1 + x if x != 1 else None]
    return shifted


def find_dog_extrema(dog_octave: np.ndarray):
    """ Finds extrema in a Difference of Gaussian octave.
    
    Args:
        dog_octave: An octave of Difference of Gaussian images.
    Returns:
        extrema_coords: The Difference of Gaussian extrema coordinates.
    """
    shifts = list(itertools.product([-1, 0, 1], repeat=3))
    shifts.remove((0, 0, 0))

    # 0 0
    # 0 1 
    # 1 2 3  7 8 9
    # 4 5 6  1 2 3 
    # 7 8 9  4 5 6
    diffs = []
    for shift_spec in shifts:
        shifted = shift(dog_octave, shift_spec)
        diff = dog_octave - shifted
        diffs.append(diff)

    diffs = np.array(diffs)
    maxima = np.where((diffs > 0).all(axis=0))
    minima = np.where((diffs < 0).all(axis=0))
    extrema_coords = np.concatenate((maxima, minima), axis=1)

    return extrema_coords


def derivatives(dog_octave: np.ndarray):
   
    o = dog_octave

    ds = (shift(o, [1, 0, 0]) - shift(o, [-1, 0, 0])) / 2
    dy = (shift(o, [0, 1, 0]) - shift(o, [0, -1, 0])) / 2
    dx = (shift(o, [0, 0, 1]) - shift(o, [0, 0, -1])) / 2

    dss = (shift(o, [1, 0, 0]) + shift(o, [-1, 0, 0]) - 2 * o)
    dyy = (shift(o, [0, 1, 0]) + shift(o, [0, -1, 0]) - 2 * o)
    dxx = (shift(o, [0, 0, 1]) + shift(o, [0, 0, -1]) - 2 * o)

    dsy = (shift(o, [1, 1, 0]) - shift(o, [1, -1, 0]) - shift(o, [-1, 1, 0]) + shift(o, [-1, -1, 0])) / 4
    dsx = (shift(o, [1, 0, 1]) - shift(o, [1, 0, -1]) - shift(o, [-1, 0, 1]) + shift(o, [-1, 0, -1])) / 4
    dyx = (shift(o, [0, 1, 1]) - shift(o, [0, 1, -1]) - shift(o, [0, -1, 1]) + shift(o, [0, -1, -1])) / 4

    derivs = np.array([ds, dy, dx])
    second_derivs = np.array([dss, dsy, dsx,
                              dsy, dyy, dyx,
                              dsx, dyx, dxx])
    return derivs, second_derivs


def pixel_dist_in_octave(octave_idx: int) -> float:
    """ Calculates the distance between adjacent pixels in an octave.
        As each octave starts with a 2x downsampled image, each successive
        octave doubles the pixel distance.

    Args:
        octave_idx: The index of the octave.
    Returns:
        The distance between adjacent pixels in an octave.
    """
    return const.min_pixel_dist * (2 ** octave_idx)
