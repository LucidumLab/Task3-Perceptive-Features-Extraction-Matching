"""
This file contains functions related to matching and visualizing SIFT features.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

import sift as const
from sift.keypoints import Keypoint
from . import const

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from PIL import Image


def match_sift_features(features1: list[Keypoint],
                        features2: list[Keypoint]) :
    """ A brute force method for finding matches between two sets of SIFT features.

    Args:
        features1: A set of SIFT features.
        features2: A set of SIFT features.
    Returns:
        matches: A list of matches. Each match is a (feature, feature) tuples.
    """

    matches = list()

    for idx1, feature1 in enumerate(features1):
        descriptor1 = feature1.descriptor

        min_dist = np.inf
        rest_min = np.inf
        min_feature = None

        for idx2, feature2 in enumerate(features2):
            descriptor2 = feature2.descriptor

            dist = np.linalg.norm(descriptor1 - descriptor2)

            if dist < min_dist:
                min_dist = dist
                min_feature = feature2

            elif dist < rest_min:
                rest_min = dist

        if min_dist < rest_min * const.rel_dist_match_thresh:
            matches.append((feature1, min_feature))

    return matches


def match_sift_features_ncc(features1: list, features2: list):
    """Brute force NCC-based matching between two sets of SIFT features."""
    matches = []

    for feature1 in features1:
        descriptor1 = feature1.descriptor
        descriptor1 = (descriptor1 - np.mean(descriptor1)) / (np.std(descriptor1) + 1e-8)

        max_ncc = -np.inf
        rest_max_ncc = -np.inf
        best_match = None

        for feature2 in features2:
            descriptor2 = feature2.descriptor
            descriptor2 = (descriptor2 - np.mean(descriptor2)) / (np.std(descriptor2) + 1e-8)

            ncc = np.dot(descriptor1, descriptor2)

            if ncc > max_ncc:
                rest_max_ncc = max_ncc
                max_ncc = ncc
                best_match = feature2
            elif ncc > rest_max_ncc:
                rest_max_ncc = ncc

        if max_ncc > rest_max_ncc * const.rel_dist_match_thresh:
            matches.append((feature1, best_match))

    return matches


def visualize_matches(matches: list[tuple[Keypoint, Keypoint]],
                      img1: np.ndarray,
                      img2: np.ndarray):
    """ Plots SIFT keypoint matches between two images.

    Args:
        matches: A list of matches. Each match is a (feature, feature) tuples.
        img1: The image in which the first match features were found.
        img2: The image in which the second match features were found.
    """

    coords_1 = [match[0].absolute_coordinate for match in matches]
    coords_1y = [coord[1] for coord in coords_1]
    coords_1x = [coord[2] for coord in coords_1]
    coords_1xy = [(x, y) for x, y in zip(coords_1x, coords_1y)]

    coords_2 = [match[1].absolute_coordinate for match in matches]
    coords_2y = [coord[1] for coord in coords_2]
    coords_2x = [coord[2] for coord in coords_2]
    coords_2xy = [(x, y) for x, y in zip(coords_2x, coords_2y)]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(img1, cmap='Greys_r')
    ax2.imshow(img2, cmap='Greys_r')

    ax1.scatter(coords_1x, coords_1y)
    ax2.scatter(coords_2x, coords_2y)

    for p1, p2 in zip(coords_1xy, coords_2xy):
        con = ConnectionPatch(xyA=p2, xyB=p1, coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)

    plt.show()



def match_sift_features_ssd(features1: list, features2: list):
    """Brute force SSD-based matching between two sets of SIFT features."""
    matches = []

    for feature1 in features1:
        descriptor1 = feature1.descriptor

        min_ssd = np.inf
        rest_min_ssd = np.inf
        best_match = None

        for feature2 in features2:
            descriptor2 = feature2.descriptor

            ssd = np.sum((descriptor1 - descriptor2) ** 2)

            if ssd < min_ssd:
                rest_min_ssd = min_ssd
                min_ssd = ssd
                best_match = feature2
            elif ssd < rest_min_ssd:
                rest_min_ssd = ssd

        if min_ssd < rest_min_ssd * const.rel_dist_match_thresh:
            matches.append((feature1, best_match))

    return matches

def visualize_matches_ndarray(matches: list[tuple[Keypoint, Keypoint]],
                      img1: np.ndarray,
                      img2: np.ndarray):
        
        """Returns the SIFT keypoint matches visualization as an image (NumPy array)."""
        
        coords_1 = [match[0].absolute_coordinate for match in matches]
        coords_1y = [coord[1] for coord in coords_1]
        coords_1x = [coord[2] for coord in coords_1]
        coords_1xy = [(x, y) for x, y in zip(coords_1x, coords_1y)]

        coords_2 = [match[1].absolute_coordinate for match in matches]
        coords_2y = [coord[1] for coord in coords_2]
        coords_2x = [coord[2] for coord in coords_2]
        coords_2xy = [(x, y) for x, y in zip(coords_2x, coords_2y)]

        fig = plt.figure(figsize=(10, 5))
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(img1, cmap='gray')
        ax2.imshow(img2, cmap='gray')

        ax1.scatter(coords_1x, coords_1y, c='cyan', s=5)
        ax2.scatter(coords_2x, coords_2y, c='cyan', s=5)

        for p1, p2 in zip(coords_1xy, coords_2xy):
            con = ConnectionPatch(xyA=p2, xyB=p1, coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color="red", linewidth=0.5)
            ax2.add_artist(con)

        # Render figure to a canvas and convert it to a NumPy image
        canvas.draw()
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_np = np.array(image)

        plt.close(fig)  # Prevent showing the figure
        return image_np
    
    