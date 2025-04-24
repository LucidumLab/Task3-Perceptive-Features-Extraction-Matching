
import cv2
import numpy as np
import sys
sys.path.append('D:/SBME/3rd Year/CV/Tasks/LucidumLab/Task3-Perceptive-Features-Extraction-Matching')

from models.sift.descriptor import assign_descriptor
from models.sift.keypoints import Keypoint, find_keypoints
from models.sift.match import match_sift_features, visualize_matches
from models.sift.octaves import build_gaussian_octaves, build_dog_octave, find_dog_extrema
from models.sift.reference_orientation import assign_reference_orientations


def detect_sift_features(img: np.ndarray) -> list[Keypoint]:
    """ Detects SIFT keypoints in an image.

    Args:
        img: a normalized grayscale image[0, 1].
    Returns:
        keypoints: A list of keypoints objects.
    """
    gauss_octaves = build_gaussian_octaves(img)

    features = list()
    for octave_idx, gauss_octave in enumerate(gauss_octaves):
        dog_octave = build_dog_octave(gauss_octave)
        extrema = find_dog_extrema(dog_octave)
        keypoint_coords = find_keypoints(extrema, dog_octave)
        keypoints = assign_reference_orientations(keypoint_coords, gauss_octave, octave_idx)
        keypoints = assign_descriptor(keypoints, gauss_octave, octave_idx)
        features += keypoints

    return features

def apply_sift(img1, img2):
    """
    Function to detect SIFT features, match them between two images, and visualize the matches.
    
    Args:
        img1 (numpy.ndarray): The first image.
        img2 (numpy.ndarray): The second image.

    Returns:
        match_image (numpy.ndarray): The image with visualized matches.
        len_keypoints1 (int): The number of keypoints in the first image.
        len_keypoints2 (int): The number of keypoints in the second image.
        len_matches (int): The number of matched keypoints.
    """
    keypoints1 = detect_sift_features(img1)  # Detect SIFT keypoints in the first image
    keypoints2 = detect_sift_features(img2)  # Detect SIFT keypoints in the second image
    
    matches = match_sift_features(keypoints1, keypoints2)  # Match keypoints between both images
    
    match_image = visualize_matches(matches, img1, img2)  # Visualize the matches on the images

    # Print the number of keypoints and matches
    len_keypoints1 = len(keypoints1)
    len_keypoints2 = len(keypoints2)
    len_matches = len(matches)

    print(f'len of keypoints 1: {len_keypoints1}')
    print(f'len of keypoints 2: {len_keypoints2}')
    print(f'len of matches: {len_matches}')

    return match_image, len_keypoints1, len_keypoints2, len_matches


def main():
    """ Detects and matches SIFT features in two images. """
    img1 = cv2.imread('box.png', flags=cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('box_in_scene.png', flags=cv2.IMREAD_GRAYSCALE)
    img1 = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img2 = cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    keypoints1 = detect_sift_features(img1)
    keypoints2 = detect_sift_features(img2)
    matches = match_sift_features(keypoints1, keypoints2)
    match_image = visualize_matches(matches, img1, img2)
    print(len(matches))

    print('len of keypoints 1: ', len(keypoints1))
    print('len of keypoints 2: ', len(keypoints2))
    print('len of matches: ', len(matches))


if __name__ == '__main__':
    main()

