from typing import List, Tuple
from numpy import ndarray
import cv2 as cv


class FeatureDatabase:
    """
    Stores the features that are identified in training images and identifies corresponding features in test images.
    """

    class DatabaseRecord:
        """
        A record of an image in the database; contains all key points and their descriptors, along with the image and
        its ID / name.
        """
        def __init__(self, ID, key_points, descriptors, image) -> None:
            self.ID = ID
            self.key_points = key_points
            self.descriptors = descriptors
            self.image = image

    def __init__(self, octave_count: int, scale_levels: int):
        """
        Construct the database
        :param octave_count:
        :param scale_levels:
        """
        self.records = []

        self.featureExtractor = cv.SIFT_create()

        self.octave_count = octave_count
        self.scale_levels = scale_levels

    def add_training_images(self, images: List[Tuple[ndarray, str]]) -> None:
        """
        Accepts a list of images or image filepaths and performs SIFT feature extraction to find points. Stores this
        information in the database.

        :param images: a list of images or image filepaths.
        """
        for image, name in images:
            keypoints, descriptors = self.featureExtractor.detectAndCompute(image, None)
            self.records.append(self.DatabaseRecord(name, keypoints, descriptors, image))

