from typing import List
from numpy import ndarray
import cv2 as cv


class FeatureDatabase:
    """
    Stores the features that are identified in training images and identifies corresponding features in test images.
    """
    def __init__(self, octave_count: int, scale_levels: int):
        """
        Construct the database
        :param octave_count:
        :param scale_levels:
        """
        self.featureExtractor = cv.SIFT_create()

        self.octave_count = octave_count
        self.scale_levels = scale_levels

    def add_training_images(self, images: List[ndarray]) -> None:
        """
        Accepts a list of images or image filepaths and performs SIFT feature extraction to find points. Stores this
        information in the database.

        :param images: a list of images or image filepaths.
        """
        for image in images:
            if type(image) == str:
                image = cv.imread(image)

            keypoints = self.featureExtractor.detect(image, None)
            image = cv.drawKeypoints(image, keypoints, image, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv.namedWindow("test")
            cv.imshow("test", image)
            cv.waitKey(0)
            cv.destroyAllWindows()

