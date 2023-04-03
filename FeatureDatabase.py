import math
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

    def __init__(self, octave_count: int, scale_levels: int, match_threshold: int = 250):
        """
        Construct the database
        :param octave_count:
        :param scale_levels:
        """
        self.training_images = []

        self.featureExtractor = cv.SIFT_create()
        self.matcher = cv.BFMatcher(crossCheck=True)

        self.octave_count = octave_count
        self.scale_levels = scale_levels

        self.MATCH_THRESHOLD = match_threshold

    def add_training_images(self, images: List[Tuple[ndarray, str]]) -> None:
        """
        Accepts a list of images or image filepaths and performs SIFT feature extraction to find points. Stores this
        information in the database.

        :param images: a list of images or image filepaths.
        """
        for image, name in images:
            keypoints, descriptors = self.featureExtractor.detectAndCompute(image, None)
            self.training_images.append(self.DatabaseRecord(name, keypoints, descriptors, image))

    def get_image_matches(self, query_image: ndarray) -> list[dict]:
        """
        Performs SIFT key point extraction on the given image and matches with all records in the database, returning
        a list of match records, sorted by the "closeness" of the match.

        :param query_image: the image to match.
        :return: the sorted matches.
        """
        # First get the SIFT points on the input image
        key_points, descriptors = self.featureExtractor.detectAndCompute(query_image, None)
        imageMatches = []

        # Now match against all records in db
        for training_image in self.training_images:
            matches = self.matcher.match(training_image.descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            normalised_match_error = sum([match.distance for match in matches]) / len(matches)

            # Reject if the match error is higher than the threshold.
            if normalised_match_error > self.MATCH_THRESHOLD:
                continue

            imageMatches.append({"matches": matches,
                                 "error": normalised_match_error,
                                 "training_image": training_image,
                                 "query_image": self.DatabaseRecord("placeholder", key_points, descriptors, training_image)})

        imageMatches = sorted(imageMatches, key=lambda match: match["error"])

        return imageMatches

    def show_matches_for_images(self, images: list[ndarray]) -> None:
        """
        Performs SIFT key point extraction on the given images and then matches with all the records in the database,
        showing the output in a window.

        :param images: the images we are trying to compare against.
        """
        for image in images:
            key_points, _ = self.featureExtractor.detectAndCompute(image, None)
            imageMatches = self.get_image_matches(image)

            for count in range(len(imageMatches)):
                if imageMatches[count]["error"] > self.MATCH_THRESHOLD:
                    continue

                print(imageMatches[count]["error"])
                i = cv.drawMatches(imageMatches[count]["image"].image, imageMatches[count]["image"].key_points, image,
                                   key_points, imageMatches[count]["matches"][:10], None,
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                cv.namedWindow("display")
                cv.imshow("display", i)
                cv.waitKey(0)
