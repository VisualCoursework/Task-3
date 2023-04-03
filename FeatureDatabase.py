import math
from typing import List, Tuple
from numpy import ndarray
import numpy as np
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
        self.POINT_MATCH_COUNT = 15

    def add_training_images(self, images: List[Tuple[ndarray, str]]) -> None:
        """
        Accepts a list of images or image filepaths and performs SIFT feature extraction to find points. Stores this
        information in the database.

        :param images: a list of images or image filepaths.
        """
        for image, name in images:
            keypoints, descriptors = self.featureExtractor.detectAndCompute(image, None)
            self.training_images.append(self.DatabaseRecord(name, keypoints, descriptors, image))

    def get_image_matches(self, query_image: ndarray, image_name: str) -> list[dict]:
        """
        Performs SIFT key point extraction on the given image and matches with all records in the database, returning
        a list of match records, sorted by the "closeness" of the match.

        :param query_image: the image to match.
        :param image_name: the name of the image.
        :return: the sorted matches.
        """
        # First get the SIFT points on the input image
        key_points, descriptors = self.featureExtractor.detectAndCompute(query_image, None)
        imageMatches = []

        # Now match against all records in db
        for training_image in self.training_images:
            # Note that the training and query images that OpenCV uses here are in the opposite order to the ones we are
            # providing. The query image we have is the "training" image in the matcher, as that's the image being
            # searched. The training images we have are being "queried".
            matches = self.matcher.match(training_image.descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            normalised_match_error = sum([match.distance for match in matches]) / len(matches)

            # Reject if the match error is higher than the threshold.
            if normalised_match_error > self.MATCH_THRESHOLD:
                continue

            match = {"matches": matches,
                     "error": normalised_match_error,
                     "training_image": training_image,
                     "query_image": self.DatabaseRecord(image_name, key_points, descriptors, query_image)}

            homography, mask = self.get_train_to_query_homography(match)
            match["homography"] = homography
            topLeft, bottomRight = self.calculate_bounding_box(match)
            match["top_left"] = topLeft
            match["bottom_right"] = bottomRight
            imageMatches.append(match)

        imageMatches = sorted(imageMatches, key=lambda match: match["error"])

        return imageMatches

    def get_train_to_query_homography(self, match: dict) -> tuple:
        """
        Calculates the homography which maps the points in the training image onto the query image.

        :return: the homography
        """
        # Counter-intuitive that we are using the trainIdx for the query points and vice versa, but this is in fact
        # correct. It's just a naming thing: we are calling the emoji the training images and the test images the query
        # images, but from OpenCV's point of view, we are querying the test images with the emoji.
        query_key_points = np.array([match["query_image"].key_points[m.trainIdx].pt for m in match["matches"][:self.POINT_MATCH_COUNT]])
        train_key_points = np.array([match["training_image"].key_points[m.queryIdx].pt for m in match["matches"][:self.POINT_MATCH_COUNT]])

        return cv.findHomography(train_key_points, query_key_points, cv.RANSAC, 1)

    def calculate_bounding_box(self, match: dict):
        """
        Calculates the bounding box for a match using its homography

        :param match: the match to get the bounding box of
        :return:
        """
        topLeft = cv.perspectiveTransform(np.float32([0, 0]).reshape(-1, 1, 2), match["homography"])[0][0]
        bottomRight = cv.perspectiveTransform(np.float32([511, 511]).reshape(-1, 1, 2), match["homography"])[0][0]

        topLeft = tuple(map(round, topLeft))
        bottomRight = tuple(map(round, bottomRight))

        return topLeft, bottomRight

    def show_boxes_around_images(self, images: list[tuple[ndarray, str]]) -> None:
        """
        For each image, finds all matches in this database which appear in the image, drawing a box around each one.

        :param images: the images to search.
        """
        for image, name in images:
            imageMatches = self.get_image_matches(image, name)

            for count, match in enumerate(imageMatches):
                # BGR colour space
                cv.rectangle(image, match["top_left"], match["bottom_right"], (0, 0, 255))

                i = cv.drawMatches(imageMatches[count]["training_image"].image,
                                   imageMatches[count]["training_image"].key_points, image,
                                   imageMatches[count]["query_image"].key_points, imageMatches[count]["matches"][:self.POINT_MATCH_COUNT],
                                   None,
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                cv.namedWindow("search")
                cv.namedWindow("found")
                cv.namedWindow("display")

                cv.imshow("search", match["query_image"].image)
                cv.imshow("found", match["training_image"].image)
                cv.imshow("display", i)

                cv.waitKey(0)

    def show_matches_for_images(self, images: list[tuple[ndarray, str]]) -> None:
        """
        Performs SIFT key point extraction on the given images and then matches with all the records in the database,
        showing the output in a window.

        :param images: the images we are trying to compare against.
        """
        for image, name in images:
            imageMatches = self.get_image_matches(image, name)

            for count in range(len(imageMatches)):
                if imageMatches[count]["error"] > self.MATCH_THRESHOLD:
                    continue

                print(imageMatches[count]["error"])
                i = cv.drawMatches(imageMatches[count]["training_image"].image, imageMatches[count]["training_image"].key_points, image,
                                   imageMatches[count]["query_image"].key_points, imageMatches[count]["matches"][:5], None,
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                cv.namedWindow("display")
                cv.imshow("display", i)
                cv.waitKey(0)
