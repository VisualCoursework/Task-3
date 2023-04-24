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

    def __init__(self, octave_layers: int, match_threshold: int = 250, points_to_match: int = 15):
        """
        Construct the database
        :param octave_layers: the number of layers to use in each octave of the Gaussian pyramid.
        """
        self.training_images = []

        self.featureExtractor = cv.SIFT_create(nOctaveLayers=octave_layers)
        self.matcher = cv.BFMatcher(crossCheck=True)

        self.OCTAVE_LAYERS = octave_layers
        self.MATCH_THRESHOLD = match_threshold
        self.POINT_MATCH_COUNT = points_to_match

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

            affine, mask = self.get_train_to_query_affine(match)
            match["affine"] = affine
            topLeft, topRight, bottomLeft, bottomRight = self.calculate_bounding_box(match)

            # Handle rotations by choosing the minimums and maximums of the bounding box.
            match["top_left"] = (min(topLeft[0], topRight[0], bottomLeft[0], bottomRight[0]),
                                 min(topLeft[1], topRight[1], bottomLeft[1], bottomRight[1]))
            match["top_right"] = (max(topLeft[0], topRight[0], bottomLeft[0], bottomRight[0]),
                                  min(topLeft[1], topRight[1], bottomLeft[1], bottomRight[1]))
            match["bottom_left"] = (min(topLeft[0], topRight[0], bottomLeft[0], bottomRight[0]),
                                    max(topLeft[1], topRight[1], bottomLeft[1], bottomRight[1]))
            match["bottom_right"] = (max(topLeft[0], topRight[0], bottomLeft[0], bottomRight[0]),
                                     max(topLeft[1], topRight[1], bottomLeft[1], bottomRight[1]))

            imageMatches.append(match)

        imageMatches = sorted(imageMatches, key=lambda match: match["error"])

        return imageMatches

    def get_train_to_query_affine(self, match: dict) -> tuple:
        """
        Calculates the affine transformation which maps the points in the training image onto the query image.

        :return: the affine transformation
        """
        # Counter-intuitive that we are using the trainIdx for the query points and vice versa, but this is in fact
        # correct. It's just a naming thing: we are calling the emoji the training images and the test images the query
        # images, but from OpenCV's point of view, we are querying the test images with the emoji.
        query_key_points = np.array([match["query_image"].key_points[m.trainIdx].pt for m in match["matches"][:self.POINT_MATCH_COUNT]])
        train_key_points = np.array([match["training_image"].key_points[m.queryIdx].pt for m in match["matches"][:self.POINT_MATCH_COUNT]])

        return cv.estimateAffinePartial2D(train_key_points, query_key_points)

    def calculate_bounding_box(self, match: dict):
        """
        Calculates the bounding box for a match using its affine transformation.

        :param match: the match to get the bounding box of.
        :return: the vertices of the bounding box.
        """
        topLeft = cv.transform(np.float32([0, 0]).reshape(-1, 1, 2), match["affine"])[0][0]
        topRight = cv.transform(np.float32([511, 0]).reshape(-1, 1, 2), match["affine"])[0][0]
        bottomLeft = cv.transform(np.float32([0, 511]).reshape(-1, 1, 2), match["affine"])[0][0]
        bottomRight = cv.transform(np.float32([511, 511]).reshape(-1, 1, 2), match["affine"])[0][0]

        topLeft = tuple(map(round, topLeft))
        topRight = tuple(map(round, topRight))
        bottomLeft = tuple(map(round, bottomLeft))
        bottomRight = tuple(map(round, bottomRight))

        return topLeft, topRight, bottomLeft, bottomRight

    def get_annotation_for_image(self, image: ndarray, name: str) -> str:
        """
        Returns the annotation for the given image

        :param image: the ndarray representing the image
        :param name: the name of the image
        :return: the annotation for the image
        """
        matches = self.get_image_matches(image, name)

        annotation = ""

        for count, match in enumerate(matches):
            annotation += f"{match['training_image'].ID}, {match['top_left']}, {match['bottom_right']} \n"

        return annotation

    def show_boxes_around_images(self, images: list[tuple[ndarray, str]]) -> None:
        """
        For each image, finds all matches in this database which appear in the image, drawing a box around each one.

        :param images: the images to search.
        """
        for image, name in images:
            imageMatches = self.get_image_matches(image, name)

            for count, match in enumerate(imageMatches):
                # BGR colour space

                points = list(map(lambda x: [float(x[0]), float(x[1])], [match["top_left"], match["top_right"], match["bottom_right"], match["bottom_left"]]))
                points = np.array(points, dtype=np.int32)
                print(points)

                image = cv.polylines(image, [points], True, (0, 0, 0), 2)
                cv.putText(image, match["training_image"].ID, (match["top_left"][0], match["top_left"][1] - 3) , cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

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
