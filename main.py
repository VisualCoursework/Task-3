import os
import cv2 as cv
from FeatureDatabase import FeatureDatabase

if __name__ == "__main__":
    print("Hello, World!")

    db = FeatureDatabase(1, 1)

    trainingImages = [(cv.imread("Training/png/" + imagePath), imagePath) for imagePath in os.listdir("Training/png/")]
    db.add_training_images(trainingImages)

    testImages = [cv.imread("TestWithoutRotations/images/" + imagePath) for imagePath in os.listdir("TestWithoutRotations/images/")]
    # testImages += [cv.imread("Task3AdditionalTestDataset/images/" + imagePath) for imagePath in os.listdir("Task3AdditionalTestDataset/images/")]
    db.show_boxes_around_images(testImages)
    # db.show_matches_for_images(testImages)
