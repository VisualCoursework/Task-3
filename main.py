import os
import cv2 as cv
from FeatureDatabase import FeatureDatabase

if __name__ == "__main__":
    print("Hello, World!")

    db = FeatureDatabase(1, 1)

    trainingImages = [(cv.imread("Training/png/" + imagePath), imagePath) for imagePath in os.listdir("Training/png/")]
    db.add_training_images(trainingImages)

    testImages = [cv.imread("TestWithoutRotations/images/" + imagePath) for imagePath in os.listdir("TestWithoutRotations/images/")]
    db.show_matches(testImages)
