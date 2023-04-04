import os
import cv2 as cv
import utils
from FeatureDatabase import FeatureDatabase

if __name__ == "__main__":
    print("Hello, World!")

    db = FeatureDatabase(1, 1)

    trainingImages = [(cv.imread("Training/png/" + imagePath), utils.get_image_name_from_file_name(imagePath))
                      for imagePath in os.listdir("Training/png/")]
    db.add_training_images(trainingImages)

    testImages = [(cv.imread("TestWithoutRotations/images/" + imagePath), utils.get_image_name_from_file_name(imagePath))
                  for imagePath in os.listdir("TestWithoutRotations/images/")]
    # testImages += [(cv.imread("Task3AdditionalTestDataset/images/" + imagePath), imagePath) for imagePath in os.listdir("Task3AdditionalTestDataset/images/")]
    db.print_annotations(testImages)
    # db.show_matches_for_images(testImages)


