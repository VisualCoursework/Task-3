import os
import cv2 as cv
import utils
import matplotlib.pyplot as plt
from FeatureDatabase import FeatureDatabase

OCTAVE_LAYERS = 3
MATCH_THRESHOLD = 250
POINTS_TO_MATCH = 15
INCLUDE_ROTATIONS = False

if __name__ == "__main__":
    # First create the lists of directories for images.
    trainingImages = [(cv.imread("Training/png/" + imagePath), utils.get_image_name_from_file_name(imagePath))
                      for imagePath in os.listdir("Training/png/")]

    testImagesNoRotation = [(cv.imread("TestWithoutRotations/images/" + imagePath), utils.get_image_name_from_file_name(imagePath))
                    for imagePath in os.listdir("TestWithoutRotations/images/")]

    testImagesWithRotations = [(cv.imread("Task3AdditionalTestDataset/images/" + imagePath), utils.get_image_name_from_file_name(imagePath)) for imagePath in
                    os.listdir("Task3AdditionalTestDataset/images/")]

    # Create the database and add all the training images to it.
    db = FeatureDatabase(OCTAVE_LAYERS, MATCH_THRESHOLD, POINTS_TO_MATCH)
    db.add_training_images(trainingImages)

    predicted_annotations = []
    actual_annotations = []

    # Now perform analysis of the test images.
    for image, name in testImagesNoRotation:
        # Read in the annotation for this test image
        with open(f"TestWithoutRotations/annotations/{name}.txt") as f:
            actual_annotations.append(f.read())

        # Calculate the annotation for the image based on the training data
        predicted_annotations.append(db.get_annotation_for_image(image, name))

    # Now create a matplotlib plot of the results by varying the IoU threshold.
    recalls = []
    thresholds = []

    for threshold_factor in [0.01 * x for x in range(1, 101)]:
        recalls.append(utils.evaluate_annotation(predicted_annotations, actual_annotations, threshold_factor)["recall"])
        thresholds.append(threshold_factor)

    # Plot recall vs threshold:
    plt.plot(thresholds, recalls)
    plt.xlabel("IoU Threshold")
    plt.ylabel("Recall")
    plt.title("Recall vs IoU Threshold")
    plt.show()

    print(recalls)
    print(thresholds)

