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

    # Now perform analysis of the non-rotated test images.
    for image, name in testImagesNoRotation:
        # Read in the annotation for this test image
        with open(f"TestWithoutRotations/annotations/{name}.txt") as f:
            actual_annotations.append(f.read())

        # Calculate the annotation for the image based on the training data
        predicted_annotations.append(db.get_annotation_for_image(image, name))

        db.show_boxes_around_images([(image, name)])

        print(f"No rotation: {name}")
        # A bit awkward, but we need to pass in a list of annotations to evaluate_annotations, so we wrap the last
        # element in a list.
        print(utils.evaluate_annotations([predicted_annotations[-1]], [actual_annotations[-1]], 0.5))
        print()

    print("All rotations:")
    print(utils.evaluate_annotations(predicted_annotations, actual_annotations, 0.5))
    print()

    # Next perform analysis of the rotated test images.
    for image, name in testImagesWithRotations:
        # Read in the annotation for this test image
        with open(f"Task3AdditionalTestDataset/annotations/{name}.csv") as f:
            actual_annotations.append(f.read())

        # Calculate the annotation for the image based on the training data
        predicted_annotations.append(db.get_annotation_for_image(image, name))

        db.show_boxes_around_images([(image, name)])

        print(f"With rotation: {name}")
        # A bit awkward, but we need to pass in a list of annotations to evaluate_annotations, so we wrap the last
        # element in a list.
        print(utils.evaluate_annotations([predicted_annotations[-1]], [actual_annotations[-1]], 0.5))
        print()

    print("Overall:")
    print(utils.evaluate_annotations(predicted_annotations, actual_annotations, 0.5))

    # Now plot the accuracy, recall and precision against the IoU threshold.
    accuracy = []
    precision = []
    recalls = []
    thresholds = []

    for threshold_factor in [0.01 * x for x in range(1, 101)]:
        accuracy.append(utils.evaluate_annotations(predicted_annotations, actual_annotations, threshold_factor)["ACC"])
        recalls.append(utils.evaluate_annotations(predicted_annotations, actual_annotations, threshold_factor)["recall"])
        precision.append(utils.evaluate_annotations(predicted_annotations, actual_annotations, threshold_factor)["precision"])

        thresholds.append(threshold_factor)

    # Plot accuracy, recall and precision vs threshold:
    plt.plot(thresholds, accuracy, label="Accuracy")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, precision, label="Precision")
    plt.xlabel("IoU Threshold")
    plt.ylabel("Accuracy / Recall / Precision")
    plt.legend()
    plt.show()



