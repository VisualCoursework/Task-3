import os
import cv2 as cv
import utils
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

        # Calcualte the annotation for the image based on the training data
        predicted_annotations.append(db.get_annotation_for_image(image, name))

        evaluation = utils.evaluate_annotation([predicted_annotations[-1]], [actual_annotations[-1]])
        print(evaluation)

    # Output the evaluation for this annotation.
    evaluation = utils.evaluate_annotation(predicted_annotations, actual_annotations)
    print(evaluation)
