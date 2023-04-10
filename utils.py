import functools
import re

def convert_annotation_to_dict(annotation: str) -> dict:
    """
    Converts the given annotation (in the format that is being used for this assignment) into a dictionary of
    coordinate pairs.

    :param annotation: the annotation to convert.
    :return: the dictionary.
    """
    output = {}

    # First remove the brackets from the coords, so we can treat this as a CSV
    annotation = annotation.replace("(", "")
    annotation = annotation.replace(")", "")

    for line in annotation.split("\n"):
        if len(line) == 0: continue

        line = line.split(",")

        # A messy unpacking, but put a record in the output of the name of the image (line[0]) and store two tuples
        # of the top left and bottom right coords.
        output[line[0]] = [(line[1], line[2]), (line[3], line[4])]

    return output


def evaluate_single_annotation(predicted: str, actual: str, IoU_threshold: float) -> tuple:
    """
    Evaluates a single annotation, returning the number of false positives and true positives.

    :param predicted: the predicted annotation.
    :param actual: the ground truth annotation.
    :param IoU_threshold: the IoU threshold to use.
    :return: the number of false positives, true positives, false negatives and true negatives.
    """
    predicted = convert_annotation_to_dict(predicted)
    actual = convert_annotation_to_dict(actual)

    FPR = TPR = 0

    for name, points in predicted.items():
        # First handle the false positives which aren't even in the ground truth.
        if name not in actual:
            FPR += 1
        else:
            # Now handle the false positives which are in the ground truth, but are not a good match (according to the
            # IoU).
            predicted_box = points
            actual_box = actual[name]
            xA = max(int(predicted_box[0][0]), int(actual_box[0][0]))
            yA = max(int(predicted_box[0][1]), int(actual_box[0][1]))
            xB = min(int(predicted_box[1][0]), int(actual_box[1][0]))
            yB = min(int(predicted_box[1][1]), int(actual_box[1][1]))
            interArea = max(0, xB - xA) * max(0, yB - yA)
            predicted_box_area = (int(predicted_box[1][0]) - int(predicted_box[0][0])) * (
                    int(predicted_box[1][1]) - int(predicted_box[0][1]))
            actual_box_area = (int(actual_box[1][0]) - int(actual_box[0][0])) * (
                    int(actual_box[1][1]) - int(actual_box[0][1]))
            iou = interArea / float(predicted_box_area + actual_box_area - interArea)

            # If the IoU exceeds a threshold, then it is a true positive. Otherwise, we classify as a false positive.
            if iou > IoU_threshold:
                TPR += 1
            else:
                FPR += 1

    FNR = len(actual) - TPR  # False negatives are any which are missed.
    TNR = 50 - (FPR + TPR)  # True negatives are any which are not predicted. 50 is used as there are 50 images in the training set.

    return FPR, TPR, FNR, TNR


def evaluate_annotation(predicted: list[str], actual: list[str], IoU_threshold: float) -> dict:
    """
    Creates an evaluation dictionary for the two given annotations.

    :param predicted: the annotation produced by the model.
    :param actual: the correct annotation.
    :param IoU_threshold: the IoU threshold to use.
    :return: the evaluation.
    """
    evaluation = {
        "FPR": 0,
        "TPR": 0,
        "FNR": 0,
        "TNR": 0
    }

    # Iterate over the predicted and actual annotations, and evaluate each one.
    for predicted, actual in zip(predicted, actual):
        FPR, TPR, FNR, TNR = evaluate_single_annotation(predicted, actual, IoU_threshold)
        evaluation["FPR"] += FPR
        evaluation["TPR"] += TPR
        evaluation["FNR"] += FNR
        evaluation["TNR"] += TNR

    # Calculate metrics for the given datasets.
    evaluation["ACC"] = (evaluation["TPR"] + evaluation["TNR"]) / (evaluation["TPR"] + evaluation["TNR"] + evaluation["FPR"] + evaluation["FNR"])
    evaluation["precision"] = evaluation["TPR"] / (evaluation["TPR"] + evaluation["FPR"])
    evaluation["recall"] = evaluation["TPR"] / (evaluation["TPR"] + evaluation["FNR"])

    return evaluation


def get_image_name_from_file_name(filename: str) -> str:
    """
    Returns the name of the image given the filename, removing the number and extension.

    e.g.:   023-traffic-light.png -> traffic-light
            036-hotel.png -> hotel
            test_image_5.png -> test_image_5

    :param filename: the name of the image's file
    :return: the image name
    """
    split = re.split(r'\.|-', filename)

    if len(split) > 2:
        image_name = functools.reduce(lambda a, b: a + "-" + b, split[1:len(split) - 1])
    else:
        image_name = split[0]

    return image_name
