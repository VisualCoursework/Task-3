import functools
import re


def evaluate_annotation(predicted: str, actual: str) -> dict:
    """
    Creates an evaluation dictionary for the two given annotations.

    :param predicted: the annotation produced by the model.
    :param actual: the correct annotation.
    :return: the evaluation.
    """
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

    evaluation = {
        "FPR": 0,
        "TPR": 0,
    }

    predicted = convert_annotation_to_dict(predicted)
    actual = convert_annotation_to_dict(actual)

    for name, points in predicted.items():
        # First handle the false positives
        if name not in actual:
            evaluation["FPR"] += 1
        else:
            evaluation["TPR"] += 1

    evaluation["FNR"] = len(actual) - evaluation["TPR"]  # False negatives are any which are missed.
    evaluation["TNR"] = 50 - len(predicted)
    evaluation["ACC"] = (evaluation["TPR"] + evaluation["TNR"]) / (evaluation["TPR"] + evaluation["TNR"] + evaluation["FPR"] + evaluation["FNR"])

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
