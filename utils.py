import re, functools

def get_image_name_from_file_name(filename: str) -> str:
    """
    Returns the name of the image given the filename, removing the number and extension.

    e.g.:   023-traffic-light.png -> traffic-light
            036-hotel.png -> hotel
            test_image_5.png -> test_image_5

    :param filename: the name of the image's file
    :return: the image name
    """
    print(filename)

    split = re.split(r'\.|-', filename)

    if len(split) > 2:
        image_name = functools.reduce(lambda a, b: a + "-" + b, split[1:len(split) - 1])
    else:
        image_name = split[0]

    print(image_name)

    return image_name
