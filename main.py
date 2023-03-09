import os
from FeatureDatabase import FeatureDatabase

if __name__ == "__main__":
    print("Hello, World!")

    db = FeatureDatabase(1, 1)

    images = ["Training/png/" + imagepath for imagepath in os.listdir("Training/png/")]
    print(images)

    db.add_training_images(images)
