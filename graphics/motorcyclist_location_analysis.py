"""
Frequency analysis to determine the optimal 
coefficients for the method of determining 
the location of the motorcyclist relative to the observer.

The way is to get the shape of the photo 
and divide the width by the height of the image.

All results are read from the keyboard (IO), 
the researcher must press the button 
to determine the type of photo.

The types are listed in the header (PHOTO_TYPES).
Change FILES_PATH before usage.
"""

# Types of turning by motorcyclist
# {type : color}
PHOTO_TYPES = {1: "r", 2: "b", 3: "g"}

# Path of photos
FILES_PATH = "..."

import os
import subprocess
from typing import Optional

import cv2
import matplotlib.pyplot as plt


def add_data(data: list, image, t: int):
    """
    Adds the ratio of the width and the height of the image in array.
    """

    height, width, channel = image.shape
    data.append((width / height, t))


def set_photo_type(image) -> Optional[str]:
    """
    Reads the pressed button on the keyboard to detect
    what is this type of photo. All types are listed in the file header.

    ESC (27) means skip the file.
    """

    cv2.imshow("T", image)

    while True:
        key = cv2.waitKey(1)

        if key == 27:
            return None
        elif key > 0 and (b := chr(key)) in PHOTO_TYPES:
            return b


def build_graphics(data: list):
    """
    Builds graphics by data.
    """

    for v, t in data:
        plt.plot(v, t, marker="o", color=PHOTO_TYPES[t], label=f"{t}")
    plt.show()


if not os.path.exists(FILES_PATH):
    raise Exception("The path does not exist!")

# Gets all files from the directory.
files = [element for element in os.popen(f"ls {FILES_PATH}").read().split()]

if not files:
    raise Exception("No files in the directory!")

data = []

for index, file in enumerate(files):

    # Read image
    img = cv2.imread(f"{FILES_PATH}/{file}")

    # Try to find categoty (type)
    photo_categoty = file.split("_")[0]

    # If type convert from string to integer
    if photo_categoty.isnumeric():
        photo_categoty = int(photo_categoty)

    # ICheck in the list
    if photo_categoty in PHOTO_TYPES:
        print(f"{file} has already got type..")
        add_data(data, img, photo_categoty)
        continue

    # Read button from keyboard
    button = set_photo_type(img)
    if button:
        print(f"{file} got {button} type..")

        # Rename file to skip for next analysis
        subprocess.call(
            [
                "mv",
                f"{FILES_PATH}/{file}",
                f"{FILES_PATH}/{button}_{file}",
            ]
        )

        add_data(data, img, int(button))
    else:
        print(f"{file} was skipped..")

    print(f"{index + 1}/{len(files)} processed..")

# Output results
build_graphics(data)

