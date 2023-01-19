"""
Image analysis to construct a normalized vector with 64 values containing color data.
It will be used to build clustering of images by color.
"""

# Global variables, to use script change this data
CSV_NAME = "... .csv"
FILES_PATH = "..."

import itertools
import os
import csv
import cv2
import numpy as np


def get_image_color_analysis(image, divider=64):
    """
    Implementation of color reduction algorithm
    with analysis by building normalize vector.
    """

    # Generate dictionary of all colors
    colors_counting = {
        xs: 0 for xs in itertools.product(range(0, 256, divider), repeat=3)
    }

    # Color reduction
    image = image // divider * divider

    # Count color of each pixel
    for row in image:
        for pixel in row:
            colors_counting[tuple(pixel)] += 1

    # Normalize vector
    width, height, _ = image.shape
    weights = [
        colors_counting[color] / (width * height)
        for color in sorted(colors_counting.keys())
    ]

    return colors_counting, weights


# Save processed image to skip it in main loop
processed_images = []
with open(CSV_NAME, "r", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=";", quoting=csv.QUOTE_NONE)
    for row in reader:
        processed_images.append(row[0])

# Reading list of files from some path
files = [
    tuple(element.split(".")) for element in os.popen(f"ls {FILES_PATH}").read().split()
]

files.sort()

data = []
try:
    for index, (file_name, file_extension) in enumerate(files):
        # Skip processed images
        if f"{file_name}.{file_extension}" in processed_images:
            print(f"{file_name} has already processed ...")
            continue

        # Reading file
        image_path = f"{FILES_PATH}/{file_name}.{file_extension}"
        img = cv2.imread(image_path)

        # Analysis
        _, weight = get_image_color_analysis(img)
        data.append([f"{file_name}.{file_extension}", *weight])

        print(f"{index}/{len(files)} processed files ...")
except KeyboardInterrupt:
    # Saving data is the main priority,
    # if the program keyboard interrupt, the script will save the data

    ...

# Writing data into csv
with open(CSV_NAME, "a", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=";", quoting=csv.QUOTE_NONE)
    for row in data:
        writer.writerow(row)
