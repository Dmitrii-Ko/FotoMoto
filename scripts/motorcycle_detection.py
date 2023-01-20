"""
Detecting motorcycles on images.

example:
    python3 motorcycle_detection.py "src_files_path" "dst_files_path"
"""

import subprocess
import sys
import os.path

import cv2

from vehicle_detector import VehicleDetector
from image import Image


def get_files(path: str) -> list[str]:
    """
    Gets all files name and extension from directory.

    Returns list of files.
    """

    command = subprocess.Popen(["ls", path], stdout=subprocess.PIPE)
    b_output = command.stdout.read()
    files = b_output.decode().split()

    return files


def detect_motorcycles_in_folder(src_files_path: str, dst_files_path: str):
    """
    Gets all images from `src_files_path`
    and detects all motorcycles on images.

    Saves all detected motorcycles in `dst_files_path`.
    """

    # Source files
    files = get_files(src_files_path)

    # Already processed files
    processed_files = set(
        [element.split(".")[0] for element in get_files(dst_files_path)]
    )

    for index, file in enumerate(files):
        print(f"{index}/{len(files)} file is processing ...")

        file_name, file_extension = file.split(".")[:2]

        # Skip processed files
        if f"{file_name}_0" in processed_files:
            continue

        # Open new image
        img = cv2.imread(f"{src_files_path}/{file}")
        image = Image(file_name)
        image.set_image(img)

        # Detecting
        motorcycles = detect_motorcycles(image)

        # Save each motorcycle from image
        for motorcycle, pnt_x, pnt_y in motorcycles:
            name = f"{dst_files_path}/{motorcycle.name}.{file_extension}"
            cv2.imwrite(name, motorcycle.img)
            print(f"{name} - OK")


def detect_motorcycles(image) -> [(Image, int, int)]:
    """
    Detects all motorcycles on the picture.
    Returns array of images.
    """

    # Create detector
    vehicle_detector = VehicleDetector()

    # Detecting
    motorcycles_images = []
    motorcycles = vehicle_detector.detect_vehicles(image)

    for index, (pnt_x, pnt_y, width, height) in enumerate(motorcycles):
        # Cut image by point and shape
        motorcycle = Image(f"{image.name}_{index}")
        motorcycle.set_image(image.img[pnt_y : pnt_y + height, pnt_x : pnt_x + width])

        # Save cutted image
        motorcycles_images.append((motorcycle, pnt_x, pnt_y))
    return motorcycles_images


if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit("2 arguments are required ...")

    path_from, path_to = sys.argv[1:]

    if not all([os.path.exists(path_from), os.path.exists(path_to)]):
        exit("Some path doesn't exist...")

    detect_motorcycles_in_folder(path_from, path_to)
    exit("Processed...")
