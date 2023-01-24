"""
Checks how YOLOV4 recognizes motorcycles in the photo 
and compares them with the marked data.
"""

# The source path where all photos are stored
SRC_FILE_PATH = "../data/motorcycles_photos"

# Already processed data, intermediate results
PROCESSED_JSON = "./motorcycle_counting.json"

# All data
ROW_JSON = "./detected_data.json"

import json
from sys import stdout
from image import Image
from motorcycle_detection import detect_motorcycles


def logging(filename, index, size, progressbar_length=30):
    # Progress bar

    interval = index // (size // progressbar_length)
    print("\u001b[100D", end="")
    print(
        f"{filename} is processing: "
        f"[{'#' * interval}{' ' * (progressbar_length - interval)}] "
        f"{round(100 * index / size):2d}%",
        end="",
    )
    stdout.flush()


def parsing_info(element: dict) -> tuple[str, list]:
    """
    Parses json file to get name of file
    and all information about captions.
    """

    captions = element["caption"].split(";")
    filename = element["captioning"].split("/")[-1].split("-")[-1]

    return filename, captions


def add_result(
    data: list, filename: str, caption: str, detector_result: int, points: list[tuple]
):
    """
    Adds new information about the difference
    between the detector and the marked data.
    """

    data.append(
        {
            "filename": filename,
            "caption": caption,
            "detector_result": detector_result,
            "caption_result": len(caption),
            "decision": "OK" if detector_result >= len(caption) else "FAIL",
            "point_x": [int(x) for x, _ in points],
            "point_y": [int(y) for _, y in points],
        }
    )


# Reading processed json
processed_elements = json.loads(open(PROCESSED_JSON, "r").read())

# Saving all names of processed files
processed_files = [element["filename"] for element in processed_elements]

# Reading unprocessed (row) json
row_elements = json.loads(open(ROW_JSON, "r").read())

try:
    for index, element in enumerate(row_elements[:40]):

        filename, captions = parsing_info(element)

        logging(filename, index + 1, len(row_elements))

        # Skipping processed files
        if filename in processed_files:
            continue

        # Reading image
        image = Image()
        image.open_image(f"{SRC_FILE_PATH}/{filename}")

        # Detecting motocycles
        motorcycles = detect_motorcycles(image)

        # Saving results
        add_result(
            processed_elements,
            filename,
            captions,
            len(motorcycles),
            [(x, y) for _, x, y in motorcycles],
        )

except BaseException as e:
    # Saves results if were some exceptions

    print(e)

print("Saving results ...")

# Saving results
with open(PROCESSED_JSON, "w") as file:
    json.dump(processed_elements, file, indent=4)

