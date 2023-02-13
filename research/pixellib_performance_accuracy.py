"""
Find out performance and accurancy for pixellib on photos with motorcycles.
"""

import os
import json
import time
import pixellib 
from pixellib.torchbackend.instance import instanceSegmentation

# Source photos path
FILES_PATH = ".."

# Path where photos will be saved
RAPID_PATH = ".."
FAST_PATH = ".."
AVERAGE_PATH = ".."

# Different tests
TESTS = {"fast": FAST_PATH, "rapid": RAPID_PATH, "average": AVERAGE_PATH}

# Performance data
JSON_PATH = ".."

# Model for cutting (pointrend_resnet50.pkl)
MODEL_PATH = ".."

# Files from source path
files = [element for element in os.popen(f"ls {FILES_PATH}").read().split()]

# Loading info about already processed photos
json_performance = []
with open(JSON_PATH, "r") as file:
    json_performance = json.loads(file.read())

# Getting processed photos
processed_files = [d["name"] for d in json_performance]

try:
    for index, file in enumerate(files):
        # Skip files that have been already processed
        if file in processed_files:
            continue

        print(f"\U0001F973 {index}/{len(files)}. processing {file}..")

        # Prepare tabel for writting results
        speed_results = {}

        for test in TESTS:
            speed_results[test] = {}

            # Init and load model
            ins = instanceSegmentation()
            ins.load_model(MODEL_PATH, detection_speed=test)

            # Processing image
            start_time = time.time()
            result, output = ins.segmentImage(
                f"{FILES_PATH}/{file}",
                show_bboxes=True,
                mask_points_values=True,
                text_size=4,
                text_thickness=10,
                box_thickness=4,
                output_image_name=f"{TESTS[test]}/{file}",
            )
            end_time = time.time()

            # Save results
            speed_results[test]["time"] = end_time - start_time
            speed_results[test]["object_counts"] = dict(result["object_counts"])

            print(f"\U0001F525 {test} test is done for {file}..")

        # Add results to the table
        json_performance.append({"name": file, **speed_results})
except BaseException as e:
    # Save all data if some exceptions
    print(e)

# Saving data
with open(JSON_PATH, "w") as file:
    file.write(json.dumps(json_performance, indent=4))
