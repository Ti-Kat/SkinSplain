import csv
import os
from .path_constants import EVAL_CSV_PATH


def write_csv_row(data):
    """
    Write a row of data to a CSV file. If the file doesn't exist, it will be created.

    Args:
    - data (list): The data to be written as a new row in the CSV file.
    """
    file_exists = os.path.exists(EVAL_CSV_PATH)

    with open(EVAL_CSV_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['image_name', 'age_approx', 'sex', 'ground_truth', 'prediction'])  # Example header row
        writer.writerow(data)

