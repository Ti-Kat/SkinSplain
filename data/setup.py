import os
import requests
import zipfile
from urllib.parse import urlparse
import shutil


IMAGE_DATA_DICT = {
    "2018": (
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip",
    ),
    "2019": (
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
    ),
    "2020": (
        "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip",
    )
}


def download_and_extract_zip(url, extract_to):
    # Parse the URL to get the file name
    file_name = os.path.basename(urlparse(url).path)
    zip_file_path = os.path.join(extract_to, file_name)
    
    # Download the zip file
    print(f"Downloading {file_name} from {url}...")
    response = requests.get(url, stream=True)
    with open(zip_file_path, 'wb') as file:
        shutil.copyfileobj(response.raw, file)
    print(f"Downloaded {file_name} to {zip_file_path}")

    # Extract the contents of the zip file
    print(f"Extracting {file_name}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {file_name}")

    # Delete the original zip file
    os.remove(zip_file_path)
    print(f"Deleted {zip_file_path}")

def process_zip_files(urls, extract_to):
    # Make extract_to relative to the script's directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    extract_to = os.path.join(script_dir, extract_to)
    
    os.makedirs(extract_to, exist_ok=True)
    for url in urls:
        download_and_extract_zip(url, extract_to)

if __name__ == "__main__":
    for extract_to, urls in IMAGE_DATA_DICT.items():
        process_zip_files(urls, extract_to)
