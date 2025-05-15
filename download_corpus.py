import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Dataset reference name
dataset = "suriyaganesh/resume-dataset-structured"
download_path = "data"

# Create folder if not exists
os.makedirs(download_path, exist_ok=True)

# Download dataset
api.dataset_download_files(dataset, path=download_path, unzip=True)

print(f"Dataset downloaded and extracted to '{download_path}' folder.")
