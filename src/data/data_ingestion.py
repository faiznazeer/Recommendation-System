import os
import urllib
import zipfile
from src.utils.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config):
        self.data_ingestion_config = data_ingestion_config
        
    def download_data(self):
        try:
            dataset_url = self.data_ingestion_config.dataset_download_url
            zip_download_dir = self.data_ingestion_config.raw_data_dir
            os.makedirs(zip_download_dir, exist_ok=True)
            data_file_name = os.path.basename(dataset_url)
            zip_file_path = os.path.join(zip_download_dir, data_file_name)
            logging.info(f"Downloading data from {dataset_url} into file {zip_file_path}")
            urllib.request.urlretrieve(dataset_url,zip_file_path)
            logging.info(f"Downloaded data from {dataset_url} into file {zip_file_path}")
            return zip_file_path
        except Exception as e:
            raise e

    def extract_zip_file(self, zip_file_path):
        try:
            ingested_dir = self.data_ingestion_config.ingested_dir
            os.makedirs(ingested_dir, exist_ok=True)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(ingested_dir)
            logging.info(f"Extracting zip file: {zip_file_path} into dir: {ingested_dir}")
        except Exception as e:
            raise e