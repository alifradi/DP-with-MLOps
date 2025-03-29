#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fetch Data"""

import os
import logging
import json
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def setup_kaggle():
    """Configure Kaggle API credentials"""
    try:
        config_path = Path("config/kaggle.json")
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        # Copy kaggle.json to standard location
        with open(config_path, "r", encoding="utf-8") as src:
            content = json.load(src)
            with open(kaggle_dir / "kaggle.json", "w", encoding="utf-8") as dest:
                json.dump(content, dest)
        
        # Set strict permissions (important for Kaggle)
        os.chmod(kaggle_dir / "kaggle.json", 0o600)
        
        logger.info("Kaggle credentials configured successfully")
        return True
    except Exception as e:
        logger.error(f"Configuration failed: {str(e)}")
        return False

def download_dataset():
    """Download and extract the multimodal dataset"""
    dataset = "linweitao/multi-label-classification-competition-2023"
    raw_data_dir = Path("data/multimodal/raw")
    
    try:
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading dataset...")
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", dataset.split('/')[-1], "-p", str(raw_data_dir)],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        logger.info(result.stdout)
        
        # Use Python's zipfile module for extraction
        zip_file = raw_data_dir / f"{dataset.split('/')[-1]}.zip"
        logger.info(f"Extracting {zip_file}...")
        
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(raw_data_dir)
            
        zip_file.unlink()
        logger.info("Dataset downloaded and extracted successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle error: {e.stderr if e.stderr else e.stdout}")
        return False
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return False

def main():
    logger.info("Starting data fetch process")
    
    if not setup_kaggle():
        logger.error("Failed to configure Kaggle API")
        return
    
    if download_dataset():
        logger.info("Data fetch completed successfully")
    else:
        logger.error("Data fetch failed")

if __name__ == "__main__":
    main()