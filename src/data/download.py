import os
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw/uci_heart")
RAW_DIR.mkdir(parents=True, exist_ok=True)

UCI_FILES = {
    "processed.cleveland.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "processed.hungarian.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "processed.switzerland.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "processed.va.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}

def download_uci():
    print("Downloading UCI Heart Disease dataset...")
    for filename, url in UCI_FILES.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"  {filename} already exists, skipping")
            continue
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")
    print("Done.")

if __name__ == "__main__":
    download_uci()
