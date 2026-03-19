from pathlib import Path 
import urllib.request 
from zipfile import ZipFile 
import os

def download_and_extract(dataset_url, zipfile_path, extract_to):
    '''Download dataset zip file and extract it to the specified directory'''
    zipfile_path = Path(zipfile_path)
    extract_to = Path(extract_to) 

    if not zipfile_path.exists():
        urllib.request.urlretrieve(dataset_url, zipfile_path)
    
    if not extract_to.exists():
        extract_to.mkdir(parents=True, exist_ok=True)
        with ZipFile(zipfile_path, 'r') as archive:
            archive.extractall(extract_to)
        
        zipfile_path.unlink()