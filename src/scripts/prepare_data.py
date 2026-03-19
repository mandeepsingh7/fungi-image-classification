import argparse 
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import DATA_DIR, DATASET_URL, RAW_DATA_DIR 
from src.data.download import download_and_extract
from src.data.dataset import prepare_dataset_from_raw 


def parse_args():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent 

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset-url',
        type=str,
        default=DATASET_URL,
        help='URL of the dataset to download'
    )

    parser.add_argument(
        '--raw_dir',
        type=str,
        default=str(RAW_DATA_DIR),
        help='Directory to save the raw data'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default=str(DATA_DIR),
        help='Directory to save the processed data'
    )

    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Dataset URL: {args.dataset_url}")
    print(f"Raw data directory: {args.raw_dir}")
    print(f"Processed data directory: {args.data_dir}")

    print("Downloading dataset...")
    download_and_extract(
        args.dataset_url,
        'data.zip',
        args.raw_dir
    )

    print("Preparing dataset...")
    prepare_dataset_from_raw(
        args.raw_dir,
        args.data_dir
    )

    print('Dataset preparation complete.')

if __name__ == '__main__':
    main()

