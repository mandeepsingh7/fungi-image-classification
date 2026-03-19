import os
import shutil 
import pandas as pd 
from pathlib import Path 
from sklearn.model_selection import train_test_split 
from shutil import copy 

def get_dataframe_from_raw(path: str) -> pd.DataFrame:
	"""Create a dataframe with `image_path` and `image_label` from raw data."""
	data = []
	for root, _, files in os.walk(path):
		for file in files:
			image_label = file[:2]
			image_path = os.path.join(root, file)
			new_row = {"image_path": image_path, "image_label": image_label}
			data.append(new_row)
	return pd.DataFrame(data)

def create_data_dirs(base_dir: str, raw_dir: str):
	"""Create train/test/valid subdirectories and class subdirectories within them
	so we can populate them later."""
	classes = sorted([
		d for d in os.listdir(raw_dir)
		if os.path.isdir(os.path.join(raw_dir, d))
	])

	base_dir = Path(base_dir)
	for split in ['train', 'test', 'valid']:
		for cls in classes:
			(base_dir / split / cls).mkdir(parents=True, exist_ok=True)

def split_and_copy(df, base_dir: str, test_size: float, valid_size: float, seed: int = 42):
	'''Split dataset and copy files to respective directories.'''
	train_val, test = train_test_split(
		df, stratify=df['image_label'], test_size=test_size, random_state=seed
	) 

	train, valid = train_test_split(
		train_val, stratify=train_val['image_label'], test_size=valid_size / (1 - test_size), random_state=seed
	)

	for split_name, split_df in zip(['train', 'valid', 'test'], [train, valid, test]):
		for _, row in split_df.iterrows():
			dst = Path(base_dir) / split_name / row['image_label']
			copy(row['image_path'], dst)

def prepare_dataset_from_raw(raw_dir: str, data_dir: str, test_size: float = 0.15, valid_size: float = 0.15, seed: int = 42):
	'''Prepare dataset from raw data by creating directories and splitting files.'''
	if os.path.exists(data_dir):
		print(f"Data directory {data_dir} already exists. Skipping preparation.")
		return

	df = get_dataframe_from_raw(raw_dir)
	create_data_dirs(data_dir, raw_dir)
	split_and_copy(df, data_dir, test_size, valid_size, seed)

	shutil.rmtree(raw_dir)