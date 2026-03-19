from pathlib import Path 


BASE_DIR = Path(__file__).resolve().parent.parent 

# DATA 

DATASET_URL = "https://archive.ics.uci.edu/static/public/773/defungi.zip"

RAW_DATA_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

TRAINING_RUNS_DIR = BASE_DIR / "training_runs"

# TRAINING 
BATCH_SIZE = 16 
DROPOUT = 0.2 

# LOGS 
LOG_DIR = BASE_DIR / "logs"

# MEAN and STD in denormalize 
# Using IMAGENET mean and std 
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

