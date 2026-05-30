from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "Durian_Leaf_Diseases"

TRAIN_ROOT = DATA_ROOT / "train"
VAL_ROOT = DATA_ROOT / "val"
TEST_ROOT = DATA_ROOT / "test"

TRAIN_CSV = TRAIN_ROOT / "metadata_train.csv"
VAL_CSV = VAL_ROOT / "metadata_val.csv"
TEST_CSV = TEST_ROOT / "metadata_test.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULT_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

NUM_DISEASE_CLASSES = 5
NUM_SEVERITY_CLASSES = 4

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = "cuda"