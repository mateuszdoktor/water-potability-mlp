from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "water_potability.csv"
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_net.pth"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

SEED = 42
BATCH_SIZE = 64
EPOCH_NUM = 100
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
SHOW_PLOTS = False

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2