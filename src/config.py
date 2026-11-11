"""Central configuration and paths for LumiRestore AI."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MANIFEST_DIR = DATA_DIR / "manifests"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

BSDS300_URL = (
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/"
    "BSDS300-images.tgz"
)
BSDS300_ARCHIVE = RAW_DATA_DIR / "BSDS300-images.tgz"
BSDS300_ROOT = RAW_DATA_DIR / "BSDS300"
MANIFEST_PATH = MANIFEST_DIR / "bsds300_source_split.csv"
MODEL_PATH = MODELS_DIR / "lumirestore_residual_unet.pt"
METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
HISTORY_PATH = REPORTS_DIR / "training_history.csv"

RANDOM_STATE = 2024
PATCH_SIZE = 128
BASE_CHANNELS = 24
TRAIN_PATCHES_PER_IMAGE = 3
EVAL_PATCHES_PER_IMAGE = 1


def create_project_directories() -> None:
    """Create every directory that scripts may write to."""
    for directory in (
        DATA_DIR,
        RAW_DATA_DIR,
        MANIFEST_DIR,
        ASSETS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
