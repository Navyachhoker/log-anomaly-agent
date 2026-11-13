"""Paths and constants shared by all scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
try:  # .env holds local settings and secrets (never committed). Optional on Colab.
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
SAMPLES_FILE = ROOT / "samples" / "samples.json"
MODEL_DIR = Path(os.getenv("LOGAGENT_MODEL_DIR", ROOT / "models" / "merged"))
THRESHOLD_FILE = RESULTS_DIR / "threshold.json"

# Official LogHub download link for HDFS_v1 (listed on https://github.com/logpai/loghub)
HDFS_URL = "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
EXPECTED_LINES = 11_175_629  # line count stated by LogHub for HDFS_v1 (used as a cross-check)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
MAX_SNIPPET_TOKENS = 768  # chosen from the real data (results/data_inspection.json)