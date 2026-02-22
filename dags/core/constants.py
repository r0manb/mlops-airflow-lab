import os
from pathlib import Path

ROOT_DIR = Path(
    os.environ.get("AIRFLOW_HOME", str(Path(__file__).resolve().parents[2]))
)
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
OUT_DATA_DIR = DATA_DIR / "out"


MLFLOW_RUNS = ROOT_DIR / "mlruns"
