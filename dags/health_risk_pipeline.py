import datetime as dt
from pathlib import Path

from airflow.sdk import dag, task

from ml.health_risk import preprocess_dataset, train_model
from core import constants


RAW_DATASET_CSV = constants.RAW_DATA_DIR / "health_risk_dataset.csv"
OUT_DIR = constants.PROCESSED_DATA_DIR / "health_risk"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dag(
    dag_id="health_risk_pipeline",
    dag_display_name="Health Risk Pipeline",
    start_date=dt.datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
)
def health_risk_pipeline() -> None:

    @task()
    def preprocess(run_id: str) -> Path:
        out_path = OUT_DIR / f"{run_id}.csv"

        preprocess_dataset(RAW_DATASET_CSV, out_path)

        return str(out_path)

    @task()
    def train(prep_path: str) -> None:
        train_model(prep_path)

    prep = preprocess()
    train(prep)


dag = health_risk_pipeline()
