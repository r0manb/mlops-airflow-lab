from pathlib import Path

import pandas as pd
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from core import constants


TEST_SIZE = 0.2


def train_model(input_csv: Path | str) -> Path:
    df = pd.read_csv(input_csv)
    X_train, X_test, y_train, y_test, scaler = _prepare_dataset(df, TEST_SIZE)

    mlflow.set_tracking_uri(f"file://{constants.MLFLOW_RUNS}")
    mlflow.sklearn.autolog(
        log_models=True, log_model_signatures=True, log_input_examples=True
    )
    mlflow.set_experiment("health_risk")
    with mlflow.start_run():
        clf = LogisticRegressionCV(
            Cs=np.logspace(-4, 2, 8),
            cv=3,
            scoring="roc_auc",
            random_state=42,
            n_jobs=-1,
            refit=True,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)

        f1, roc_auc, accuracy, precision, recall = _evaluate_model(
            y_pred, y_pred_proba, y_test
        )

        mlflow.sklearn.log_model(scaler, artifact_path="scaler")

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc_auc)
        mlflow.log_metric("precision_score", precision)
        mlflow.log_metric("recall_score", recall)
        mlflow.log_metric("accuracy", accuracy)


def _prepare_dataset(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df.drop("health_risk", axis=1)
    y = df["health_risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    scaler = MinMaxScaler((0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def _evaluate_model(
    y_pred: np.ndarray, y_pred_proba: np.ndarray, y_true: np.ndarray
) -> tuple[float, float, float, float, float]:
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return f1, roc_auc, accuracy, precision, recall
