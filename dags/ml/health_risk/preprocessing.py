from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


EXERCISE_MAP = {"none": 0, "low": 1, "medium": 2, "high": 3}
SUGAR_INTAKE_MAP = {k: v - 1 for k, v in EXERCISE_MAP.items() if k != "none"}
BIN_COLS = ["smoking", "alcohol", "married"]


def preprocess_dataset(input_csv: Path | str, out_csv: Path | str) -> None:
    df = pd.read_csv(input_csv)

    df.drop("height", axis=1, inplace=True)

    df["exercise"] = df["exercise"].map(EXERCISE_MAP).astype("int8")
    df["sugar_intake"] = df["sugar_intake"].map(SUGAR_INTAKE_MAP).astype("int8")

    df[BIN_COLS] = df[BIN_COLS].replace({"yes": 1, "no": 0}).astype("int8")
    df["health_risk"] = (df["health_risk"] == "high").astype("int8")

    encoder = OneHotEncoder(sparse_output=False)
    encoded_professions = encoder.fit_transform(df[["profession"]])

    encoded_df = pd.DataFrame(
        encoded_professions,
        columns=encoder.get_feature_names_out(["profession"]),
        dtype="int8",
    )
    df = pd.concat([df, encoded_df], axis=1)
    df.drop("profession", axis=1, inplace=True)

    df.to_csv(out_csv)
