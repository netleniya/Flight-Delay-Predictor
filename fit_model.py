import json
import os
import pandas as pd
from joblib import dump, load
from model import create_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_dataset() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_parquet("clean_data.parquet")
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["delay"]), df["delay"], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def fit_model() -> Pipeline:
    X_train, _, y_train, _ = load_dataset()

    with open("hyperparams.json", "r") as f:
        hyperparameters: dict = json.load(f)

    model: Pipeline = create_model(X=X_train, y=y_train, params=hyperparameters)

    model.fit(X_train, y_train)

    if os.path.exists("model.joblib"):
        pass
    else:
        dump(model, "model.joblib")


if __name__ == "__main__":
    fit_model()
