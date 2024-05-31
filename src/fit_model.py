import json
import os
import pandas as pd
from joblib import dump
from model import create_model
from pathlib import Path
from sklearn.pipeline import Pipeline


def fit_model() -> Pipeline:

    train_df = Path.cwd().joinpath("data", "processed", "train.parquet")
    hyper_params = Path.cwd().joinpath("src", "hyperparameters.json")
    trained_model = Path.cwd().joinpath("models", "model.joblib")

    df: pd.DataFrame = pd.read_parquet(train_df)
    X_train, y_train = df.drop(columns=["delay"]), df["delay"]

    with open(hyper_params, "r") as f:
        hyperparameters: dict = json.load(f)

    model: Pipeline = create_model(X=X_train, y=y_train, params=hyperparameters)

    model.fit(X_train, y_train)

    if os.path.exists(trained_model):
        os.remove(trained_model)
    dump(model, trained_model)


if __name__ == "__main__":
    fit_model()
