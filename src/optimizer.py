import json
import optuna
import os
import pandas as pd
from pathlib import Path
from model import create_model

from typing import Any
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


class Objective:

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.df: pd.DataFrame = dataset

    def __call__(self, trial, *args: Any, **kwargs: Any) -> float | Any:
        X, y = self.df.drop(columns=["delay"]), self.df["delay"]

        params: dict[str:Any] = {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e0),
            "penalty": trial.suggest_categorical("penalty", ["l2", "elasticnet"]),
            "loss": trial.suggest_categorical("loss", ["modified_huber", "log_loss"]),
        }

        model: Pipeline = create_model(X, y, params)

        score: float = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()

        return score


def main() -> None:
    file_path = Path.cwd().joinpath("data", "processed", "train.parquet")
    param_file = Path.cwd().joinpath("src", "hyperparameters.json")

    dataset: pd.DataFrame = pd.read_parquet(file_path)

    database_url = "sqlite:///flights.db"

    storage = optuna.storages.RDBStorage(
        url=database_url,
        engine_kwargs={"pool_pre_ping": True},
    )
    objective = Objective(dataset=dataset)

    study: optuna.Study = optuna.create_study(
        study_name="Trial",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
    )

    study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=-1)
    if os.path.exists(param_file):
        os.remove(param_file)

    json_object: str = json.dumps(study.best_params)

    with open(param_file, "w") as file:
        file.write(json_object)


if __name__ == "__main__":
    main()
