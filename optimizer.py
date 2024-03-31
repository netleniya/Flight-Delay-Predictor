import json
import optuna
import os
import pandas as pd

from typing import Any
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from sklearn.linear_model import SGDClassifier


class Objective:

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.df = dataset

    def __call__(self, trial, *args: Any, **kwds: Any) -> float | Any:
        X, y = self.df.drop(columns=["delay"]), self.df["delay"]

        num_cols = X.select_dtypes(exclude=["object"]).columns
        cat_cols = X.select_dtypes(include=["object"]).columns

        scalae: list[tuple] = [
            (
                "one_hot",
                OneHotEncoder(sparse_output=True, handle_unknown="infrequent_if_exist"),
                cat_cols,
            ),
            ("scaler", RobustScaler(), num_cols),
        ]
        transformer: ColumnTransformer = ColumnTransformer(transformers=scalae)

        params: dict[str:Any] = {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e0),
            "penalty": trial.suggest_categorical("penalty", ["l2", "elasticnet"]),
            "loss": trial.suggest_categorical("loss", ["modified_huber", "log_loss"]),
        }
        model: Pipeline = Pipeline(
            steps=[
                ("transformer", transformer),
                ("estimator", SGDClassifier(**params)),
            ]
        )
        score: float = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()

        return score


def main() -> None:
    dataset = pd.read_parquet("clean_data.parquet")

    storage = optuna.storages.RDBStorage(
        url="sqlite:///flights.db", engine_kwargs={"pool_pre_ping": True}
    )
    objective = Objective(dataset=dataset)

    study = optuna.create_study(
        study_name="Flight Delays",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
    )

    study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=-1)
    param_file = "hyperparams.json"
    if os.path.exists(param_file):
        os.remove(param_file)

    json_object = json.dumps(study.best_params)

    with open(param_file, "w") as file:
        file.write(json_object)


if __name__ == "__main__":
    main()
