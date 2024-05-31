import os
import numpy as np
import pandas as pd
from pathlib import Path


datafile = Path.cwd().joinpath("data", "raw", "Airlines.csv")


def make_dataset() -> None:
    """
    Generates a dataset for training and testing a machine learning model.

    The function reads a CSV file containing flight data, preprocesses the data, and
    saves the resulting train and test datasets as Parquet files.

    The preprocessing steps include:
    - Converting column data types to more efficient types
    - Extracting a "flight" column from the "Airline" and "Flight" columns
    - Selecting the top 20 most frequent "AirportTo" destinations
    - Filtering the data to only include flights to the top 20 destinations with at
      least 30 flights
    - Splitting the filtered data into 80% training and 20% testing sets
    - Saving the train and test datasets as Parquet files
    """
    df: pd.DataFrame = pd.read_csv(
        datafile,
        dtype={
            "id": "int32",
            "Flight": "int16",
            "DayOfWeek": "int8",
            "Time": "int16",
            "Length": "int16",
            "Delay": "int8",
        },
    )

    df["flight"] = df["Airline"] + "-" + df["Flight"].astype(str)
    top20: pd.DataFrame = (
        df.groupby("AirportTo").count().nlargest(20, columns="id").reset_index()
    )

    data: pd.DataFrame = df[df["AirportTo"].isin(top20["AirportTo"])]
    data = data.drop(columns=["id", "Airline", "Flight"])

    data.columns = data.columns.str.lower()
    data = data.rename(
        columns={
            "airportfrom": "origin",
            "airportto": "destination",
            "dayofweek": "day",
        }
    )

    filtered_df: pd.DataFrame = data.loc[
        data.groupby(["flight", "destination"]).transform("size") >= 30
    ]

    mask = np.random.random(len(filtered_df)) < 0.8
    train_df: pd.DataFrame = filtered_df[mask]
    test_df: pd.DataFrame = filtered_df[~mask]

    train_path = Path.cwd().joinpath("data", "processed", "train.parquet")
    test_path = Path.cwd().joinpath("data", "processed", "test.parquet")

    if os.path.exists(train_path):
        os.remove(train_path)
        print(f"Old instance of {train_path} removed")
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)


if __name__ == "__main__":
    make_dataset()
