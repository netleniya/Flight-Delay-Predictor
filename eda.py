import os
import pandas as pd


def make_dataset() -> None:
    df: pd.DataFrame = pd.read_csv(
        "Airlines.csv",
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

    filename = "clean_data.parquet"

    if os.path.exists(filename):
        os.remove(filename)
        print(f"Old instance of {filename} removed")
    filtered_df.to_parquet("clean_data.parquet")


if __name__ == "__main__":
    make_dataset()
