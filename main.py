import json
import pandas as pd
import streamlit as st
from model import create_model
from sklearn.pipeline import Pipeline


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_parquet("clean_data.parquet")
    return df


def fit_model() -> Pipeline:
    df: pd.DataFrame = load_dataset()
    X, y = df.drop(columns=["delay"]), df["delay"]

    with open("hyperparams.json", "r") as f:
        hyperparameters: dict = json.load(f)

    model: Pipeline = create_model(X=X, y=y, params=hyperparameters)

    model.fit(X, y)

    return model


def main() -> None:
    st.title("Flight Delay Prediction")

    st.sidebar.header("Model Parameters")

    origin = st.sidebar.selectbox("Origin", load_dataset()["origin"].unique())
    destination = st.sidebar.selectbox(
        "Destination", load_dataset()["destination"].unique()
    )
    day = st.sidebar.selectbox("Day", load_dataset()["day"].unique())
    time = st.sidebar.selectbox("Time", load_dataset()["time"].unique())
    length = st.sidebar.slider("Flight Length", 0, 1000, 100)
    flight = st.sidebar.selectbox("Flight", load_dataset()["flight"].unique())

    test = pd.DataFrame(
        {
            "origin": [origin],
            "destination": [destination],
            "day": [day],
            "time": [time],
            "length": [length],
            "flight": [flight],
        }
    )

    model: Pipeline = fit_model()
    submit: bool = st.sidebar.button("Predict")

    if submit:
        st.write("Delayed" if model.predict(test)[0] > 0.5 else "On Time")
        return


if __name__ == "__main__":
    main()
