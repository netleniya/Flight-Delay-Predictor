import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from joblib import load
from pathlib import Path


def main() -> None:

    st.title("Flight delay prediction")

    test_data = Path.cwd().parent.joinpath("data", "processed", "test.parquet")
    trained_model = Path.cwd().parent.joinpath("models", "model.joblib")

    df: pd.DataFrame = pd.read_parquet(test_data)
    X_test, _ = df.drop(columns=["delay"]), df["delay"]

    data = X_test.sample(1)
    st.write(data)
    model = load(trained_model)

    st.sidebar.header("Hyperparameters")

    origin = st.sidebar.selectbox("Origin", data["origin"].unique())
    destination = st.sidebar.selectbox("Destination", data["destination"].unique())
    day = st.sidebar.selectbox("Day", data["day"].unique())
    time = st.sidebar.selectbox("Time", data["time"].unique())
    length = st.sidebar.selectbox("Length", data["length"].unique())
    flight = st.sidebar.selectbox("Flight", data["flight"].unique())

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

    if submit := st.sidebar.button("Submit"):
        st.write("Delayed" if model.predict(test)[0] > 0.5 else "On Time")
        st.write(f"Probability of delay: {model.predict_proba(test)[:, 1][0]:.2%}")


if __name__ == "__main__":
    main()
