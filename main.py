import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from joblib import load


def main() -> None:

    st.title("Flight delay prediction")

    df: pd.DataFrame = pd.read_parquet("clean_data.parquet")
    _, X_test, _, _ = train_test_split(
        df.drop(columns=["delay"]), df["delay"], test_size=0.2, random_state=42
    )

    data = X_test.sample(1)
    st.write(data)
    model = load("model.joblib")

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
