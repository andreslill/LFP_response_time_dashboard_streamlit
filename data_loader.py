import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    df = pd.read_parquet("Data/lfb_streamlit.parquet")

    # Feature Engineering

    # Extract year and month
    df["Year"] = df["CallDate"].dt.year
    df["Month"] = df["CallDate"].dt.month
    df["MonthName"] = df["CallDate"].dt.month_name()

    # Create weekday feature
    df["Weekday"] = df["CallDate"].dt.day_name()

    # Compliance flag
    df["FirstPump_Within_6min"] = (
        df["FirstPumpArriving_AttendanceTime"] <= 360
    )

    return df
