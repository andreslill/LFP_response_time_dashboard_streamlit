import streamlit as st

st.set_page_config(
    page_title="London Fire Brigade – Operational Performance", layout="wide")

st.title("🚒 London Fire Brigade Incident & Response Time Analysis Dashboard")

st.markdown("""
This dashboard analyzes operational response performance of the London Fire Brigade using incident and mobilisation data from 2021 to 2025.
The goal is to better understand how reliably response targets are met and how response performance how performance varies over time and across London.


Use the sidebar to navigate through the analytical sections:
- Executive Summary
- Incident Composition
- Response Time Analysis
- Geographic Performance
- Operational Drivers
""")
