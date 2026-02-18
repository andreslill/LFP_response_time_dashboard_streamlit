import streamlit as st

st.set_page_config(
    page_title="London Fire Brigade – Operational Performance", layout="wide")

st.title("🚒 London Fire Brigade Incident & Response Time Analysis Dashboard")

st.markdown("""
This dashboard analyses London Fire Brigade incident and mobilisation data (2021–2025).

Use the sidebar to navigate through the analytical sections:
- Executive Summary
- Incident Composition
- Response Time Analysis
- Geographic Performance
- Operational Drivers
""")
