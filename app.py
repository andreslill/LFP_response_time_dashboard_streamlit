import streamlit as st

st.set_page_config(
    page_title="London Fire Brigade – Operational Performance", layout="wide")

st.title("🚒 London Fire Brigade Incident & Response Time Analysis Dashboard")

st.markdown("""
Using mobilisation and incident data from 2021–2025, this dashboard analyses response performance of the London Fire Brigade.
It evaluates how consistently response targets are met and tracks performance trends over time and across different areas of London.


Use the sidebar to navigate through the analytical sections:
- Executive Summary
- Incident Composition
- Response Time Analysis
- Geographic Performance
- Operational Drivers
""")
