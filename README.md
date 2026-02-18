# 🚒 London Fire Brigade -- Incident & Response Time Analysis (2021--2025)

## 📊 Project Overview

This project analyses operational performance data from the London Fire
Brigade (LFB) between 2021 and 2025.

The goal is to explore response performance patterns, identify
structural differences across boroughs, and evaluate compliance with the
6-minute response target.

The project is implemented as a multi-page Streamlit application.

⚠️ Note: The dashboard is currently still under development. Additional
refinements, visual improvements, and extended analyses are planned.

------------------------------------------------------------------------

## 🎯 Key Analytical Focus Areas

-   Median First Pump Response Time\
-   6-Minute Target Compliance Rate\
-   Borough-Level Performance Differences\
-   Incident Type Comparison (Fire, Special Service, False Alarm)\
-   Temporal Patterns (Yearly, Monthly, Hourly)\
-   Geographic Performance Distribution

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python\
-   Pandas\
-   NumPy\
-   Streamlit\
-   Matplotlib / Seaborn\
-   Plotly\
-   GeoPandas\
-   Folium

Data is stored in compressed Parquet (Snappy) format for performance
optimization.

------------------------------------------------------------------------

## 🗂 Project Structure

    lfb-streamlit-app/
    │
    ├── app.py
    ├── data_loader.py
    ├── pages/
    ├── data/
    │   └── lfb_streamlit.parquet
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🚀 Running the App Locally

``` bash
pip install -r requirements.txt
streamlit run app.py
```

------------------------------------------------------------------------

## 🔄 Current Development Status

The application is functional but still evolving. Planned improvements
include:

-   UI refinements\
-   Enhanced interactivity\
-   Further geographic optimization\
-   Additional KPI deep dives

------------------------------------------------------------------------

## 👤 Author

Andrés Lill\
Data Analyst / Analytics Engineering Trainee
