import streamlit as st
from data_loader import load_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# theme

st.set_page_config(layout="wide")
sns.set_theme(style="white", context="notebook")

# ---------------------------------------------------------------------
# Plot constants

FIG_WIDTH = 10
FIG_HEIGHT_SMALL = 3.5
FIG_HEIGHT_MEDIUM = 4.5
FIG_HEIGHT_LARGE = 6

def style_axes(ax):
    ax.title.set_fontsize(16)
    ax.title.set_weight("bold")

    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)

    ax.tick_params(axis="both", which="major", labelsize=11)

# ---------------------------------------------------------------------
# Title

st.title("Executive Summary")
st.markdown("London Fire Brigade Response Performance (2021–2025)")

st.markdown("""
*Note: In this dashboard, "Response Time" refers to First Pump Attendance Time
(time from call to arrival of the first pump).*
""")
# ---------------------------------------------------------------------
# Load Data

df = load_data()

# ---------------------------------------------------------------------
# Filters

# Year and Month Filters 

st.sidebar.header("Filters")

# Available years
available_years = ["All"] + sorted(df["Year"].unique())

# Available months (mit All Option)
available_months = ["All"] + [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# Year filter
selected_year = st.sidebar.selectbox("Select Year",options=available_years)

# Month filter
selected_month = st.sidebar.selectbox("Select Month",options=available_months)

# Incident type Filter
incident_options = ["All"] + sorted(df["IncidentGroup"].dropna().unique())

selected_incident = st.sidebar.selectbox(
    "Select Incident Type",
    options=incident_options,
    key="geo_incident"
)

# ---------------------------------------------------------------------
# Apply Filters

# Year + Month
if selected_year == "All" and selected_month == "All":
    filtered_df = df.copy()

elif selected_year == "All":
    filtered_df = df[df["MonthName"] == selected_month]

elif selected_month == "All":
    filtered_df = df[df["Year"] == selected_year]

else:
    filtered_df = df[
        (df["Year"] == selected_year) &
        (df["MonthName"] == selected_month)
    ]

# Incident Type
if selected_incident != "All":
    filtered_df = filtered_df[
        filtered_df["IncidentGroup"] == selected_incident
    ]

# Empty check
if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

if selected_year == "All":
    year_text = "All Years"
else:
    year_text = selected_year

if selected_month == "All":
    month_text = "All Months"
else:
    month_text = selected_month

# Dynamic Period Label
min_year = df["Year"].min()
max_year = df["Year"].max()

if selected_year == "All" and selected_month == "All":
   period_label = f"{min_year}–{max_year}"

elif selected_year != "All" and selected_month == "All":
     period_label = f"{selected_year}, January–December"

elif selected_year == "All" and selected_month != "All":
     period_label = f"{selected_month} months between {min_year} and {max_year}"

else:
    period_label = f"{selected_month} {selected_year}"

# Dynamic Incident Label
if selected_incident == "All":
    incident_label = "All Incident Types"
else:
    incident_label = f"{selected_incident} Incidents"
    

st.caption(f"Data shown: {period_label}, {incident_label} ")

# ---------------------------------------------------------------------
# KPIs

total_incidents = len(filtered_df)
median_response = filtered_df["FirstPumpArriving_AttendanceTime"].median() / 60
response_within_6min = (filtered_df["FirstPumpArriving_AttendanceTime"] <= 360).mean() * 100
p90_response = filtered_df["FirstPumpArriving_AttendanceTime"].quantile(0.90) / 60
p90_response = filtered_df["FirstPumpArriving_AttendanceTime"].quantile(0.90) / 60
extreme_delay_rate = (filtered_df["FirstPumpArriving_AttendanceTime"] > 600).mean() * 100

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Incidents", f"{total_incidents:,}")
col2.metric("Median Response Time (min)", f"{median_response:.2f} min")
col3.metric("Response within 6 min (%)", f"{response_within_6min:.1f}%")
col4.metric("90th Percentile Response Time (min)", f"{p90_response:.2f} min")
col5.metric(">10 min Delays (%)", f"{extreme_delay_rate:.1f}%")

# ---------------------------------------------------------------------
# Distribution Plot

st.subheader("Distribution of Response Time")

response_minutes = filtered_df["FirstPumpArriving_AttendanceTime"] / 60

median = response_minutes.median()
mean = response_minutes.mean()
p90 = response_minutes.quantile(0.90)

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_MEDIUM))

sns.histplot(
    response_minutes,
    bins=25,
    kde=False,
    stat="percent",
    ax=ax
)

# Reference lines
ax.axvline(6, color="red", linestyle="--", linewidth=2, label="6-min target")
ax.axvline(median, color="black", linewidth=2, label=f"Median ({median:.2f})")
ax.axvline(mean, color="blue", linestyle="--", label=f"Mean ({mean:.2f})")
ax.axvline(p90, color="purple", linestyle=":", label=f"P90 ({p90:.2f})")

ax.set_xlabel("Response Time (minutes)")
ax.set_ylabel("Incidents (%)")  # ← Y-Achse angepasst

style_axes(ax)

ax.legend(frameon=False, fontsize=11)

sns.despine()
fig.tight_layout()

st.pyplot(fig)

# ---------------------------------------------------------------------
# Key Insights

st.markdown("**Key Insights:**")

st.markdown(f"""
- Across {period_label} ({incident_label.lower()}), response performance remains stable,
  with the 6-minute target achieved in approximately three out of four incidents.
- The right-skewed distribution indicates that performance variability is
 mainly driven by a smaller share of longer incidents.
- Extreme delays exceeding 10 min affect only a small share of incidents,
  yet they represent a relevant risk.
""")







