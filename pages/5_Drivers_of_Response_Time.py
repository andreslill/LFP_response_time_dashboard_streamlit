# pages/6_Operational_Drivers.py

import streamlit as st
from data_loader import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Page config + theme


st.set_page_config(layout="wide")
sns.set_theme(style="white", context="notebook")

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
#Title + Intro

st.title("Operational and Structural Drivers of Response Time")

st.markdown("""
This page analyses the drivers of Response Time by decomposing it into turnout and travel components across boroughs and incident types.

**Response Time = Turnout Time (Station alerted → First vehicle leaves) + Travel Time (First vehicle leaves → Arrival at scene)** 

In addition, the recorded causes of 6-minute target exceedance are examined to identify recurring operational and external delay factors.

The objective is to distinguish operational drivers (station mobilisation) from external constraints (e.g. traffic conditions and travel distance)
and assess their contribution to 6-minute target exceedance.
""")


# ---------------------------------------------------------------------
# Load Data

df = load_data()

# ---------------------------------------------------------------------
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

# Apply Filters
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

st.caption(f"Data shown: {period_label}")

# ------------------------------------------------------------
# Feature engineering for this page

# Minutes versions (clean + consistent)
filtered_df = filtered_df.copy()

# Attendance in minutes
filtered_df["AttendanceMinutes"] = filtered_df["FirstPumpArriving_AttendanceTime"] / 60

# Turnout + Travel (only if available)
has_turnout = "TurnoutTimeSeconds" in filtered_df.columns
has_travel  = "TravelTimeSeconds" in filtered_df.columns

if has_turnout:
    filtered_df["TurnoutMinutes"] = filtered_df["TurnoutTimeSeconds"] / 60
if has_travel:
    filtered_df["TravelMinutes"] = filtered_df["TravelTimeSeconds"] / 60

filtered_df["Over6"] = filtered_df["FirstPumpArriving_AttendanceTime"] > 360


# ------------------------------------------------------------
# KPIs

overall_turnout = filtered_df["TurnoutMinutes"].median()
overall_travel = filtered_df["TravelMinutes"].median()

total_component = overall_turnout + overall_travel
travel_share_pct = (overall_travel / total_component) * 100

col1, col2, col3 = st.columns(3)

col1.metric(
    "Median Turnout Time",
    f"{overall_turnout:.2f} min"
)

col2.metric(
    "Median Travel Time",
    f"{overall_travel:.2f} min"
)

col3.metric(
    "Travel Share of Response",
    f"{travel_share_pct:.0f}%"
)

# ------------------------------------------------------------
st.header("1. What Drives Borough Differences? (Turnout vs Travel)")  
# ------------------------------------------------------------
# Borough-level decomposition (Top 10 slowest boroughs)
st.subheader("Slowest Boroughs: Response Time Decomposition")


# Borough-level medians
borough_decomp = (
    filtered_df
    .groupby("IncGeo_BoroughName")
    .agg(
        TurnoutMedian=("TurnoutMinutes", "median"),
        TravelMedian=("TravelMinutes", "median"),
    )
    .reset_index()
)


# Exact total = turnout + travel
borough_decomp["TotalMedian"] = (
    borough_decomp["TurnoutMedian"] +
    borough_decomp["TravelMedian"]
)


# Sort by total descending
borough_decomp = borough_decomp.sort_values(
    "TotalMedian",
    ascending=False
)

# Optional: show only slowest 10
borough_decomp = borough_decomp.head(10)


# Plot
fig, ax = plt.subplots(figsize=(7.5, 6))

cb = sns.color_palette("colorblind")
turnout_color = cb[2]
travel_color = cb[0]

ax.barh(
    borough_decomp["IncGeo_BoroughName"],
    borough_decomp["TurnoutMedian"],
    label="Turnout (median)"
)

ax.barh(
    borough_decomp["IncGeo_BoroughName"],
    borough_decomp["TravelMedian"],
    left=borough_decomp["TurnoutMedian"],
    label="Travel (median)"
)


ax.set_xlabel("Minutes (median)")
ax.set_ylabel("")

# Reverse y-axis so slowest on top
ax.invert_yaxis()

# ️Legend 
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=2,
    frameon=False
)

sns.despine()
fig.tight_layout()

st.pyplot(fig)


# Dynamic insight for the shown borough subset
borough_decomp["TravelShare"] = (
    borough_decomp["TravelMedian"] /
    borough_decomp["TotalMedian"]
)

avg_travel_share = borough_decomp["TravelShare"].mean() * 100

st.markdown(f"""
**Key Insight ({period_label})**

- Overall, travel time accounts for **{travel_share_pct:.0f}%** of the median response structure.
- Among the slowest boroughs, this share increases to approximately
  **{avg_travel_share:.0f}%**, reinforcing the structural dominance of travel-related factors.
- Turnout times remain comparatively stable, indicating that extended response
  times are primarily driven by factors associated with travel time such as distance and traffic.
""")



# ------------------------------------------------------------
# Turnout Time Stability Check 

if has_turnout:

    # Overall turnout median (minutes)
    overall_turnout_median = filtered_df["TurnoutMinutes"].median()

    turnout_stats = (
        filtered_df
        .groupby("IncGeo_BoroughName")["TurnoutMinutes"]
        .agg(["median", "std"])
        .reset_index()
    )

    avg_borough_std = turnout_stats["std"].mean()
    max_borough_std = turnout_stats["std"].max()

    # Convert to seconds
    overall_turnout_sec = overall_turnout_median * 60
    avg_borough_std_sec = avg_borough_std * 60
    max_borough_std_sec = max_borough_std * 60

    with st.expander("Turnout Time Stability Check"):

        st.markdown(f"""
**Turnout Time Stability Check ({period_label})**

- Overall median turnout time: **{overall_turnout_median:.2f} minutes ({overall_turnout_sec:.0f} seconds)**
- Average turnout variability across boroughs (std): **{avg_borough_std:.2f} minutes ({avg_borough_std_sec:.0f} seconds)**
- Maximum turnout variability across boroughs: **{max_borough_std:.2f} minutes ({max_borough_std_sec:.0f} seconds)**

Turnout time shows relatively low variability across boroughs compared to travel time,
suggesting that station mobilisation performance is structurally stable.
""")

else:
    st.info("TurnoutMinutes column not available in the dataset.")



# ------------------------------------------------------------
#Methodological Note


with st.expander("Methodological Note"):
    st.markdown("""
    Median turnout and median travel time are calculated independently.
    As medians are not additive, Median(A) + Median(B) does not necessarily equal 
    Median(A + B). Therefore, their sum may differ slightly from the median attendance time.
    """)

# ------------------------------------------------------------
st.header("2. How does Hour of Day influence Response Time")

st.subheader("Turnout vs Travel Time by Hour of Day")

hourly_components = (
    filtered_df
    .groupby("HourOfCall")
    .agg(
        TurnoutMedian=("TurnoutMinutes", "median"),
        TravelMedian=("TravelMinutes", "median")
    )
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 5))

cb = sns.color_palette("colorblind")
turnout_color = cb[2]
travel_color = cb[0]

ax.plot(
    hourly_components["HourOfCall"],
    hourly_components["TurnoutMedian"],
    label="Turnout (median)",
    linewidth=2.5,
    marker="o"
)

ax.plot(
    hourly_components["HourOfCall"],
    hourly_components["TravelMedian"],
    label="Travel (median)",
    linewidth=2.5,
    marker="o"
)

ax.set_xlabel("Hour of Call")
ax.set_ylabel("Minutes (median)")
ax.set_xticks(range(0, 24))

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=2,
    frameon=False
)

sns.despine()
fig.tight_layout()

st.pyplot(fig)


# Peak travel hour
peak_hour = hourly_components.loc[
    hourly_components["TravelMedian"].idxmax(),
    "HourOfCall"
]

travel_range = (
    hourly_components["TravelMedian"].max() -
    hourly_components["TravelMedian"].min()
)

turnout_range = (
    hourly_components["TurnoutMedian"].max() -
    hourly_components["TurnoutMedian"].min()
)

st.markdown(f"""
**Key Insight ({period_label})**

- Travel time varies by approximately **{travel_range:.2f} minutes** across the day, peaking around **{peak_hour}:00**.  
- Turnout time shows substantially lower variation (**{turnout_range:.2f} minutes** range).  
- This pattern supports the interpretation that **traffic-related factors, rather than station mobilisation, drive intra-day performance differences.**
""")






# ------------------------------------------------------------


st.header("3. Why Do Incidents Exceed the 6-Minute Target?")


# ---------------------------------------------------------------------

st.subheader("Breakdown of Recorded Delay Reasons")

# Filter incidents exceeding 6-minute target
delayed_df = filtered_df[
    filtered_df["FirstPumpArriving_AttendanceTime"] > 360
].copy()

# Remove missing delay codes
delayed_df = delayed_df[
    delayed_df["DelayCode_Description"].notna()
]

# Count delay codes
delay_counts = (
    delayed_df
    .groupby("DelayCode_Description")
    .size()
    .reset_index(name="IncidentCount")
    .sort_values("IncidentCount", ascending=False)
)

delay_counts["DelayCode_Description"] = delay_counts["DelayCode_Description"].replace(
    {"No delay": "No recorded delay code"}
)


total_exceedances = delay_counts["IncidentCount"].sum()

top_n = 5

top_delay = delay_counts.head(top_n).copy()
others_delay = delay_counts.iloc[top_n:].copy()

# Calculate percentages
top_delay["Percent"] = (
    top_delay["IncidentCount"] / total_exceedances * 100
)

others_percent = (
    others_delay["IncidentCount"].sum() / total_exceedances * 100
)

# Add Others row
others_row = pd.DataFrame({
    "DelayCode_Description": ["Other Delay Codes"],
    "IncidentCount": [others_delay["IncidentCount"].sum()],
    "Percent": [others_percent]
})

final_delay = pd.concat([top_delay, others_row], ignore_index=True)

# Sort ascending for horizontal bar plot
final_delay = final_delay.sort_values("Percent", ascending=True)

# Context

exceedances = f"{len(delayed_df):,}".replace(",", ".")

st.caption(
    f"{exceedances} incidents exceeded the 6-minute target "
    f"({len(delayed_df)/len(filtered_df)*100:.1f}% of total incidents) in {period_label}."
)

# Plot

fig, ax = plt.subplots(figsize=(10, 6))

cb = sns.color_palette("colorblind")
main_color = cb[0]  # consistent dashboard color

bars = ax.barh(
    final_delay["DelayCode_Description"],
    final_delay["Percent"],
    color=main_color
)

# Add labels
for i, val in enumerate(final_delay["Percent"]):
    ax.text(
        val - 0.5,
        i,
        f"{val:.1f}%",
        va="center",
        ha="right",
        fontsize=10,
        weight="bold",
        color="white"
    )

ax.set_xlabel("Share of Incidents Exceeding 6-Minute Response Time Target (%)")

sns.despine()
plt.tight_layout()
st.pyplot(fig)

# Calculate Share of "Not held up" for the Insights

not_held_up_row = delay_counts[
    delay_counts["DelayCode_Description"] == "Not held up"
]

if not not_held_up_row.empty:
    not_held_up_percent = (
        not_held_up_row["IncidentCount"].values[0] /
        total_exceedances * 100
    )
else:
    not_held_up_percent = 0

# Insights

top_driver = top_delay.iloc[0]

st.markdown(
    f"""

- A substantial share of exceedances (**{not_held_up_percent:.1f}%**) are recorded without a specific delay factor ("Not held up"), suggesting potential limitations in delay attribution rather than a single dominant operational cause.
- The remaining delay factors collectively account for approximately **{others_percent:.1f}%**, indicating a moderate long-tail distribution of operational causes.

"""
)

# ---------------------------------------------------------------------
# Expandable explaining "Others" Category

with st.expander("Show delay codes included in 'Other Delay Codes'"):

    if not others_delay.empty:

        others_delay["Percent"] = (
            others_delay["IncidentCount"] / total_exceedances * 100
        )

        others_delay = others_delay.sort_values("Percent", ascending=False)

        for _, row in others_delay.iterrows():
            st.markdown(
                f"- {row['DelayCode_Description']} "
                f"– {row['Percent']:.1f}%"
            )


st.markdown("""
### Key Takeaway:

- The findings suggest that variability in response performance is
  mainly driven by constraints related to travel, while turnout
  processes remain comparatively stable across boroughs and time.



### Implication:

- Improvements in overall response performance will likely require
  measures addressing travel constraints, such as distance and traffic
  conditions, rather than further optimisation of station mobilisation.
""")












