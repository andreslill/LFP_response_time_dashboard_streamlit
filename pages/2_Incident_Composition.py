import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_data

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

st.title("Incident Composition")

st.markdown("""
This section analyses the structural composition of incidents, 
including workload distribution and temporal patterns.
""")

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
selected_year = st.sidebar.selectbox(
    "Select Year",
    options=available_years
)

# Month filter
selected_month = st.sidebar.selectbox(
    "Select Month",
    options=available_months
)

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
    period_label = f"{selected_month}, {min_year}–{max_year}"

else:
    period_label = f"{selected_month} {selected_year}"

st.caption(f"Data shown: {period_label}")

# ---------------------------------------------------------------------
# Total incidents

total_incidents = len(filtered_df)

st.markdown(
    f"<span style='font-size:16px;'><b>{total_incidents:,}</b> incidents analysed in the selected period.</span>",
    unsafe_allow_html=True
)
# ---------------------------------------------------------------------
# Barplot 


st.subheader("Incident Mix Distribution (%)")

st.markdown("<br>", unsafe_allow_html=True) # space

incident_mix = (
    filtered_df["IncidentGroup"]
    .value_counts(normalize=True)
    .mul(100)
    .round(1)
    .reset_index()
)

incident_mix.columns = ["IncidentGroup", "Percentage"]

# Sort incident types
order = ["Fire", "Special Service", "False Alarm"]

incident_mix["IncidentGroup"] = pd.Categorical(
    incident_mix["IncidentGroup"],
    categories=order,
    ordered=True
)

incident_mix = incident_mix.sort_values("IncidentGroup")

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_SMALL))

colors = [
    sns.color_palette("colorblind")[1],  # Fire
    sns.color_palette("colorblind")[2],  # Special Service
    sns.color_palette("colorblind")[0],  # False Alarm
]

bars = ax.barh(
    incident_mix["IncidentGroup"],
    incident_mix["Percentage"],
    color=colors
)

# Value labels
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}%",
        va="center"
    )

ax.set_xlabel("Percentage of Total Incidents")
ax.set_ylabel("")
ax.set_xlim(0, 100)

style_axes(ax)
sns.despine(ax=ax)
fig.tight_layout()

st.pyplot(fig)

# ---------------------------------------------------------------------
st.markdown("---")
# ---------------------------------------------------------------------
# Lineplot

st.subheader("Monthly Incident Trends by Incident Type")

st.markdown("<br>", unsafe_allow_html=True) # space

# legend
spacer, col1, col2, col3, col4 = st.columns([0.4, 1, 1, 1, 1])

col1.markdown("<span style='color:black;'>●</span> All Incidents", unsafe_allow_html=True)
col2.markdown("<span style='color:#1f77b4;'>●</span> False Alarm", unsafe_allow_html=True)
col3.markdown("<span style='color:#2ca02c;'>●</span> Special Service", unsafe_allow_html=True)
col4.markdown("<span style='color:#ff7f0e;'>●</span> Fire", unsafe_allow_html=True)

# Monthly unique incident counts by incident type
monthly_incidents_by_type = (
    filtered_df
    .groupby(["Month", "IncidentGroup"])["IncidentNumber"]
    .nunique()
    .reset_index(name="IncidentCount")
)

# Monthly unique incident counts across all incident types
monthly_incidents_total = (
    filtered_df
    .groupby("Month")["IncidentNumber"]
    .nunique()
    .reset_index(name="IncidentCount")
)

# Label totals so they can be plotted together with incident types
monthly_incidents_total["IncidentGroup"] = "All Incidents"

# Combine into long format
monthly_incident_counts_long = pd.concat(
    [monthly_incidents_by_type, monthly_incidents_total],
    ignore_index=True
)

palette = {
    "All Incidents": "black",
    "False Alarm": sns.color_palette("colorblind")[0],
    "Fire": sns.color_palette("colorblind")[1],
    "Special Service": sns.color_palette("colorblind")[2],
}

sns.set_theme(style="white") # removes grid automatically

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_MEDIUM))

hue_order = [
    "All Incidents",
    "False Alarm",
    "Special Service",
    "Fire"
]

# Plot all incident types EXCEPT totals
sns.lineplot(
    data=monthly_incident_counts_long[monthly_incident_counts_long["IncidentGroup"] != "All Incidents"],
    x="Month",
    y="IncidentCount",
    hue="IncidentGroup",
    hue_order=hue_order[1:],  # exclude All Incidents
    palette=palette,
    linewidth=2.5,
    marker="o",
    alpha=0.7,
    ax=ax,
    legend=False,
)

# Plot ALL INCIDENTS separately with thicker line
sns.lineplot(
    data=monthly_incident_counts_long[monthly_incident_counts_long["IncidentGroup"] == "All Incidents"],
    x="Month",
    y="IncidentCount",
    color="black",
    linewidth=4,
    alpha=1,
    ax=ax,
    legend=False
)

ax.set_xlabel("Month")
ax.set_ylabel("Number of Incidents")

ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

style_axes(ax)
sns.despine(ax=ax)
fig.tight_layout()

st.pyplot(fig, use_container_width=True)

# ---------------------------------------------------------------------
st.markdown("---")
# ---------------------------------------------------------------------
# Heatmap

st.subheader("Daily and Hourly Incident Heatmap")

incident_options = [
    "All",
    "False Alarm",
    "Special Service",
    "Fire"
]

selected_incident_type = st.radio(
    "Incident Type",
    options=incident_options,
    horizontal=True
)

# -------------------------
# Filter
# -------------------------

if selected_incident_type == "All":
    heatmap_df = filtered_df.copy()
else:
    heatmap_df = filtered_df[
        filtered_df["IncidentGroup"] == selected_incident_type
    ]

# -------------------------
# Pivot table (Hour x Weekday)
# -------------------------

daily_hourly_incidents = heatmap_df.pivot_table(
    index="HourOfCall",
    columns="Weekday",
    values="IncidentNumber",
    aggfunc="nunique"
)

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

daily_hourly_incidents = (
    daily_hourly_incidents
    .reindex(index=range(24))          # ensure 0–23
    .reindex(columns=weekday_order)    # Monday → Sunday
    .T                                 # 🔁 swap axes
)

# -------------------------
# Plot
# -------------------------

fig, ax = plt.subplots(figsize=(FIG_WIDTH, 10))

sns.heatmap(
    daily_hourly_incidents,
    cmap="coolwarm",
    square=True,
    linewidths=0.3,
    linecolor="white",
    cbar_kws={"label": "Number of Incidents",
              "shrink": 0.2},
    ax=ax
)

ax.set_xlabel("Hour of Call")
ax.set_ylabel("Day of Week")

ax.title.set_fontsize(10)
ax.title.set_weight("bold")

ax.xaxis.label.set_size(13)
ax.yaxis.label.set_size(13)

ax.tick_params(axis="both", labelsize=10)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
cbar.set_label("Number of Incidents", fontsize=13)

fig.tight_layout()

st.pyplot(fig)





