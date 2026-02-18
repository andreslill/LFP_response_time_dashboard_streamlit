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

st.title("Response Performance")

st.caption(
    "Response Time refers to First Pump Attendance Time "
    "(seconds from call to arrival)."
)

st.markdown("""
This section analyses response performance, including its distribution, temporal variation and differences across incident types.
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
    period_label = f"{selected_month}, {min_year}–{max_year}"

else:
    period_label = f"{selected_month} {selected_year}"

st.caption(f"Data shown: {period_label}")

# ---------------------------------------------------------------------
#KPIs

median_response = filtered_df["FirstPumpArriving_AttendanceTime"].median() / 60
response_within_6min = ((filtered_df["FirstPumpArriving_AttendanceTime"] <= 360).mean() * 100)
p90_response = filtered_df["FirstPumpArriving_AttendanceTime"].quantile(0.90) / 60
extreme_delay_rate = (filtered_df["FirstPumpArriving_AttendanceTime"] > 600).mean() * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Median Response Time (min)", f"{median_response:.2f} min")
col2.metric("Response within 6 min (%)", f"{response_within_6min:.1f}%")
col3.metric("90th Percentile Response Time (min)", f"{p90_response:.2f} min")
col4.metric(">10 min Delays (%)", f"{extreme_delay_rate:.1f}%")

# ---------------------------------------------------------------------
# Stacked Barplot

st.subheader("Distribution of Response Times")

st.markdown("<br>", unsafe_allow_html=True) # space

# Legend

left, col1, col2, col3, col4, right = st.columns([0.8, 1, 1, 1, 1, 1.5])

col1.markdown("<span style='color:#2ca02c;'>●</span> ≤ 6 min", unsafe_allow_html=True)
col2.markdown("<span style='color:#f1ce63;'>●</span> 6–8 min", unsafe_allow_html=True)
col3.markdown("<span style='color:#ff7f0e;'>●</span> 8–10 min", unsafe_allow_html=True)
col4.markdown("<span style='color:#d62728;'>●</span> > 10 min", unsafe_allow_html=True)


# Create response time in minutes
filtered_df["ResponseMinutes"] = (
    filtered_df["FirstPumpArriving_AttendanceTime"] / 60
)

# Extreme delay KPI (GLOBAL)
extreme_delay_rate = (
    (filtered_df["ResponseMinutes"] > 10).mean() * 100
)

# Define bands
bins = [0, 6, 8, 10, float("inf")]
labels = ["≤ 6 min", "6–8 min", "8–10 min", "> 10 min"]

filtered_df["ResponseBand"] = pd.cut(
    filtered_df["ResponseMinutes"],
    bins=bins,
    labels=labels,
    right=True
)

# Set categorical order BEFORE grouping / pivoting
filtered_df["IncidentGroup"] = pd.Categorical(
    filtered_df["IncidentGroup"],
    categories=["False Alarm", "Special Service", "Fire"],
    ordered=True
)

# Count incidents per band & type
band_counts = (
    filtered_df
    .groupby(["IncidentGroup", "ResponseBand"], observed=True)
    .size()
    .reset_index(name="Count")
)

# Calculate percentage within each IncidentGroup
band_counts["Percent"] = (
    band_counts.groupby("IncidentGroup")["Count"]
    .transform(lambda x: 100 * x / x.sum())
)

# Pivot for stacked bar
band_pivot = band_counts.pivot(
    index="IncidentGroup",
    columns="ResponseBand",
    values="Percent"
).fillna(0)


# Plot
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_SMALL))

palette = ["#2ca02c", "#f1c40f", "#e67e22", "#c0392b"]

left = None

for i, band in enumerate(labels):
    ax.barh(
        band_pivot.index,
        band_pivot[band],
        left=left,
        color=palette[i],
        label=band
    )
    if left is None:
        left = band_pivot[band]
    else:
        left += band_pivot[band]
        

ax.set_xlim(0, 100)
ax.set_xlabel("Percentage of Incidents (%)")

ax.invert_yaxis()

style_axes(ax)
sns.despine()
fig.tight_layout()

st.pyplot(fig)


extreme_delay_rate = (
    filtered_df["ResponseMinutes"] > 10
).mean() * 100

st.markdown("""
- False Alarm incidents show the highest compliance with the 6-minute target.
- Special Service incidents show the largest proportion of responses exceeding 6 minutes.
- Fire incidents follow a similar distribution pattern, with the majority handled within target but a noticeable share falling outside the 6-min target.
- Delays exceeding 10 minutes response time remain limited across all incident types, representing a small yet operationally relevant risk segment.
""")

# ---------------------------------------------------------------------
st.markdown("---")
# ---------------------------------------------------------------------
# Lineplot

st.subheader("Seasonal Patterns in Response Times")

st.markdown("<br>", unsafe_allow_html=True) # space

# legend
spacer, col1, col2, col3, col4 = st.columns([0.3, 1, 1, 1, 1])

col1.markdown("<span style='color:black;'>●</span> All Incidents", unsafe_allow_html=True)
col2.markdown("<span style='color:#1f77b4;'>●</span> False Alarm", unsafe_allow_html=True)
col3.markdown("<span style='color:#2ca02c;'>●</span> Special Service", unsafe_allow_html=True)
col4.markdown("<span style='color:#ff7f0e;'>●</span> Fire", unsafe_allow_html=True)


median_firstpump_attendance_by_type = (
    filtered_df
    .groupby(["Month", "IncidentGroup"])["FirstPumpArriving_AttendanceTime"]
    .median()
    .div(60)
    .reset_index(name="MedianFirstPumpMinutes")
)

median_firstpump_attendance_total = (
    filtered_df
    .groupby("Month")["FirstPumpArriving_AttendanceTime"]
    .median()
    .div(60)
    .reset_index(name="MedianFirstPumpMinutes")
)

median_firstpump_attendance_total["IncidentGroup"] = "All Incidents"

median_firstpump_attendance_long = pd.concat(
    [median_firstpump_attendance_by_type, median_firstpump_attendance_total],
    ignore_index=True
)

palette = {
    "All Incidents": "black",
    "False Alarm": sns.color_palette("colorblind")[0],
    "Fire": sns.color_palette("colorblind")[1],
    "Special Service": sns.color_palette("colorblind")[2],
}

hue_order = ["Fire", "Special Service", "False Alarm", "All Incidents"]

sns.set_theme(style="white")  # removes background grid

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_MEDIUM))

# Plot incident types
sns.lineplot(
    data=median_firstpump_attendance_long[
        median_firstpump_attendance_long["IncidentGroup"] != "All Incidents"
    ],
    x="Month",
    y="MedianFirstPumpMinutes",
    hue="IncidentGroup",
    hue_order=hue_order[:-1],
    palette=palette,
    linewidth=2.5,
    marker="o",
    markeredgewidth=0,
    alpha=0.7,
    ax=ax,
    legend=False
)

# Plot ALL incidents separately thicker
sns.lineplot(
    data=median_firstpump_attendance_long[
        median_firstpump_attendance_long["IncidentGroup"] == "All Incidents"
    ],
    x="Month",
    y="MedianFirstPumpMinutes",
    color="black",
    linewidth=4,
    alpha=1,
    ax=ax,
    legend=False
)

ax.set_xlabel("Month")
ax.set_ylabel("Median Response Time (minutes)")

ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])


sns.despine()
fig.tight_layout()

st.pyplot(fig)

st.markdown("""
- Median response times remain stable throughout the analysed period.
- Special Service incidents show moderate seasonal variation, with increases in early summer and again in late autumn.
- Fire incidents showing a pronounced peak in summer.
- False Alarms display a similar increase towards summer and an additional increase in late autumn.
- Fire and Special Service incidents** consistently show slightly longer attendance times than False Alarms.
""")

# ---------------------------------------------------------------------
st.markdown("---")
# ---------------------------------------------------------------------

st.subheader("Response Times by Hour of Day")

# calculate Attendence time in minutes
df_hour = filtered_df.copy()

df_hour["AttendanceMinutes"] = (
    df_hour["FirstPumpArriving_AttendanceTime"] / 60
)

# Calculate median per hour
hourly_median = (
    df_hour
    .groupby("HourOfCall")["AttendanceMinutes"]
    .median()
    .reset_index(name="MedianMinutes")
)

hourly_median = hourly_median.sort_values("HourOfCall")

# Calculations for filters
min_hour = hourly_median.loc[hourly_median["MedianMinutes"].idxmin()]
max_hour = hourly_median.loc[hourly_median["MedianMinutes"].idxmax()]

min_val = min_hour["MedianMinutes"]
max_val = max_hour["MedianMinutes"]
diff = max_val - min_val
diff_seconds = diff * 60

min_hour_label = int(min_hour["HourOfCall"])
max_hour_label = int(max_hour["HourOfCall"])


# Plot

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_MEDIUM))

sns.lineplot(
    data=hourly_median,
    x="HourOfCall",
    y="MedianMinutes",
    marker="o",
    linewidth=2.5,
    ax=ax
)

# Minimun and Maximum
max_hour = hourly_median.loc[hourly_median["MedianMinutes"].idxmax()]
min_hour = hourly_median.loc[hourly_median["MedianMinutes"].idxmin()]

ax.scatter(max_hour["HourOfCall"], max_hour["MedianMinutes"], s=80)
ax.scatter(min_hour["HourOfCall"], min_hour["MedianMinutes"], s=80)


ax.set_xlabel("Hour of Call")
ax.set_ylabel("Median Response Time (minutes)")

ax.set_xticks(range(0, 24, 2))  # nur jede 2. Stunde anzeigen

sns.despine()
plt.tight_layout()
st.pyplot(fig)


st.markdown(
    f"""

- The median response time shows a moderate variation by hour of day, ranging from **{min_val:.2f} minutes (minimum at {min_hour_label}:00)** 
  to **{max_val:.2f} minutes (maximum at {max_hour_label}:00)**:  A difference of approximately **{diff:.2f} minutes (~{diff_seconds:.0f} seconds)**.
- Although observable, this difference is unlikely to be critical, suggesting consistent response performance throughout the day.

"""
)

# ---------------------------------------------------------------------
st.markdown("---")
# ---------------------------------------------------------------------
# Key Takeaways


st.markdown("""
### Key Takeaways

- Overall response time remains stable across the analysed period.
- False Alarm incidents show the fastest response times.
- Delays exeeding 10 minutes remain rare, accounting for approximately 3% of incidents.
""")


# ---------------------------------------------------------------------
st.markdown("---")
# ---------------------------------------------------------------------
# Boxplot

with st.expander("Show detailed response time distribution (Boxplot)"):

    st.subheader("First Pump Attendance Time Distribution")

    st.markdown("<br>", unsafe_allow_html=True) # space

    # First Pump Attendance Time
    first_pump_df = filtered_df[
        ["IncidentGroup", "FirstPumpArriving_AttendanceTime"]
    ].copy()

    # Convert to minutes
    first_pump_df["AttendanceTimeMinutes"] = (
        first_pump_df["FirstPumpArriving_AttendanceTime"] / 60
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_MEDIUM))

    palette = {
        "False Alarm": sns.color_palette("colorblind")[0],
        "Fire": sns.color_palette("colorblind")[1],
        "Special Service": sns.color_palette("colorblind")[2],
    }

    sns.boxplot(
        data=first_pump_df,
        x="IncidentGroup",
        y="AttendanceTimeMinutes",
        order=["False Alarm", "Special Service", "Fire"],
        palette=palette,
        showfliers=False,
        ax=ax
    )

    # Median annotations
    medians = first_pump_df.groupby("IncidentGroup")["AttendanceTimeMinutes"].median()

    for tick, label in zip(ax.get_xticks(), medians.index):
        ax.text(
            tick,
            medians[label],
            f"{medians[label]:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            weight="bold"
        )

    # Formatting
    ax.set_xlabel("")
    ax.set_ylabel("Response Time (minutes)")

    ax.axhline(
        6,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="6-min target"
    )

    ax.legend(frameon=False)

    sns.despine()
    fig.tight_layout()

    st.pyplot(fig)

    st.markdown("""

    - Response time variability appears to differ across incident types.  
    - Special Service incidents display a wider interquartile range, while False Alarms exhibit the most consistent response performance.
    """)


















