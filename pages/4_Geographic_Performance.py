import streamlit as st
from data_loader import load_data
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import linregress
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import statsmodels.api as sm


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

st.title("Geographic Performance")


st.markdown("""
This section analyses geographic variation in response performance
across London boroughs, identifying spatial patterns and structural differences
in operational response
""")

# ---------------------------------------------------------------------
# Load Data

df = load_data()

# London Boroughs
boroughs = gpd.read_file("Data/london_boroughs/London_Borough_Excluding_MHW.shp")

# London Population
pop = pd.read_csv("Data/london_population_borough.csv")
boroughs["Area_km2"] = boroughs["HECTARES"] / 100

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

st.header("1. Do Response Times Differ Between Inner and Outer London?")

# ---------------------------------------------------------------------
# Prepare dataset for analysis

# Normalize borough names for merging
boroughs["NAME_clean"] = (
    boroughs["NAME"]
    .str.strip()
    .str.upper()
)

# Create borough-level Median Response Time (base dataframe for multiple plots)
# based on filtered_df
borough_spatial_extent = (
    filtered_df
    .groupby("IncGeo_BoroughName")
    .agg(
        MedianResponseMinutes=(
            "FirstPumpArriving_AttendanceTime",
            lambda x: x.median() / 60
        )
    )
    .reset_index()
)

# Standardise borough names for merging with geo dataset
borough_spatial_extent["NAME_clean"] = (
    borough_spatial_extent["IncGeo_BoroughName"]
    .str.strip()
    .str.upper()
)

# Merge area size
borough_spatial_extent = borough_spatial_extent.merge(
    boroughs[["NAME_clean", "Area_km2"]],
    on="NAME_clean",
    how="left"
)


# Add compliance rate per borough

borough_compliance = (
    filtered_df
    .assign(Within6 = filtered_df["FirstPumpArriving_AttendanceTime"] <= 360)
    .groupby("IncGeo_BoroughName")["Within6"]
    .mean()
    .mul(100)
    .reset_index(name="ComplianceRate")
)

borough_spatial_extent = borough_spatial_extent.merge(
    borough_compliance,
    on="IncGeo_BoroughName"
)

# Add incident volume per borough (bubble size)
borough_volume = (
    filtered_df
    .groupby("IncGeo_BoroughName")
    .size()
    .reset_index(name="IncidentCount")
)

borough_spatial_extent = borough_spatial_extent.merge(
    borough_volume,
    on="IncGeo_BoroughName"
)

# Prepare borough-level dataset for inner-outer London comparison
df = borough_spatial_extent.copy()


# Merge official Inner / Outer London classification (ONS)
df = df.merge(
    boroughs[["NAME_clean", "ONS_INNER"]],
    on="NAME_clean",
    how="left"
)

# Map ONS indicator to Inner and Outer London 
df["AreaType"] = df["ONS_INNER"].map({
    "T": "Inner London",
    "F": "Outer London"
})


# Split dataset by Inner and Outer London 
inner_df = df[df["AreaType"] == "Inner London"]
outer_df = df[df["AreaType"] == "Outer London"]


# ---------------------------------------------------------------------
# Barplot
st.subheader("Median Response Time: Inner vs. Outer London")


# Prepare Inner vs Outer dataframe
inner_outer_df = (
    borough_spatial_extent
    .merge(
        boroughs[["NAME_clean", "ONS_INNER"]],
        on="NAME_clean",
        how="left"
    )
)

# Map readable labels
inner_outer_df["AreaType"] = inner_outer_df["ONS_INNER"].map({
    "T": "Inner London",
    "F": "Outer London"
})


# Aggregate
inner_outer_summary = (
    inner_outer_df
    .groupby("AreaType")["MedianResponseMinutes"]
    .mean()
    .reset_index()
)


# Set categorical order
inner_outer_summary["AreaType"] = pd.Categorical(
    inner_outer_summary["AreaType"],
    categories=["Inner London", "Outer London"],
    ordered=True
)

# Optional defensive sorting (not required but clean)
inner_outer_summary = inner_outer_summary.sort_values("AreaType")


# Plot 

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_SMALL))

# Colorblind-safe palette (2 colors)
palette = sns.color_palette("colorblind", 2)

ax = sns.barplot(
    data=inner_outer_summary,
    y="AreaType",
    x="MedianResponseMinutes",
    palette=palette
)

plt.title(
    "Median Response Time: Inner vs Outer London",
    weight="bold"
)

plt.xlabel("Median Response Time (minutes)")
plt.ylabel("")

# Value labels
for i, v in enumerate(inner_outer_summary["MedianResponseMinutes"]):
    ax.text(v + 0.03, i, f"{v:.2f} min", va="center")

sns.despine(left=True, bottom=True)

plt.tight_layout()
st.pyplot(fig)


# Dynamic Markdown Inner vs Outer London

inner_value = inner_outer_summary.loc[
    inner_outer_summary["AreaType"] == "Inner London",
    "MedianResponseMinutes"
].values[0]

outer_value = inner_outer_summary.loc[
    inner_outer_summary["AreaType"] == "Outer London",
    "MedianResponseMinutes"
].values[0]

difference_minutes = outer_value - inner_value
difference_seconds = difference_minutes * 60
percent_difference = (difference_minutes / inner_value) * 100

# Format gap text intelligently
if abs(difference_minutes) >= 1:
    gap_text = f"{difference_minutes:.2f} minutes"
else:
    gap_text = f"{difference_seconds:.0f} seconds"

st.markdown(f"""

**Key Insight:**

- Outer London has a higher median response time **({outer_value:.2f} min)** 
  than Inner London **({inner_value:.2f} min)**.
- The gap of **{gap_text} ({percent_difference:.1f}% difference)**  highlights how borough density and travel distance directly affect response performance.
""")

with st.expander("Show Inner vs Outer Borough Classification"):

    
    import folium

    st.subheader("Classification: Inner vs Outer London")

    st.markdown("""London’s Inner and Outer boroughs, according to official ONS classification.
    """)
    
    # Map T/F to readable labels
    boroughs["AreaType"] = boroughs["ONS_INNER"].map({
        "T": "Inner London",
        "F": "Outer London"
    })

    # Create base map
    m = folium.Map(
    location=[51.5074, -0.1278],
    zoom_start=10,
    min_zoom=10,
    max_zoom=10,
    zoom_control=False,      # removes zoom buttons
    scrollWheelZoom=False,   # disables mouse wheel zoom
    dragging=False,          # disables map dragging
    doubleClickZoom=False,   # disables double-click zoom
    touchZoom=False          # disables touch zoom (mobile)
)

    # Color mapping
    color_dict = {
        "Inner London": "#1f77b4",  # blue
        "Outer London": "#2ca02c"   # green
    }

    def style_function(feature):
        area = feature["properties"]["AreaType"]
        return {
            "fillColor": color_dict.get(area, "gray"),
            "color": "black",
            "weight": 0.8,
            "fillOpacity": 0.7
        }

    # Add GeoJSON layer
    folium.GeoJson(
        boroughs,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["NAME", "AreaType"],
            aliases=["Borough:", "Classification:"]
        )
    ).add_to(m)

    # Add custom legend
    legend_html = """
    <div style="
    position: fixed; 
    bottom: 50px; left: 50px; 
    width: 180px; 
    background-color: white; 
    border:2px solid grey; 
    z-index:9999; 
    font-size:14px;
    padding: 10px;
    ">
    <b>London Classification</b><br>
    <i style="background:#1f77b4;width:12px;height:12px;display:inline-block;"></i> Inner London<br>
    <i style="background:#2ca02c;width:12px;height:12px;display:inline-block;"></i> Outer London
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    # Render
    st.components.v1.html(m._repr_html_(), height=600)
    
# ---------------------------------------------------------------------
st.markdown("---")
# ---------------------------------------------------------------------


st.header("2. How Are Demand and Response Performance Distributed Across Boroughs?")


# INTERACTIVE GEOGRAPHIC PERFORMANCE MAP


# Calculate median response time per borough
median_response_by_borough = (
    filtered_df
    .groupby("IncGeo_BoroughName")["FirstPumpArriving_AttendanceTime"]
    .median()
    .div(60)
    .reset_index(name="MedianResponseMinutes")
)


# Calculate compliance rate per borough
compliance_by_borough = (
    filtered_df
    .groupby("IncGeo_BoroughName")["FirstPump_Within_6min"]
    .mean()
    .mul(100)
    .reset_index(name="CompliancePercent")
)

# Calculate incident volume per borough
incident_volume_by_borough = (
    filtered_df
    .groupby("IncGeo_BoroughName")
    .size()
    .reset_index(name="IncidentCount")
)

#Normalize (clean) Borough Names for Merging response time (uppercase + trim)
boroughs["NAME_clean"] = (
    boroughs["NAME"]
    .str.strip()
    .str.upper()
)

median_response_by_borough["IncGeo_BoroughName_clean"] = (
    median_response_by_borough["IncGeo_BoroughName"]
    .str.strip()
    .str.upper()
)

# Merge Median Response Times with Geodataframe
boroughs = boroughs.merge(
    median_response_by_borough,
    left_on="NAME_clean",
    right_on="IncGeo_BoroughName_clean",
    how="left"
)

#Normalize (clean) Borough Names for Merging compliance rate (uppercase + trim)
compliance_by_borough["NAME_clean"] = (
    compliance_by_borough["IncGeo_BoroughName"]
    .str.strip()
    .str.upper()
)

# Remove old column if rerun (avoid _x/_y)
boroughs = boroughs.drop(
    columns=["CompliancePercent"],
    errors="ignore"
)

# Merge Compliance Rate with Geodataframe
boroughs = boroughs.merge(
    compliance_by_borough[["NAME_clean", "CompliancePercent"]],
    on="NAME_clean",
    how="left"
)

# Round Values for Tooltip Display
boroughs["CompliancePercent"] = boroughs["CompliancePercent"].round(1)

# Delete redundant and empty Columns
boroughs = boroughs.drop(columns=[
    "IncGeo_BoroughName_clean",
    "SUB_2009",
    "SUB_2006",
    ], errors="ignore")

# Clean names in incident volume df
incident_volume_by_borough["NAME_clean"] = (
    incident_volume_by_borough["IncGeo_BoroughName"]
    .str.strip()
    .str.upper()
)

# Remove old column if rerun (avoid _x/_y)
boroughs = boroughs.drop(
    columns=["IncidentCount"],
    errors="ignore"
)

# Merge Incident Volume with Geodataframe
boroughs = boroughs.merge(
    incident_volume_by_borough[["NAME_clean", "IncidentCount"]],
    on="NAME_clean",
    how="left"
)

# Metric Toggle
metric_choice = st.radio(
    "Select Geographic Metric",
    ["Incident Volume", "Median Response Time", "Compliance Rate"],
    horizontal=True,
    key="geo_metric_toggle"
)

# # Non-interactive (fixed London view) dashboard map 
m = folium.Map(
    location=[51.5074, -0.1278],
    zoom_start=10,
    min_zoom=10,
    max_zoom=10,
    zoom_control=False,      # removes zoom buttons
    scrollWheelZoom=False,   # disables mouse wheel zoom
    dragging=False,          # disables map dragging
    doubleClickZoom=False,   # disables double-click zoom
    touchZoom=False          # disables touch zoom (mobile
)

# Select Metric
if metric_choice == "Median Response Time":
    value_column = "MedianResponseMinutes"
    legend_name = "Median Response Time (minutes)"
    fill_color = "YlOrRd"

    boroughs["MedianResponse_display"] = (
        boroughs["MedianResponseMinutes"]
        .round(2)
        .astype(str) + " min"
    )

    tooltip_fields = ["NAME", "MedianResponse_display"]
    tooltip_aliases = ["Borough:", "Median Response Time:"]

elif metric_choice == "Compliance Rate":
    value_column = "CompliancePercent"
    legend_name = "Compliance Rate (%)"
    fill_color = "YlGn"

    boroughs["CompliancePercent_display"] = (
        boroughs["CompliancePercent"]
        .round(1)
        .astype(str) + "%"
    )
    
    tooltip_fields = ["NAME", "CompliancePercent_display"]
    tooltip_aliases = ["Borough:", "Compliance Rate:"]

else:  # Incident Volume

    value_column = "IncidentCount"
    legend_name = "Incident Volume"
    fill_color = "Blues"

    boroughs["IncidentVolume_display"] = (
        boroughs["IncidentCount"]
        .astype(int)
        .astype(str)
    )

    tooltip_fields = ["NAME", "IncidentVolume_display"]
    tooltip_aliases = ["Borough:", "Number of Incidents:"]

# Map

# Choropleth Layer
folium.Choropleth(
    geo_data=boroughs,
    data=boroughs,
    columns=["NAME", value_column],
    key_on="feature.properties.NAME",
    fill_color=fill_color,
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=legend_name
).add_to(m)


# Tooltip Layer 
folium.GeoJson(
    boroughs,
    style_function=lambda x: {
        "fillOpacity": 0,
        "color": "black",
        "weight": 0.5
    },
    tooltip=folium.GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=tooltip_aliases,
        style=(
            "background-color: white;"
            "color: black;"
            "font-family: Arial;"
            "font-size: 12px;"
            "padding: 6px;"
        ),
        localize=True
    )
).add_to(m)


st_folium(m, use_container_width=True, height=600)

# ---------------------------------------------------
# Map Insight (direct interpretation of the map) 
if metric_choice == "Incident Volume":
    st.markdown(f"""
    **Map Insight**

    - **Westminster** records the highest incident volume.
    - Incident demand is concentrated in major central boroughs.
    - Outer boroughs show comparatively lower incident numbers
    """)

elif metric_choice == "Median Response Time":
    st.markdown(f"""
    **Map Insight**

    - Median response times vary across boroughs.  
    - Longer response times are clustered in larger outer boroughs
    - Central areas generally demonstrate faster attendance.
    """)
elif metric_choice == "Compliance Rate":
    st.markdown(f"""
    **Map Insight**

    - 6-minute compliance varies across boroughs.  
    - Higher compliance rates cluster in central boroughs, 
    - whereas several outer boroughs show lower target achievement.
    """)


st.markdown("")


# ---------------------------------------------------------------------
# Expandable Compliance Rate Ranking

with st.expander("Incident Volume by Borough Ranking"):

    st.subheader("Borough Ranking: Incident Volume")

    incident_volume_by_borough = (
        filtered_df
        .groupby("IncGeo_BoroughName")
        .size()
        .reset_index(name="IncidentCount")
        .sort_values("IncidentCount", ascending=False)  
    )

    fig, ax = plt.subplots(figsize=(10, 12))

    sns.barplot(
        data=incident_volume_by_borough,
        y="IncGeo_BoroughName",
        x="IncidentCount",
        order=incident_volume_by_borough["IncGeo_BoroughName"],
        palette="Blues_r",   
        ax=ax
    )

    ax.set_xlabel("Number of Incidents")
    ax.set_ylabel("")

    sns.despine()

    st.pyplot(fig)


    # Dynamic Ranking Insight

    highest_borough = incident_volume_by_borough.iloc[0]
    lowest_borough = incident_volume_by_borough.iloc[-1]

    spread = (
        highest_borough["IncidentCount"]
        - lowest_borough["IncidentCount"]
    )

    st.markdown(f"""
    **Ranking Insight**

    - Highest incident volume: **{highest_borough['IncGeo_BoroughName']}** ({highest_borough['IncidentCount']:,})
    - Lowest incident volume: **{lowest_borough['IncGeo_BoroughName']}** ({lowest_borough['IncidentCount']:,})
    - Volume spread: **{spread:,} incidents**
    """)

# ---------------------------------------------------
# Expandable Response Time Ranking

with st.expander("Show Response Time by Borough Ranking"):

    st.subheader("Borough Ranking: Median Response Time")

    # Median berechnen und sauber sortieren
    median_response_by_borough = (
        filtered_df
        .groupby("IncGeo_BoroughName")["FirstPumpArriving_AttendanceTime"]
        .median()
        .div(60)
        .reset_index(name="MedianResponseMinutes")
        .sort_values("MedianResponseMinutes", ascending=True)
    )

    sns.set_theme(style="white")

    fig, ax = plt.subplots(figsize=(10, 12))

    # Green (fast) → Red (slow)
    palette = sns.color_palette(
        "RdYlGn_r",   
        n_colors=len(median_response_by_borough)
    )

    sns.barplot(
        data=median_response_by_borough,
        y="IncGeo_BoroughName",
        x="MedianResponseMinutes",
        order=median_response_by_borough["IncGeo_BoroughName"],  
        palette=palette,
        ax=ax
    )

    # 6-Minuten-Reference Line
    ax.axvline(
        6,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7
    )

    # Put Text next to the Reference line (left)
    ax.text(
        5.95,                                 
        -0.8,                                  
        "6-minute target",
        fontsize=10,
        ha="right"
    )

    ax.set_xlabel("Median Response Time (minutes)")
    ax.set_ylabel("")

    sns.despine()
    fig.tight_layout()

    st.pyplot(fig)

    # Dynamic Ranking Insight

    ranking_df = median_response_by_borough.copy()

    top_fast = ranking_df.iloc[0]
    top_slow = ranking_df.iloc[-1]

    spread = (
        top_slow["MedianResponseMinutes"]
        - top_fast["MedianResponseMinutes"]
    )

    st.markdown(f"""
    **Ranking Insight**

    - Fastest borough: **{top_fast['IncGeo_BoroughName']}**
      ({top_fast['MedianResponseMinutes']:.2f} min)

    - Slowest borough: **{top_slow['IncGeo_BoroughName']}**
      ({top_slow['MedianResponseMinutes']:.2f} min)

    - Overall performance spread: **{spread:.2f} minutes**

    This variation reflects geographic differences 
    in median response performance across boroughs.
    """)

# ---------------------------------------------------------------------
# Expandable Compliance Rate Ranking

with st.expander("Show Compliance Rate by Borough Ranking"):
    
    st.subheader(" Borough Ranking: Compliance Rate (Response Time ≤ 6 minutes)")

    borough_compliance = (
        filtered_df
        .assign(Within6 = filtered_df["FirstPumpArriving_AttendanceTime"] <= 360)
        .groupby("IncGeo_BoroughName")["Within6"]
        .mean()
        .mul(100)
        .reset_index(name="ComplianceRate")
        .sort_values("ComplianceRate", ascending=False)  # highest compliance on top
    )

    fig, ax = plt.subplots(figsize=(10, 12))

    palette = sns.color_palette("RdYlGn_r", len(borough_compliance))

    sns.barplot(
        data=borough_compliance,
        y="IncGeo_BoroughName",
        x="ComplianceRate",
        order=borough_compliance["IncGeo_BoroughName"],  # force correct order
        palette=palette,
        ax=ax
    )

    ax.set_xlabel("Compliance Rate (%)")
    ax.set_ylabel("")

    sns.despine()
    fig.tight_layout()

    st.pyplot(fig)

    # Dynamic Ranking Insight
    top_high = borough_compliance.iloc[0]
    top_low = borough_compliance.iloc[-1]
    gap = top_high["ComplianceRate"] - top_low["ComplianceRate"]

    st.markdown(f"""
    **Ranking Insight**

    - Highest compliance: **{top_high['IncGeo_BoroughName']}** ({top_high['ComplianceRate']:.1f}%)
    - Lowest compliance: **{top_low['IncGeo_BoroughName']}** ({top_low['ComplianceRate']:.1f}%)
    - Compliance differs by up to **{gap:.1f}%** across boroughs.

    This variation reflects geographic differences 
    in 6-minute target compliance across boroughs.
    """)

# ---------------------------------------------------------------------
st.markdown("---")
# ---------------------------------------------------------------------


st.header("3. What Drives Geographic Differences in Response Time?")


# ---------------------------------------------------------------------
# Borough Size vs. Median Response Time by Area Type
# Investigates structural relationship between geographic size and performance

st.markdown("### 3.1 Impact on Response Time")
st.markdown("") 
st.subheader("Borough Size vs. Median Response Time by Area Type")


# Linear Regression (Borough Area vs Median Response Time
slope, intercept, r_value, p_value, std_err = linregress(
    df["Area_km2"],
    df["MedianResponseMinutes"]
)

# Statistical calculations
r = r_value
p = p_value
r_squared = r_value ** 2
r_squared_percent = r_squared * 100

x_range = np.linspace(
    df["Area_km2"].min(),
    df["Area_km2"].max(),
    100
)

y_range = slope * x_range + intercept



# Within-group regression (Inner vs Outer London)
_, _, r_inner_val, p_inner, _ = linregress(
    inner_df["Area_km2"],
    inner_df["MedianResponseMinutes"]
)

_, _, r_outer_val, p_outer, _ = linregress(
    outer_df["Area_km2"],
    outer_df["MedianResponseMinutes"]
)



# Colorblind-friendly palette
inner_color = sns.color_palette("colorblind")[1]
outer_color = sns.color_palette("colorblind")[2]



# Create scatter plot
fig = go.Figure()



# Inner London scatter
fig.add_trace(go.Scatter(
    x=inner_df["Area_km2"],
    y=inner_df["MedianResponseMinutes"],
    mode="markers",
    marker=dict(
        size=12,
        color="#1f77b4",
        line=dict(width=1.2, color="black"),
        opacity=0.85
    ),
    name="Inner London",
    customdata=np.stack(
        (
            inner_df["NAME_clean"],
            inner_df["Area_km2"],
            inner_df["MedianResponseMinutes"],
            inner_df["AreaType"]
        ),
        axis=-1
    ),
    hovertemplate=
    "<b>%{customdata[0]}</b><br><br>" +
    "Median Response Time: %{customdata[2]:.1f} min<br>" +
    "Area: %{customdata[1]:.1f} km²<br>" +
    "Area Type: %{customdata[3]}" +
    "<extra></extra>"
))



# Outer London scatter
fig.add_trace(go.Scatter(
    x=outer_df["Area_km2"],
    y=outer_df["MedianResponseMinutes"],
    mode="markers",
    marker=dict(
        size=12,
        color="#ff7f0e",
        line=dict(width=1.2, color="black"),
        opacity=0.85
    ),
    name="Outer London",
    customdata=np.stack(
        (
            outer_df["NAME_clean"],
            outer_df["Area_km2"],
            outer_df["MedianResponseMinutes"],
            outer_df["AreaType"]
        ),
        axis=-1
    ),
    hovertemplate=
       "<b>%{customdata[0]}</b><br><br>" +
       "Median Response Time: %{customdata[2]:.1f} min<br>" +
       "Area: %{customdata[1]:.1f} km²<br>" +
       "Area Type: %{customdata[3]}" +
       "<extra></extra>"
))

# Regression line
fig.add_trace(go.Scatter(
    x=x_range,
    y=y_range,
    mode="lines",
    line=dict(color="black", width=2),
    showlegend=False   
))

# Layout
fig.update_layout(
    height=650,
    xaxis_title="Borough Area (km²)",
    yaxis_title="Median Response Time (minutes)",
    template="simple_white",
    legend_title_text="Area Type"
)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

st.plotly_chart(fig, use_container_width=True)


# Regression Summary Table 
def format_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def interpret_strength(r):
    abs_r = abs(r)
    if abs_r >= 0.7:
        return "Strong"
    elif abs_r >= 0.4:
        return "Moderate"
    elif abs_r >= 0.2:
        return "Weak"
    else:
        return "Very weak"


def significance_label(p):
    return "Yes" if p < 0.05 else "No"

def interpret_direction(r):
    return "Negative" if r < 0 else "Positive"


summary_table = pd.DataFrame({
    "Group": [
        "All Boroughs",
        "Inner London",
        "Outer London"
    ],
    "Correlation (r)": [
        round(r, 2),
        round(r_inner_val, 2),
        round(r_outer_val, 2)
    ],
    "R²": [
        round(r_squared, 2),
        round(r_inner_val**2, 2),
        round(r_outer_val**2, 2)
    ],
    "p-value": [
        format_p(p),
        format_p(p_inner),
        format_p(p_outer)
    ],
    "Statistically Significant (α = 0.05)": [
        significance_label(p),
        significance_label(p_inner),
        significance_label(p_outer)
    ],
    "Effect Strength": [
        interpret_strength(r),
        interpret_strength(r_inner_val),
        interpret_strength(r_outer_val)
    ]
})

with st.expander("Show Regression Summary (Statistical Details)"):

    col1, col2, col3 = st.columns(3)


    # All Boroughs
    with col1:
        st.markdown("### All Boroughs")
        st.metric("Correlation (r)", f"{r:.2f}")
        st.metric("R²", f"{r_squared:.2f}")
        st.metric("p-value", format_p(p))
        st.metric("Statistically Significant", significance_label(p))
        st.metric("Effect Strength", interpret_strength(r))
        st.metric("Effect Direction", interpret_direction(r))


    # Inner London
    with col2:
        st.markdown("### Inner London")
        st.metric("Correlation (r)", f"{r_inner_val:.2f}")
        st.metric("R²", f"{r_inner_val**2:.2f}")
        st.metric("p-value", format_p(p_inner))
        st.metric("Statistically Significant", significance_label(p_inner))
        st.metric("Effect Strength", interpret_strength(r_inner_val))
        st.metric("Effect Direction", interpret_direction(r))


    # Outer London
    with col3:
        st.markdown("### Outer London")
        st.metric("Correlation (r)", f"{r_outer_val:.2f}")
        st.metric("R²", f"{r_outer_val**2:.2f}")
        st.metric("p-value", format_p(p_outer))
        st.metric("Statistically Significant", significance_label(p_outer))
        st.metric("Effect Strength", interpret_strength(r_outer_val))
        st.metric("Effect Direction", interpret_direction(r))

# ---------------------------------------------------------------------
# Dynamic Markdown

significance_text = (
    "statistically significant"
    if p < 0.05
    else "not statistically significant"
)

significance_inner = (
    "statistically significant"
    if p_inner < 0.05
    else "not statistically significant"
)

significance_outer = (
    "statistically significant"
    if p_outer < 0.05
    else "not statistically significant"
)


st.markdown(f"""
**Key Insight:**

- Borough size is **strongly associated** with median response time 
  **(r = {r:.2f}, R² = {r_squared:.2f})**, explaining approximately 
  **{r_squared_percent:.0f}%** of the observed variation 
  **(p {format_p(p)}; {significance_text})**; {significance_text}).
- The positive relationship remains observable within both groups,
  Inner London **(r = {r_inner_val:.2f}; {significance_inner})** and 
  Outer London **(r = {r_outer_val:.2f}; {significance_outer})**.
- This suggests that borough size contributes to response performance 
  regardless of structural classification.
""")

# ---------------------------------------------------------------------
st.markdown("---")
# ---------------------------------------------------------------------
st.markdown("### 3.2 Impact on Target Compliance")
st.markdown("") 
st.subheader("Borough Size vs. 6-Minute Compliance Rate")

df_comp = borough_spatial_extent.copy()

# Regression
slope_c, intercept_c, r_c, p_c, std_err_c = linregress(
    df_comp["Area_km2"],
    df_comp["ComplianceRate"]
)

r2_c = r_c ** 2

# Interpretation helpers for expander

strength_c = (
    "very strong" if abs(r_c) >= 0.7 else
    "strong" if abs(r_c) >= 0.5 else
    "moderate" if abs(r_c) >= 0.3 else
    "weak"
)

direction_c = "Negative" if r_c < 0 else "Positive"

significance_c = (
    "statistically significant"
    if p_c < 0.05
    else "not statistically significant"
)

x_range_c = np.linspace(
    df_comp["Area_km2"].min(),
    df_comp["Area_km2"].max(),
    100
)

y_range_c = slope_c * x_range_c + intercept_c

fig_comp = go.Figure()

# Scatter
fig_comp.add_trace(go.Scatter(
    x=df_comp["Area_km2"],
    y=df_comp["ComplianceRate"],
    mode="markers",
    marker=dict(
        size=12,
        color=df_comp["ComplianceRate"],
        colorscale="YlGn",
        line=dict(width=1, color="black"),
        opacity=0.8,
        colorbar=dict(title="Compliance Rate (%)")
    ),
    text=df_comp["IncGeo_BoroughName"],
    hovertemplate=
        "<b>%{text}</b><br><br>" +
        "Compliance: %{y:.1f}%<br>" +
        "Area: %{x:.1f} km²<br>" +
        "<extra></extra>",
    showlegend=False
))

# Regression line
fig_comp.add_trace(go.Scatter(
    x=x_range_c,
    y=y_range_c,
    mode="lines",
    line=dict(color="black", width=2),
    showlegend=False
))

fig_comp.update_layout(
    height=600,
    xaxis_title="Borough Area (km²)",
    yaxis_title="Compliance Rate (%)",
    template="simple_white"
)

st.plotly_chart(fig_comp)


# Regression Summary Expander (Statistical Details)

with st.expander("Show Regression Summary (Statistical Details)"):

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Correlation (r)", f"{r_c:.3f}")
        st.metric("R²", f"{r2_c:.3f}")
        st.metric("p-value", f"{format_p(p)}")
        
    with col2:
        st.metric("Statistically Significant", "Yes" if p_c < 0.05 else "No" )
        st.metric("Effect Strength", strength_c.capitalize())
        st.metric("Effect Direction", direction_c)

# ----------------------------------------------------------
# Dynamic Markdown Size vs Compliance

strength_c = (
    "very strong" if abs(r_c) >= 0.7 else
    "strong" if abs(r_c) >= 0.5 else
    "moderate" if abs(r_c) >= 0.3 else
    "weak"
)

significance_c = (
    "statistically significant"
    if p_c < 0.05
    else "not statistically significant"
)

st.markdown(f"""

**Key Insight:**

- The relationship between borough size and 6-minute compliance is **{strength_c} and {direction_c}**
(r = {r_c:.2f}, R² = {r2_c:.2f}).
- This indicates that the impact of borough size is not limited to response time but is also associated with lower target compliance.
- The effect is **{significance_c}** (p {format_p(p)}), indicating that larger boroughs are less likely to meet the 6-minutes response target.
""")

# ---------------------------------------------------------------------
st.markdown("---")
# ---------------------------------------------------------------------


st.markdown("""
### Key Takeaways

- Incident volume is unevenly distributed, with demand concentrated in major central boroughs, likely reflecting
  higher population density, economic activity, and tourism. 
- Borough size is the primary structural driver, strongly associated with both longer response times and lower compliance.
- Inner London boroughs generally outperform Outer London boroughs. However, the size effect persists within both groups,
  suggesting that borough size influences response performance independently of the Inner–Outer grouping.
- Geographic structure explains a significant portion of variation in response times across boroughs. However, performance
  differences cannot be attributed to geography alone and are also influenced by other factors (e.g. operational capacity, station allocation,
  and traffic conditions).
- “Future analysis could incorporate station ground boundaries to assess whether operational allocation
   explains part of the observed geographic performance differences.”
""")

st.markdown("<br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
st.markdown("---")
# ---------------------------------------------------------------------

# Bubbleplot


# Define volume categories using quantiles (for legend only)

low_threshold = borough_spatial_extent["IncidentCount"].quantile(0.33)
high_threshold = borough_spatial_extent["IncidentCount"].quantile(0.66)

low_label = f"Low volume: ≤ {int(low_threshold)} incidents"
medium_label = f"Medium volume: {int(low_threshold)+1}–{int(high_threshold)} incidents"
high_label = f"High volume: > {int(high_threshold)} incidents"

# ----------------------------------------------------------
with st.expander("Show Borough Level Performance Overview"):

    st.subheader("Borough Size vs. Median Response Time")

    df = borough_spatial_extent.copy()

    # ----------------------------------------------------------
    # Linear regression model (Area_km2 as predictor)
    slope, intercept, r_value, p_value, std_err = linregress(
        df["Area_km2"],
        df["MedianResponseMinutes"]
    )

    # Model statistics
    r = r_value
    p = p_value
    r_squared = r_value ** 2

    x_range = np.linspace(
        df["Area_km2"].min(),
        df["Area_km2"].max(),
        100
    )

    y_range = slope * x_range + intercept

    # ----------------------------------------------------------
    # Bubble size scaling

    min_size = 15
    max_size = 60

    size_scaled = (
        (df["IncidentCount"] - df["IncidentCount"].min()) /
        (df["IncidentCount"].max() - df["IncidentCount"].min())
    )

    size_scaled = size_scaled * (max_size - min_size) + min_size

    # ----------------------------------------------------------
    # Create figure

    fig = go.Figure()

    # Main bubble layer
    fig.add_trace(go.Scatter(
        x=df["Area_km2"], 
        y=df["MedianResponseMinutes"],
        mode="markers",
        marker=dict(
            size=size_scaled,
            color=df["ComplianceRate"],
            colorscale="RdYlGn",
            reversescale=False,
            line=dict(width=1.2, color="black"),
            colorbar=dict(
                title="Compliance Rate (%)"
            ),
            opacity=0.75
        ),
        hovertemplate=
            "<b>%{text}</b><br><br>" +
            "Median Response: %{y:.2f} min<br>" +
            "Compliance: %{marker.color:.1f}%<br>" +
            "Incident Count: %{customdata[0]:,.0f}<br>" +
            "Area: %{customdata[1]:.1f} km²" +
            "<extra></extra>",
        text=df["IncGeo_BoroughName"],
        customdata=np.stack(
        (
            df["IncidentCount"],
            df["Area_km2"]
        ),
        axis=-1
    ),
        showlegend=False
    ))

    # Regression line
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False
    ))

    # ----------------------------------------------------------
    # Layout styling

    fig.update_layout(
        height=700,
        width=900,
        xaxis_title="Borough Area (km²)",
        yaxis_title="Median Response Time (minutes)",
        template="simple_white",
    )

    fig.update_traces(marker=dict(opacity=0.75))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)


    # Add annotation

    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        text=(
            f"y = {slope:.4f}x + {intercept:.2f}<br>"
            f"r = {r:.2f} | R² = {r_squared:.2f}<br>"
            f"p = {p:.4f}"
        ),
        showarrow=False,
        align="center",
        font=dict(size=14),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="black",
        borderwidth=1,
        borderpad=8
    )


    # Custom size legend (manual dummy traces)

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=3, color="grey", line=dict(color="black", width=1)),
        name=low_label
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=6, color="grey", line=dict(color="black", width=1)),
        name=medium_label
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=9, color="grey", line=dict(color="black", width=1)),
        name=high_label
    ))

    fig.update_layout(
        legend_title_text="     Bubble Size: Incident Volume",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        )
    )


    st.plotly_chart(fig)

    # ----------------------------------------------------------
    # Dynamic Statistics 

    # Correlation strength
    strength = (
        "very strong" if abs(r) >= 0.7 else
        "strong" if abs(r) >= 0.5 else
        "moderate" if abs(r) >= 0.3 else
        "weak"
    )

    # Correlation direction
    direction = "positive" if r > 0 else "negative"

    # Significance
    if p < 0.05:
        significance = "statistically significant"
    else:
        significance = "not statistically significant"

    # ----------------------------------------------------------
    # Dynamic Markdown

    st.markdown(
        f"""
    - Within the 32 boroughs, a **{strength} {direction} relationship** (r = {r:.2f}, R² = {r_squared:.2f}) appears between borough size
      and median response time.
    - Larger boroughs tend to have longer median response times.
    - The relationship is **{significance}** (p = {p:.7f}).

    """
    )





