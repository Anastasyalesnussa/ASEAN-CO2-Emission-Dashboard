import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------------------------------
# 🧭 PAGE SETUP
# -----------------------------------------------------
st.set_page_config(
    page_title="ASEAN CO₂ Emission Forecast Dashboard",
    layout="wide"
)

st.title("🌏 ASEAN CO₂ emission per capita")
st.markdown("""
Carbon dioxide (CO₂) emissions from burning fossil fuels and industrial processes. This includes emissions from transport, electricity
generation, and heating, but not land-use change.

Data source: Global Carbon Budget (2024); Population based on various sources (2024)
""")

# -----------------------------------------------------
# 🧩 LOAD DATA
# -----------------------------------------------------
df = pd.read_csv("co2_emission_asean_clean.csv")

# Check if the file contains coordinates; if not, we can assign manually for map plotting
if "latitude" not in df.columns or "longitude" not in df.columns:
    country_coords = {
        "Indonesia": (-0.7893, 113.9213),
        "Malaysia": (4.2105, 101.9758),
        "Thailand": (15.8700, 100.9925),
        "Vietnam": (14.0583, 108.2772),
        "Philippines": (12.8797, 121.7740),
        "Singapore": (1.3521, 103.8198),
        "Myanmar": (21.9162, 95.9560),
        "Cambodia": (12.5657, 104.9910),
        "Laos": (19.8563, 102.4955),
        "Brunei": (4.5353, 114.7277)
    }
    df["latitude"] = df["country"].map(lambda x: country_coords.get(x, (0, 0))[0])
    df["longitude"] = df["country"].map(lambda x: country_coords.get(x, (0, 0))[1])

# -----------------------------------------------------
# ⚙️ SIDEBAR CONTROLS
# -----------------------------------------------------
view_mode = st.sidebar.radio(
    "Select Visualization Type",
    ["🗺️ Map", "📈 Line Chart", "📊 Bar Chart"]
)

# Use slider for timeline across all charts
min_year, max_year = int(df["year"].min()), int(df["year"].max())
selected_year = st.sidebar.slider("Select Year (for time-lapse)", min_year, max_year, 2020, step=1)

# -----------------------------------------------------
# 🌍 MAP VIEW
# -----------------------------------------------------
if view_mode == "🗺️ Map":
    st.subheader(f"CO₂ Emissions per Capita — {selected_year}")
    year_df = df[df["year"] == selected_year]

    fig = px.scatter_mapbox(
    year_df,
    lat="latitude",
    lon="longitude",
    size="co2_per_capita",
    color="co2_per_capita",
    hover_name="country",
    color_continuous_scale="Reds",
    size_max=40,
    zoom=4,
    height=600
    )
    fig.update_layout(
    mapbox_style="carto-positron",
    margin={"r":0,"t":30,"l":0,"b":0}
)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("▶️ Use the **slider** on the sidebar to simulate a time-lapse effect year by year.")

# -----------------------------------------------------
# 📈 LINE CHART VIEW
# -----------------------------------------------------
elif view_mode == "📈 Line Chart":
    st.subheader("Historical CO₂ Emissions Trends")

    fig = px.line(
        df,
        x="year",
        y="co2_per_capita",
        color="country",
        title="CO₂ Emission per Capita Over Time (ASEAN)",
        labels={"co2_per_capita": "CO₂ (tons per capita)", "year": "Year"},
        height=600
    )
    fig.add_vline(x=selected_year, line_dash="dash", line_color="red")

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("▶️ Move the slider to change the focus year and animate trends over time.")

# -----------------------------------------------------
# 📊 BAR CHART VIEW
# -----------------------------------------------------
elif view_mode == "📊 Bar Chart":
    st.subheader(f"CO₂ Emission Comparison — {selected_year}")

    year_df = df[df["year"] == selected_year].sort_values(by="co2_per_capita", ascending=False)

    fig = px.bar(
        year_df,
        x="country",
        y="co2_per_capita",
        color="country",
        text_auto=".2f",
        title=f"CO₂ Emission per Capita ({selected_year})",
        labels={"co2_per_capita": "CO₂ (tons per capita)", "country": "Country"},
        height=600
    )
    fig.update_traces(textfont_size=12)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("▶️ Adjust the slider to see emissions ranking by year.")
