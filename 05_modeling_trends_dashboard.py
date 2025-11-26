import streamlit as st
import pandas as pd
import plotly.express as px

# Page setup
st.set_page_config(
    page_title="ASEAN CO₂ Emission Dashboard",
    layout="wide"
)

# Title
st.title("ASEAN CO₂ Emission per Capita")
st.markdown("""
Carbon dioxide (CO₂) emissions from burning fossil fuels and industrial processes.  
Includes emissions from transport, electricity generation, and heating, but excludes land-use change.  

**Data source:** Global Carbon Budget (2024); Population data from various sources (2024)
""")

# Load data
df = pd.read_csv("co2_emission_asean_clean.csv")

# Coordinates
if "latitude" not in df.columns or "longitude" not in df.columns:
    coords = {
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
    df["latitude"] = df["country"].map(lambda x: coords.get(x, (0, 0))[0])
    df["longitude"] = df["country"].map(lambda x: coords.get(x, (0, 0))[1])

# Sidebar Controls
st.sidebar.header("Dashboard Controls")

# Country selector
countries = ["All ASEAN Countries"] + sorted(df["country"].unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", countries)

# Year slider
min_year, max_year = int(df["year"].min()), int(df["year"].max())
# Year sliders for historical & forecast
selected_year_hist = st.sidebar.slider("Select Historical Year", min_year, max_year, 2020)
future_year = st.sidebar.slider("Select Future Year to Forecast", max_year + 1, 2040, 2035)


# Filter based on country
if selected_country != "All ASEAN Countries":
    df_filtered = df[df["country"] == selected_country]
else:
    df_filtered = df.copy()

# Row 1 - Bar Chart
st.subheader(f"CO₂ Emission Comparison Over Years")

bar_df = df_filtered.copy()
bar_df["highlight"] = bar_df["year"].apply(
    lambda x: "Selected Year" if x == selected_year_hist else "Other Years"
)

fig_bar = px.bar(
    bar_df,
    x="year",
    y="co2_per_capita",
    color="highlight",
    color_discrete_map={"Selected Year": "crimson", "Other Years": "lightblue"},
    labels={"co2_per_capita": "CO₂ (tons per capita)", "year": "Year"},
    text_auto=".2f",
    height=500
)
fig_bar.update_traces(textfont_size=10)
fig_bar.update_layout(
    legend_title_text="",
    title=f"CO₂ Emission per Capita ({selected_country}) — Highlighted: {selected_year_hist}",
    xaxis=dict(tickmode="linear", dtick=5),
)
st.plotly_chart(fig_bar, use_container_width=True)

# Row 2 - Line chart + map side by side
col1, col2 = st.columns(2)

# Line chart
with col1:
    st.subheader("CO₂ Emission Trends Over Time")
    fig_line = px.line(
        df_filtered,
        x="year",
        y="co2_per_capita",
        color="country" if selected_country == "All ASEAN Countries" else None,
        labels={"co2_per_capita": "CO₂ (tons per capita)", "year": "Year"},
        height=500
    )
    fig_line.add_vline(x=selected_year_hist, line_dash="dash", line_color="red")
    st.plotly_chart(fig_line, use_container_width=True)

# Map chart
with col2:
    st.subheader(f"CO₂ Emission Distribution — {selected_year_hist}")
    year_df = df[df["year"] == selected_year_hist]

    fig_map = px.scatter_mapbox(
        year_df,
        lat="latitude",
        lon="longitude",
        size="co2_per_capita",
        color="co2_per_capita",
        hover_name="country",
        color_continuous_scale="Reds",
        size_max=40,
        zoom=4,
        height=500
    )
    fig_map.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown("▶️ Use the sidebar to switch between countries and explore annual trends dynamically.")

# Forecast Section
from prophet import Prophet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go

# Prepare filtered data for selected country
if selected_country == "All ASEAN Countries":
    df_filtered_forecast = (
        df.groupby("year", as_index=False)["co2_per_capita"]
        .mean()
        .sort_values("year")
    )
else:
    df_filtered_forecast = (
        df[df["country"] == selected_country][["year", "co2_per_capita"]]
        .sort_values("year")
    )

# Polynomial Regression Forecast
X_hist = df_filtered_forecast["year"].values.reshape(-1, 1)
y_hist = df_filtered_forecast["co2_per_capita"].values.reshape(-1, 1)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_hist)
model = LinearRegression().fit(X_poly, y_hist)

min_year = int(df_filtered_forecast["year"].min())
max_hist_year = int(df_filtered_forecast["year"].max())
future_year_int = int(future_year)

X_future = np.arange(min_year, future_year_int + 1).reshape(-1, 1)
X_future_poly = poly.transform(X_future)
y_pred = model.predict(X_future_poly).flatten()

poly_df = pd.DataFrame({
    "year": np.arange(min_year, future_year_int + 1),
    "poly_prediction": y_pred
})

# Prophet Forecast
df_prophet = df_filtered_forecast.rename(columns={"year": "ds", "co2_per_capita": "y"})
df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")
prophet_model = Prophet(yearly_seasonality=False, daily_seasonality=False)
prophet_model.fit(df_prophet)
future_df = prophet_model.make_future_dataframe(periods=future_year_int - max_hist_year, freq="Y")
forecast = prophet_model.predict(future_df)
forecast_result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_result["year"] = forecast_result["ds"].dt.year

# Combine predictions
compare_df = poly_df.merge(
    forecast_result[["year", "yhat", "yhat_lower", "yhat_upper"]],
    on="year",
    how="outer"
).sort_values("year")

# Add observed data
compare_df = compare_df.merge(df_filtered_forecast, on="year", how="left")

# forecast comparisson
fig = go.Figure()

# Observed data
fig.add_trace(go.Scatter(
    x=compare_df["year"], y=compare_df["co2_per_capita"],
    mode="markers+lines", name="Observed", 
    line=dict(color="#00BFFF", width=2),
    marker=dict(size=6)
))

# Polynomial Regression
fig.add_trace(go.Scatter(
    x=compare_df["year"], y=compare_df["poly_prediction"],
    mode="lines", name="Polynomial Regression", 
    line=dict(color="#FFD700", dash="dot", width=2)
))

# Prophet Forecast
fig.add_trace(go.Scatter(
    x=compare_df["year"], y=compare_df["yhat"],
    mode="lines", name="Prophet Forecast", 
    line=dict(color="#FF4B4B", width=3)
))

# Prophet Uncertainty Band
fig.add_trace(go.Scatter(
    x=pd.concat([compare_df["year"], compare_df["year"][::-1]]),
    y=pd.concat([compare_df["yhat_upper"], compare_df["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(255,75,75,0.15)",
    line=dict(color="rgba(255,255,255,0)"), showlegend=False
))

# Detect Streamlit theme
try:
    theme = st.get_option("theme.base")
except Exception:
    theme = "dark"

# Force consistent themes
fig.update_layout(
    title=dict(
        text=f"CO₂ Emission Forecast — {selected_country}",
        x=0.5,
        font=dict(size=18, color="#f0f0f0")
    ),
    xaxis_title="Year",
    yaxis_title="CO₂ Emission per Capita (tons)",
    hovermode="x unified",
    template="plotly_dark",
    # dark blen background
    plot_bgcolor="#1c1f24",
    paper_bgcolor="#1c1f24",
    font=dict(color="#f0f0f0", size=13),
    legend=dict(
        bgcolor="rgba(28,28,28,0.8)",
        bordercolor="#444444",
        borderwidth=1
    ),
    xaxis=dict(
        gridcolor="#333333",
        linecolor="#777777"
    ),
    yaxis=dict(
        gridcolor="#333333",
        linecolor="#777777"
    ),
)

# Show forecast chart
st.subheader("CO₂ Emission Forecast Comparison")
st.plotly_chart(fig, use_container_width=True)

# Optional expandable forecast data
with st.expander("Show Forecast Data Table"):
    st.dataframe(compare_df.style.format({
        "poly_prediction": "{:.2f}",
        "yhat": "{:.2f}",
        "yhat_lower": "{:.2f}",
        "yhat_upper": "{:.2f}",
        "co2_per_capita": "{:.2f}"
    }))

