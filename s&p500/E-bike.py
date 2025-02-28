# Import libraries
import gdown

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely import wkt
import ipywidgets as widgets
from IPython.display import display
import h3
import geopandas as gpd
import folium
import plotly.graph_objects as go

import pydeck as pdk

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import panel as pn

pn.extension()

# Convert the Google Drive link to a direct download link
url = 'https://drive.google.com/uc?id=11zxZiKfNG65YbU2xP7HuPojRzh_P-Ztu'
output = 'lyft_data.csv'

# Download the file
gdown.download(url, output, quiet=False)

# Load the CSV into a Pandas DataFrame stored in trip_data_2019
lyft_data = pd.read_csv(output)

# Convert the Google Drive link to a direct download link
url = 'https://drive.google.com/uc?id=18tuorMTVZEtzdVBbJhgibapGZ7GvIxjQ'
output = 'lime_data.csv'

# Download the file
gdown.download(url, output, quiet=False)

# Load the CSV into a Pandas DataFrame stored in trip_data_2019
lime_data = pd.read_csv(output)

# Prepare the dataframe for SPATIAL Forecasting and Plotting

lime_df = pd.DataFrame(lime_data)
lime_df['Start Time'] = pd.to_datetime(lime_df['Start Time'])


# Extract date-related features
lime_df["month"] = lime_df["Start Time"].dt.month
lime_df["date"] = lime_df["Start Time"].dt.date
lime_df["weekday"] = lime_df["Start Time"].dt.weekday  # 0 = Monday, 6 = Sunday
lime_df["hour"] = lime_df["Start Time"].dt.hour

# Prepare the dataframe for TEMPORAL Forcasting and Plotting

lyft_df = pd.DataFrame(lyft_data)
lyft_df['Start Time'] = pd.to_datetime(lyft_df['Start Time'])


# Extract date-related features
lyft_df["month"] = lyft_df["Start Time"].dt.month
lyft_df["date"] = lyft_df["Start Time"].dt.date
lyft_df["weekday"] = lyft_df["Start Time"].dt.weekday  # 0 = Monday, 6 = Sunday
lyft_df["hour"] = lyft_df["Start Time"].dt.hour

# Group by date and location and count trips.
spt_daily_trips = (
    lime_df.groupby(
        [lime_df['Start Time'].dt.date, 'Start Centroid Location'] # Group by day and location
        ).agg(trip_count=('Trip ID', 'count') # Count trips in the spatiotemporal groups
        ).reset_index()
)

# Rename the date column (originally from the dt.date extraction) to "date"
spt_daily_trips.rename(columns={'Start Time': 'date'}, inplace=True)

# Now, extract +++full-precision+++ lat and lon directly from the "Start Centroid Location" WKT string.
# Shapely will preserve the full decimal precision as given in the WKT (Rounding let to weird grouping).
spt_daily_trips['lat'] = (spt_daily_trips['Start Centroid Location']
                          .apply(lambda pt: wkt.loads(pt).y))
spt_daily_trips['lon'] = (spt_daily_trips['Start Centroid Location']
                          .apply(lambda pt: wkt.loads(pt).x))

# Create a hexbin map of the scooter trips in Chicago
# We will use the H3 library (Uber) to convert lat/lon to hex IDs.
resolution = 7  # Lower resolution = bigger hexagons.
trips_h3 = []
for idx, row in spt_daily_trips.iterrows():
    lon = row["lon"]
    lat = row["lat"]
    # Compute H3 cell from the exact lat and lon coordinates:
    hex_id = h3.latlng_to_cell(lat, lon, resolution)
    trips_h3.append({
        "hex_id": hex_id,
        "trip_count": row["trip_count"],
        "date": row["date"]
    })
    # We have now created trips_h3 with trips per day being binned spatially into hexes!

df_hex = pd.DataFrame(trips_h3)
hex_trips = df_hex.groupby(["hex_id", "date"])["trip_count"].sum().reset_index()
hex_trips.columns = ["hex_id", "date", "trip_count"]

hex_trips.to_csv('hex_trips.csv', index=False)

# Get the map data of Chicago

# Convert the Google Drive link to a direct download link
url = 'https://drive.google.com/uc?id=1KrUyLtr8_KN_ZXJIjxVMS_yANg2yNYz8'
output = 'CommAreas_20250226.csv'

# Download the file
gdown.download(url, output, quiet=False)

# Load the CSV into a Pandas DataFrame stored for the deck script to use
comm_areas = pd.read_csv(output)
comm_areas.to_csv('CommAreas_20250226.csv', index=False)

import pandas as pd
import pydeck as pdk
import panel as pn
import json
from shapely import wkt
from shapely.geometry import mapping

pn.extension('deckgl')

# Load CSVs (Assuming you've uploaded them manually)
hex_trips = pd.read_csv("hex_trips.csv", parse_dates=["date"])
comm_areas = pd.read_csv("CommAreas_20250226.csv")

# Process hex_trips
hex_trips["year_month"] = hex_trips["date"].dt.to_period("M").astype(str)
monthly_data = hex_trips.groupby(["hex_id", "year_month"], as_index=False)["trip_count"].sum()
monthly_groups = {ym: df.drop(columns="year_month") for ym, df in monthly_data.groupby("year_month")}
available_months = sorted(monthly_groups.keys())
# Process GeoJSON from Community Areas
features = []
for _, row in comm_areas.iterrows():
    shape = wkt.loads(row["the_geom"])  # Ensure "the_geom" exists
    geom_geojson = mapping(shape)
    feature = {"type": "Feature", "properties": {}, "geometry": geom_geojson}
    features.append(feature)

chicago_fc = {"type": "FeatureCollection", "features": features}

# Function to compute RGBA color
def compute_rgba(count, cmin, cmax):
    if cmax == cmin:
        return (128, 128, 128, 255)
    ratio = (count - cmin) / (cmax - cmin)
    return (int(255 * ratio), int(255 * (1 - ratio)), 0, 255)

# Function to build Pydeck map
def build_deck_for_month(ym_str):
    df = monthly_groups.get(ym_str, pd.DataFrame({"hex_id": [], "trip_count": []}))
    cmin, cmax = df["trip_count"].min(), df["trip_count"].max() if not df.empty else (0, 1)

    df = df.copy()
    df[["colorR", "colorG", "colorB", "colorA"]] = df.apply(lambda row: compute_rgba(row["trip_count"], cmin, cmax), axis=1, result_type="expand")

    hex_layer = pdk.Layer(
        "H3HexagonLayer",
        data=df.to_dict(orient="records"),
        get_hexagon="hex_id",
        get_elevation="trip_count",
        elevation_scale=1,
        extruded=True,
        coverage=1,
        pickable=True,
        auto_highlight=False,
        get_fill_color=["colorR", "colorG", "colorB", "colorA"],
        opacity=1.0
    )

    boundary_layer = pdk.Layer(
        "GeoJsonLayer",
        data=chicago_fc,
        stroked=True,
        filled=True,
        get_line_color=[80, 80, 80],
        get_fill_color=[200, 200, 200],
        opacity=1.0
    )

    view_state = pdk.ViewState(latitude=41.8781, longitude=-87.6298, zoom=10, pitch=45)
    return pdk.Deck(layers=[boundary_layer, hex_layer], initial_view_state=view_state, map_provider="carto", map_style="light")
# Initial render
initial_month = available_months[0]
deck_pane = pn.pane.DeckGL(build_deck_for_month(initial_month).to_json(), width=800, height=600)

# Create a dropdown for month selection
month_selector = pn.widgets.Select(name="Month", options=available_months, value=initial_month)

# Callback function
def update_map(event):
    deck_pane.object = build_deck_for_month(event.new).to_json()

month_selector.param.watch(update_map, "value")

# Create the UI layout
ui = pn.Column(
    "# Scooter Trips: 3D Extruded Hex Map",
    "Select a month to view trip density:",
    month_selector,
    deck_pane
)

# Serve the UI on localhost
ui.servable()
#ui.show()

# To run the Panel server
pn.serve(ui, port=8509, show=True)  # Opens browser automatically
