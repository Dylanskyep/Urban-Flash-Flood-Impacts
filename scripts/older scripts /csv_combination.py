"""
Flash Flood Data Scrubber: Groundsource + NOAA Storm Events
Cleans and filters both sources down to Miami, Tampa, Orlando, Gainesville,
and Jacksonville, then exports CSV files shaped for upload to online
charting tools rather than plotting anything in Python directly.

Assumes you already have NOAA Storm Events data split into one CSV per
city/county (set each city's noaa_file path below to match your files),
and that you've downloaded the Groundsource parquet file yourself from
Zenodo (set GROUNDSOURCE_PARQUET_PATH near the bottom to match).

Output files:
    combined_events.csv   -> one row per event, for a RAWGraphs dot/beeswarm plot
    yearly_summary.csv    -> event counts per city/source/year, for a RAWGraphs box plot
    stats_summary.csv     -> mean and std dev per city/source, for a Google Sheets error bar chart
    per_city_files/*.csv  -> combined_events.csv split into one file per city

Requirements:
    pip install pandas pyarrow shapely
"""

import os
from dataclasses import dataclass

import pandas as pd
from shapely import wkb


# ---------------------------------------------------------------------------
# STEP 1: Define the cities as named structs
# ---------------------------------------------------------------------------

@dataclass
class CityBoundingBox:
    name: str
    noaa_file: str           # path to this city's own, already-separated NOAA CSV
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float


# Fill in the actual file paths/names below to match your own local files.
# They can be full paths (e.g. "C:/Users/Dylan/Downloads/miami_flash_floods.csv")
# or relative paths if the CSVs sit in the same folder as this script.

MIAMI = CityBoundingBox(
    name="Miami",
    noaa_file="/Users/dylanpanganiban/Downloads/NOAA_storm_data(Miami).csv",
    min_lon=-80.9, max_lon=-80.1,
    min_lat=25.1, max_lat=25.9,
)

TAMPA = CityBoundingBox(
    name="Tampa",
    noaa_file="/Users/dylanpanganiban/Downloads/NOAA_storm_data(Tampa).csv",
    min_lon=-82.7, max_lon=-82.3,
    min_lat=27.7, max_lat=28.1,
)

ORLANDO = CityBoundingBox(
    name="Orlando",
    noaa_file="/Users/dylanpanganiban/Downloads/NOAA_storm_data(Orlando).csv",
    min_lon=-81.6, max_lon=-81.2,
    min_lat=28.3, max_lat=28.7,
)

GAINESVILLE = CityBoundingBox(
    name="Gainesville",
    noaa_file="/Users/dylanpanganiban/Downloads/NOAA_storm_data(Gainesville).csv",
    min_lon=-82.45, max_lon=-82.25,
    min_lat=29.55, max_lat=29.75,
)

JACKSONVILLE = CityBoundingBox(
    name="Jacksonville",
    noaa_file="/Users/dylanpanganiban/Downloads/NOAA_storm_data(Jacksonville).csv",
    min_lon=-81.90, max_lon=-81.35,
    min_lat=30.10, max_lat=30.60,
)

CITIES = [MIAMI, TAMPA, ORLANDO, GAINESVILLE, JACKSONVILLE]


# ---------------------------------------------------------------------------
# STEP 2: Scrub the Groundsource dataset
# Loaded ONCE for the whole run, then filtered per city, rather than
# reloading the full 2.6 million row file five separate times.
# ---------------------------------------------------------------------------

def load_all_groundsource_events(parquet_path: str) -> pd.DataFrame:
    """Loads the Groundsource parquet file you've uploaded/downloaded
    yourself, decodes its WKB geometry column into plain latitude and
    longitude, and parses its date columns. Point GROUNDSOURCE_PARQUET_PATH
    at wherever you saved the file from Zenodo (the ~667MB single .parquet
    file, not the enriched version some packages build on top of it)."""

    raw_events = pd.read_parquet(parquet_path)

    raw_events["start_date"] = pd.to_datetime(raw_events["start_date"])
    raw_events["end_date"] = pd.to_datetime(raw_events["end_date"])
    raw_events["geometry"] = raw_events["geometry"].apply(wkb.loads)

    raw_events["centroid_lon"] = raw_events["geometry"].apply(lambda g: g.centroid.x)
    raw_events["centroid_lat"] = raw_events["geometry"].apply(lambda g: g.centroid.y)

    return raw_events


def filter_groundsource_events_for_city(
    all_groundsource_events: pd.DataFrame, city: CityBoundingBox
) -> pd.DataFrame:
    """Filters the already-loaded Groundsource dataframe down to one
    city's bounding box and returns just the columns needed for the charts."""

    in_city = all_groundsource_events[
        (all_groundsource_events["centroid_lon"] >= city.min_lon)
        & (all_groundsource_events["centroid_lon"] <= city.max_lon)
        & (all_groundsource_events["centroid_lat"] >= city.min_lat)
        & (all_groundsource_events["centroid_lat"] <= city.max_lat)
    ].copy()

    in_city["event_date"] = in_city["start_date"]
    in_city["year"] = in_city["event_date"].dt.year
    in_city["city"] = city.name
    in_city["source"] = "Groundsource"

    return in_city[["event_date", "year", "city", "source"]]


# ---------------------------------------------------------------------------
# STEP 3: Scrub the NOAA Storm Events dataset
# Each city already has its own separated CSV, so we just read that one
# file per city directly instead of filtering a shared statewide file.
# ---------------------------------------------------------------------------

def load_noaa_events_for_city(city: CityBoundingBox) -> pd.DataFrame:
    """Reads one city's own NOAA CSV and returns just the columns
    needed for the charts, using the real column names from the
    Storm-Data-Columns.pdf schema (event_type, cz_type, begin_date_time)."""

    raw = pd.read_csv(city.noaa_file, low_memory=False)
    raw.columns = raw.columns.str.lower()

    required_columns = ["event_type", "cz_type", "begin_date"]
    missing_columns = [col for col in required_columns if col not in raw.columns]

    if missing_columns:
        raise KeyError(
            f"{city.name}'s file ({city.noaa_file}) is missing the "
            f"column(s) {missing_columns}. The columns it actually has are:\n"
            f"{sorted(raw.columns.tolist())}\n"
            "Update the column names referenced in load_noaa_events_for_city "
            "to match whichever of these are the real event type, county/zone "
            "type, and begin-date columns in your export."
        )

    # cz_type is included as a safety check: Flash Flood is always
    # county-designated ("C"), so this catches any stray zone/marine rows
    flash_floods = raw[
        (raw["event_type"] == "Flash Flood")
        & (raw["cz_type"] == "C")
    ].copy()

    flash_floods["event_date"] = pd.to_datetime(flash_floods["begin_date"])
    flash_floods["year"] = flash_floods["event_date"].dt.year
    flash_floods["city"] = city.name
    flash_floods["source"] = "NOAA Storm Events"

    return flash_floods[["event_date", "year", "city", "source"]]


# ---------------------------------------------------------------------------
# STEP 4: Combine both sources for every city
# ---------------------------------------------------------------------------

def build_combined_events(groundsource_parquet_path: str) -> pd.DataFrame:
    all_groundsource_events = load_all_groundsource_events(groundsource_parquet_path)

    all_city_frames = []
    for city in CITIES:
        groundsource_events = filter_groundsource_events_for_city(all_groundsource_events, city)
        noaa_events = load_noaa_events_for_city(city)
        all_city_frames.append(groundsource_events)
        all_city_frames.append(noaa_events)

    return pd.concat(all_city_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# STEP 5: Build the three export tables
# ---------------------------------------------------------------------------

def build_yearly_summary(combined_events: pd.DataFrame) -> pd.DataFrame:
    """One row per city/source/year, with an event count. Feed this to
    RAWGraphs as a box plot: put 'city' or 'source' on one axis and
    'event_count' on the other, to see the year-to-year spread."""

    return (
        combined_events.groupby(["city", "source", "year"])
        .size()
        .reset_index(name="event_count")
    )


def build_stats_summary(yearly_summary: pd.DataFrame) -> pd.DataFrame:
    """One row per city/source, with the mean and standard deviation of
    yearly event counts. Feed this to Google Sheets: plot 'mean_events'
    as your bar or point value, and use 'std_dev_events' as the custom
    error bar value."""

    return (
        yearly_summary.groupby(["city", "source"])["event_count"]
        .agg(mean_events="mean", std_dev_events="std")
        .reset_index()
    )


# ---------------------------------------------------------------------------
# STEP 6: Write one combined file (both sources, all cities, one upload)
#         plus a separate file per city (both sources, one city each)
# ---------------------------------------------------------------------------

def write_per_city_files(combined_events: pd.DataFrame, output_directory: str) -> None:
    """Splits the combined events table into one CSV per city, each still
    containing both Groundsource and NOAA rows, distinguished by the
    'source' column. Use these when you want a clean, single-city upload
    to RAWGraphs rather than filtering a full combined file by hand."""

    os.makedirs(output_directory, exist_ok=True)

    for city in CITIES:
        city_events = combined_events[combined_events["city"] == city.name]
        file_path = os.path.join(output_directory, f"combined_events_{city.name.lower()}.csv")
        city_events.to_csv(file_path, index=False)
        print(f"Wrote {file_path} ({len(city_events)} events)")


# ---------------------------------------------------------------------------
# STEP 7: Run everything and write all the CSVs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Point this at the Groundsource parquet file you downloaded from
    # Zenodo (https://zenodo.org/records/18647054) and uploaded yourself.
    # os.path.expanduser("~") resolves to your home folder automatically,
    # so this works without hardcoding your username or a drive letter.
    GROUNDSOURCE_PARQUET_PATH = os.path.expanduser("~/Downloads/groundsource_2026.parquet")
    PER_CITY_OUTPUT_DIRECTORY = "./per_city_files"

    combined_events = build_combined_events(GROUNDSOURCE_PARQUET_PATH)
    yearly_summary = build_yearly_summary(combined_events)
    stats_summary = build_stats_summary(yearly_summary)

    # One file with everything: both sources, all three cities, one upload
    combined_events.to_csv("combined_events.csv", index=False)

    # Aggregated summaries, also covering all cities in one file each
    yearly_summary.to_csv("yearly_summary.csv", index=False)
    stats_summary.to_csv("stats_summary.csv", index=False)

    # One file per city, each still containing both sources
    write_per_city_files(combined_events, PER_CITY_OUTPUT_DIRECTORY)

    print("Wrote combined_events.csv, yearly_summary.csv, stats_summary.csv, "
          f"and per-city files in {PER_CITY_OUTPUT_DIRECTORY}/")