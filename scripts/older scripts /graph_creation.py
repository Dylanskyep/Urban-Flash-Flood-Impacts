"""
SIMPLE FLASH-FLOOD CSV BUILDER
NOAA Storm Events + Groundsource

HOW TO USE
1. Put this script in the SAME folder as:
   - your Groundsource .parquet file
   - NOAA_storm_data(Miami).csv
   - NOAA_storm_data(Tampa).csv
   - NOAA_storm_data(Orlando).csv
   - NOAA_storm_data(Gainesville).csv
   - NOAA_storm_data(Jacksonville).csv

2. Install once:
   pip install pandas pyarrow shapely numpy

3. Run:
   python simple_flash_flood_builder.py

4. Open the new folder:
   flash_flood_outputs

No file paths need to be edited if the files are in the same folder.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from shapely import wkb


# ============================================================
# SETTINGS
# ============================================================

# Change only this if you want a different single-year comparison.
FOCUS_YEAR = 2024

CITIES = {
    "Miami": {
        "min_lon": -80.9,
        "max_lon": -80.1,
        "min_lat": 25.1,
        "max_lat": 25.9,
    },

    "Tampa": {
        "min_lon": -82.7,
        "max_lon": -82.3,
        "min_lat": 27.7,
        "max_lat": 28.1,
    },

    "Orlando": {
        "min_lon": -81.6,
        "max_lon": -81.2,
        "min_lat": 28.3,
        "max_lat": 28.7,
    },

    "Gainesville": {
        "min_lon": -82.45,
        "max_lon": -82.25,
        "min_lat": 29.55,
        "max_lat": 29.75,
    },

    "Jacksonville": {
        "min_lon": -81.90,
        "max_lon": -81.35,
        "min_lat": 30.10,
        "max_lat": 30.60,
    },
}


# ============================================================
# FILE LOCATIONS
# ============================================================

# Automatically use the folder this script is inside.
BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "flash_flood_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def find_groundsource_file():
    """
    Automatically finds the Groundsource parquet file.
    The filename just needs to contain the word 'groundsource'.
    """

    files = [
        file
        for file in BASE_DIR.glob("*.parquet")
        if "groundsource" in file.name.lower()
    ]

    if len(files) == 1:
        return files[0]

    if len(files) == 0:
        raise FileNotFoundError(
            "No Groundsource parquet file was found.\n"
            "Put the Groundsource .parquet file in the same folder "
            "as this Python script."
        )

    raise RuntimeError(
        "More than one Groundsource parquet file was found.\n"
        "Keep only the Groundsource file you want to analyze in this folder."
    )


def find_noaa_file(city):
    """
    Automatically finds each NOAA CSV.
    The filename needs to contain:
    NOAA + the city name
    """

    files = [
        file
        for file in BASE_DIR.glob("*.csv")
        if "noaa" in file.name.lower()
        and city.lower() in file.name.lower()
    ]

    if len(files) == 1:
        return files[0]

    if len(files) == 0:
        raise FileNotFoundError(
            f"No NOAA CSV was found for {city}.\n"
            f"Make sure its filename contains NOAA and {city}."
        )

    raise RuntimeError(
        f"More than one NOAA CSV was found for {city}.\n"
        "Keep only the file you want to use."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def first_existing_column(df, possible_names):
    """
    Finds the first column that exists from a list of possible names.
    """

    for name in possible_names:
        if name in df.columns:
            return name

    return None


def parse_damage(value):
    """
    Converts NOAA damage values such as:

    500
    5K
    2.5M
    1B

    into numeric dollar values.
    """

    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (int, float, np.integer, np.floating)
    ):
        return float(value)

    text = str(value).strip().upper()

    text = (
        text
        .replace("$", "")
        .replace(",", "")
    )

    if text in {
        "",
        "NAN",
        "NONE"
    }:
        return np.nan

    match = re.fullmatch(
        r"(-?\d+(?:\.\d+)?)\s*([KMB])?",
        text
    )

    if not match:
        return np.nan

    number = float(
        match.group(1)
    )

    suffix = match.group(2)

    multiplier = {
        None: 1,
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
    }[suffix]

    return number * multiplier


# ============================================================
# LOAD GROUNDSOURCE
# ============================================================

def load_groundsource():

    groundsource_file = find_groundsource_file()

    print(
        "Reading Groundsource:",
        groundsource_file.name
    )

    df = pd.read_parquet(
        groundsource_file
    )

    required_columns = [
        "start_date",
        "end_date",
        "geometry"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Groundsource is missing columns: {missing}"
        )

    df["start_date"] = pd.to_datetime(
        df["start_date"],
        errors="coerce"
    )

    df["end_date"] = pd.to_datetime(
        df["end_date"],
        errors="coerce"
    )

    def decode_geometry(geometry):

        if geometry is None:
            return None

        # Already a shapely geometry.
        if hasattr(
            geometry,
            "centroid"
        ):
            return geometry

        # Try normal WKB.
        try:
            return wkb.loads(
                geometry
            )
        except Exception:
            pass

        # Try hexadecimal WKB.
        try:
            return wkb.loads(
                bytes.fromhex(
                    str(geometry)
                )
            )
        except Exception:
            return None

    df["decoded_geometry"] = (
        df["geometry"]
        .apply(
            decode_geometry
        )
    )

    df = df[
        df["decoded_geometry"].notna()
    ].copy()

    df["longitude"] = (
        df["decoded_geometry"]
        .apply(
            lambda geometry:
            geometry.centroid.x
        )
    )

    df["latitude"] = (
        df["decoded_geometry"]
        .apply(
            lambda geometry:
            geometry.centroid.y
        )
    )

    return df


# ============================================================
# FILTER GROUNDSOURCE BY CITY
# ============================================================

def groundsource_city_events(
    all_groundsource,
    city
):

    bounds = CITIES[city]

    df = all_groundsource[
        (
            all_groundsource["longitude"]
            >= bounds["min_lon"]
        )
        &
        (
            all_groundsource["longitude"]
            <= bounds["max_lon"]
        )
        &
        (
            all_groundsource["latitude"]
            >= bounds["min_lat"]
        )
        &
        (
            all_groundsource["latitude"]
            <= bounds["max_lat"]
        )
    ].copy()

    df["event_date"] = (
        df["start_date"]
    )

    df["duration_hours"] = (
        df["end_date"]
        - df["start_date"]
    ).dt.total_seconds() / 3600

    # Remove impossible negative durations.
    df.loc[
        df["duration_hours"] < 0,
        "duration_hours"
    ] = np.nan

    df["city"] = city

    df["source"] = (
        "Groundsource"
    )

    df["event_type"] = (
        "Flash Flood"
    )

    # Groundsource does not automatically contain
    # comparable NOAA damage values.
    df["property_damage_usd"] = np.nan
    df["crop_damage_usd"] = np.nan

    return df[
        [
            "event_date",
            "end_date",
            "city",
            "source",
            "event_type",
            "latitude",
            "longitude",
            "duration_hours",
            "property_damage_usd",
            "crop_damage_usd",
        ]
    ]


# ============================================================
# LOAD NOAA
# ============================================================

def load_noaa_city(city):

    noaa_file = find_noaa_file(
        city
    )

    print(
        f"Reading NOAA {city}:",
        noaa_file.name
    )

    df = pd.read_csv(
        noaa_file,
        low_memory=False
    )

    # Make column names easier to work with.
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    event_type_col = (
        first_existing_column(
            df,
            [
                "event_type",
                "eventtype"
            ]
        )
    )

    cz_type_col = (
        first_existing_column(
            df,
            [
                "cz_type",
                "cztype"
            ]
        )
    )

    if event_type_col is None:

        raise KeyError(
            f"{city}: "
            "Could not find the NOAA event type column."
        )

    # Keep only Flash Flood events.
    keep = (
        df[event_type_col]
        .astype(str)
        .str.strip()
        .str.casefold()
        == "flash flood"
    )

    # Your original script also restricted
    # NOAA flash floods to county-designated C rows.
    if cz_type_col is not None:

        keep &= (
            df[cz_type_col]
            .astype(str)
            .str.strip()
            .str.upper()
            == "C"
        )

    df = df[
        keep
    ].copy()

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    begin_col = (
        first_existing_column(
            df,
            [
                "begin_date_time",
                "begin_datetime",
                "begin_date",
                "start_date",
            ]
        )
    )

    end_col = (
        first_existing_column(
            df,
            [
                "end_date_time",
                "end_datetime",
                "end_date",
                "finish_date",
            ]
        )
    )

    if begin_col is None:

        raise KeyError(
            f"{city}: "
            "Could not find a NOAA begin-date column."
        )

    df["event_date"] = (
        pd.to_datetime(
            df[begin_col],
            errors="coerce"
        )
    )

    if end_col is not None:

        df["end_date"] = (
            pd.to_datetime(
                df[end_col],
                errors="coerce"
            )
        )

    else:

        df["end_date"] = pd.NaT

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    lat_col = (
        first_existing_column(
            df,
            [
                "begin_lat",
                "begin_latitude",
                "latitude",
                "lat",
            ]
        )
    )

    lon_col = (
        first_existing_column(
            df,
            [
                "begin_lon",
                "begin_longitude",
                "longitude",
                "lon",
                "lng",
            ]
        )
    )

    if lat_col is not None:

        df["latitude"] = (
            pd.to_numeric(
                df[lat_col],
                errors="coerce"
            )
        )

    else:

        df["latitude"] = np.nan

    if lon_col is not None:

        df["longitude"] = (
            pd.to_numeric(
                df[lon_col],
                errors="coerce"
            )
        )

    else:

        df["longitude"] = np.nan

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    df["duration_hours"] = (
        df["end_date"]
        - df["event_date"]
    ).dt.total_seconds() / 3600

    df.loc[
        df["duration_hours"] < 0,
        "duration_hours"
    ] = np.nan

    # --------------------------------------------------------
    # NOAA damage
    # --------------------------------------------------------

    property_col = (
        first_existing_column(
            df,
            [
                "damage_property",
                "property_damage",
                "damage_property_num",
            ]
        )
    )

    crop_col = (
        first_existing_column(
            df,
            [
                "damage_crops",
                "crop_damage",
                "damage_crops_num",
            ]
        )
    )

    if property_col is not None:

        df["property_damage_usd"] = (
            df[property_col]
            .apply(
                parse_damage
            )
        )

    else:

        df["property_damage_usd"] = np.nan

    if crop_col is not None:

        df["crop_damage_usd"] = (
            df[crop_col]
            .apply(
                parse_damage
            )
        )

    else:

        df["crop_damage_usd"] = np.nan

    df["city"] = city

    df["source"] = (
        "NOAA Storm Events"
    )

    df["event_type"] = (
        "Flash Flood"
    )

    return df[
        [
            "event_date",
            "end_date",
            "city",
            "source",
            "event_type",
            "latitude",
            "longitude",
            "duration_hours",
            "property_damage_usd",
            "crop_damage_usd",
        ]
    ]


# ============================================================
# COMBINE NOAA + GROUNDSOURCE
# ============================================================

def build_master_events():

    groundsource = (
        load_groundsource()
    )

    all_events = []

    for city in CITIES:

        print(
            f"Processing {city}..."
        )

        groundsource_events = (
            groundsource_city_events(
                groundsource,
                city
            )
        )

        noaa_events = (
            load_noaa_city(
                city
            )
        )

        all_events.append(
            groundsource_events
        )

        all_events.append(
            noaa_events
        )

    master = pd.concat(
        all_events,
        ignore_index=True
    )

    # Remove rows without a usable date.
    master = master[
        master["event_date"].notna()
    ].copy()

    master["year"] = (
        master["event_date"]
        .dt.year
        .astype("Int64")
    )

    master["month"] = (
        master["event_date"]
        .dt.month
        .astype("Int64")
    )

    master["year_month"] = (
        master["event_date"]
        .dt.to_period("M")
        .astype(str)
    )

    # NOAA property + crop damage.
    master["total_damage_usd"] = (
        master[
            [
                "property_damage_usd",
                "crop_damage_usd",
            ]
        ]
        .sum(
            axis=1,
            min_count=1
        )
    )

    master = (
        master
        .sort_values(
            [
                "city",
                "source",
                "event_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # Add a unique ID.
    master.insert(
        0,
        "event_id",
        [
            f"E{i:06d}"
            for i in range(
                1,
                len(master) + 1
            )
        ]
    )

    return master


# ============================================================
# CREATE CSV FILES FOR GRAPHS
# ============================================================

def make_outputs(master):

    # ========================================================
    # CSV 1
    # INDIVIDUAL EVENTS
    #
    # USE FOR:
    # Beeswarm plot
    # individual event analysis
    # ========================================================

    master.to_csv(
        OUTPUT_DIR
        / "01_master_events.csv",
        index=False
    )

    # ========================================================
    # CSV 2
    # YEARLY EVENT COUNTS
    #
    # USE FOR:
    # Line Chart
    # Matrix Plot
    # Box Plot
    # Bar Chart
    # ========================================================

    yearly = (
        master
        .groupby(
            [
                "city",
                "source",
                "year",
            ]
        )
        .size()
        .reset_index(
            name="event_count"
        )
        .sort_values(
            [
                "source",
                "city",
                "year",
            ]
        )
    )

    yearly.to_csv(
        OUTPUT_DIR
        / "02_yearly_counts.csv",
        index=False
    )
    # ========================================================
# CREATE ONE YEARLY CSV FOR EACH CITY
# ========================================================

    for city in CITIES:

        city_data = yearly[
            yearly["city"] == city
        ].copy()

        city_data.to_csv(
            OUTPUT_DIR / f"yearly_{city.lower()}.csv",
            index=False
        )
    # ========================================================
    # CSV 3
    # MONTHLY COUNTS
    #
    # USE FOR:
    # More detailed time-series graphs
    # ========================================================

    monthly = (
        master
        .groupby(
            [
                "city",
                "source",
                "year",
                "month",
                "year_month",
            ]
        )
        .size()
        .reset_index(
            name="event_count"
        )
        .sort_values(
            [
                "source",
                "city",
                "year",
                "month",
            ]
        )
    )

    monthly.to_csv(
        OUTPUT_DIR
        / "03_monthly_counts.csv",
        index=False
    )

    # ========================================================
    # CSV 4
    # SPATIAL EVENT DATA
    #
    # USE FOR:
    # longitude vs latitude plot
    # maps
    # shapefile overlays
    # ========================================================

    spatial = master[
        master["latitude"].notna()
        &
        master["longitude"].notna()
    ].copy()

    spatial.to_csv(
        OUTPUT_DIR
        / "04_spatial_events.csv",
        index=False
    )

    # ========================================================
    # CSV 5
    # INDIVIDUAL EVENT DURATIONS
    #
    # USE FOR:
    # Box Plot
    # Violin Plot
    # ========================================================

    duration = master[
        master[
            "duration_hours"
        ].notna()
    ].copy()

    duration.to_csv(
        OUTPUT_DIR
        / "05_duration_events.csv",
        index=False
    )

    # ========================================================
    # CSV 6
    # DURATION STATISTICS
    #
    # USE FOR:
    # Standard deviation graph
    # average duration comparison
    # ========================================================

    if len(duration) > 0:

        duration_summary = (
            duration
            .groupby(
                [
                    "city",
                    "source",
                ]
            )["duration_hours"]
            .agg(
                event_count="count",
                mean_duration_hours="mean",
                std_duration_hours="std",
                median_duration_hours="median",
                min_duration_hours="min",
                max_duration_hours="max",
            )
            .reset_index()
        )

    else:

        duration_summary = (
            pd.DataFrame(
                columns=[
                    "city",
                    "source",
                    "event_count",
                    "mean_duration_hours",
                    "std_duration_hours",
                    "median_duration_hours",
                    "min_duration_hours",
                    "max_duration_hours",
                ]
            )
        )

    duration_summary.to_csv(
        OUTPUT_DIR
        / "06_duration_summary.csv",
        index=False
    )

    # ========================================================
    # CSV 7
    # ONE SPECIFIC YEAR
    #
    # USE FOR:
    # Multi-set Bar Chart
    # ========================================================

    focus_year = (
        master[
            master["year"]
            == FOCUS_YEAR
        ]
        .groupby(
            [
                "city",
                "source",
            ]
        )
        .size()
        .reset_index(
            name="event_count"
        )
    )

    focus_year.to_csv(
        OUTPUT_DIR
        / f"07_focus_year_{FOCUS_YEAR}.csv",
        index=False
    )

    # ========================================================
    # CSV 8
    # NOAA VS GROUNDSOURCE
    #
    # USE FOR:
    # Bubble/scatter-style comparison
    # difference-over-time graph
    # ========================================================

    comparison = (
        yearly
        .pivot_table(
            index=[
                "city",
                "year",
            ],
            columns="source",
            values="event_count",
            aggfunc="sum"
        )
        .reset_index()
    )

    comparison.columns.name = None

    if (
        "Groundsource"
        not in comparison.columns
    ):
        comparison[
            "Groundsource"
        ] = np.nan

    if (
        "NOAA Storm Events"
        not in comparison.columns
    ):
        comparison[
            "NOAA Storm Events"
        ] = np.nan

    comparison[
        "count_difference_groundsource_minus_noaa"
    ] = (
        comparison["Groundsource"]
        -
        comparison["NOAA Storm Events"]
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "08_source_comparison.csv",
        index=False
    )

    # ========================================================
    # CSV 9
    # DAMAGE BY YEAR
    #
    # Mainly NOAA unless Groundsource also has
    # comparable damage values.
    # ========================================================

    damage = master[
        master[
            "total_damage_usd"
        ].notna()
    ].copy()

    if len(damage) > 0:

        damage_year = (
            damage
            .groupby(
                [
                    "city",
                    "source",
                    "year",
                ]
            )["total_damage_usd"]
            .agg(
                events_with_damage="count",
                total_damage_usd="sum",
                mean_damage_usd="mean",
                median_damage_usd="median",
            )
            .reset_index()
        )

    else:

        damage_year = (
            pd.DataFrame(
                columns=[
                    "city",
                    "source",
                    "year",
                    "events_with_damage",
                    "total_damage_usd",
                    "mean_damage_usd",
                    "median_damage_usd",
                ]
            )
        )

    damage_year.to_csv(
        OUTPUT_DIR
        / "09_damage_by_year.csv",
        index=False
    )


# ============================================================
# RUN EVERYTHING
# ============================================================

def main():

    master = (
        build_master_events()
    )

    make_outputs(
        master
    )

    print()
    print("DONE")
    print()

    print(
        "Output folder:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "CSV files created:"
    )

    for file in sorted(
        OUTPUT_DIR.glob(
            "*.csv"
        )
    ):

        print(
            " -",
            file.name
        )

    print()

    print(
        "Most useful RAWGraphs files:"
    )

    print(
        "02_yearly_counts.csv "
        "-> line, matrix, box, bar"
    )

    print(
        "03_monthly_counts.csv "
        "-> monthly line chart"
    )

    print(
        "04_spatial_events.csv "
        "-> longitude/latitude plot"
    )

    print(
        "05_duration_events.csv "
        "-> box or violin plot"
    )

    print(
        "06_duration_summary.csv "
        "-> standard deviation graph"
    )

    print(
        f"07_focus_year_{FOCUS_YEAR}.csv "
        "-> one-year bar chart"
    )

    print(
        "08_source_comparison.csv "
        "-> NOAA vs Groundsource comparison"
    )


if __name__ == "__main__":
    main()