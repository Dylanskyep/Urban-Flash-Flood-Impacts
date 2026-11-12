"""
Additional relationship and source-only analysis tables
for Urban Flash-Flood Impacts.

Inputs:
    data/processed/14_harmonized_overlap_events.csv
    data/processed/24_matched_event_pairs_enriched.csv
    data/processed/28_high_confidence_matched_events.csv

Existing relationship outputs:
    38_damage_probability_by_duration_bin.csv
    39_high_confidence_area_vs_damage.csv
    40_high_confidence_damaging_events_duration_vs_damage.csv
    41_match_distance_vs_temporal_gap.csv
    42_damage_per_event_by_county_high_confidence.csv
    43_annual_frequency_damage_severity.csv

Source-only outputs:
    44_noaa_events_source_only.csv
    45_groundsource_episodes_source_only.csv
    46_source_overall_summary.csv
    47_source_events_by_city.csv
    48_source_events_by_year.csv
    49_source_events_by_city_and_year.csv
    50_noaa_damage_summary_by_city.csv
    51_noaa_damage_by_year.csv
    52_groundsource_area_summary_by_city.csv
    53_mapping_events.csv

Common-period outputs:
    54_common_period_source_summary.csv
    55_common_period_events_by_year.csv
    56_common_period_events_by_city.csv
    57_common_period_events_by_city_and_year.csv

Separate graph-friendly outputs:
    58_noaa_events_by_city.csv
    59_groundsource_events_by_city.csv
    60_noaa_events_by_year.csv
    61_groundsource_events_by_year.csv

Notes:
    - NOAA rows represent NOAA flash-flood events.
    - Groundsource rows represent clustered candidate flood episodes.
    - NOAA financial impacts remain NOAA-reported attributes.
    - Source-only Groundsource statistics do not use NOAA damage.
    - Direct NOAA vs Groundsource comparisons use the common
      2005-2025 period.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HARMONIZED_FILE = DATA_DIR / "14_harmonized_overlap_events.csv"
PAIR_FILE = DATA_DIR / "24_matched_event_pairs_enriched.csv"
HIGH_FILE = DATA_DIR / "28_high_confidence_matched_events.csv"


# ==========================================================
# COMMON COMPARISON PERIOD
# ==========================================================

COMMON_START_YEAR = 2005
COMMON_END_YEAR = 2025


# ==========================================================
# HELPERS
# ==========================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find required file: {path}"
        )


def to_numeric(df, column):
    if column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64"
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def first_existing_column(df, names):
    lookup = {
        str(column).lower(): column
        for column in df.columns
    }

    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def numeric_from_candidates(df, names):
    column = first_existing_column(
        df,
        names
    )

    if column is None:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64"
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def safe_sum(series):
    return series.sum(min_count=1)


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ------------------------------------------------------
    # CHECK INPUT FILES
    # ------------------------------------------------------

    require_file(HARMONIZED_FILE)
    require_file(PAIR_FILE)
    require_file(HIGH_FILE)

    print(f"Reading: {HARMONIZED_FILE}")
    events = pd.read_csv(HARMONIZED_FILE)

    print(f"Reading: {PAIR_FILE}")
    pairs = pd.read_csv(PAIR_FILE)

    print(f"Reading: {HIGH_FILE}")
    high = pd.read_csv(HIGH_FILE)

    # ------------------------------------------------------
    # STANDARDIZE EXISTING MATCHED DATA
    # ------------------------------------------------------

    high["noaa_duration_hours"] = to_numeric(
        high,
        "noaa_duration_hours"
    )

    high["noaa_total_damage_usd"] = to_numeric(
        high,
        "noaa_total_damage_usd"
    )

    high["groundsource_area_max_km2"] = to_numeric(
        high,
        "groundsource_area_max_km2"
    )

    pairs["distance_km"] = to_numeric(
        pairs,
        "distance_km"
    )

    pairs["temporal_gap_hours"] = to_numeric(
        pairs,
        "temporal_gap_hours"
    )

    # ======================================================
    # EXISTING OUTPUTS 38-43
    # ======================================================

    # ------------------------------------------------------
    # 38. DAMAGE PROBABILITY BY DURATION BIN
    # ------------------------------------------------------

    duration_bins = [
        -np.inf,
        1,
        3,
        6,
        12,
        np.inf
    ]

    duration_labels = [
        "<1 hour",
        "1–3 hours",
        "3–6 hours",
        "6–12 hours",
        ">12 hours"
    ]

    damage_probability_events = high[
        high["noaa_duration_hours"].notna()
        & high["noaa_total_damage_usd"].notna()
    ].copy()

    damage_probability_events["duration_bin"] = pd.cut(
        damage_probability_events["noaa_duration_hours"],
        bins=duration_bins,
        labels=duration_labels,
        right=False
    )

    damage_probability_events["has_reported_damage"] = (
        damage_probability_events[
            "noaa_total_damage_usd"
        ] > 0
    )

    damage_probability = (
        damage_probability_events
        .groupby(
            "duration_bin",
            observed=False
        )
        .agg(
            total_events=(
                "match_id",
                "count"
            ),
            damaging_events=(
                "has_reported_damage",
                "sum"
            )
        )
        .reset_index()
    )

    damage_probability[
        "damage_probability_percent"
    ] = np.where(
        damage_probability["total_events"] > 0,
        (
            100
            * damage_probability["damaging_events"]
            / damage_probability["total_events"]
        ).round(1),
        np.nan
    )

    damage_probability["duration_bin"] = (
        damage_probability["duration_bin"]
        .astype(str)
    )

    # ------------------------------------------------------
    # 39. GROUNDSOURCE AREA VS NOAA DAMAGE
    # ------------------------------------------------------

    area_vs_damage = high[
        high["groundsource_area_max_km2"].notna()
        & high["noaa_total_damage_usd"].notna()
    ].copy()

    area_columns = [
        "match_id",
        "city",
        "match_confidence",
        "noaa_event_date",
        "groundsource_event_date",
        "groundsource_area_max_km2",
        "noaa_duration_hours",
        "noaa_total_damage_usd"
    ]

    area_columns = [
        column
        for column in area_columns
        if column in area_vs_damage.columns
    ]

    area_vs_damage = area_vs_damage[
        area_columns
    ]

    sort_columns = [
        column
        for column in [
            "city",
            "groundsource_area_max_km2"
        ]
        if column in area_vs_damage.columns
    ]

    if sort_columns:
        area_vs_damage = (
            area_vs_damage
            .sort_values(sort_columns)
        )

    # ------------------------------------------------------
    # 40. DURATION VS DAMAGE AMONG DAMAGING EVENTS ONLY
    # ------------------------------------------------------

    damaging_duration = high[
        high["noaa_duration_hours"].notna()
        & high["noaa_total_damage_usd"].notna()
        & (
            high["noaa_total_damage_usd"] > 0
        )
    ].copy()

    damaging_columns = [
        "match_id",
        "city",
        "match_confidence",
        "noaa_event_date",
        "noaa_duration_hours",
        "noaa_total_damage_usd",
        "groundsource_area_max_km2"
    ]

    damaging_columns = [
        column
        for column in damaging_columns
        if column in damaging_duration.columns
    ]

    damaging_duration = damaging_duration[
        damaging_columns
    ]

    sort_columns = [
        column
        for column in [
            "city",
            "noaa_duration_hours"
        ]
        if column in damaging_duration.columns
    ]

    if sort_columns:
        damaging_duration = (
            damaging_duration
            .sort_values(sort_columns)
        )

    # ------------------------------------------------------
    # 41. MATCH DISTANCE VS TEMPORAL GAP
    # ------------------------------------------------------

    match_diagnostic = pairs[
        pairs["distance_km"].notna()
        & pairs["temporal_gap_hours"].notna()
    ].copy()

    diagnostic_columns = [
        "match_id",
        "city",
        "match_confidence",
        "distance_km",
        "temporal_gap_hours",
        "intervals_overlap",
        "match_score_0_100"
    ]

    diagnostic_columns = [
        column
        for column in diagnostic_columns
        if column in match_diagnostic.columns
    ]

    match_diagnostic = match_diagnostic[
        diagnostic_columns
    ]

    sort_columns = [
        column
        for column in [
            "match_confidence",
            "distance_km",
            "temporal_gap_hours"
        ]
        if column in match_diagnostic.columns
    ]

    if sort_columns:
        match_diagnostic = (
            match_diagnostic
            .sort_values(sort_columns)
        )

    # ------------------------------------------------------
    # 42. DAMAGE PER EVENT BY STUDY AREA
    # ------------------------------------------------------

    county_damage_events = high[
        high["noaa_total_damage_usd"].notna()
    ].copy()

    county_damage = (
        county_damage_events
        .groupby("city")
        .agg(
            matched_events=(
                "match_id",
                "count"
            ),

            damaging_events=(
                "noaa_total_damage_usd",
                lambda values:
                int((values > 0).sum())
            ),

            total_noaa_damage_usd=(
                "noaa_total_damage_usd",
                safe_sum
            ),

            mean_noaa_damage_per_event_usd=(
                "noaa_total_damage_usd",
                "mean"
            ),

            median_noaa_damage_per_event_usd=(
                "noaa_total_damage_usd",
                "median"
            ),

            max_noaa_damage_usd=(
                "noaa_total_damage_usd",
                "max"
            )
        )
        .reset_index()
        .sort_values("city")
    )

    county_damage[
        "percent_events_with_damage"
    ] = np.where(
        county_damage["matched_events"] > 0,
        (
            100
            * county_damage["damaging_events"]
            / county_damage["matched_events"]
        ).round(1),
        np.nan
    )

    # ------------------------------------------------------
    # 43. ANNUAL FREQUENCY VS EVENT SEVERITY
    # ------------------------------------------------------

    high["noaa_year"] = pd.to_datetime(
        high["noaa_event_date"],
        errors="coerce"
    ).dt.year

    annual_events = high[
        high["noaa_year"].notna()
        & high["noaa_total_damage_usd"].notna()
    ].copy()

    annual_severity = (
        annual_events
        .groupby("noaa_year")
        .agg(
            event_count=(
                "match_id",
                "count"
            ),

            damaging_events=(
                "noaa_total_damage_usd",
                lambda values:
                int((values > 0).sum())
            ),

            total_noaa_damage_usd=(
                "noaa_total_damage_usd",
                safe_sum
            ),

            mean_noaa_damage_usd=(
                "noaa_total_damage_usd",
                "mean"
            ),

            median_noaa_damage_usd=(
                "noaa_total_damage_usd",
                "median"
            ),

            max_noaa_damage_usd=(
                "noaa_total_damage_usd",
                "max"
            )
        )
        .reset_index()
        .sort_values("noaa_year")
    )

    annual_severity[
        "percent_events_with_damage"
    ] = np.where(
        annual_severity["event_count"] > 0,
        (
            100
            * annual_severity["damaging_events"]
            / annual_severity["event_count"]
        ).round(1),
        np.nan
    )

    # ======================================================
    # SOURCE-ONLY ANALYSIS
    # ======================================================

    # ------------------------------------------------------
    # STANDARDIZE HARMONIZED EVENTS
    # ------------------------------------------------------

    events["event_date"] = pd.to_datetime(
        events["event_date"],
        errors="coerce"
    )

    if "end_date" in events.columns:
        events["end_date"] = pd.to_datetime(
            events["end_date"],
            errors="coerce"
        )

    events["year"] = (
        events["event_date"].dt.year
    )

    events["latitude"] = to_numeric(
        events,
        "latitude"
    )

    events["longitude"] = to_numeric(
        events,
        "longitude"
    )

    events["duration_hours"] = to_numeric(
        events,
        "duration_hours"
    )

    source_text = (
        events["source"]
        .astype(str)
        .str.strip()
    )

    noaa = events[
        source_text.str.contains(
            "NOAA",
            case=False,
            na=False
        )
    ].copy()

    groundsource = events[
        source_text.str.contains(
            "Groundsource",
            case=False,
            na=False
        )
    ].copy()

    # ======================================================
    # COMMON-PERIOD DATASETS: 2005-2025
    # ======================================================

    noaa_common = noaa[
        noaa["year"].between(
            COMMON_START_YEAR,
            COMMON_END_YEAR,
            inclusive="both"
        )
    ].copy()

    groundsource_common = groundsource[
        groundsource["year"].between(
            COMMON_START_YEAR,
            COMMON_END_YEAR,
            inclusive="both"
        )
    ].copy()

    common_year_span = (
        COMMON_END_YEAR
        - COMMON_START_YEAR
        + 1
    )

    # ------------------------------------------------------
    # STANDARDIZE NOAA DAMAGE FIELDS
    # ------------------------------------------------------

    noaa["noaa_property_damage_usd"] = (
        numeric_from_candidates(
            noaa,
            [
                "noaa_property_damage_usd",
                "property_damage_usd",
                "damage_property",
                "property_damage"
            ]
        )
    )

    noaa["noaa_crop_damage_usd"] = (
        numeric_from_candidates(
            noaa,
            [
                "noaa_crop_damage_usd",
                "crop_damage_usd",
                "damage_crops",
                "crop_damage"
            ]
        )
    )

    existing_total_damage = (
        numeric_from_candidates(
            noaa,
            [
                "noaa_total_damage_usd",
                "total_damage_usd",
                "damage_total",
                "total_damage"
            ]
        )
    )

    calculated_total_damage = (
        noaa["noaa_property_damage_usd"].fillna(0)
        + noaa["noaa_crop_damage_usd"].fillna(0)
    )

    both_damage_missing = (
        noaa["noaa_property_damage_usd"].isna()
        & noaa["noaa_crop_damage_usd"].isna()
    )

    calculated_total_damage.loc[
        both_damage_missing
    ] = np.nan

    noaa["noaa_total_damage_usd"] = (
        existing_total_damage
        .combine_first(
            calculated_total_damage
        )
    )

    noaa_common = noaa[
        noaa["year"].between(
            COMMON_START_YEAR,
            COMMON_END_YEAR,
            inclusive="both"
        )
    ].copy()

    # ------------------------------------------------------
    # STANDARDIZE GROUNDSOURCE AREA FIELDS
    # ------------------------------------------------------

    groundsource[
        "groundsource_area_max_km2"
    ] = numeric_from_candidates(
        groundsource,
        [
            "groundsource_area_max_km2",
            "area_km2",
            "area_max_km2"
        ]
    )

    groundsource[
        "groundsource_area_mean_km2"
    ] = numeric_from_candidates(
        groundsource,
        [
            "groundsource_area_mean_km2",
            "area_mean_km2"
        ]
    )

    groundsource[
        "groundsource_area_sum_km2"
    ] = numeric_from_candidates(
        groundsource,
        [
            "groundsource_area_sum_km2",
            "area_sum_km2"
        ]
    )

    groundsource_common = groundsource[
        groundsource["year"].between(
            COMMON_START_YEAR,
            COMMON_END_YEAR,
            inclusive="both"
        )
    ].copy()

    # ------------------------------------------------------
    # 44. NOAA EVENT-LEVEL DATASET
    # ------------------------------------------------------

    noaa_source_only = (
        noaa
        .sort_values(
            ["city", "event_date"]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------
    # 45. GROUNDSOURCE EPISODE-LEVEL DATASET
    # ------------------------------------------------------

    groundsource_source_only = (
        groundsource
        .sort_values(
            ["city", "event_date"]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------
    # 46. OVERALL SOURCE SUMMARY
    # ------------------------------------------------------

    source_summary_rows = []

    for source_name, dataframe in [
        ("NOAA Storm Events", noaa),
        ("Groundsource", groundsource)
    ]:

        valid_years = (
            dataframe["year"]
            .dropna()
            .astype(int)
        )

        total_events = len(dataframe)

        if len(valid_years) > 0:

            first_year = int(
                valid_years.min()
            )

            last_year = int(
                valid_years.max()
            )

            year_span = (
                last_year
                - first_year
                + 1
            )

            years_with_events = (
                valid_years.nunique()
            )

            average_events_per_year = (
                total_events
                / year_span
            )

        else:

            first_year = np.nan
            last_year = np.nan
            year_span = np.nan
            years_with_events = 0
            average_events_per_year = np.nan

        cities_present = (
            dataframe["city"]
            .dropna()
            .nunique()
        )

        average_events_per_city = (
            total_events
            / cities_present
            if cities_present > 0
            else np.nan
        )

        row = {
            "source": source_name,
            "total_events": total_events,
            "first_year": first_year,
            "last_year": last_year,
            "year_span_inclusive": year_span,
            "years_with_events": years_with_events,
            "cities_present": cities_present,
            "average_events_per_year": round(
                average_events_per_year,
                2
            ) if pd.notna(
                average_events_per_year
            ) else np.nan,
            "average_events_per_city": round(
                average_events_per_city,
                2
            ) if pd.notna(
                average_events_per_city
            ) else np.nan
        }

        if source_name == "NOAA Storm Events":

            total_damage = safe_sum(
                dataframe[
                    "noaa_total_damage_usd"
                ]
            )

            row[
                "total_reported_damage_usd"
            ] = total_damage

            row[
                "mean_damage_per_event_usd"
            ] = dataframe[
                "noaa_total_damage_usd"
            ].mean()

            row[
                "median_damage_per_event_usd"
            ] = dataframe[
                "noaa_total_damage_usd"
            ].median()

            row[
                "average_annual_damage_usd"
            ] = (
                total_damage / year_span
                if (
                    pd.notna(total_damage)
                    and pd.notna(year_span)
                    and year_span > 0
                )
                else np.nan
            )

        else:

            row[
                "total_reported_damage_usd"
            ] = np.nan

            row[
                "mean_damage_per_event_usd"
            ] = np.nan

            row[
                "median_damage_per_event_usd"
            ] = np.nan

            row[
                "average_annual_damage_usd"
            ] = np.nan

        source_summary_rows.append(row)

    source_overall_summary = pd.DataFrame(
        source_summary_rows
    )

    # ------------------------------------------------------
    # 47. EVENT COUNTS BY SOURCE AND CITY
    # ------------------------------------------------------

    noaa_city_counts = (
        noaa
        .groupby("city")
        .size()
        .reset_index(
            name="event_count"
        )
    )

    noaa_city_counts[
        "source"
    ] = "NOAA Storm Events"

    groundsource_city_counts = (
        groundsource
        .groupby("city")
        .size()
        .reset_index(
            name="event_count"
        )
    )

    groundsource_city_counts[
        "source"
    ] = "Groundsource"

    source_events_by_city = pd.concat(
        [
            noaa_city_counts,
            groundsource_city_counts
        ],
        ignore_index=True
    )

    source_events_by_city = (
        source_events_by_city[
            [
                "source",
                "city",
                "event_count"
            ]
        ]
        .sort_values(
            ["source", "city"]
        )
    )

    # ------------------------------------------------------
    # 48. EVENT COUNTS BY SOURCE AND YEAR
    # ------------------------------------------------------

    noaa_year_counts = (
        noaa
        .dropna(subset=["year"])
        .groupby("year")
        .size()
        .reset_index(
            name="event_count"
        )
    )

    noaa_year_counts[
        "source"
    ] = "NOAA Storm Events"

    groundsource_year_counts = (
        groundsource
        .dropna(subset=["year"])
        .groupby("year")
        .size()
        .reset_index(
            name="event_count"
        )
    )

    groundsource_year_counts[
        "source"
    ] = "Groundsource"

    source_events_by_year = pd.concat(
        [
            noaa_year_counts,
            groundsource_year_counts
        ],
        ignore_index=True
    )

    source_events_by_year["year"] = (
        source_events_by_year[
            "year"
        ].astype(int)
    )

    source_events_by_year = (
        source_events_by_year[
            [
                "source",
                "year",
                "event_count"
            ]
        ]
        .sort_values(
            ["source", "year"]
        )
    )

    # ------------------------------------------------------
    # 49. EVENT COUNTS BY SOURCE, CITY, AND YEAR
    # ------------------------------------------------------

    noaa_city_year = (
        noaa
        .dropna(subset=["year"])
        .groupby(
            ["city", "year"]
        )
        .size()
        .reset_index(
            name="event_count"
        )
    )

    noaa_city_year[
        "source"
    ] = "NOAA Storm Events"

    groundsource_city_year = (
        groundsource
        .dropna(subset=["year"])
        .groupby(
            ["city", "year"]
        )
        .size()
        .reset_index(
            name="event_count"
        )
    )

    groundsource_city_year[
        "source"
    ] = "Groundsource"

    source_events_by_city_year = pd.concat(
        [
            noaa_city_year,
            groundsource_city_year
        ],
        ignore_index=True
    )

    source_events_by_city_year[
        "year"
    ] = (
        source_events_by_city_year[
            "year"
        ].astype(int)
    )

    source_events_by_city_year = (
        source_events_by_city_year[
            [
                "source",
                "city",
                "year",
                "event_count"
            ]
        ]
        .sort_values(
            [
                "source",
                "city",
                "year"
            ]
        )
    )

    # ------------------------------------------------------
    # 50. NOAA DAMAGE SUMMARY BY CITY
    # ------------------------------------------------------

    noaa_damage_by_city = (
        noaa
        .groupby("city")
        .agg(
            event_count=(
                "city",
                "size"
            ),

            events_with_damage_values=(
                "noaa_total_damage_usd",
                "count"
            ),

            damaging_events=(
                "noaa_total_damage_usd",
                lambda values:
                int((values > 0).sum())
            ),

            total_damage_usd=(
                "noaa_total_damage_usd",
                safe_sum
            ),

            mean_damage_per_event_usd=(
                "noaa_total_damage_usd",
                "mean"
            ),

            median_damage_per_event_usd=(
                "noaa_total_damage_usd",
                "median"
            ),

            max_damage_usd=(
                "noaa_total_damage_usd",
                "max"
            )
        )
        .reset_index()
        .sort_values("city")
    )

    noaa_damage_by_city[
        "percent_events_with_reported_damage"
    ] = np.where(
        noaa_damage_by_city[
            "events_with_damage_values"
        ] > 0,
        (
            100
            * noaa_damage_by_city[
                "damaging_events"
            ]
            / noaa_damage_by_city[
                "events_with_damage_values"
            ]
        ).round(1),
        np.nan
    )

    # ------------------------------------------------------
    # 51. NOAA DAMAGE BY YEAR
    # ------------------------------------------------------

    noaa_damage_by_year = (
        noaa
        .dropna(subset=["year"])
        .groupby("year")
        .agg(
            event_count=(
                "city",
                "size"
            ),

            damaging_events=(
                "noaa_total_damage_usd",
                lambda values:
                int((values > 0).sum())
            ),

            total_damage_usd=(
                "noaa_total_damage_usd",
                safe_sum
            ),

            mean_damage_per_event_usd=(
                "noaa_total_damage_usd",
                "mean"
            ),

            median_damage_per_event_usd=(
                "noaa_total_damage_usd",
                "median"
            ),

            max_damage_usd=(
                "noaa_total_damage_usd",
                "max"
            )
        )
        .reset_index()
        .sort_values("year")
    )

    noaa_damage_by_year[
        "year"
    ] = noaa_damage_by_year[
        "year"
    ].astype(int)

    # ------------------------------------------------------
    # 52. GROUNDSOURCE AREA SUMMARY BY CITY
    # ------------------------------------------------------

    groundsource_area_by_city = (
        groundsource
        .groupby("city")
        .agg(
            episode_count=(
                "city",
                "size"
            ),

            episodes_with_area=(
                "groundsource_area_max_km2",
                "count"
            ),

            mean_episode_max_area_km2=(
                "groundsource_area_max_km2",
                "mean"
            ),

            median_episode_max_area_km2=(
                "groundsource_area_max_km2",
                "median"
            ),

            max_episode_area_km2=(
                "groundsource_area_max_km2",
                "max"
            ),

            mean_duration_hours=(
                "duration_hours",
                "mean"
            ),

            median_duration_hours=(
                "duration_hours",
                "median"
            )
        )
        .reset_index()
        .sort_values("city")
    )

    # ------------------------------------------------------
    # 53. CLEAN MAPPING DATASET
    # ------------------------------------------------------

    mapping_rows = []

    for _, row in noaa.iterrows():

        mapping_rows.append({
            "source": "NOAA Storm Events",
            "city": row.get(
                "city",
                np.nan
            ),
            "event_id": row.get(
                "episode_id",
                np.nan
            ),
            "event_date": row.get(
                "event_date",
                pd.NaT
            ),
            "end_date": row.get(
                "end_date",
                pd.NaT
            ),
            "year": row.get(
                "year",
                np.nan
            ),
            "latitude": row.get(
                "latitude",
                np.nan
            ),
            "longitude": row.get(
                "longitude",
                np.nan
            ),
            "duration_hours": row.get(
                "duration_hours",
                np.nan
            ),
            "noaa_total_damage_usd": row.get(
                "noaa_total_damage_usd",
                np.nan
            ),
            "groundsource_area_max_km2": np.nan
        })

    for _, row in groundsource.iterrows():

        mapping_rows.append({
            "source": "Groundsource",
            "city": row.get(
                "city",
                np.nan
            ),
            "event_id": row.get(
                "episode_id",
                np.nan
            ),
            "event_date": row.get(
                "event_date",
                pd.NaT
            ),
            "end_date": row.get(
                "end_date",
                pd.NaT
            ),
            "year": row.get(
                "year",
                np.nan
            ),
            "latitude": row.get(
                "latitude",
                np.nan
            ),
            "longitude": row.get(
                "longitude",
                np.nan
            ),
            "duration_hours": row.get(
                "duration_hours",
                np.nan
            ),
            "noaa_total_damage_usd": np.nan,
            "groundsource_area_max_km2": row.get(
                "groundsource_area_max_km2",
                np.nan
            )
        })

    mapping_events = pd.DataFrame(
        mapping_rows
    )

    mapping_events = (
        mapping_events
        .dropna(
            subset=[
                "latitude",
                "longitude"
            ]
        )
        .sort_values(
            [
                "source",
                "city",
                "event_date"
            ]
        )
    )

    # ======================================================
    # COMMON-PERIOD ANALYSIS: 2005-2025
    # ======================================================

    # ------------------------------------------------------
    # 54. COMMON-PERIOD SOURCE SUMMARY
    # ------------------------------------------------------

    common_summary_rows = []

    for source_name, dataframe in [
        ("NOAA Storm Events", noaa_common),
        ("Groundsource", groundsource_common)
    ]:

        total_events = len(dataframe)

        cities_present = (
            dataframe["city"]
            .dropna()
            .nunique()
        )

        average_events_per_year = (
            total_events
            / common_year_span
        )

        average_events_per_city = (
            total_events
            / cities_present
            if cities_present > 0
            else np.nan
        )

        common_summary_rows.append({
            "source": source_name,
            "start_year": COMMON_START_YEAR,
            "end_year": COMMON_END_YEAR,
            "year_span": common_year_span,
            "total_events": total_events,
            "average_events_per_year": round(
                average_events_per_year,
                2
            ),
            "cities_present": cities_present,
            "average_events_per_city": round(
                average_events_per_city,
                2
            ) if pd.notna(
                average_events_per_city
            ) else np.nan
        })

    common_period_source_summary = pd.DataFrame(
        common_summary_rows
    )

    # ------------------------------------------------------
    # 55. COMMON-PERIOD EVENTS BY YEAR
    # ------------------------------------------------------

    noaa_common_year = (
        noaa_common
        .groupby("year")
        .size()
        .reset_index(
            name="event_count"
        )
    )

    noaa_common_year[
        "source"
    ] = "NOAA Storm Events"

    groundsource_common_year = (
        groundsource_common
        .groupby("year")
        .size()
        .reset_index(
            name="event_count"
        )
    )

    groundsource_common_year[
        "source"
    ] = "Groundsource"

    common_period_events_by_year = pd.concat(
        [
            noaa_common_year,
            groundsource_common_year
        ],
        ignore_index=True
    )

    common_period_events_by_year[
        "year"
    ] = (
        common_period_events_by_year[
            "year"
        ].astype(int)
    )

    common_period_events_by_year = (
        common_period_events_by_year[
            [
                "source",
                "year",
                "event_count"
            ]
        ]
        .sort_values(
            [
                "source",
                "year"
            ]
        )
    )

    # ------------------------------------------------------
    # 56. COMMON-PERIOD EVENTS BY CITY
    # ------------------------------------------------------

    noaa_common_city = (
        noaa_common
        .groupby("city")
        .size()
        .reset_index(
            name="event_count"
        )
    )

    noaa_common_city[
        "source"
    ] = "NOAA Storm Events"

    groundsource_common_city = (
        groundsource_common
        .groupby("city")
        .size()
        .reset_index(
            name="event_count"
        )
    )

    groundsource_common_city[
        "source"
    ] = "Groundsource"

    common_period_events_by_city = pd.concat(
        [
            noaa_common_city,
            groundsource_common_city
        ],
        ignore_index=True
    )

    common_period_events_by_city = (
        common_period_events_by_city[
            [
                "source",
                "city",
                "event_count"
            ]
        ]
        .sort_values(
            [
                "source",
                "city"
            ]
        )
    )

    # ------------------------------------------------------
    # 57. COMMON-PERIOD EVENTS BY CITY AND YEAR
    # ------------------------------------------------------

    noaa_common_city_year = (
        noaa_common
        .groupby(
            [
                "city",
                "year"
            ]
        )
        .size()
        .reset_index(
            name="event_count"
        )
    )

    noaa_common_city_year[
        "source"
    ] = "NOAA Storm Events"

    groundsource_common_city_year = (
        groundsource_common
        .groupby(
            [
                "city",
                "year"
            ]
        )
        .size()
        .reset_index(
            name="event_count"
        )
    )

    groundsource_common_city_year[
        "source"
    ] = "Groundsource"

    common_period_events_by_city_year = pd.concat(
        [
            noaa_common_city_year,
            groundsource_common_city_year
        ],
        ignore_index=True
    )

    common_period_events_by_city_year[
        "year"
    ] = (
        common_period_events_by_city_year[
            "year"
        ].astype(int)
    )

    common_period_events_by_city_year = (
        common_period_events_by_city_year[
            [
                "source",
                "city",
                "year",
                "event_count"
            ]
        ]
        .sort_values(
            [
                "source",
                "city",
                "year"
            ]
        )
    )

    # ======================================================
    # SEPARATE GRAPH-FRIENDLY SOURCE TABLES
    # ======================================================

    # ------------------------------------------------------
    # 58. NOAA EVENTS BY CITY
    # ------------------------------------------------------

    noaa_events_by_city = (
        noaa_common
        .groupby("city")
        .size()
        .reset_index(
            name="event_count"
        )
        .sort_values("city")
    )

    # ------------------------------------------------------
    # 59. GROUNDSOURCE EPISODES BY CITY
    # ------------------------------------------------------

    groundsource_events_by_city = (
        groundsource_common
        .groupby("city")
        .size()
        .reset_index(
            name="event_count"
        )
        .sort_values("city")
    )

    # ------------------------------------------------------
    # 60. NOAA EVENTS BY YEAR
    # ------------------------------------------------------

    noaa_events_by_year = (
        noaa_common
        .groupby("year")
        .size()
        .reset_index(
            name="event_count"
        )
        .sort_values("year")
    )

    noaa_events_by_year[
        "year"
    ] = (
        noaa_events_by_year[
            "year"
        ].astype(int)
    )

    # ------------------------------------------------------
    # 61. GROUNDSOURCE EPISODES BY YEAR
    # ------------------------------------------------------

    groundsource_events_by_year = (
        groundsource_common
        .groupby("year")
        .size()
        .reset_index(
            name="event_count"
        )
        .sort_values("year")
    )

    groundsource_events_by_year[
        "year"
    ] = (
        groundsource_events_by_year[
            "year"
        ].astype(int)
    )

    # ------------------------------------------------------
    # 62. NOAA DURATION SUMMARY BY CITY
    # ------------------------------------------------------

    noaa_duration_summary_by_city = (
        noaa_common
        .groupby("city")
        .agg(
            event_count=("city", "size"),
            mean_duration_hours=("duration_hours", "mean"),
            median_duration_hours=("duration_hours", "median"),
            min_duration_hours=("duration_hours", "min"),
            max_duration_hours=("duration_hours", "max"),
        )
        .reset_index()
        .sort_values("city")
    )


    # ------------------------------------------------------
    # 63. AVERAGE EVENTS PER YEAR BY CITY
    # ------------------------------------------------------

    avg_events_rows = []

    for source_name, dataframe in [
        ("NOAA Storm Events", noaa_common),
        ("Groundsource", groundsource_common)
    ]:

        city_counts = (
            dataframe
            .groupby("city")
            .size()
            .reset_index(name="total_events")
        )

        city_counts["source"] = source_name

        city_counts["average_events_per_year"] = (
            city_counts["total_events"]
            / common_year_span
        ).round(2)

        avg_events_rows.append(city_counts)

    average_events_per_year_by_city = pd.concat(
        avg_events_rows,
        ignore_index=True
    )

    average_events_per_year_by_city = (
        average_events_per_year_by_city[
            [
                "source",
                "city",
                "total_events",
                "average_events_per_year"
            ]
        ]
        .sort_values(
            [
                "source",
                "city"
            ]
        )
    )

    # ======================================================
    # WRITE FILES
    # ======================================================

    # Existing outputs 38-43

    damage_probability.to_csv(
        DATA_DIR
        / "38_damage_probability_by_duration_bin.csv",
        index=False
    )

    area_vs_damage.to_csv(
        DATA_DIR
        / "39_high_confidence_area_vs_damage.csv",
        index=False
    )

    damaging_duration.to_csv(
        DATA_DIR
        / "40_high_confidence_damaging_events_duration_vs_damage.csv",
        index=False
    )

    match_diagnostic.to_csv(
        DATA_DIR
        / "41_match_distance_vs_temporal_gap.csv",
        index=False
    )

    county_damage.to_csv(
        DATA_DIR
        / "42_damage_per_event_by_county_high_confidence.csv",
        index=False
    )

    annual_severity.to_csv(
        DATA_DIR
        / "43_annual_frequency_damage_severity.csv",
        index=False
    )

    # Source-only outputs 44-53

    noaa_source_only.to_csv(
        DATA_DIR
        / "44_noaa_events_source_only.csv",
        index=False
    )

    groundsource_source_only.to_csv(
        DATA_DIR
        / "45_groundsource_episodes_source_only.csv",
        index=False
    )

    source_overall_summary.to_csv(
        DATA_DIR
        / "46_source_overall_summary.csv",
        index=False
    )

    source_events_by_city.to_csv(
        DATA_DIR
        / "47_source_events_by_city.csv",
        index=False
    )

    source_events_by_year.to_csv(
        DATA_DIR
        / "48_source_events_by_year.csv",
        index=False
    )

    source_events_by_city_year.to_csv(
        DATA_DIR
        / "49_source_events_by_city_and_year.csv",
        index=False
    )

    noaa_damage_by_city.to_csv(
        DATA_DIR
        / "50_noaa_damage_summary_by_city.csv",
        index=False
    )

    noaa_damage_by_year.to_csv(
        DATA_DIR
        / "51_noaa_damage_by_year.csv",
        index=False
    )

    groundsource_area_by_city.to_csv(
        DATA_DIR
        / "52_groundsource_area_summary_by_city.csv",
        index=False
    )

    mapping_events.to_csv(
        DATA_DIR
        / "53_mapping_events.csv",
        index=False
    )

    # Common-period outputs 54-57

    common_period_source_summary.to_csv(
        DATA_DIR
        / "54_common_period_source_summary.csv",
        index=False
    )

    common_period_events_by_year.to_csv(
        DATA_DIR
        / "55_common_period_events_by_year.csv",
        index=False
    )

    common_period_events_by_city.to_csv(
        DATA_DIR
        / "56_common_period_events_by_city.csv",
        index=False
    )

    common_period_events_by_city_year.to_csv(
        DATA_DIR
        / "57_common_period_events_by_city_and_year.csv",
        index=False
    )

    # Separate graph-friendly outputs 58-61

    noaa_events_by_city.to_csv(
        DATA_DIR
        / "58_noaa_events_by_city.csv",
        index=False
    )

    groundsource_events_by_city.to_csv(
        DATA_DIR
        / "59_groundsource_events_by_city.csv",
        index=False
    )

    noaa_events_by_year.to_csv(
        DATA_DIR
        / "60_noaa_events_by_year.csv",
        index=False
    )

    groundsource_events_by_year.to_csv(
        DATA_DIR
        / "61_groundsource_events_by_year.csv",
        index=False
    )
    
    noaa_duration_summary_by_city.to_csv(
    DATA_DIR
    / "62_noaa_duration_summary_by_city.csv",
    index=False
    )

    average_events_per_year_by_city.to_csv(
        DATA_DIR
        / "63_average_events_per_year_by_city.csv",
        index=False
    )

    # ======================================================
    # TERMINAL SUMMARY
    # ======================================================

    print("\nDONE")
    print(f"Output folder: {DATA_DIR}")

    print("\nSOURCE-ONLY COUNTS")

    print(
        f"NOAA flash-flood events: "
        f"{len(noaa):,}"
    )

    print(
        f"Groundsource candidate episodes: "
        f"{len(groundsource):,}"
    )

    print(
        f"Combined event/episode rows: "
        f"{len(noaa) + len(groundsource):,}"
    )

    print("\nNOAA EVENTS BY STUDY AREA")

    print(
        noaa
        .groupby("city")
        .size()
        .to_string()
    )

    print("\nGROUNDSOURCE EPISODES BY STUDY AREA")

    print(
        groundsource
        .groupby("city")
        .size()
        .to_string()
    )

    print("\nYEAR COVERAGE")

    if noaa["year"].notna().any():
        print(
            "NOAA:",
            int(noaa["year"].min()),
            "-",
            int(noaa["year"].max())
        )

    if groundsource["year"].notna().any():
        print(
            "Groundsource:",
            int(
                groundsource[
                    "year"
                ].min()
            ),
            "-",
            int(
                groundsource[
                    "year"
                ].max()
            )
        )

    print("\nCOMMON COMPARISON PERIOD")

    print(
        f"{COMMON_START_YEAR}-{COMMON_END_YEAR} "
        f"({common_year_span} years)"
    )

    print(
        f"NOAA common-period events: "
        f"{len(noaa_common):,}"
    )

    print(
        f"Groundsource common-period episodes: "
        f"{len(groundsource_common):,}"
    )

    print("\nAVERAGE EVENTS PER YEAR DURING COMMON PERIOD")

    print(
        "NOAA:",
        round(
            len(noaa_common)
            / common_year_span,
            2
        )
    )

    print(
        "Groundsource:",
        round(
            len(groundsource_common)
            / common_year_span,
            2
        )
    )

    print("\nCreated outputs 38-61.")


if __name__ == "__main__":
    main()