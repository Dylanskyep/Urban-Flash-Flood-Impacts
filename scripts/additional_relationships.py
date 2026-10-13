"""
Additional relationship tables for Urban Flash-Flood Impacts.

Inputs:
    flash_flood_outputs/24_matched_event_pairs_enriched.csv
    flash_flood_outputs/28_high_confidence_matched_events.csv

Outputs:
    38_damage_probability_by_duration_bin.csv
    39_high_confidence_area_vs_damage.csv
    40_high_confidence_damaging_events_duration_vs_damage.csv
    41_match_distance_vs_temporal_gap.csv
    42_damage_per_event_by_county_high_confidence.csv
    43_annual_frequency_damage_severity.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_CANDIDATES = [
    BASE_DIR / "flash_flood_outputs",
    BASE_DIR.parent / "flash_flood_outputs",
]

OUTPUT_DIR = next((p for p in OUTPUT_CANDIDATES if p.exists()), BASE_DIR / "flash_flood_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PAIR_CANDIDATES = [
    OUTPUT_DIR / "24_matched_event_pairs_enriched.csv",
    BASE_DIR / "24_matched_event_pairs_enriched.csv",
    BASE_DIR.parent / "data" / "processed" / "24_matched_event_pairs_enriched.csv",
]

HIGH_CANDIDATES = [
    OUTPUT_DIR / "28_high_confidence_matched_events.csv",
    BASE_DIR / "28_high_confidence_matched_events.csv",
    BASE_DIR.parent / "data" / "processed" / "28_high_confidence_matched_events.csv",
]


def find_file(candidates, label):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {label}. Checked:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def to_numeric(df, column):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def main():
    pair_path = find_file(PAIR_CANDIDATES, "24_matched_event_pairs_enriched.csv")
    high_path = find_file(HIGH_CANDIDATES, "28_high_confidence_matched_events.csv")

    print(f"Reading: {pair_path}")
    pairs = pd.read_csv(pair_path)
    print(f"Reading: {high_path}")
    high = pd.read_csv(high_path)

    high["noaa_duration_hours"] = to_numeric(high, "noaa_duration_hours")
    high["noaa_total_damage_usd"] = to_numeric(high, "noaa_total_damage_usd")
    high["groundsource_area_max_km2"] = to_numeric(high, "groundsource_area_max_km2")
    pairs["distance_km"] = to_numeric(pairs, "distance_km")
    pairs["temporal_gap_hours"] = to_numeric(pairs, "temporal_gap_hours")

    duration_bins = [-np.inf, 1, 3, 6, 12, np.inf]
    duration_labels = ["<1 hour", "1–3 hours", "3–6 hours", "6–12 hours", ">12 hours"]

    damage_probability_events = high[
        high["noaa_duration_hours"].notna()
        & high["noaa_total_damage_usd"].notna()
    ].copy()
    damage_probability_events["duration_bin"] = pd.cut(
        damage_probability_events["noaa_duration_hours"],
        bins=duration_bins,
        labels=duration_labels,
        right=False,
    )
    damage_probability_events["has_reported_damage"] = (
        damage_probability_events["noaa_total_damage_usd"] > 0
    )
    damage_probability = (
        damage_probability_events
        .groupby("duration_bin", observed=False)
        .agg(
            total_events=("match_id", "count"),
            damaging_events=("has_reported_damage", "sum"),
        )
        .reset_index()
    )
    damage_probability["damage_probability_percent"] = np.where(
        damage_probability["total_events"] > 0,
        (
            100
            * damage_probability["damaging_events"]
            / damage_probability["total_events"]
        ).round(1),
        np.nan,
    )
    damage_probability["duration_bin"] = damage_probability["duration_bin"].astype(str)

    area_vs_damage = high[
        high["groundsource_area_max_km2"].notna()
        & high["noaa_total_damage_usd"].notna()
    ].copy()
    area_cols = [
        "match_id", "city", "match_confidence", "noaa_event_date",
        "groundsource_event_date", "groundsource_area_max_km2",
        "noaa_duration_hours", "noaa_total_damage_usd"
    ]
    area_vs_damage = area_vs_damage[[c for c in area_cols if c in area_vs_damage.columns]]
    area_vs_damage = area_vs_damage.sort_values(
        [c for c in ["city", "groundsource_area_max_km2"] if c in area_vs_damage.columns]
    )

    damaging_duration = high[
        high["noaa_duration_hours"].notna()
        & high["noaa_total_damage_usd"].notna()
        & (high["noaa_total_damage_usd"] > 0)
    ].copy()
    damaging_cols = [
        "match_id", "city", "match_confidence", "noaa_event_date",
        "noaa_duration_hours", "noaa_total_damage_usd",
        "groundsource_area_max_km2"
    ]
    damaging_duration = damaging_duration[
        [c for c in damaging_cols if c in damaging_duration.columns]
    ].sort_values(
        [c for c in ["city", "noaa_duration_hours"] if c in damaging_duration.columns]
    )

    match_diagnostic = pairs[
        pairs["distance_km"].notna()
        & pairs["temporal_gap_hours"].notna()
    ].copy()
    diagnostic_columns = [
        "match_id", "city", "match_confidence", "distance_km",
        "temporal_gap_hours", "intervals_overlap", "match_score_0_100"
    ]
    diagnostic_columns = [c for c in diagnostic_columns if c in match_diagnostic.columns]
    match_diagnostic = match_diagnostic[diagnostic_columns].sort_values(
        [c for c in ["match_confidence", "distance_km", "temporal_gap_hours"]
         if c in diagnostic_columns]
    )

    county_damage_events = high[high["noaa_total_damage_usd"].notna()].copy()
    county_damage = (
        county_damage_events
        .groupby("city")
        .agg(
            matched_events=("match_id", "count"),
            damaging_events=("noaa_total_damage_usd", lambda s: int((s > 0).sum())),
            total_noaa_damage_usd=("noaa_total_damage_usd", "sum"),
            mean_noaa_damage_per_event_usd=("noaa_total_damage_usd", "mean"),
            median_noaa_damage_per_event_usd=("noaa_total_damage_usd", "median"),
            max_noaa_damage_usd=("noaa_total_damage_usd", "max"),
        )
        .reset_index()
        .sort_values("city")
    )
    county_damage["percent_events_with_damage"] = np.where(
        county_damage["matched_events"] > 0,
        (100 * county_damage["damaging_events"] / county_damage["matched_events"]).round(1),
        np.nan,
    )

    high["noaa_year"] = pd.to_datetime(high["noaa_event_date"], errors="coerce").dt.year
    annual_events = high[
        high["noaa_year"].notna()
        & high["noaa_total_damage_usd"].notna()
    ].copy()
    annual_severity = (
        annual_events
        .groupby("noaa_year")
        .agg(
            event_count=("match_id", "count"),
            damaging_events=("noaa_total_damage_usd", lambda s: int((s > 0).sum())),
            total_noaa_damage_usd=("noaa_total_damage_usd", "sum"),
            mean_noaa_damage_usd=("noaa_total_damage_usd", "mean"),
            median_noaa_damage_usd=("noaa_total_damage_usd", "median"),
            max_noaa_damage_usd=("noaa_total_damage_usd", "max"),
        )
        .reset_index()
        .sort_values("noaa_year")
    )
    annual_severity["percent_events_with_damage"] = np.where(
        annual_severity["event_count"] > 0,
        (100 * annual_severity["damaging_events"] / annual_severity["event_count"]).round(1),
        np.nan,
    )

    damage_probability.to_csv(OUTPUT_DIR / "38_damage_probability_by_duration_bin.csv", index=False)
    area_vs_damage.to_csv(OUTPUT_DIR / "39_high_confidence_area_vs_damage.csv", index=False)
    damaging_duration.to_csv(
        OUTPUT_DIR / "40_high_confidence_damaging_events_duration_vs_damage.csv",
        index=False,
    )
    match_diagnostic.to_csv(OUTPUT_DIR / "41_match_distance_vs_temporal_gap.csv", index=False)
    county_damage.to_csv(
        OUTPUT_DIR / "42_damage_per_event_by_county_high_confidence.csv",
        index=False,
    )
    annual_severity.to_csv(
        OUTPUT_DIR / "43_annual_frequency_damage_severity.csv",
        index=False,
    )

    print("\nDONE")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Created 38-43 additional relationship CSVs.")


if __name__ == "__main__":
    main()
