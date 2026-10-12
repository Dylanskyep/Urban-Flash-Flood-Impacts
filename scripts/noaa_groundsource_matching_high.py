"""
Optimized NOAA <-> Groundsource event matcher and impact linker.
Input: flash_flood_outputs/14_harmonized_overlap_events.csv
Outputs:
  24_matched_event_pairs_enriched.csv
  25_groundsource_with_noaa_impacts.csv
  26_noaa_event_match_status.csv
  27_match_summary_by_county.csv
28_high_confidence_matched_events.csv
29_high_confidence_groundsource_with_noaa_impacts.csv
  30_noaa_damage_by_duration_and_county.csv
  31_noaa_damage_by_duration.csv
 32_noaa_damage_by_duration_bin.csv
 33_match_confidence_by_county.csv
 34_noaa_damage_by_year_high_confidence.csv
 35_match_rate_by_confidence.csv
 36_damage_status_by_county_high_confidence.csv
 37_duration_by_damage_status_high_confidence.csv
A match is a likely corresponding flood, not proof of identity.
NOAA impact values remain NOAA-reported attributes.
Requirements: pandas, numpy, scipy
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "flash_flood_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_CANDIDATES = [
    OUTPUT_DIR / "14_harmonized_overlap_events.csv",
    BASE_DIR / "14_harmonized_overlap_events.csv",
]
MAX_TIME_GAP_HOURS = 24.0
MAX_DISTANCE_KM = 10.0
HIGH_TIME_GAP_HOURS = 12.0
HIGH_DISTANCE_KM = 5.0
def find_input_file():
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find 14_harmonized_overlap_events.csv in the project folder "
        "or flash_flood_outputs/."
    )
def first_existing_column(df, names):
    lookup = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None
def numeric_series(df, names):
    col = first_existing_column(df, names)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")
def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    r = 6371.0088
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
def normalize_interval(row):
    start = row["event_date"]
    end = row.get("end_date", start)
    if pd.isna(start):
        return pd.NaT, pd.NaT
    if pd.isna(end):
        end = start
    if end < start:
        start, end = end, start
    return start, end
def temporal_relationship(a, b):
    a0, a1 = normalize_interval(a)
    b0, b1 = normalize_interval(b)
    if pd.isna(a0) or pd.isna(b0):
        return False, np.nan
    overlap = (a0 <= b1) and (b0 <= a1)
    if overlap:
        return True, 0.0
    if a1 < b0:
        gap = (b0 - a1).total_seconds() / 3600
    else:
        gap = (a0 - b1).total_seconds() / 3600
    return False, float(max(gap, 0.0))
def match_score(overlap, gap_hours, distance_km):
    # Heuristic ranking score, not a probability.
    time_component = max(0.0, 1.0 - gap_hours / MAX_TIME_GAP_HOURS)
    distance_component = max(0.0, 1.0 - distance_km / MAX_DISTANCE_KM)
    overlap_bonus = 1.0 if overlap else 0.0
    score = 100 * (0.50 * time_component + 0.40 * distance_component + 0.10 * overlap_bonus)
    return round(min(max(score, 0.0), 100.0), 1)
def confidence_level(overlap, gap_hours, distance_km):
    if distance_km <= HIGH_DISTANCE_KM and (overlap or gap_hours <= HIGH_TIME_GAP_HOURS):
        return "High"
    return "Medium"
def load_events():
    path = find_input_file()
    print(f"Reading: {path}")
    df = pd.read_csv(path)
    required = ["city", "source", "event_date", "latitude", "longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if "end_date" not in df.columns:
        df["end_date"] = df["event_date"]
    else:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df
def add_standardized_noaa_impacts(noaa):
    noaa = noaa.copy()
    noaa["noaa_property_damage_usd"] = numeric_series(noaa, ["property_damage_usd", "damage_property", "property_damage"])
    noaa["noaa_crop_damage_usd"] = numeric_series(noaa, ["crop_damage_usd", "damage_crops", "crop_damage"])
    existing_total = numeric_series(noaa, ["total_damage_usd", "damage_total", "total_damage"])
    calc_total = noaa["noaa_property_damage_usd"].fillna(0) + noaa["noaa_crop_damage_usd"].fillna(0)
    both_missing = noaa["noaa_property_damage_usd"].isna() & noaa["noaa_crop_damage_usd"].isna()
    calc_total.loc[both_missing] = np.nan
    noaa["noaa_total_damage_usd"] = existing_total.combine_first(calc_total)
    noaa["noaa_direct_deaths"] = numeric_series(noaa, ["deaths_direct", "direct_deaths", "direct_fatalities", "fatalities_direct"])
    noaa["noaa_indirect_deaths"] = numeric_series(noaa, ["deaths_indirect", "indirect_deaths", "indirect_fatalities", "fatalities_indirect"])
    total_deaths = numeric_series(noaa, ["total_deaths", "total_fatalities", "fatalities"])
    calc_deaths = noaa["noaa_direct_deaths"].fillna(0) + noaa["noaa_indirect_deaths"].fillna(0)
    both_death_missing = noaa["noaa_direct_deaths"].isna() & noaa["noaa_indirect_deaths"].isna()
    calc_deaths.loc[both_death_missing] = np.nan
    noaa["noaa_total_deaths"] = total_deaths.combine_first(calc_deaths)
    noaa["noaa_direct_injuries"] = numeric_series(noaa, ["injuries_direct", "direct_injuries"])
    noaa["noaa_indirect_injuries"] = numeric_series(noaa, ["injuries_indirect", "indirect_injuries"])
    total_inj = numeric_series(noaa, ["total_injuries", "injuries"])
    calc_inj = noaa["noaa_direct_injuries"].fillna(0) + noaa["noaa_indirect_injuries"].fillna(0)
    both_inj_missing = noaa["noaa_direct_injuries"].isna() & noaa["noaa_indirect_injuries"].isna()
    calc_inj.loc[both_inj_missing] = np.nan
    noaa["noaa_total_injuries"] = total_inj.combine_first(calc_inj)
    return noaa
IMPACT_COLS = [
    "noaa_property_damage_usd", "noaa_crop_damage_usd", "noaa_total_damage_usd",
    "noaa_direct_deaths", "noaa_indirect_deaths", "noaa_total_deaths",
    "noaa_direct_injuries", "noaa_indirect_injuries", "noaa_total_injuries",
]
def candidate_pairs(noaa, gs):
    rows = []
    for ni, n in noaa.iterrows():
        if pd.isna(n["latitude"]) or pd.isna(n["longitude"]):
            continue
        for gi, g in gs.iterrows():
            if pd.isna(g["latitude"]) or pd.isna(g["longitude"]):
                continue
            overlap, gap = temporal_relationship(n, g)
            if pd.isna(gap) or gap > MAX_TIME_GAP_HOURS:
                continue
            dist = haversine_km(n["latitude"], n["longitude"], g["latitude"], g["longitude"])
            if pd.isna(dist) or dist > MAX_DISTANCE_KM:
                continue
            rows.append({
                "noaa_index": ni,
                "groundsource_index": gi,
                "intervals_overlap": overlap,
                "temporal_gap_hours": round(gap, 2),
                "distance_km": round(dist, 3),
                "match_score_0_100": match_score(overlap, gap, dist),
                "match_confidence": confidence_level(overlap, gap, dist),
            })
    return pd.DataFrame(rows)
def global_one_to_one(noaa, gs, candidates):
    if candidates.empty or noaa.empty or gs.empty:
        return pd.DataFrame()
    noaa_ids = list(noaa.index)
    gs_ids = list(gs.index)
    npos = {idx: i for i, idx in enumerate(noaa_ids)}
    gpos = {idx: j for j, idx in enumerate(gs_ids)}
    n, m = len(noaa_ids), len(gs_ids)
    # Real GS columns + one dummy unmatched column per NOAA row.
    cost = np.full((n, m + n), 999.0)
    for i in range(n):
        cost[i, m + i] = 1.01
    lookup = {}
    for _, c in candidates.iterrows():
        i, j = npos[c["noaa_index"]], gpos[c["groundsource_index"]]
        cost[i, j] = 1.0 - float(c["match_score_0_100"]) / 100.0
        lookup[(c["noaa_index"], c["groundsource_index"])] = c
    rows, cols = linear_sum_assignment(cost)
    selected = []
    for i, j in zip(rows, cols):
        if j >= m or cost[i, j] >= 999:
            continue
        selected.append(lookup[(noaa_ids[i], gs_ids[j])].to_dict())
    return pd.DataFrame(selected)
def getv(row, col):
    return row[col] if col in row.index else np.nan
def build_pair_rows(city, noaa, gs, selected):
    rows = []
    if selected.empty:
        return pd.DataFrame()
    for k, (_, match) in enumerate(selected.iterrows(), start=1):
        n = noaa.loc[int(match["noaa_index"])]
        g = gs.loc[int(match["groundsource_index"])]
        row = {
            "match_id": f"{city.upper()}_MATCH_{k:04d}",
            "city": city,
            "match_confidence": match["match_confidence"],
            "match_score_0_100": match["match_score_0_100"],
            "intervals_overlap": bool(match["intervals_overlap"]),
            "temporal_gap_hours": match["temporal_gap_hours"],
            "distance_km": match["distance_km"],
            "groundsource_episode_id": getv(g, "episode_id"),
            "groundsource_event_date": g["event_date"],
            "groundsource_end_date": g["end_date"],
            "groundsource_latitude": g["latitude"],
            "groundsource_longitude": g["longitude"],
            "groundsource_duration_hours": getv(g, "duration_hours"),
            "groundsource_episode_member_count": getv(g, "episode_member_count"),
            "groundsource_area_sum_km2": getv(g, "groundsource_area_sum_km2"),
            "groundsource_area_mean_km2": getv(g, "groundsource_area_mean_km2"),
            "groundsource_area_max_km2": getv(g, "groundsource_area_max_km2"),
            "noaa_episode_id": getv(n, "episode_id"),
            "noaa_event_date": n["event_date"],
            "noaa_end_date": n["end_date"],
            "noaa_latitude": n["latitude"],
            "noaa_longitude": n["longitude"],
            "noaa_duration_hours": getv(n, "duration_hours"),
        }
        for c in IMPACT_COLS:
            row[c] = getv(n, c)
        rows.append(row)
    return pd.DataFrame(rows)
def build_groundsource_enriched(gs, pairs):
    out = gs.copy()
    out["matched_to_noaa"] = False
    out["match_id"] = pd.NA
    out["match_confidence"] = pd.NA
    out["match_score_0_100"] = np.nan
    out["intervals_overlap"] = pd.NA
    out["temporal_gap_hours"] = np.nan
    out["match_distance_km"] = np.nan
    for c in IMPACT_COLS:
        out[c] = np.nan
    if pairs.empty:
        return out
    for _, p in pairs.iterrows():
        mask = out["episode_id"] == p["groundsource_episode_id"]
        out.loc[mask, "matched_to_noaa"] = True
        out.loc[mask, "match_id"] = p["match_id"]
        out.loc[mask, "match_confidence"] = p["match_confidence"]
        out.loc[mask, "match_score_0_100"] = p["match_score_0_100"]
        out.loc[mask, "intervals_overlap"] = p["intervals_overlap"]
        out.loc[mask, "temporal_gap_hours"] = p["temporal_gap_hours"]
        out.loc[mask, "match_distance_km"] = p["distance_km"]
        for c in IMPACT_COLS:
            out.loc[mask, c] = p[c]
    return out
def build_noaa_status(noaa, pairs):
    out = noaa.copy()
    out["has_usable_coordinates"] = out["latitude"].notna() & out["longitude"].notna()
    out["matched_to_groundsource"] = False
    out["match_id"] = pd.NA
    out["match_confidence"] = pd.NA
    out["match_score_0_100"] = np.nan
    out["temporal_gap_hours"] = np.nan
    out["match_distance_km"] = np.nan
    if pairs.empty:
        return out
    for _, p in pairs.iterrows():
        mask = out["episode_id"] == p["noaa_episode_id"]
        out.loc[mask, "matched_to_groundsource"] = True
        out.loc[mask, "match_id"] = p["match_id"]
        out.loc[mask, "match_confidence"] = p["match_confidence"]
        out.loc[mask, "match_score_0_100"] = p["match_score_0_100"]
        out.loc[mask, "temporal_gap_hours"] = p["temporal_gap_hours"]
        out.loc[mask, "match_distance_km"] = p["distance_km"]
    return out
def build_summary(cities, noaa, gs, pairs):
    rows = []
    for city in cities:
        n = noaa[noaa["city"] == city]
        g = gs[gs["city"] == city]
        m = pairs[pairs["city"] == city] if not pairs.empty else pd.DataFrame()
        usable = n[n["latitude"].notna() & n["longitude"].notna()]
        matched = len(m)
        rows.append({
            "city": city,
            "groundsource_episodes": len(g),
            "noaa_events_total": len(n),
            "noaa_events_with_coordinates": len(usable),
            "matched_pairs": matched,
            "high_confidence_matches": int((m["match_confidence"] == "High").sum()) if not m.empty else 0,
            "medium_confidence_matches": int((m["match_confidence"] == "Medium").sum()) if not m.empty else 0,
            "groundsource_unmatched": len(g) - matched,
            "noaa_with_coords_unmatched": len(usable) - matched,
            "noaa_match_rate_percent": round(100 * matched / len(usable), 1) if len(usable) else np.nan,
        })
    return pd.DataFrame(rows)
def main():
    events = load_events()
    noaa = add_standardized_noaa_impacts(events[events["source"] == "NOAA Storm Events"].copy())
    gs = events[events["source"] == "Groundsource"].copy()
    cities = sorted(events["city"].dropna().unique())
    pair_frames = []
    print("\nMATCHING EVENTS...")
    for city in cities:
        n = noaa[noaa["city"] == city]
        g = gs[gs["city"] == city]
        cand = candidate_pairs(n, g)
        selected = global_one_to_one(n, g, cand)
        pairs = build_pair_rows(city, n, g, selected)
        if not pairs.empty:
            pair_frames.append(pairs)
        high = int((pairs["match_confidence"] == "High").sum()) if not pairs.empty else 0
        print(f"{city}: {len(pairs)} matched pairs ({high} high confidence)")
    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    match_confidence_summary = (
    pairs
    .groupby(["city", "match_confidence"])
    .size()
    .reset_index(name="event_count")
)
    gs_enriched = build_groundsource_enriched(gs, pairs)
    noaa_status = build_noaa_status(noaa, pairs)
    summary = build_summary(cities, noaa, gs, pairs)
    # High-confidence subsets for the strongest impact-analysis graphs.
    high_confidence_pairs = (
        pairs[pairs["match_confidence"] == "High"].copy()
        if not pairs.empty
        else pairs.copy()
    )
    high_confidence_pairs["noaa_duration_hours"] = (
    pd.to_numeric(
        high_confidence_pairs["noaa_duration_hours"],
        errors="coerce"
    ).round(2)
    )
    high_confidence_pairs["noaa_year"] = pd.to_datetime(
    high_confidence_pairs["noaa_event_date"],
    errors="coerce").dt.year
    damage_by_year = (
    high_confidence_pairs
    .groupby("noaa_year")
    .agg(
        event_count=("match_id", "count"),
        total_noaa_damage_usd=("noaa_total_damage_usd", "sum"),
        mean_noaa_damage_usd=("noaa_total_damage_usd", "mean"),
        median_noaa_damage_usd=("noaa_total_damage_usd", "median"),
    )
    .reset_index()
    .sort_values("noaa_year")
    )
    # ----------------------------------------------------------
    # Match-rate sensitivity summary: High only vs High + Medium.
    match_rate_rows = []
    for _, r in summary.iterrows():
        denom = r["noaa_events_with_coordinates"]
        high_count = r["high_confidence_matches"]
        all_count = r["matched_pairs"]
        for match_type, matched_count in [
            ("High confidence", high_count),
            ("High + Medium", all_count),
        ]:
            match_rate_rows.append({
                "city": r["city"],
                "match_type": match_type,
                "matched_events": int(matched_count),
                "noaa_events_with_coordinates": int(denom),
                "match_rate_percent": (
                    round(100 * matched_count / denom, 1)
                    if denom else np.nan
                ),
            })
    match_rate_by_confidence = pd.DataFrame(match_rate_rows)
    # Damage/no-damage status among high-confidence matched events.
    # Missing NOAA damage values are excluded rather than treated as zero.
    damage_status_events = high_confidence_pairs[
        high_confidence_pairs["noaa_total_damage_usd"].notna()
    ].copy()
    damage_status_events["damage_status"] = np.where(
        pd.to_numeric(
            damage_status_events["noaa_total_damage_usd"],
            errors="coerce"
        ) > 0,
        "Reported damage",
        "No reported damage"
    )
    damage_status_by_county = (
        damage_status_events
        .groupby(["city", "damage_status"])
        .size()
        .reset_index(name="event_count")
        .sort_values(["city", "damage_status"])
    )
    # Event-level data for duration distribution by damage status.
    duration_by_damage_status = damage_status_events[
        damage_status_events["noaa_duration_hours"].notna()
    ].copy()
    # Duration-bin summary for NOAA damage
    # HIGH-CONFIDENCE matched NOAA events only
    # ----------------------------------------------------------
    duration_bins = [-np.inf, 1, 3, 6, 12, np.inf]
    duration_labels = [
        "<1 hour",
        "1–3 hours",
        "3–6 hours",
        "6–12 hours",
        ">12 hours",
    ]
    high_confidence_pairs["duration_bin"] = pd.cut(
        high_confidence_pairs["noaa_duration_hours"],
        bins=duration_bins,
        labels=duration_labels,
        right=False
    )
    damage_by_duration_bin = (
        high_confidence_pairs
        .groupby("duration_bin", observed=False)
        .agg(
            event_count=("match_id", "count"),
            events_with_damage=("noaa_total_damage_usd", "count"),
            mean_noaa_total_damage_usd=("noaa_total_damage_usd", "mean"),
            median_noaa_total_damage_usd=("noaa_total_damage_usd", "median"),
            sum_noaa_total_damage_usd=("noaa_total_damage_usd", "sum"),
            max_noaa_total_damage_usd=("noaa_total_damage_usd", "max"),
        )
        .reset_index()
    )
    damage_by_duration_bin["duration_bin"] = (
        damage_by_duration_bin["duration_bin"].astype(str)
    )
    high_confidence_groundsource = gs_enriched[
        gs_enriched["match_confidence"] == "High"
    ].copy()
    # RAWGraphs-ready summaries for NOAA flood duration vs NOAA damage.
    # These use HIGH-CONFIDENCE matched NOAA events only.
    if not high_confidence_pairs.empty:
        damage_by_duration_county = (
            high_confidence_pairs
            .groupby(["city", "noaa_duration_hours"], dropna=False)
            .agg(
                event_count=("match_id", "count"),
                mean_noaa_total_damage_usd=("noaa_total_damage_usd", "mean"),
                median_noaa_total_damage_usd=("noaa_total_damage_usd", "median"),
                sum_noaa_total_damage_usd=("noaa_total_damage_usd", "sum"),
            )
            .reset_index()
            .sort_values(["city", "noaa_duration_hours"])
        )
        damage_by_duration = (
            high_confidence_pairs
            .groupby(["noaa_duration_hours"], dropna=False)
            .agg(
                event_count=("match_id", "count"),
                mean_noaa_total_damage_usd=("noaa_total_damage_usd", "mean"),
                median_noaa_total_damage_usd=("noaa_total_damage_usd", "median"),
                sum_noaa_total_damage_usd=("noaa_total_damage_usd", "sum"),
            )
            .reset_index()
            .sort_values(["noaa_duration_hours"])
        )
    else:
        damage_by_duration_county = pd.DataFrame(columns=[
            "city", "noaa_duration_hours", "event_count",
            "mean_noaa_total_damage_usd", "median_noaa_total_damage_usd",
            "sum_noaa_total_damage_usd"
        ])
        damage_by_duration = pd.DataFrame(columns=[
            "noaa_duration_hours", "event_count",
            "mean_noaa_total_damage_usd", "median_noaa_total_damage_usd",
            "sum_noaa_total_damage_usd"
        ])
    pairs.to_csv(OUTPUT_DIR / "24_matched_event_pairs_enriched.csv", index=False)
    gs_enriched.to_csv(OUTPUT_DIR / "25_groundsource_with_noaa_impacts.csv", index=False)
    noaa_status.to_csv(OUTPUT_DIR / "26_noaa_event_match_status.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "27_match_summary_by_county.csv", index=False)
    high_confidence_pairs.to_csv(
        OUTPUT_DIR / "28_high_confidence_matched_events.csv",
        index=False
    )
    high_confidence_groundsource.to_csv(
        OUTPUT_DIR / "29_high_confidence_groundsource_with_noaa_impacts.csv",
        index=False
    )
    damage_by_duration_county.to_csv(
        OUTPUT_DIR / "30_noaa_damage_by_duration_and_county.csv",
        index=False
    )
    damage_by_duration.to_csv(
        OUTPUT_DIR / "31_noaa_damage_by_duration.csv",
        index=False
    )
    damage_by_duration_bin.to_csv(
    OUTPUT_DIR / "32_noaa_damage_by_duration_bin.csv",
    index=False
    )
    match_confidence_summary.to_csv(
    OUTPUT_DIR / "33_match_confidence_by_county.csv",
    index=False
    )
    damage_by_year.to_csv(
    OUTPUT_DIR / "34_noaa_damage_by_year_high_confidence.csv",
    index=False
    )
    match_rate_by_confidence.to_csv(
        OUTPUT_DIR / "35_match_rate_by_confidence.csv",
        index=False
    )
    damage_status_by_county.to_csv(
        OUTPUT_DIR / "36_damage_status_by_county_high_confidence.csv",
        index=False
    )
    duration_by_damage_status.to_csv(
        OUTPUT_DIR / "37_duration_by_damage_status_high_confidence.csv",
        index=False
    )
    print("\nDONE")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Created 24-37 matching/impact CSVs.")
    print("  28_high_confidence_matched_events.csv")
    print("  29_high_confidence_groundsource_with_noaa_impacts.csv")
    print("  30_noaa_damage_by_duration_and_county.csv")
    print("  31_noaa_damage_by_duration.csv")
    print("  32_noaa_damage_by_duration_bin.csv")
    print("  33_match_confidence_by_county.csv")
    print("  34_noaa_damage_by_year_high_confidence.csv")
    print("  35_match_rate_by_confidence.csv")
    print("  36_damage_status_by_county_high_confidence.csv")
    print("  37_duration_by_damage_status_high_confidence.csv")
    print(f"High-confidence matched pairs available for impact graphs: {len(high_confidence_pairs)}")
    print("Match score is heuristic, not a probability.")
    print("For the strongest impact graphs, use 28_high_confidence_matched_events.csv.")
    print("NOAA impacts remain NOAA-reported impacts linked to likely matching Groundsource episodes.")
    missing = [c for c in IMPACT_COLS if noaa[c].notna().sum() == 0]
    if missing:
        print("\nImpact fields absent from the harmonized input and left blank:")
        for c in missing:
            print(f"  - {c}")
        print("Carry these fields into 14_harmonized_overlap_events.csv from the original NOAA files if needed.")
if __name__ == "__main__":
    main()
