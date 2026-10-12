"""
FLASH-FLOOD RAW + HARMONIZED CSV BUILDER
NOAA Storm Events + Groundsource
Keeps your existing raw outputs and adds harmonized outputs.
Groundsource records are grouped into candidate flood episodes when they are:
- in the same city
- within 24 hours
- within 10 km
Put this script in the same folder as the Groundsource parquet and NOAA city CSVs.\nAlso place an extracted Florida Census PVS county shapefile (PVS_*_county_12.shp) somewhere inside the project folder.\nRun: python graph_creation_harmonized_county_boundaries.py
"""
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb
FOCUS_YEAR = 2024
CLUSTER_TIME_HOURS = 24
CLUSTER_DISTANCE_KM = 10.0
CITIES = [
    "Miami",
    "Tampa",
    "Orlando",
    "Gainesville",
    "Jacksonville",
]
COUNTY_FOR_CITY = {
    "Miami": "Miami-Dade",
    "Tampa": "Hillsborough",
    "Orlando": "Orange",
    "Gainesville": "Alachua",
    "Jacksonville": "Duval",
}
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "flash_flood_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
# Census PVS county shapefile for Florida.
# Your extracted statewide ZIP contains a file such as:
# PVS_25_v2_county_12.shp
_COUNTY_BOUNDARIES = None
def find_county_shapefile():
    candidates = list(BASE_DIR.rglob("PVS_*_county_12.shp"))
    # Also accept a TIGER/Line county shapefile if one is used later.
    candidates += list(BASE_DIR.rglob("tl_*_12_county.shp"))
    if len(candidates) == 1:
        print(f"Using county boundary shapefile: {candidates[0]}")
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "No Florida county shapefile found. "
            "Expected a file such as PVS_25_v2_county_12.shp inside the project folder."
        )
    raise RuntimeError(
        f"More than one Florida county shapefile found: {candidates}"
    )
def load_county_boundaries():
    global _COUNTY_BOUNDARIES
    if _COUNTY_BOUNDARIES is not None:
        return _COUNTY_BOUNDARIES
    shp = find_county_shapefile()
    print(f"Reading county boundaries: {shp.name}")
    counties = gpd.read_file(shp)
    if "NAME" not in counties.columns:
        raise KeyError("The county shapefile does not contain a NAME column.")
    wanted = set(COUNTY_FOR_CITY.values())
    counties = counties[counties["NAME"].isin(wanted)].copy()
    missing = sorted(wanted - set(counties["NAME"]))
    if missing:
        raise ValueError(f"County boundaries not found in shapefile: {missing}")
    counties = counties.to_crs("EPSG:4326")
    _COUNTY_BOUNDARIES = counties
    return _COUNTY_BOUNDARIES
def county_polygon(city):
    counties = load_county_boundaries()
    county_name = COUNTY_FOR_CITY[city]
    row = counties[counties["NAME"] == county_name]
    if row.empty:
        raise ValueError(f"No polygon found for {county_name} County ({city}).")
    # union_all() is the current GeoPandas/Shapely method.
    return row.geometry.union_all()
def points_inside_county(df, city, lon_col="longitude", lat_col="latitude"):
    """
    Return a boolean mask indicating whether each valid point lies
    inside or on the boundary of the county associated with the city label.
    Rows with missing coordinates are False.
    """
    polygon = county_polygon(city)
    valid = df[lon_col].notna() & df[lat_col].notna()
    mask = pd.Series(False, index=df.index)
    if valid.any():
        pts = gpd.GeoSeries(
            gpd.points_from_xy(df.loc[valid, lon_col], df.loc[valid, lat_col]),
            index=df.index[valid],
            crs="EPSG:4326",
        )
        mask.loc[valid] = pts.apply(polygon.covers)
    return mask
def find_groundsource_file():
    files = [p for p in BASE_DIR.glob("*.parquet") if "groundsource" in p.name.lower()]
    if len(files) == 1:
        return files[0]
    if not files:
        raise FileNotFoundError("No Groundsource parquet file found in this folder.")
    raise RuntimeError("More than one Groundsource parquet file found.")
def find_noaa_file(city):
    files = [p for p in BASE_DIR.glob("*.csv") if "noaa" in p.name.lower() and city.lower() in p.name.lower()]
    if len(files) == 1:
        return files[0]
    if not files:
        raise FileNotFoundError(f"No NOAA CSV found for {city}.")
    raise RuntimeError(f"More than one NOAA CSV found for {city}.")
def first_existing_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None
def parse_damage(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().upper().replace("$", "").replace(",", "")
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*([KMB])?", text)
    if not m:
        return np.nan
    number = float(m.group(1))
    multiplier = {None: 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[m.group(2)]
    return number * multiplier
def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    r = 6371.0088
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
def load_groundsource():
    file = find_groundsource_file()
    print(f"Reading Groundsource: {file.name}")
    df = pd.read_parquet(file)
    required = ["uuid", "area_km2", "geometry", "start_date", "end_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Groundsource is missing: {missing}")
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    def decode(g):
        if g is None:
            return None
        if hasattr(g, "centroid"):
            return g
        try:
            return wkb.loads(g)
        except Exception:
            try:
                return wkb.loads(bytes.fromhex(str(g)))
            except Exception:
                return None
    df["decoded_geometry"] = df["geometry"].apply(decode)
    df = df[df["decoded_geometry"].notna() & df["start_date"].notna()].copy()
    df["longitude"] = df["decoded_geometry"].apply(lambda g: g.centroid.x)
    df["latitude"] = df["decoded_geometry"].apply(lambda g: g.centroid.y)
    return df
def groundsource_city_records(all_groundsource, city):
    # Use the county polygon associated with this city label.
    mask = points_inside_county(all_groundsource, city)
    df = all_groundsource[mask].copy()
    print(f"  Groundsource inside {COUNTY_FOR_CITY[city]} County ({city}): {len(df):,}")
    df["event_date"] = df["start_date"]
    df["duration_hours"] = (
        (df["end_date"] - df["event_date"]).dt.total_seconds() / 3600
    ).round(2)
    df.loc[df["duration_hours"] < 0, "duration_hours"] = np.nan
    df["city"] = city
    df["source"] = "Groundsource"
    df["event_type"] = "Flood record"
    df["property_damage_usd"] = np.nan
    df["crop_damage_usd"] = np.nan
    df["deaths_direct"] = np.nan
    df["deaths_indirect"] = np.nan
    df["injuries_direct"] = np.nan
    df["injuries_indirect"] = np.nan
    return df[[
        "uuid", "area_km2", "event_date", "end_date", "city", "source",
        "event_type", "latitude", "longitude", "duration_hours",
        "property_damage_usd", "crop_damage_usd",
        "deaths_direct", "deaths_indirect",
        "injuries_direct", "injuries_indirect",
    ]]
def combine_noaa_date_time(date_series, time_series):
    time_text = (
        pd.to_numeric(time_series, errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(4)
    )
    return pd.to_datetime(
        date_series.astype(str)
        + " "
        + time_text.str[:2]
        + ":"
        + time_text.str[2:],
        errors="coerce",
    )
def load_noaa_city(city):
    file = find_noaa_file(city)
    print(f"Reading NOAA {city}: {file.name}")
    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    event_type_col = first_existing_column(df, ["event_type", "eventtype"])
    cz_type_col = first_existing_column(df, ["cz_type", "cztype"])
    if event_type_col is None:
        raise KeyError(f"{city}: no NOAA event type column found.")
    keep = df[event_type_col].astype(str).str.strip().str.casefold().eq("flash flood")
    if cz_type_col is not None:
        keep &= df[cz_type_col].astype(str).str.strip().str.upper().eq("C")
    df = df[keep].copy()
    begin_col = first_existing_column(df,["begin_date_time", "begin_datetime", "begin_date", "start_date"])
    end_col = first_existing_column(df,["end_date_time", "end_datetime", "end_date", "finish_date"])
    if begin_col is None:
        raise KeyError(f"{city}: no NOAA begin-date column found.")
    begin_time_col = first_existing_column(df,["begin_time", "begintime"])
    end_time_col = first_existing_column(df,["end_time", "endtime"])
    if begin_time_col is not None:
        df["event_date"] = combine_noaa_date_time(
            df[begin_col],
            df[begin_time_col],
        )
    else:
        df["event_date"] = pd.to_datetime(
            df[begin_col],
            errors="coerce",
        )
    if end_col is not None and end_time_col is not None:
        df["end_date"] = combine_noaa_date_time(
            df[end_col],
            df[end_time_col],
        )
    elif end_col is not None:
        df["end_date"] = pd.to_datetime(
            df[end_col],
            errors="coerce",
        )
    else:
        df["end_date"] = pd.NaT
    lat_col = first_existing_column(df, ["begin_lat", "begin_latitude", "latitude", "lat"])
    lon_col = first_existing_column(df, ["begin_lon", "begin_longitude", "longitude", "lon", "lng"])
    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan
    # NOAA rows without usable coordinates cannot be assigned to a county polygon.
    print(
    f"  NOAA {COUNTY_FOR_CITY[city]} County ({city}): "
    f"{len(df):,} filtered flash-flood rows"
)
    df["duration_hours"] = (
        (df["end_date"] - df["event_date"]).dt.total_seconds() / 3600
    ).round(2)
    df.loc[df["duration_hours"] < 0, "duration_hours"] = np.nan
    property_col = first_existing_column(df, ["damage_property", "property_damage", "damage_property_num"])
    crop_col = first_existing_column(df, ["damage_crops", "crop_damage", "damage_crops_num"])
    df["property_damage_usd"] = df[property_col].apply(parse_damage) if property_col else np.nan
    df["crop_damage_usd"] = df[crop_col].apply(parse_damage) if crop_col else np.nan
    # NOAA human-impact fields
    deaths_direct_col = first_existing_column(
        df,
        ["deaths_direct", "direct_deaths"]
    )
    deaths_indirect_col = first_existing_column(
        df,
        ["deaths_indirect", "indirect_deaths"]
    )
    injuries_direct_col = first_existing_column(
        df,
        ["injuries_direct", "direct_injuries"]
    )
    injuries_indirect_col = first_existing_column(
        df,
        ["injuries_indirect", "indirect_injuries"]
    )
    df["deaths_direct"] = (
        pd.to_numeric(df[deaths_direct_col], errors="coerce")
        if deaths_direct_col else np.nan
    )
    df["deaths_indirect"] = (
        pd.to_numeric(df[deaths_indirect_col], errors="coerce")
        if deaths_indirect_col else np.nan
    )
    df["injuries_direct"] = (
        pd.to_numeric(df[injuries_direct_col], errors="coerce")
        if injuries_direct_col else np.nan
    )
    df["injuries_indirect"] = (
        pd.to_numeric(df[injuries_indirect_col], errors="coerce")
        if injuries_indirect_col else np.nan
    )
    df["city"] = city
    df["source"] = "NOAA Storm Events"
    df["event_type"] = "Flash Flood"
    df["uuid"] = np.nan
    df["area_km2"] = np.nan
    return df[[
        "uuid", "area_km2", "event_date", "end_date", "city", "source",
        "event_type", "latitude", "longitude", "duration_hours",
        "property_damage_usd", "crop_damage_usd",
        "deaths_direct", "deaths_indirect",
        "injuries_direct", "injuries_indirect",
    ]]
def build_raw_master():
    gs = load_groundsource()
    pieces = []
    for city in CITIES:
        pieces.append(groundsource_city_records(gs, city))
        pieces.append(load_noaa_city(city))
    master = pd.concat(pieces, ignore_index=True)
    master = master[master["event_date"].notna()].copy()
    master["year"] = master["event_date"].dt.year.astype("Int64")
    master["month"] = master["event_date"].dt.month.astype("Int64")
    master["year_month"] = master["event_date"].dt.to_period("M").astype(str)
    master["total_damage_usd"] = master[["property_damage_usd", "crop_damage_usd"]].sum(axis=1, min_count=1)
    master = master.sort_values(["city", "source", "event_date"]).reset_index(drop=True)
    master.insert(0, "event_id", [f"RAW{i:06d}" for i in range(1, len(master) + 1)])
    return master
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
def cluster_groundsource_city(city_df):
    df = city_df.sort_values("event_date").reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return pd.DataFrame()
    uf = UnionFind(n)
    for i in range(n):
        j = i + 1
        while j < n:
            hours = (df.loc[j, "event_date"] - df.loc[i, "event_date"]).total_seconds() / 3600
            if hours > CLUSTER_TIME_HOURS:
                break
            d = haversine_km(df.loc[i, "latitude"], df.loc[i, "longitude"], df.loc[j, "latitude"], df.loc[j, "longitude"])
            if not pd.isna(d) and d <= CLUSTER_DISTANCE_KM:
                uf.union(i, j)
            j += 1
    roots = [uf.find(i) for i in range(n)]
    mapping = {}
    next_id = 1
    cluster_ids = []
    for root in roots:
        if root not in mapping:
            mapping[root] = next_id
            next_id += 1
        cluster_ids.append(mapping[root])
    df["cluster_number"] = cluster_ids
    rows = []
    for cluster_number, g in df.groupby("cluster_number"):
        start = g["event_date"].min()
        end = g["end_date"].max()
        if pd.isna(end):
            end = g["event_date"].max()
        duration = (end - start).total_seconds() / 3600 if pd.notna(start) and pd.notna(end) else np.nan
        rows.append({
            "episode_id": f"GS_{g['city'].iloc[0].upper()}_{int(cluster_number):05d}",
            "city": g["city"].iloc[0],
            "source": "Groundsource",
            "event_type": "Candidate flood episode",
            "event_date": start,
            "end_date": end,
            "latitude": g["latitude"].mean(),
            "longitude": g["longitude"].mean(),
            "duration_hours": duration,
            "episode_member_count": len(g),
            "groundsource_area_sum_km2": g["area_km2"].sum(min_count=1),
            "groundsource_area_mean_km2": g["area_km2"].mean(),
            "groundsource_area_max_km2": g["area_km2"].max(),
            "property_damage_usd": np.nan,
            "crop_damage_usd": np.nan,
            "total_damage_usd": np.nan,
            "deaths_direct": np.nan,
            "deaths_indirect": np.nan,
            "injuries_direct": np.nan,
            "injuries_indirect": np.nan,
        })
    return pd.DataFrame(rows)
def build_groundsource_episodes(raw_master):
    pieces = []
    for city in CITIES:
        df = raw_master[(raw_master["city"] == city) & (raw_master["source"] == "Groundsource")].copy()
        print(f"Clustering Groundsource {city}: {len(df):,} raw records")
        out = cluster_groundsource_city(df)
        if not out.empty:
            pieces.append(out)
    episodes = pd.concat(pieces, ignore_index=True)
    episodes["year"] = episodes["event_date"].dt.year.astype("Int64")
    episodes["month"] = episodes["event_date"].dt.month.astype("Int64")
    episodes["year_month"] = episodes["event_date"].dt.to_period("M").astype(str)
    return episodes
def build_noaa_episodes(raw_master):
    noaa = raw_master[raw_master["source"] == "NOAA Storm Events"].copy().reset_index(drop=True)
    noaa["episode_id"] = [f"NOAA{i:06d}" for i in range(1, len(noaa) + 1)]
    noaa["episode_member_count"] = 1
    noaa["groundsource_area_sum_km2"] = np.nan
    noaa["groundsource_area_mean_km2"] = np.nan
    noaa["groundsource_area_max_km2"] = np.nan
    cols = [
        "episode_id", "city", "source", "event_type", "event_date", "end_date",
        "latitude", "longitude", "duration_hours", "episode_member_count",
        "groundsource_area_sum_km2", "groundsource_area_mean_km2",
        "groundsource_area_max_km2", "property_damage_usd", "crop_damage_usd",
        "total_damage_usd", "deaths_direct", "deaths_indirect",
        "injuries_direct", "injuries_indirect", "year", "month", "year_month",
    ]
    return noaa[cols]
def build_harmonized_master(gs_episodes, noaa_episodes):
    cols = list(noaa_episodes.columns)
    return pd.concat([gs_episodes[cols], noaa_episodes[cols]], ignore_index=True).sort_values(["city", "source", "event_date"]).reset_index(drop=True)
def overlap_years(harmonized):
    rows = []
    for city in CITIES:
        c = harmonized[harmonized["city"] == city]
        g = c[c["source"] == "Groundsource"]["year"].dropna()
        n = c[c["source"] == "NOAA Storm Events"]["year"].dropna()
        if len(g) and len(n):
            rows.append({"city": city, "overlap_start_year": max(int(g.min()), int(n.min())), "overlap_end_year": min(int(g.max()), int(n.max()))})
    return pd.DataFrame(rows)
def restrict_to_overlap(harmonized, overlaps):
    pieces = []
    for _, row in overlaps.iterrows():
        pieces.append(harmonized[(harmonized["city"] == row["city"]) & (harmonized["year"] >= row["overlap_start_year"]) & (harmonized["year"] <= row["overlap_end_year"])].copy())
    return pd.concat(pieces, ignore_index=True) if pieces else harmonized.iloc[0:0].copy()
def yearly_counts(df, name="event_count"):
    return df.groupby(["city", "source", "year"]).size().reset_index(name=name).sort_values(["source", "city", "year"])
def monthly_counts(df, name="event_count"):
    monthly = (
        df.groupby(
            ["city", "source", "year", "month", "year_month"]
        )
        .size()
        .reset_index(name=name)
    )
    # Create a real date for RAWGraphs
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str)
        + "-"
        + monthly["month"].astype(str)
        + "-01"
    )
    return monthly.sort_values(
        ["source", "city", "date"]
    )
def duration_summary(df):
    d = df[df["duration_hours"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    return d.groupby(["city", "source"])["duration_hours"].agg(event_count="count", mean_duration_hours="mean", std_duration_hours="std", median_duration_hours="median", min_duration_hours="min", max_duration_hours="max").reset_index()
def save_raw_outputs(raw_master):
    raw_master.to_csv(OUTPUT_DIR / "01_master_events.csv", index=False)
    yearly = yearly_counts(raw_master)
    yearly.to_csv(OUTPUT_DIR / "02_yearly_counts.csv", index=False)
    monthly_counts(raw_master).to_csv(OUTPUT_DIR / "03_monthly_counts.csv", index=False)
    raw_master[raw_master["latitude"].notna() & raw_master["longitude"].notna()].to_csv(OUTPUT_DIR / "04_spatial_events.csv", index=False)
    raw_master[raw_master["duration_hours"].notna()].to_csv(OUTPUT_DIR / "05_duration_events.csv", index=False)
    duration_summary(raw_master).to_csv(OUTPUT_DIR / "06_duration_summary.csv", index=False)
    raw_master[raw_master["year"] == FOCUS_YEAR].groupby(["city", "source"]).size().reset_index(name="event_count").to_csv(OUTPUT_DIR / f"07_focus_year_{FOCUS_YEAR}.csv", index=False)
    comp = yearly.pivot_table(index=["city", "year"], columns="source", values="event_count", aggfunc="sum").reset_index()
    comp.columns.name = None
    if "Groundsource" not in comp.columns: comp["Groundsource"] = np.nan
    if "NOAA Storm Events" not in comp.columns: comp["NOAA Storm Events"] = np.nan
    comp["count_difference_groundsource_minus_noaa"] = comp["Groundsource"] - comp["NOAA Storm Events"]
    comp.to_csv(OUTPUT_DIR / "08_source_comparison.csv", index=False)
    damage = raw_master[raw_master["total_damage_usd"].notna()].copy()
    if not damage.empty:
        damage.groupby(["city", "source", "year"])["total_damage_usd"].agg(events_with_damage="count", total_damage_usd="sum", mean_damage_usd="mean", median_damage_usd="median").reset_index().to_csv(OUTPUT_DIR / "09_damage_by_year.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "09_damage_by_year.csv", index=False)
def build_event_match_outputs(overlap_master):
    """
    Match NOAA events to Groundsource candidate episodes.
    A strict match requires:
    - same county/city label
    - event start times within 24 hours
    - locations within 10 km
    Matching is one-to-one: one Groundsource episode cannot
    match multiple NOAA events.
    """
    summary_rows = []
    rate_rows = []
    for city in CITIES:
        city_df = overlap_master[
            overlap_master["city"] == city
        ].copy()
        gs = city_df[
            city_df["source"] == "Groundsource"
        ].reset_index(drop=True)
        noaa = city_df[
            city_df["source"] == "NOAA Storm Events"
        ].reset_index(drop=True)
        # Only NOAA rows with coordinates can be used for
        # strict spatial + temporal matching.
        noaa_with_coords = noaa[
            noaa["latitude"].notna()
            & noaa["longitude"].notna()
        ].reset_index(drop=True)
        candidates = []
        for ni, nrow in noaa_with_coords.iterrows():
            for gi, grow in gs.iterrows():
                hours = abs(
                    (
                        nrow["event_date"]
                        - grow["event_date"]
                    ).total_seconds()
                    / 3600
                )
                if hours > CLUSTER_TIME_HOURS:
                    continue
                distance = haversine_km(
                    nrow["latitude"],
                    nrow["longitude"],
                    grow["latitude"],
                    grow["longitude"]
                )
                if pd.isna(distance):
                    continue
                if distance <= CLUSTER_DISTANCE_KM:
                    candidates.append(
                        (
                            hours,
                            distance,
                            ni,
                            gi
                        )
                    )
        # Match closest events first.
        candidates.sort(
            key=lambda x: (x[0], x[1])
        )
        matched_noaa = set()
        matched_gs = set()
        for hours, distance, ni, gi in candidates:
            if ni in matched_noaa:
                continue
            if gi in matched_gs:
                continue
            matched_noaa.add(ni)
            matched_gs.add(gi)
        matched_count = len(matched_noaa)
        noaa_only = (
            len(noaa_with_coords)
            - matched_count
        )
        gs_only = (
            len(gs)
            - len(matched_gs)
        )
        summary_rows.extend([
            {
                "city": city,
                "match_status": "Matched NOAA + Groundsource",
                "event_count": matched_count
            },
            {
                "city": city,
                "match_status": "Groundsource only",
                "event_count": gs_only
            },
            {
                "city": city,
                "match_status": "NOAA only",
                "event_count": noaa_only
            }
        ])
        match_rate = (
            matched_count
            / len(noaa_with_coords)
            * 100
            if len(noaa_with_coords) > 0
            else np.nan
        )
        rate_rows.append({
            "city": city,
            "total_noaa_events": len(noaa),
            "noaa_events_with_coordinates":
                len(noaa_with_coords),
            "matched_noaa_events": matched_count,
            "noaa_match_rate_percent": match_rate
        })
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(rate_rows)
    )
def save_harmonized_outputs(gs_episodes, noaa_episodes, harmonized, overlaps, overlap_master):
    gs_episodes.to_csv(OUTPUT_DIR / "10_groundsource_clustered_episodes.csv", index=False)
    noaa_episodes.to_csv(OUTPUT_DIR / "11_noaa_flash_flood_events.csv", index=False)
    harmonized.to_csv(OUTPUT_DIR / "12_harmonized_master_events.csv", index=False)
    overlaps.to_csv(OUTPUT_DIR / "13_city_overlap_years.csv", index=False)
    overlap_master.to_csv(OUTPUT_DIR / "14_harmonized_overlap_events.csv", index=False)
    match_summary, match_rates = (
        build_event_match_outputs(overlap_master)
    )
    match_summary.to_csv(
        OUTPUT_DIR / "22_event_match_summary.csv",
        index=False
    )
    match_rates.to_csv(
        OUTPUT_DIR / "23_noaa_match_rate.csv",
        index=False
    )
    yearly = yearly_counts(overlap_master, "episode_count")
    yearly.to_csv(OUTPUT_DIR / "15_harmonized_yearly_counts.csv", index=False)
    groundsource_only = yearly[
        yearly["source"] == "Groundsource"
    ]
    groundsource_only.to_csv(
        OUTPUT_DIR / "15a_groundsource_yearly_counts.csv",
        index=False
    )
    noaa_only = yearly[
        yearly["source"] == "NOAA Storm Events"
    ]
    noaa_only.to_csv(
        OUTPUT_DIR / "15b_noaa_yearly_counts.csv",
        index=False
    )
    monthly_counts(overlap_master, "episode_count").to_csv(OUTPUT_DIR / "16_harmonized_monthly_counts.csv", index=False)
    overlap_master[overlap_master["latitude"].notna() & overlap_master["longitude"].notna()].to_csv(OUTPUT_DIR / "17_harmonized_spatial_events.csv", index=False)
    duration_events = overlap_master[
        overlap_master["duration_hours"].notna()
    ].copy()
    duration_events["city_source"] = (
        duration_events["city"]
        + " — "
        + duration_events["source"]
    )
    duration_events.to_csv(
        OUTPUT_DIR / "18_harmonized_duration_events.csv",
        index=False
    )
    duration_summary(overlap_master).to_csv(OUTPUT_DIR / "19_harmonized_duration_summary.csv", index=False)
    overlap_master[overlap_master["year"] == FOCUS_YEAR].groupby(["city", "source"]).size().reset_index(name="episode_count").to_csv(OUTPUT_DIR / f"20_harmonized_focus_year_{FOCUS_YEAR}.csv", index=False)
    comp = yearly.pivot_table(index=["city", "year"], columns="source", values="episode_count", aggfunc="sum").reset_index()
    comp.columns.name = None
    if "Groundsource" not in comp.columns: comp["Groundsource"] = np.nan
    if "NOAA Storm Events" not in comp.columns: comp["NOAA Storm Events"] = np.nan
    comp["difference_groundsource_minus_noaa"] = comp["Groundsource"] - comp["NOAA Storm Events"]
    comp["Groundsource"] = pd.to_numeric(
        comp["Groundsource"],
        errors="coerce"
    ).fillna(0)
    comp["NOAA Storm Events"] = pd.to_numeric(
        comp["NOAA Storm Events"],
        errors="coerce"
    ).fillna(0)
    # Signed difference
    comp["difference_groundsource_minus_noaa"] = (
        comp["Groundsource"] - comp["NOAA Storm Events"]
    )
    # Always-positive numeric difference for bubble size
    comp["absolute_difference"] = (
        comp["difference_groundsource_minus_noaa"].abs()
    )
    comp.to_csv(OUTPUT_DIR / "21_harmonized_source_comparison.csv", index=False)
def main():
    print("\nBUILDING RAW OUTPUTS...")
    raw_master = build_raw_master()
    save_raw_outputs(raw_master)
    print("\nBUILDING HARMONIZED OUTPUTS...")
    gs_episodes = build_groundsource_episodes(raw_master)
    noaa_episodes = build_noaa_episodes(raw_master)
    harmonized = build_harmonized_master(gs_episodes, noaa_episodes)
    overlaps = overlap_years(harmonized)
    overlap_master = restrict_to_overlap(harmonized, overlaps)
    save_harmonized_outputs(gs_episodes, noaa_episodes, harmonized, overlaps, overlap_master)
    print("\nDONE")
    print("Output folder:")
    print(OUTPUT_DIR)
    print("\nGeography: official Census county polygons applied to BOTH sources")
    print("City labels use these county boundaries:")
    for city, county in COUNTY_FOR_CITY.items():
        print(f"  {city} -> {county} County")
    print(f"Groundsource cluster rule: {CLUSTER_TIME_HOURS} hours + {CLUSTER_DISTANCE_KM} km")
    print("\nBest new files for RAWGraphs:")
    print("15_harmonized_yearly_counts.csv -> annual line/matrix/box plots")
    print("16_harmonized_monthly_counts.csv -> monthly line chart")
    print("17_harmonized_spatial_events.csv -> lat/lon bubble plot or map")
    print("18_harmonized_duration_events.csv -> box/violin duration plot")
    print("19_harmonized_duration_summary.csv -> duration SD bar chart")
    print(f"20_harmonized_focus_year_{FOCUS_YEAR}.csv -> one-year grouped bar chart")
    print("21_harmonized_source_comparison.csv -> NOAA vs Groundsource scatter/difference")
if __name__ == "__main__":
    main()
