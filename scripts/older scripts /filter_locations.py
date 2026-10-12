"""
FLASH-FLOOD RAW + HARMONIZED CSV BUILDER
NOAA Storm Events + Groundsource

Keeps your existing raw outputs and adds harmonized outputs.
Groundsource records are grouped into candidate flood episodes when they are:
- in the same city
- within 24 hours
- within 10 km

Put this script in the same folder as the Groundsource parquet and NOAA city CSVs.
Run: python graph_creation_harmonized.py
"""

from pathlib import Path
import math
import re
import numpy as np
import pandas as pd
from shapely import wkb

FOCUS_YEAR = 2024
CLUSTER_TIME_HOURS = 24
CLUSTER_DISTANCE_KM = 10.0

CITIES = {
    "Miami": {"min_lon": -80.9, "max_lon": -80.1, "min_lat": 25.1, "max_lat": 25.9},
    "Tampa": {"min_lon": -82.7, "max_lon": -82.3, "min_lat": 27.7, "max_lat": 28.1},
    "Orlando": {"min_lon": -81.6, "max_lon": -81.2, "min_lat": 28.3, "max_lat": 28.7},
    "Gainesville": {"min_lon": -82.45, "max_lon": -82.25, "min_lat": 29.55, "max_lat": 29.75},
    "Jacksonville": {"min_lon": -81.90, "max_lon": -81.35, "min_lat": 30.10, "max_lat": 30.60},
}

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "flash_flood_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


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
    b = CITIES[city]
    df = all_groundsource[
        (all_groundsource["longitude"] >= b["min_lon"]) &
        (all_groundsource["longitude"] <= b["max_lon"]) &
        (all_groundsource["latitude"] >= b["min_lat"]) &
        (all_groundsource["latitude"] <= b["max_lat"])
    ].copy()
    df["event_date"] = df["start_date"]
    df["duration_hours"] = (df["end_date"] - df["start_date"]).dt.total_seconds() / 3600
    df.loc[df["duration_hours"] < 0, "duration_hours"] = np.nan
    df["city"] = city
    df["source"] = "Groundsource"
    df["event_type"] = "Flood record"
    df["property_damage_usd"] = np.nan
    df["crop_damage_usd"] = np.nan
    return df[["uuid", "area_km2", "event_date", "end_date", "city", "source", "event_type", "latitude", "longitude", "duration_hours", "property_damage_usd", "crop_damage_usd"]]


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

    begin_col = first_existing_column(df, ["begin_date_time", "begin_datetime", "begin_date", "start_date"])
    end_col = first_existing_column(df, ["end_date_time", "end_datetime", "end_date", "finish_date"])
    if begin_col is None:
        raise KeyError(f"{city}: no NOAA begin-date column found.")

    df["event_date"] = pd.to_datetime(df[begin_col], errors="coerce")
    df["end_date"] = pd.to_datetime(df[end_col], errors="coerce") if end_col else pd.NaT

    lat_col = first_existing_column(df, ["begin_lat", "begin_latitude", "latitude", "lat"])
    lon_col = first_existing_column(df, ["begin_lon", "begin_longitude", "longitude", "lon", "lng"])
    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan

    df["duration_hours"] = (df["end_date"] - df["event_date"]).dt.total_seconds() / 3600
    df.loc[df["duration_hours"] < 0, "duration_hours"] = np.nan

    property_col = first_existing_column(df, ["damage_property", "property_damage", "damage_property_num"])
    crop_col = first_existing_column(df, ["damage_crops", "crop_damage", "damage_crops_num"])
    df["property_damage_usd"] = df[property_col].apply(parse_damage) if property_col else np.nan
    df["crop_damage_usd"] = df[crop_col].apply(parse_damage) if crop_col else np.nan
    df["city"] = city
    df["source"] = "NOAA Storm Events"
    df["event_type"] = "Flash Flood"
    df["uuid"] = np.nan
    df["area_km2"] = np.nan
    return df[["uuid", "area_km2", "event_date", "end_date", "city", "source", "event_type", "latitude", "longitude", "duration_hours", "property_damage_usd", "crop_damage_usd"]]


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
    cols = ["episode_id", "city", "source", "event_type", "event_date", "end_date", "latitude", "longitude", "duration_hours", "episode_member_count", "groundsource_area_sum_km2", "groundsource_area_mean_km2", "groundsource_area_max_km2", "property_damage_usd", "crop_damage_usd", "total_damage_usd", "year", "month", "year_month"]
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
    return df.groupby(["city", "source", "year", "month", "year_month"]).size().reset_index(name=name).sort_values(["source", "city", "year", "month"])


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


def save_harmonized_outputs(gs_episodes, noaa_episodes, harmonized, overlaps, overlap_master):
    gs_episodes.to_csv(OUTPUT_DIR / "10_groundsource_clustered_episodes.csv", index=False)
    noaa_episodes.to_csv(OUTPUT_DIR / "11_noaa_flash_flood_events.csv", index=False)
    harmonized.to_csv(OUTPUT_DIR / "12_harmonized_master_events.csv", index=False)
    overlaps.to_csv(OUTPUT_DIR / "13_city_overlap_years.csv", index=False)
    overlap_master.to_csv(OUTPUT_DIR / "14_harmonized_overlap_events.csv", index=False)

    yearly = yearly_counts(overlap_master, "episode_count")
    yearly.to_csv(OUTPUT_DIR / "15_harmonized_yearly_counts.csv", index=False)
# ============================================================
# CREATE ONE HARMONIZED YEARLY CSV FOR EACH CITY
# ============================================================

    for city in CITIES:

        city_yearly = yearly[
            yearly["city"] == city
        ].copy()

        city_yearly.to_csv(
            OUTPUT_DIR / f"harmonized_yearly_{city.lower()}.csv",
            index=False
        )
    monthly_counts(overlap_master, "episode_count").to_csv(OUTPUT_DIR / "16_harmonized_monthly_counts.csv", index=False)
    overlap_master[overlap_master["latitude"].notna() & overlap_master["longitude"].notna()].to_csv(OUTPUT_DIR / "17_harmonized_spatial_events.csv", index=False)
    overlap_master[overlap_master["duration_hours"].notna()].to_csv(OUTPUT_DIR / "18_harmonized_duration_events.csv", index=False)
    duration_summary(overlap_master).to_csv(OUTPUT_DIR / "19_harmonized_duration_summary.csv", index=False)
    overlap_master[overlap_master["year"] == FOCUS_YEAR].groupby(["city", "source"]).size().reset_index(name="episode_count").to_csv(OUTPUT_DIR / f"20_harmonized_focus_year_{FOCUS_YEAR}.csv", index=False)

    comp = yearly.pivot_table(index=["city", "year"], columns="source", values="episode_count", aggfunc="sum").reset_index()
    comp.columns.name = None
    if "Groundsource" not in comp.columns: comp["Groundsource"] = np.nan
    if "NOAA Storm Events" not in comp.columns: comp["NOAA Storm Events"] = np.nan
    comp["difference_groundsource_minus_noaa"] = comp["Groundsource"] - comp["NOAA Storm Events"]
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
    print(f"\nGroundsource cluster rule: {CLUSTER_TIME_HOURS} hours + {CLUSTER_DISTANCE_KM} km")
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
