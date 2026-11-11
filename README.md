# Urban Flash-Flood Impacts

## F.L.O.O.D.
### Florida Flood Location, Observation, Outcomes, and Dynamics

This project examines urban flash-flood observations, reported impacts, spatial correspondence, and event profiles across five Florida study areas: Miami, Tampa, Orlando, Gainesville, and Jacksonville.

## Overview

The project compares NOAA Storm Events records with Groundsource flood observations to study:

- flood-event frequency and timing;
- flood duration;
- spatial correspondence between independent flood datasets;
- NOAA-reported financial impacts;
- damaging versus non-damaging events;
- geographic differences in reported impacts; and
- recurring NOAA flash-flood event profiles identified using K-means clustering.

NOAA is treated as the primary source for official flash-flood characteristics and reported impacts. Groundsource serves as an independent observational source used to identify and corroborate likely corresponding flood episodes.

Matched NOAA-Groundsource pairs represent likely corresponding events based on spatial and temporal proximity. They should not be interpreted as confirmed event identities.

---

## Research Questions

1. How do NOAA and Groundsource differ in recorded flood frequency and timing?
2. What proportion of NOAA flash-flood events can be corroborated by Groundsource?
3. How does match coverage change between High-confidence and High + Medium criteria?
4. How is flood duration related to NOAA-reported financial damage?
5. Are longer floods more likely to have reported financial damage?
6. Do years with more matched floods also have greater total or maximum event damage?
7. How do financial impacts differ across the five study areas?
8. Do the main impact relationships remain similar when Medium-confidence matches are included?
9. Can NOAA flash-flood events be grouped into distinct profiles based on seasonal timing, duration, and NOAA-reported financial damage?
10. Do similar-duration flood profiles differ substantially in their reported impacts?

---

## Processing Workflow

The final analysis uses four primary Python scripts.

### 1. Build and Harmonize Event Records

`/scripts/01_build_harmonized_events.py`

This script:

- loads NOAA and Groundsource data;
- filters NOAA to Flash Flood events;
- filters Groundsource using official county polygons;
- calculates NOAA flood duration;
- preserves NOAA impact fields;
- groups nearby Groundsource observations into candidate flood episodes;
- restricts records to overlapping study years; and
- creates harmonized analysis files.

Main matching input:

`14_harmonized_overlap_events.csv`

### 2. Match NOAA and Groundsource Events

`/scripts/02_match_noaa_groundsource.py`

A candidate match must:

- occur in the same study area;
- be within 10 km; and
- have overlapping intervals or a temporal gap of no more than 24 hours.

Global one-to-one optimization is used so that one record cannot be assigned to multiple counterparts.

The match score is a heuristic ranking measure and is not a probability.

### 3. Create Additional Relationship Tables

`/scripts/additional_relationships.py`

This script creates graph-ready datasets for:

- probability of reported damage by duration category;
- Groundsource flooded area vs. NOAA-reported damage;
- duration vs. damage among damaging events only;
- match distance vs. temporal gap;
- county-level damage per matched event; and
- annual flood frequency vs. event severity.

### 4. Create NOAA Flash-Flood Event Profiles

`/scripts/kmeans_event_profiles.py`

This script performs exploratory K-means clustering using NOAA flash-flood events from the common 2005–2025 analysis period.

The clustering variables represent three conceptual event characteristics:

- seasonal timing;
- flood duration; and
- NOAA-reported financial damage.

Flood duration and financial damage are transformed using `log(1 + x)` to reduce the influence of extreme values.

Seasonal timing is represented using sine and cosine transformations of day of year so that dates near the end and beginning of the calendar year remain close in feature space.

The four numerical clustering features are therefore:

- `log(1 + duration_hours)`;
- `log(1 + noaa_total_damage_usd)`;
- sine of day of year; and
- cosine of day of year.

All clustering features are standardized before K-means is fitted.

Candidate solutions from `k = 2` through `k = 10` are evaluated using:

- within-cluster sum of squares (inertia); and
- silhouette score.

The five-cluster solution is retained as the primary exploratory interpretation based on a balance of cluster separation, cluster size, and substantive interpretability. The four-cluster solution is retained as a sensitivity comparison.

Cluster descriptions are assigned only after K-means assigns the events. These descriptions summarize the original NOAA event characteristics and are not predefined categories used to determine cluster membership.

---

## K-Means Event Profiles

The primary five-cluster solution identifies distinct NOAA flash-flood profiles based on combinations of seasonal timing, flood duration, and reported financial impact.

| Cluster | Post-hoc description | Median duration | Events with reported damage |
|---|---|---:|---:|
| Cluster 1 | May–Jun, ~3-hour, non-damaging | 2.95 hr | 0% |
| Cluster 2 | Sep–Nov, ~4-hour, mostly non-damaging | 4.25 hr | 5.7% |
| Cluster 3 | Aug–Oct, very-short-duration, mixed-impact | ~5 min | 33.3% |
| Cluster 4 | May–Jun, ~2.5-hour, high-impact | 2.53 hr | 100% |
| Cluster 5 | Oct–Dec, ~4.5-hour, high-impact | 4.50 hr | 100% |

The clustering suggests that NOAA-reported flood impacts are associated with multiple combinations of timing and duration rather than one simple duration-to-damage relationship.

In particular:

- high-impact profiles occur at both shorter and longer durations;
- similar-duration clusters can have very different reported-damage frequencies;
- longer duration alone does not consistently correspond to greater reported financial loss; and
- seasonal timing alone does not determine impact severity.

These clusters are exploratory descriptive profiles and should not be interpreted as causal categories.

---

## Interactive Dashboard

The project includes an interactive Streamlit dashboard:

`dashboard/app.py`

Run it with:

```bash
streamlit run dashboard/app.py
```

The dashboard contains two main sections.

### Source Overview

Users can filter NOAA and Groundsource records by study area and year.

The dashboard displays:

- source-level event and episode counts;
- NOAA-reported financial damage;
- spatial flood-observation maps;
- separate NOAA and Groundsource flood-activity time-series panels; and
- recorded observations by study area.

NOAA and Groundsource annual activity are displayed separately because the two datasets contain substantially different numbers of observations.

### K-Means Clusters

Users can filter NOAA events by:

- cluster;
- study area; and
- year.

Interactive features include:

- elbow and silhouette model-selection graphs;
- event counts by cluster;
- log-transformed flood duration vs. log-transformed reported damage;
- duration and damage distributions;
- monthly cluster patterns;
- annual cluster composition using stacked bars;
- damage-frequency and damage-severity summaries;
- an interactive three-dimensional cluster visualization;
- spatial cluster maps; and
- event-level tables.

Cluster colors and Cluster 1–5 labels are standardized across the clustering figures and dashboard.

---

## Clustering Figures

### `figures/clustering/poster/`

Poster-focused K-means figures are condensed into multi-panel layouts to reduce visual repetition and conserve space.

Primary outputs include:

- **Model Selection** — elbow and silhouette analysis for candidate cluster solutions from `k = 2` through `k = 10`;
- **Cluster Description Guide** — lookup table describing Clusters 1–5;
- **Cluster Characteristics** — event counts, duration vs. damage, duration distribution, and damage distribution;
- **Temporal Patterns** — monthly cluster activity and annual stacked cluster composition;
- **Cluster Distributions** — log-transformed duration and damage distributions; and
- **3D Cluster Structure** — visualization of duration, damage, and seasonal timing.

Standalone percentage-damaging and median-damage figures are retained for supporting analysis but can be summarized as text on the research poster to reduce graphical redundancy.

---

## Current Clustering Findings

The clustering analysis adds several findings to the earlier matched-event analysis:

- Cluster 1 contains no events with positive NOAA-reported financial damage.
- Approximately 5.7% of Cluster 2 events report financial damage.
- Approximately 33.3% of Cluster 3 events report financial damage.
- Clusters 4 and 5 consist entirely of events with reported financial damage.
- Cluster 4 has a median duration of approximately 2.53 hours despite being a high-impact profile.
- Cluster 5 has a longer median duration of approximately 4.50 hours and is also a high-impact profile.
- Clusters 2 and 5 have relatively similar typical durations but substantially different impact patterns.
- These patterns reinforce the earlier finding that flood duration alone does not consistently explain reported financial impact.

The results describe associations and event profiles rather than causal relationships.

---

## Repository Structure

```text
Urban-Flash-Flood-Impacts/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 01_build_harmonized_events.py
│   ├── 02_match_noaa_groundsource.py
│   ├── additional_relationships.py
│   ├── kmeans_event_profiles.py
│   └── older_scripts/
├── data/
│   ├── README.md
│   └── processed/
├── figures/
│   ├── temporal/
│   ├── matching/
│   ├── impacts/
│   ├── sensitivity/
│   └── clustering/
│       ├── model_selection/
│       ├── k5/
│       └── poster/
└── dashboard/
    └── app.py
```

---

## Reproducing the Analysis

1. Obtain the NOAA Storm Events data used for the five Florida study areas.
2. Obtain the Groundsource flood-observation dataset.
3. Add the Florida county-boundary data used for spatial filtering.
4. Install the required Python packages.
5. Run the analysis scripts in order:

```bash
python scripts/01_build_harmonized_events.py
python scripts/02_match_noaa_groundsource.py
python scripts/additional_relationships.py
python scripts/kmeans_event_profiles.py
```

6. Launch the interactive dashboard with:

```bash
streamlit run dashboard/app.py
```

### Core Python Requirements

```text
pandas
numpy
scipy
scikit-learn
matplotlib
plotly
streamlit
folium
streamlit-folium
geopandas
shapely
pyarrow
```

---

## Generative AI Assistance

Generative AI tools were used during project development as assistants for Python coding, data-cleaning workflows, debugging, code organization, visualization development, and documentation.

AI-generated suggestions were not treated as analytical results. All code included in the final workflow was reviewed, executed, tested, and verified by the project authors against the underlying NOAA and Groundsource data.

Data-cleaning decisions, event-matching criteria, clustering variables, figures, statistical interpretations, and final conclusions were reviewed and confirmed by the project authors.

Generative AI was not used as a source of flood-event observations, financial-impact records, or other empirical research data.

---

## Limitations

- NOAA and Groundsource use different event definitions and reporting systems.
- Some NOAA events lack usable coordinates and therefore cannot participate in strict spatial matching.
- Multiple Groundsource records may represent one broader flood episode.
- Event matching is based on spatial and temporal heuristics rather than confirmed event identity.
- High-confidence sample sizes are small in some study areas.
- Gainesville currently has no High-confidence matches and therefore does not appear in some High-confidence impact figures.
- NOAA-reported damage is strongly influenced by a small number of high-loss events.
- Missing damage values are not equivalent to zero reported damage.
- Groundsource flooded-area geometry may represent broad observational footprints rather than exact physical inundation boundaries.
- K-means clustering is exploratory and depends on the selected features, transformations, scaling, and number of clusters.
- Cluster names are post-hoc descriptions and are not predefined classifications.
- Duration, frequency, damage, geographic differences, and clustering patterns are descriptive associations and should not be interpreted as causal.
