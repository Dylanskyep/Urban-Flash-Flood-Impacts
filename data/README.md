## NOAA Source-Only, Mapping, and K-Means Files

### `44_noaa_events_source_only.csv`

Contains NOAA flash-flood events used for NOAA-only descriptive analyses and K-means event profiling.

This file is the primary input to the clustering analysis.

Groundsource records are not included in K-means because Groundsource does not contain directly comparable NOAA financial-impact attributes. Groundsource damage is therefore not assumed or imputed to be zero.

### `53_mapping_events.csv`

Graph-ready dataset used for spatial visualization in the Streamlit dashboard.

Contains mapped NOAA and Groundsource observations with source, date, study-area, coordinate, and source-specific attributes.

This file supports:

- NOAA and Groundsource spatial maps;
- source filtering;
- study-area filtering; and
- dashboard visualizations.

### `65_kmeans_model_selection.csv`

Contains model-selection statistics for candidate K-means solutions.

Includes:

- number of clusters (`k`);
- inertia; and
- silhouette score.

Candidate solutions from `k = 2` through `k = 10` are evaluated.

Inertia measures within-cluster compactness and naturally decreases as more clusters are added.

Silhouette score evaluates how well separated an already fitted clustering solution is. It does not assign events to clusters.

### `66_noaa_kmeans_event_clusters_k5.csv`

Event-level NOAA output for the primary five-cluster K-means solution.

Contains each original NOAA flash-flood record with its assigned:

- cluster number;
- Cluster 1–5 label;
- transformed duration;
- transformed damage;
- seasonal timing variables; and
- post-hoc descriptive profile.

The original NOAA duration, damage, date, location, and impact values remain unchanged. K-means adds a cluster assignment based on similarity in the transformed feature space.

### `67_kmeans_cluster_summary_k5.csv`

Summarizes the original NOAA event characteristics within each five-cluster group.

Includes:

- event count;
- duration minimum, quartiles, median, mean, and maximum;
- financial-damage statistics;
- damaging-event count;
- percentage of events with reported damage;
- damage statistics among damaging events only;
- dominant month;
- dominant season; and
- dominant study area.

These summaries are calculated after cluster membership has been assigned and are used to interpret the event profiles.

### `68_kmeans_cluster_month_distribution_k5.csv`

Contains monthly NOAA event counts for each K-means cluster.

Used for:

- monthly cluster-pattern figures;
- seasonal interpretation; and
- Streamlit dashboard visualizations.

### `69_kmeans_cluster_season_distribution_k5.csv`

Contains broad season-level event counts for each K-means cluster.

Season labels are post-hoc descriptive summaries. Broad categorical season labels are not direct K-means inputs.

### `70_kmeans_cluster_city_distribution_k5.csv`

Contains the study-area distribution of NOAA events within each cluster.

Geographic study area is not used as a K-means input variable.

This file is used only for post-hoc geographic interpretation of the resulting clusters.

### `71_kmeans_cluster_year_distribution_k5.csv`

Contains annual NOAA event counts for each K-means cluster.

Used for:

- annual cluster-frequency analysis;
- annual stacked-bar figures; and
- dashboard visualizations.

### `72_kmeans_k4_vs_k5_comparison.csv`

Contains a sensitivity comparison between the four-cluster and five-cluster K-means solutions.

Includes information such as:

- silhouette score;
- inertia;
- number of clusters;
- smallest cluster size;
- largest cluster size; and
- average cluster size.

The five-cluster model is retained as the primary exploratory interpretation, while the four-cluster model provides a sensitivity comparison.

---

## K-Means Feature Construction

K-means clustering is applied only to NOAA flash-flood records because NOAA contains the comparable timing, duration, and reported financial-impact variables required for the analysis.

The clustering feature matrix contains four numerical variables representing three conceptual event characteristics.

### Flood Duration

```text
log(1 + duration_hours)
```

The log transformation reduces the influence of unusually long-duration events while retaining differences among shorter events.

### NOAA-Reported Financial Damage

```text
log(1 + noaa_total_damage_usd)
```

The log transformation reduces the influence of a small number of very high-loss events and allows zero-damage events to remain in the analysis.

### Seasonal Timing

Event timing is represented using:

```text
sin(2π × day_of_year / 365.25)
cos(2π × day_of_year / 365.25)
```

Using sine and cosine preserves the cyclical structure of the calendar. For example, an event occurring near December 31 remains close in seasonal feature space to an event occurring near January 1.

The two timing variables represent one conceptual characteristic: seasonal timing.

All four numerical clustering features are standardized before K-means is fitted.

---

## K-Means Interpretation

K-means creates clusters based on similarity in:

- transformed duration;
- transformed financial damage; and
- cyclical seasonal timing.

The algorithm does not use:

- predefined duration categories;
- predefined damage categories;
- cluster names;
- study area;
- latitude or longitude; or
- Groundsource observations.

After K-means assigns each NOAA event to a cluster, the original NOAA values are summarized within each group.

Post-hoc summaries include:

- median and mean duration;
- duration range;
- percentage of damaging events;
- median and mean financial damage;
- damage among damaging events only;
- month and seasonal timing;
- study area; and
- annual frequency.

Cluster descriptions such as `May-Jun, ~3-Hour, Non-Damaging` are therefore interpretive labels added after clustering. They are not rules used to create cluster membership.

---

## Primary Five-Cluster Interpretation

| Cluster | Description | Median Duration | Percent Damaging |
|---|---|---:|---:|
| Cluster 1 | May–Jun, ~3-hour, non-damaging | 2.95 hr | 0% |
| Cluster 2 | Sep–Nov, ~4-hour, mostly non-damaging | 4.25 hr | 5.7% |
| Cluster 3 | Aug–Oct, very-short-duration, mixed-impact | ~5 min | 33.3% |
| Cluster 4 | May–Jun, ~2.5-hour, high-impact | 2.53 hr | 100% |
| Cluster 5 | Oct–Dec, ~4.5-hour, high-impact | 4.50 hr | 100% |

These profiles suggest that similar-duration floods can have substantially different reported impacts and that high-impact events are not limited to the longest-duration floods.

The clusters should be interpreted as exploratory descriptive profiles rather than causal or predictive categories.

---

## Dashboard Use

The processed datasets also support the Streamlit dashboard located at:

```text
dashboard/app.py
```

The dashboard uses the processed data to display:

- NOAA and Groundsource spatial observations;
- separate NOAA and Groundsource annual flood-activity time series;
- study-area comparisons;
- K-means model-selection diagnostics;
- cluster counts;
- log-transformed duration and damage relationships;
- cluster distributions;
- monthly patterns;
- annual stacked cluster composition;
- damage summaries;
- interactive three-dimensional clustering; and
- spatial cluster distributions.

---

## Development and Generative AI Note

Generative AI tools were used as assistants for Python coding, debugging, data-cleaning workflows, visualization development, and documentation.

AI-generated code or processing suggestions were not automatically accepted. All final scripts and analytical steps were reviewed, executed, tested, and verified by the project authors using the NOAA and Groundsource datasets.

The empirical flood observations, financial-impact values, event durations, spatial data, and matching results come from the project datasets rather than generative AI.

Final methodological decisions and interpretations were reviewed and confirmed by the project authors.

---

## Regenerating the Data

Processed files are generated by running:

```bash
python scripts/01_build_harmonized_events.py
python scripts/02_match_noaa_groundsource.py
python scripts/additional_relationships.py
python scripts/kmeans_event_profiles.py
```

The interactive dashboard can then be launched with:

```bash
streamlit run dashboard/app.py
```
