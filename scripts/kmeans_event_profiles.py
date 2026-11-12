from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

FIGURE_DIR = BASE_DIR / "figures" / "clustering"
MODEL_SELECTION_DIR = FIGURE_DIR / "model_selection"
K4_DIR = FIGURE_DIR / "k4"
K5_DIR = FIGURE_DIR / "k5"
POSTER_DIR = FIGURE_DIR / "poster"

for folder in [
    FIGURE_DIR,
    MODEL_SELECTION_DIR,
    K4_DIR,
    K5_DIR,
    POSTER_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "44_noaa_events_source_only.csv"

COMMON_START_YEAR = 2005
COMMON_END_YEAR = 2025

RANDOM_STATE = 42
N_INIT = 50

# Test 2 through 10 clusters.
MODEL_SELECTION_K_VALUES = list(range(2, 11))

# Preserve k=4 as sensitivity analysis and k=5 as primary.
K_VALUES_TO_SAVE = [4, 5]

CLUSTER_ORDER = [1, 2, 3, 4, 5]


# ============================================================
# CONSISTENT COLORS
# ============================================================

CLUSTER_COLORS = {
    1: "#1f77b4",  # blue
    2: "#ff7f0e",  # orange
    3: "#2ca02c",  # green
    4: "#d62728",  # red
    5: "#9467bd",  # purple
}

PLOTLY_CLUSTER_COLORS = {
    "Cluster 1": "#1f77b4",
    "Cluster 2": "#ff7f0e",
    "Cluster 3": "#2ca02c",
    "Cluster 4": "#d62728",
    "Cluster 5": "#9467bd",
}


# ============================================================
# POST-HOC CLUSTER DESCRIPTIONS
# ============================================================

K5_PROFILE_DESCRIPTIONS = {
    1: "May-Jun, ~3-Hour, Non-Damaging",
    2: "Sep-Nov, ~4-Hour, Mostly Non-Damaging",
    3: "Aug-Oct, Very-Short (~5-Min), Mixed-Impact",
    4: "May-Jun, ~2.5-Hour, High-Impact",
    5: "Oct-Dec, ~4.5-Hour, High-Impact",
}


# ============================================================
# HELPERS
# ============================================================

def cluster_label(cluster_id):
    return f"Cluster {int(cluster_id)}"


def assign_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Spring"

    if month in [6, 7, 8]:
        return "Summer"

    return "Fall"


# ============================================================
# LOAD NOAA DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "event_date",
    "duration_hours",
    "noaa_total_damage_usd",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# CLEAN DATA
# ============================================================

df["event_date"] = pd.to_datetime(
    df["event_date"],
    errors="coerce",
)

df["duration_hours"] = pd.to_numeric(
    df["duration_hours"],
    errors="coerce",
)

df["noaa_total_damage_usd"] = pd.to_numeric(
    df["noaa_total_damage_usd"],
    errors="coerce",
)

df["year"] = df["event_date"].dt.year


analysis_base = df[
    df["year"].between(
        COMMON_START_YEAR,
        COMMON_END_YEAR,
        inclusive="both",
    )
].copy()


analysis_base = analysis_base.dropna(
    subset=[
        "event_date",
        "duration_hours",
        "noaa_total_damage_usd",
    ]
).copy()


analysis_base = analysis_base[
    (analysis_base["duration_hours"] >= 0)
    &
    (analysis_base["noaa_total_damage_usd"] >= 0)
].copy()


# ============================================================
# DATE VARIABLES
# ============================================================

analysis_base["month"] = (
    analysis_base["event_date"].dt.month
)

analysis_base["day_of_year"] = (
    analysis_base["event_date"].dt.dayofyear
)

analysis_base["season"] = (
    analysis_base["month"]
    .apply(assign_season)
)


# ============================================================
# CYCLICAL DATE REPRESENTATION
# ============================================================

analysis_base["season_sin"] = np.sin(
    2
    * np.pi
    * analysis_base["day_of_year"]
    / 365.25
)

analysis_base["season_cos"] = np.cos(
    2
    * np.pi
    * analysis_base["day_of_year"]
    / 365.25
)


# ============================================================
# LOG TRANSFORMS
# ============================================================

analysis_base["duration_log"] = np.log1p(
    analysis_base["duration_hours"]
)

analysis_base["damage_log"] = np.log1p(
    analysis_base["noaa_total_damage_usd"]
)


# ============================================================
# FEATURE MATRIX
# ============================================================

FEATURE_COLUMNS = [
    "duration_log",
    "damage_log",
    "season_sin",
    "season_cos",
]

X = analysis_base[
    FEATURE_COLUMNS
].copy()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


print(
    "\nUsable NOAA events:",
    len(analysis_base),
)

print(
    "Feature matrix:",
    X.shape,
)


# ============================================================
# MODEL SELECTION: K = 2 THROUGH 10
# ============================================================

model_selection_rows = []

for k in MODEL_SELECTION_K_VALUES:

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

    labels = model.fit_predict(
        X_scaled
    )

    model_selection_rows.append(
        {
            "k": k,

            "inertia":
                model.inertia_,

            "silhouette_score":
                silhouette_score(
                    X_scaled,
                    labels,
                ),
        }
    )


model_selection = pd.DataFrame(
    model_selection_rows
)


model_selection.to_csv(
    DATA_DIR
    / "65_kmeans_model_selection.csv",
    index=False,
)


print("\nMODEL SELECTION: K=2 THROUGH 10")

print(
    model_selection.to_string(
        index=False
    )
)


# ============================================================
# POSTER PANEL 1:
# MODEL SELECTION
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.8),
)


# ELBOW

axes[0].plot(
    model_selection["k"],
    model_selection["inertia"],
    marker="o",
)

axes[0].set_title(
    "A. Elbow"
)

axes[0].set_xlabel(
    "Number of clusters (k)"
)

axes[0].set_ylabel(
    "Within-cluster sum of squares"
)

axes[0].set_xticks(
    MODEL_SELECTION_K_VALUES
)

axes[0].grid(
    alpha=0.2
)


# SILHOUETTE

axes[1].plot(
    model_selection["k"],
    model_selection[
        "silhouette_score"
    ],
    marker="o",
)

axes[1].set_title(
    "B. Silhouette"
)

axes[1].set_xlabel(
    "Number of clusters (k)"
)

axes[1].set_ylabel(
    "Silhouette score"
)

axes[1].set_xticks(
    MODEL_SELECTION_K_VALUES
)

axes[1].grid(
    alpha=0.2
)


fig.suptitle(
    "Model Selection",
    fontsize=14,
)

fig.tight_layout()

fig.savefig(
    POSTER_DIR
    / "01_model_selection_panel.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FUNCTION TO RUN K-MEANS
# ============================================================

def run_kmeans_solution(k):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"RUNNING K-MEANS: K={k}"
    )

    print(
        "=" * 70
    )


    analysis = (
        analysis_base.copy()
    )


    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )


    analysis[
        "cluster"
    ] = (
        model.fit_predict(
            X_scaled
        )
        + 1
    )


    analysis[
        "cluster_label"
    ] = (
        analysis[
            "cluster"
        ]
        .apply(
            cluster_label
        )
    )


    analysis[
        "damaging_event"
    ] = (
        analysis[
            "noaa_total_damage_usd"
        ]
        > 0
    )


    if k == 5:

        analysis[
            "profile_description"
        ] = (
            analysis[
                "cluster"
            ]
            .map(
                K5_PROFILE_DESCRIPTIONS
            )
        )

    else:

        analysis[
            "profile_description"
        ] = (
            analysis[
                "cluster_label"
            ]
        )


    # ========================================================
    # EVENT-LEVEL CSV
    # ========================================================

    run_label = f"k{k}"

    analysis.to_csv(
        DATA_DIR
        / (
            f"66_noaa_kmeans_event_clusters_"
            f"{run_label}.csv"
        ),
        index=False,
    )


    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    cluster_summary = (
        analysis
        .groupby("cluster")
        .agg(

            event_count=(
                "cluster",
                "size",
            ),

            damaging_event_count=(
                "damaging_event",
                "sum",
            ),

            duration_min_hours=(
                "duration_hours",
                "min",
            ),

            duration_q1_hours=(
                "duration_hours",
                lambda x: x.quantile(
                    0.25
                ),
            ),

            duration_median_hours=(
                "duration_hours",
                "median",
            ),

            duration_mean_hours=(
                "duration_hours",
                "mean",
            ),

            duration_q3_hours=(
                "duration_hours",
                lambda x: x.quantile(
                    0.75
                ),
            ),

            duration_max_hours=(
                "duration_hours",
                "max",
            ),

            damage_min_usd=(
                "noaa_total_damage_usd",
                "min",
            ),

            damage_q1_usd=(
                "noaa_total_damage_usd",
                lambda x: x.quantile(
                    0.25
                ),
            ),

            damage_median_usd=(
                "noaa_total_damage_usd",
                "median",
            ),

            damage_mean_usd=(
                "noaa_total_damage_usd",
                "mean",
            ),

            damage_q3_usd=(
                "noaa_total_damage_usd",
                lambda x: x.quantile(
                    0.75
                ),
            ),

            damage_max_usd=(
                "noaa_total_damage_usd",
                "max",
            ),

            total_damage_usd=(
                "noaa_total_damage_usd",
                "sum",
            ),

            median_day_of_year=(
                "day_of_year",
                "median",
            ),
        )
        .reset_index()
    )


    cluster_summary[
        "percent_damaging"
    ] = (
        100
        * cluster_summary[
            "damaging_event_count"
        ]
        / cluster_summary[
            "event_count"
        ]
    )


    # ========================================================
    # DAMAGING EVENTS ONLY
    # ========================================================

    damaging_only = analysis[
        analysis[
            "noaa_total_damage_usd"
        ]
        > 0
    ].copy()


    damaging_only_summary = (
        damaging_only
        .groupby("cluster")
        .agg(

            damaging_only_event_count=(
                "cluster",
                "size",
            ),

            damaging_only_median_damage_usd=(
                "noaa_total_damage_usd",
                "median",
            ),

            damaging_only_mean_damage_usd=(
                "noaa_total_damage_usd",
                "mean",
            ),
        )
        .reset_index()
    )


    cluster_summary = (
        cluster_summary
        .merge(
            damaging_only_summary,
            on="cluster",
            how="left",
        )
    )


    # ========================================================
    # MONTH DISTRIBUTION
    # ========================================================

    month_distribution = (
        analysis
        .groupby(
            [
                "cluster",
                "month",
            ]
        )
        .size()
        .reset_index(
            name="event_count"
        )
    )


    dominant_month = (
        month_distribution
        .sort_values(
            [
                "cluster",
                "event_count",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset="cluster"
        )
        [
            [
                "cluster",
                "month",
            ]
        ]
        .rename(
            columns={
                "month":
                    "dominant_month"
            }
        )
    )


    # ========================================================
    # SEASON DISTRIBUTION
    # ========================================================

    season_distribution = (
        analysis
        .groupby(
            [
                "cluster",
                "season",
            ]
        )
        .size()
        .reset_index(
            name="event_count"
        )
    )


    dominant_season = (
        season_distribution
        .sort_values(
            [
                "cluster",
                "event_count",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset="cluster"
        )
        [
            [
                "cluster",
                "season",
            ]
        ]
        .rename(
            columns={
                "season":
                    "dominant_season"
            }
        )
    )


    cluster_summary = (
        cluster_summary
        .merge(
            dominant_month,
            on="cluster",
            how="left",
        )
        .merge(
            dominant_season,
            on="cluster",
            how="left",
        )
    )


    # ========================================================
    # CITY DISTRIBUTION
    # ========================================================

    if "city" in analysis.columns:

        city_distribution = (
            analysis
            .groupby(
                [
                    "cluster",
                    "city",
                ]
            )
            .size()
            .reset_index(
                name="event_count"
            )
        )


        dominant_city = (
            city_distribution
            .sort_values(
                [
                    "cluster",
                    "event_count",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .drop_duplicates(
                subset="cluster"
            )
            [
                [
                    "cluster",
                    "city",
                ]
            ]
            .rename(
                columns={
                    "city":
                        "dominant_city"
                }
            )
        )


        cluster_summary = (
            cluster_summary
            .merge(
                dominant_city,
                on="cluster",
                how="left",
            )
        )

    else:

        city_distribution = (
            pd.DataFrame()
        )


    # ========================================================
    # YEAR DISTRIBUTION
    # ========================================================

    year_distribution = (
        analysis
        .groupby(
            [
                "cluster",
                "year",
            ]
        )
        .size()
        .reset_index(
            name="event_count"
        )
    )


    # ========================================================
    # LABELS
    # ========================================================

    cluster_summary[
        "cluster_label"
    ] = (
        cluster_summary[
            "cluster"
        ]
        .apply(
            cluster_label
        )
    )


    if k == 5:

        cluster_summary[
            "profile_description"
        ] = (
            cluster_summary[
                "cluster"
            ]
            .map(
                K5_PROFILE_DESCRIPTIONS
            )
        )

    else:

        cluster_summary[
            "profile_description"
        ] = (
            cluster_summary[
                "cluster_label"
            ]
        )


    # ========================================================
    # SAVE TABLES
    # ========================================================

    cluster_summary.to_csv(
        DATA_DIR
        / (
            f"67_kmeans_cluster_summary_"
            f"{run_label}.csv"
        ),
        index=False,
    )


    month_distribution.to_csv(
        DATA_DIR
        / (
            f"68_kmeans_cluster_month_distribution_"
            f"{run_label}.csv"
        ),
        index=False,
    )


    season_distribution.to_csv(
        DATA_DIR
        / (
            f"69_kmeans_cluster_season_distribution_"
            f"{run_label}.csv"
        ),
        index=False,
    )


    if not city_distribution.empty:

        city_distribution.to_csv(
            DATA_DIR
            / (
                f"70_kmeans_cluster_city_distribution_"
                f"{run_label}.csv"
            ),
            index=False,
        )


    year_distribution.to_csv(
        DATA_DIR
        / (
            f"71_kmeans_cluster_year_distribution_"
            f"{run_label}.csv"
        ),
        index=False,
    )


    return {
        "analysis":
            analysis,

        "summary":
            cluster_summary,

        "month_distribution":
            month_distribution,

        "season_distribution":
            season_distribution,

        "city_distribution":
            city_distribution,

        "year_distribution":
            year_distribution,
    }


# ============================================================
# RUN K=4 AND K=5
# ============================================================

results = {}

for k in K_VALUES_TO_SAVE:

    results[k] = (
        run_kmeans_solution(k)
    )


analysis = (
    results[5]["analysis"]
    .copy()
)

cluster_summary = (
    results[5]["summary"]
    .copy()
)

month_distribution = (
    results[5][
        "month_distribution"
    ]
    .copy()
)


# ============================================================
# K4 VS K5 TABLE
# ============================================================

comparison_rows = []

for k in K_VALUES_TO_SAVE:

    summary = (
        results[k]["summary"]
    )

    model_row = (
        model_selection.loc[
            model_selection["k"] == k
        ]
        .iloc[0]
    )

    comparison_rows.append(
        {
            "k":
                k,

            "silhouette_score":
                model_row[
                    "silhouette_score"
                ],

            "inertia":
                model_row[
                    "inertia"
                ],

            "number_of_clusters":
                len(summary),

            "smallest_cluster_n":
                summary[
                    "event_count"
                ].min(),

            "largest_cluster_n":
                summary[
                    "event_count"
                ].max(),

            "mean_cluster_size":
                summary[
                    "event_count"
                ].mean(),
        }
    )


comparison_table = pd.DataFrame(
    comparison_rows
)


comparison_table.to_csv(
    DATA_DIR
    / "72_kmeans_k4_vs_k5_comparison.csv",
    index=False,
)


# ============================================================
# CLUSTER DESCRIPTION TABLE
# ============================================================

cluster_table = (
    cluster_summary[
        [
            "cluster",
            "profile_description",
            "event_count",
            "duration_median_hours",
            "percent_damaging",
            "damage_median_usd",
            "damaging_only_median_damage_usd",
        ]
    ]
    .copy()
)


cluster_table = (
    cluster_table.rename(
        columns={
            "cluster":
                "Cluster",

            "profile_description":
                "Description",

            "event_count":
                "N",

            "duration_median_hours":
                "Median Duration (hr)",

            "percent_damaging":
                "% Damaging",

            "damage_median_usd":
                "Median Damage ($)",

            "damaging_only_median_damage_usd":
                "Median Damage if Damaging ($)",
        }
    )
)


cluster_table.to_csv(
    POSTER_DIR
    / "cluster_description_table.csv",
    index=False,
)


# ============================================================
# POSTER TABLE IMAGE
# ============================================================

display_table = (
    cluster_table.copy()
)


display_table[
    "Cluster"
] = (
    display_table[
        "Cluster"
    ]
    .apply(
        lambda x:
            f"Cluster {int(x)}"
    )
)


display_table[
    "Median Duration (hr)"
] = (
    display_table[
        "Median Duration (hr)"
    ]
    .map(
        lambda x:
            f"{x:.2f}"
    )
)


display_table[
    "% Damaging"
] = (
    display_table[
        "% Damaging"
    ]
    .map(
        lambda x:
            f"{x:.1f}%"
    )
)


display_table[
    "Median Damage ($)"
] = (
    display_table[
        "Median Damage ($)"
    ]
    .map(
        lambda x:
            f"${x:,.0f}"
    )
)


display_table[
    "Median Damage if Damaging ($)"
] = (
    display_table[
        "Median Damage if Damaging ($)"
    ]
    .map(
        lambda x:
            "N/A"
            if pd.isna(x)
            else f"${x:,.0f}"
    )
)


fig, ax = plt.subplots(
    figsize=(13, 3.5)
)

ax.axis("off")


table = ax.table(
    cellText=(
        display_table.values
    ),

    colLabels=(
        display_table.columns
    ),

    loc="center",

    cellLoc="center",
)


table.auto_set_font_size(
    False
)

table.set_fontsize(
    8
)

table.scale(
    1,
    1.5,
)


ax.set_title(
    "Cluster Description Guide",
    fontsize=13,
    pad=12,
)


fig.tight_layout()


fig.savefig(
    POSTER_DIR
    / "02_cluster_description_table.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# POSTER PANEL 2:
# CLUSTER CHARACTERISTICS - 2X2
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
)


# ------------------------------------------------------------
# A. EVENT COUNTS
# ------------------------------------------------------------

counts = (
    analysis[
        "cluster"
    ]
    .value_counts()
    .reindex(
        CLUSTER_ORDER,
        fill_value=0,
    )
)


axes[0, 0].bar(
    [
        cluster_label(c)
        for c in CLUSTER_ORDER
    ],

    counts.values,

    color=[
        CLUSTER_COLORS[c]
        for c in CLUSTER_ORDER
    ],
)


axes[0, 0].set_title(
    "A. Events"
)

axes[0, 0].set_ylabel(
    "Count"
)


# ------------------------------------------------------------
# B. LOG DURATION VS LOG DAMAGE
# ------------------------------------------------------------

for cluster_id in CLUSTER_ORDER:

    subset = analysis[
        analysis["cluster"]
        == cluster_id
    ]

    axes[0, 1].scatter(

        subset[
            "duration_log"
        ],

        subset[
            "damage_log"
        ],

        color=CLUSTER_COLORS[
            cluster_id
        ],

        alpha=0.7,

        s=30,

        label=cluster_label(
            cluster_id
        ),
    )


axes[0, 1].set_title(
    "B. Duration vs. Damage"
)

axes[0, 1].set_xlabel(
    "log(1 + duration [hr])"
)

axes[0, 1].set_ylabel(
    "log(1 + damage [$])"
)

axes[0, 1].legend(
    title="Cluster",
    fontsize=8,
)


# ------------------------------------------------------------
# C. DURATION BOXPLOT
# ------------------------------------------------------------

duration_groups = [

    analysis.loc[
        analysis["cluster"]
        == cluster_id,

        "duration_hours",
    ].values

    for cluster_id
    in CLUSTER_ORDER
]


box = axes[1, 0].boxplot(

    duration_groups,

    tick_labels=[
        cluster_label(c)
        for c in CLUSTER_ORDER
    ],

    patch_artist=True,
)


for patch, cluster_id in zip(
    box["boxes"],
    CLUSTER_ORDER,
):

    patch.set_facecolor(
        CLUSTER_COLORS[
            cluster_id
        ]
    )

    patch.set_alpha(
        0.45
    )


axes[1, 0].set_title(
    "C. Duration"
)

axes[1, 0].set_ylabel(
    "Hours"
)


# ------------------------------------------------------------
# D. DAMAGE BOXPLOT
# ------------------------------------------------------------

damage_groups = [

    analysis.loc[
        analysis["cluster"]
        == cluster_id,

        "damage_log",
    ].values

    for cluster_id
    in CLUSTER_ORDER
]


box = axes[1, 1].boxplot(

    damage_groups,

    tick_labels=[
        cluster_label(c)
        for c in CLUSTER_ORDER
    ],

    patch_artist=True,
)


for patch, cluster_id in zip(
    box["boxes"],
    CLUSTER_ORDER,
):

    patch.set_facecolor(
        CLUSTER_COLORS[
            cluster_id
        ]
    )

    patch.set_alpha(
        0.45
    )


axes[1, 1].set_title(
    "D. Damage"
)

axes[1, 1].set_ylabel(
    "log(1 + damage [$])"
)


for ax in axes.flat:

    ax.tick_params(
        axis="x",
        labelrotation=20,
    )

    ax.grid(
        axis="y",
        alpha=0.15,
    )


fig.suptitle(
    "Cluster Characteristics",
    fontsize=15,
)


fig.tight_layout()


fig.savefig(
    POSTER_DIR
    / "03_cluster_characteristics_panel.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# POSTER PANEL 3:
# TEMPORAL PATTERNS
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5),
)


# ------------------------------------------------------------
# A. MONTHLY LINE GRAPH
# ------------------------------------------------------------

month_pivot = (
    month_distribution
    .pivot(
        index="month",
        columns="cluster",
        values="event_count",
    )
    .fillna(0)
    .reindex(
        index=range(1, 13),
        fill_value=0,
    )
)


for cluster_id in CLUSTER_ORDER:

    if cluster_id not in (
        month_pivot.columns
    ):

        month_pivot[
            cluster_id
        ] = 0


    axes[0].plot(

        month_pivot.index,

        month_pivot[
            cluster_id
        ],

        marker="o",

        linewidth=1.8,

        color=CLUSTER_COLORS[
            cluster_id
        ],

        label=cluster_label(
            cluster_id
        ),
    )


axes[0].set_title(
    "A. Monthly Pattern"
)

axes[0].set_xlabel(
    "Month"
)

axes[0].set_ylabel(
    "Events"
)

axes[0].set_xticks(
    range(1, 13)
)

axes[0].legend(
    title="Cluster",
    fontsize=8,
)

axes[0].grid(
    alpha=0.2
)


# ------------------------------------------------------------
# B. ANNUAL STACKED BAR
# ------------------------------------------------------------

annual_cluster = (

    analysis

    .groupby(
        [
            "year",
            "cluster",
        ]
    )

    .size()

    .unstack(
        fill_value=0
    )

    .reindex(
        columns=CLUSTER_ORDER,
        fill_value=0,
    )

    .reindex(
        range(
            COMMON_START_YEAR,
            COMMON_END_YEAR + 1,
        ),
        fill_value=0,
    )
)


bottom = np.zeros(
    len(
        annual_cluster
    )
)


for cluster_id in CLUSTER_ORDER:

    values = (
        annual_cluster[
            cluster_id
        ]
        .values
    )


    axes[1].bar(

        annual_cluster.index,

        values,

        bottom=bottom,

        color=CLUSTER_COLORS[
            cluster_id
        ],

        label=cluster_label(
            cluster_id
        ),

        width=0.8,
    )


    bottom += values


axes[1].set_title(
    "B. Annual Composition"
)

axes[1].set_xlabel(
    "Year"
)

axes[1].set_ylabel(
    "Events"
)

axes[1].legend(
    title="Cluster",
    fontsize=8,
)

axes[1].grid(
    axis="y",
    alpha=0.2,
)


fig.suptitle(
    "Temporal Patterns",
    fontsize=15,
)


fig.tight_layout()


fig.savefig(
    POSTER_DIR
    / "04_temporal_patterns_panel.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# POSTER PANEL 4:
# HISTOGRAMS
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.8),
)


# ------------------------------------------------------------
# A. DURATION HISTOGRAM
# ------------------------------------------------------------

for cluster_id in CLUSTER_ORDER:

    subset = analysis[
        analysis["cluster"]
        == cluster_id
    ]


    axes[0].hist(

        subset[
            "duration_log"
        ],

        bins=12,

        histtype="step",

        linewidth=2,

        color=CLUSTER_COLORS[
            cluster_id
        ],

        label=cluster_label(
            cluster_id
        ),
    )


axes[0].set_title(
    "A. Duration"
)

axes[0].set_xlabel(
    "log(1 + duration [hr])"
)

axes[0].set_ylabel(
    "Events"
)

axes[0].legend(
    title="Cluster",
    fontsize=8,
)


# ------------------------------------------------------------
# B. DAMAGE HISTOGRAM
# ------------------------------------------------------------

for cluster_id in CLUSTER_ORDER:

    subset = analysis[
        analysis["cluster"]
        == cluster_id
    ]


    axes[1].hist(

        subset[
            "damage_log"
        ],

        bins=12,

        histtype="step",

        linewidth=2,

        color=CLUSTER_COLORS[
            cluster_id
        ],

        label=cluster_label(
            cluster_id
        ),
    )


axes[1].set_title(
    "B. Damage"
)

axes[1].set_xlabel(
    "log(1 + damage [$])"
)

axes[1].set_ylabel(
    "Events"
)

axes[1].legend(
    title="Cluster",
    fontsize=8,
)


for ax in axes:

    ax.grid(
        axis="y",
        alpha=0.2,
    )


fig.suptitle(
    "Cluster Distributions",
    fontsize=15,
)


fig.tight_layout()


fig.savefig(
    POSTER_DIR
    / "05_distribution_histograms_panel.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# OPTIONAL SUMMARY PANEL
#
# You said Figures 6-8 should not be on the poster.
# This panel is still created for reference.
# ============================================================

plot_summary = (

    cluster_summary

    .set_index(
        "cluster"
    )

    .reindex(
        CLUSTER_ORDER
    )
)


fig, axes = plt.subplots(
    2,
    2,
    figsize=(11, 8),
)


# ------------------------------------------------------------
# A. PERCENT DAMAGING
# ------------------------------------------------------------

axes[0, 0].bar(

    [
        cluster_label(c)
        for c in CLUSTER_ORDER
    ],

    plot_summary[
        "percent_damaging"
    ].values,

    color=[
        CLUSTER_COLORS[c]
        for c in CLUSTER_ORDER
    ],
)


axes[0, 0].set_title(
    "A. Damaging Events"
)

axes[0, 0].set_ylabel(
    "Percent"
)

axes[0, 0].set_ylim(
    0,
    100
)


# ------------------------------------------------------------
# B. MEDIAN DURATION
# ------------------------------------------------------------

axes[0, 1].bar(

    [
        cluster_label(c)
        for c in CLUSTER_ORDER
    ],

    plot_summary[
        "duration_median_hours"
    ].values,

    color=[
        CLUSTER_COLORS[c]
        for c in CLUSTER_ORDER
    ],
)


axes[0, 1].set_title(
    "B. Median Duration"
)

axes[0, 1].set_ylabel(
    "Hours"
)


# ------------------------------------------------------------
# C. MEDIAN DAMAGE
# ------------------------------------------------------------

axes[1, 0].bar(

    [
        cluster_label(c)
        for c in CLUSTER_ORDER
    ],

    plot_summary[
        "damage_median_usd"
    ].values,

    color=[
        CLUSTER_COLORS[c]
        for c in CLUSTER_ORDER
    ],
)


axes[1, 0].set_title(
    "C. Median Damage"
)

axes[1, 0].set_ylabel(
    "USD"
)


# ------------------------------------------------------------
# D. MEDIAN DAMAGE AMONG DAMAGING EVENTS
# ------------------------------------------------------------

damaging_medians = (

    plot_summary[
        "damaging_only_median_damage_usd"
    ]

    .fillna(0)

    .values
)


axes[1, 1].bar(

    [
        cluster_label(c)
        for c in CLUSTER_ORDER
    ],

    damaging_medians,

    color=[
        CLUSTER_COLORS[c]
        for c in CLUSTER_ORDER
    ],
)


axes[1, 1].set_title(
    "D. Damage if Damaging"
)

axes[1, 1].set_ylabel(
    "USD"
)


for ax in axes.flat:

    ax.tick_params(
        axis="x",
        labelrotation=20,
    )

    ax.grid(
        axis="y",
        alpha=0.15,
    )


fig.suptitle(
    "Cluster Summary Metrics",
    fontsize=15,
)


fig.tight_layout()


fig.savefig(
    POSTER_DIR
    / "06_summary_metrics_panel_optional.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# STATIC 3D CLUSTER GRAPH
# ============================================================

fig = plt.figure(
    figsize=(9, 7)
)

ax = fig.add_subplot(
    111,
    projection="3d",
)


for cluster_id in CLUSTER_ORDER:

    subset = analysis[
        analysis["cluster"]
        == cluster_id
    ]


    ax.scatter(

        subset[
            "duration_log"
        ],

        subset[
            "damage_log"
        ],

        subset[
            "day_of_year"
        ],

        color=CLUSTER_COLORS[
            cluster_id
        ],

        label=cluster_label(
            cluster_id
        ),

        alpha=0.72,

        s=30,
    )


ax.set_xlabel(
    "log(1 + duration [hr])"
)

ax.set_ylabel(
    "log(1 + damage [$])"
)

ax.set_zlabel(
    "Day of year"
)

ax.set_title(
    "3D Cluster Structure"
)

ax.legend(
    title="Cluster",
    fontsize=8,
)


fig.tight_layout()


fig.savefig(
    POSTER_DIR
    / "07_3d_cluster_view.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# INTERACTIVE 3D HTML
# ============================================================

hover_columns = [
    "event_date",
    "duration_hours",
    "noaa_total_damage_usd",
    "month",
    "season",
]

if "city" in analysis.columns:

    hover_columns.append(
        "city"
    )


fig_3d = px.scatter_3d(

    analysis,

    x="duration_log",

    y="damage_log",

    z="day_of_year",

    color="cluster_label",

    color_discrete_map=(
        PLOTLY_CLUSTER_COLORS
    ),

    category_orders={
        "cluster_label": [
            cluster_label(c)
            for c in CLUSTER_ORDER
        ]
    },

    hover_data=(
        hover_columns
    ),

    labels={

        "duration_log":
            "log(1 + Flood Duration [hours])",

        "damage_log":
            "log(1 + NOAA Damage [USD])",

        "day_of_year":
            "Day of Year",

        "cluster_label":
            "Cluster",
    },

    title=(
        "K-Means Event Clusters: "
        "Duration, Damage, and Seasonal Timing"
    ),
)


fig_3d.write_html(
    K5_DIR
    / "11_duration_damage_date_3d.html"
)


# ============================================================
# POSTER CALLOUT TEXT
# ============================================================

summary_lookup = (

    cluster_summary

    .set_index(
        "cluster"
    )
)


c1_pct = (
    summary_lookup.loc[
        1,
        "percent_damaging"
    ]
)

c2_pct = (
    summary_lookup.loc[
        2,
        "percent_damaging"
    ]
)

c3_pct = (
    summary_lookup.loc[
        3,
        "percent_damaging"
    ]
)

c4_pct = (
    summary_lookup.loc[
        4,
        "percent_damaging"
    ]
)

c5_pct = (
    summary_lookup.loc[
        5,
        "percent_damaging"
    ]
)


c2_damage = (
    summary_lookup.loc[
        2,
        "damaging_only_median_damage_usd"
    ]
)

c3_damage = (
    summary_lookup.loc[
        3,
        "damaging_only_median_damage_usd"
    ]
)

c4_damage = (
    summary_lookup.loc[
        4,
        "damaging_only_median_damage_usd"
    ]
)

c5_damage = (
    summary_lookup.loc[
        5,
        "damaging_only_median_damage_usd"
    ]
)


poster_text = f"""
K-MEANS POSTER CALLOUTS

3D CLUSTER DESCRIPTION
K-means grouped NOAA flash-flood events using three conceptual
dimensions: seasonal timing, flood duration, and reported financial
damage. Duration and damage were log-transformed, while seasonal timing
was represented using cyclical sine and cosine transformations of day
of year.

DAMAGE OCCURRENCE
Damage occurrence differed strongly across clusters. Cluster 1 had
{c1_pct:.1f}% damaging events, Cluster 2 had {c2_pct:.1f}%, Cluster 3
had {c3_pct:.1f}%, Cluster 4 had {c4_pct:.1f}%, and Cluster 5 had
{c5_pct:.1f}%.

DAMAGE SEVERITY
When zero-damage events were included, median reported damage was $0
for Clusters 1-3, compared with approximately
${summary_lookup.loc[4, "damage_median_usd"]:,.0f} for Cluster 4 and
${summary_lookup.loc[5, "damage_median_usd"]:,.0f} for Cluster 5.

Among damaging events only, median reported damage was approximately
${c2_damage:,.0f} in Cluster 2,
${c3_damage:,.0f} in Cluster 3,
${c4_damage:,.0f} in Cluster 4, and
${c5_damage:,.0f} in Cluster 5.

INTERPRETATION
High reported impact was not limited to the longest-duration floods.
The five-cluster solution includes both shorter-duration and
longer-duration high-impact profiles. Similar-duration clusters also
show different damage occurrence, suggesting that duration alone does
not explain NOAA-reported financial impact.

MODEL SELECTION NOTE
The expanded model-selection analysis evaluates k=2 through k=10.
The five-cluster solution should be described as the primary
interpretation based on a balance of separation, cluster size, and
interpretability. Do not claim k=5 has the maximum silhouette score
unless the new k=2 through k=10 results show that.
"""


(
    POSTER_DIR
    / "poster_cluster_callouts.txt"
).write_text(
    poster_text,
    encoding="utf-8",
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\nPoster outputs created:"
)

for path in sorted(
    POSTER_DIR.iterdir()
):

    print(path)


print(
    "\nK-means poster analysis complete."
)