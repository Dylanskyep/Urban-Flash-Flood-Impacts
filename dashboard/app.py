from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="F.L.O.O.D. | Florida Flash-Flood Analysis",
    page_icon="🌧️",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

MAPPING_FILE = (
    DATA_DIR
    / "53_mapping_events.csv"
)

KMEANS_EVENTS_FILE = (
    DATA_DIR
    / "66_noaa_kmeans_event_clusters_k5.csv"
)

KMEANS_SUMMARY_FILE = (
    DATA_DIR
    / "67_kmeans_cluster_summary_k5.csv"
)

MODEL_SELECTION_FILE = (
    DATA_DIR
    / "65_kmeans_model_selection.csv"
)


# ============================================================
# PROJECT NAME
# ============================================================

PROJECT_ACRONYM = "F.L.O.O.D."

PROJECT_NAME = (
    "Florida Flood Location, Observation, "
    "Outcomes, and Dynamics"
)


# ============================================================
# CLUSTER SETTINGS
# ============================================================

CLUSTER_ORDER = [
    1,
    2,
    3,
    4,
    5,
]

CLUSTER_LABELS = {
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
    4: "Cluster 4",
    5: "Cluster 5",
}

CLUSTER_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
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

PROFILE_DESCRIPTIONS = {
    1: "May-Jun, ~3-Hour, Non-Damaging",
    2: "Sep-Nov, ~4-Hour, Mostly Non-Damaging",
    3: "Aug-Oct, Very-Short (~5-Min), Mixed-Impact",
    4: "May-Jun, ~2.5-Hour, High-Impact",
    5: "Oct-Dec, ~4.5-Hour, High-Impact",
}


# ============================================================
# SOURCE COLORS
# ============================================================

SOURCE_COLORS = {
    "NOAA Storm Events": "#1f77b4",
    "Groundsource": "#7f7f7f",
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_mapping_data():

    data = pd.read_csv(
        MAPPING_FILE
    )

    data["event_date"] = pd.to_datetime(
        data["event_date"],
        errors="coerce",
    )

    data["year"] = (
        data[
            "event_date"
        ]
        .dt.year
    )

    numeric_columns = [
        "latitude",
        "longitude",
        "duration_hours",
        "noaa_total_damage_usd",
        "groundsource_area_max_km2",
    ]

    for col in numeric_columns:

        if col in data.columns:

            data[col] = (
                pd.to_numeric(
                    data[col],
                    errors="coerce",
                )
            )

    return data


@st.cache_data
def load_kmeans_events():

    data = pd.read_csv(
        KMEANS_EVENTS_FILE
    )

    data["event_date"] = pd.to_datetime(
        data["event_date"],
        errors="coerce",
    )

    data["year"] = (
        data[
            "event_date"
        ]
        .dt.year
    )

    data["month"] = (
        data[
            "event_date"
        ]
        .dt.month
    )

    data["cluster"] = (
        pd.to_numeric(
            data["cluster"],
            errors="coerce",
        )
        .astype(
            "Int64"
        )
    )

    data = data[
        data[
            "cluster"
        ]
        .isin(
            CLUSTER_ORDER
        )
    ].copy()

    data[
        "cluster_label"
    ] = (
        data[
            "cluster"
        ]
        .map(
            CLUSTER_LABELS
        )
    )

    data[
        "profile_description"
    ] = (
        data[
            "cluster"
        ]
        .map(
            PROFILE_DESCRIPTIONS
        )
    )

    numeric_columns = [
        "duration_hours",
        "noaa_total_damage_usd",
        "latitude",
        "longitude",
        "day_of_year",
    ]

    for col in numeric_columns:

        if col in data.columns:

            data[col] = (
                pd.to_numeric(
                    data[col],
                    errors="coerce",
                )
            )

    # --------------------------------------------------------
    # LOG TRANSFORMS
    # --------------------------------------------------------

    data[
        "duration_log"
    ] = (
        np.log1p(
            data[
                "duration_hours"
            ]
            .clip(
                lower=0
            )
        )
    )

    data[
        "damage_log"
    ] = (
        np.log1p(
            data[
                "noaa_total_damage_usd"
            ]
            .fillna(0)
            .clip(
                lower=0
            )
        )
    )

    data[
        "damaging_event"
    ] = (
        data[
            "noaa_total_damage_usd"
        ]
        > 0
    )

    return data


@st.cache_data
def load_kmeans_summary():

    data = pd.read_csv(
        KMEANS_SUMMARY_FILE
    )

    data["cluster"] = (
        pd.to_numeric(
            data["cluster"],
            errors="coerce",
        )
        .astype(
            "Int64"
        )
    )

    data[
        "cluster_label"
    ] = (
        data[
            "cluster"
        ]
        .map(
            CLUSTER_LABELS
        )
    )

    data[
        "profile_description"
    ] = (
        data[
            "cluster"
        ]
        .map(
            PROFILE_DESCRIPTIONS
        )
    )

    return data


@st.cache_data
def load_model_selection():

    return pd.read_csv(
        MODEL_SELECTION_FILE
    )


mapping = (
    load_mapping_data()
)

kmeans = (
    load_kmeans_events()
)

kmeans_summary = (
    load_kmeans_summary()
)

model_selection = (
    load_model_selection()
)


# ============================================================
# HELPERS
# ============================================================

def cluster_category_order():

    return [
        CLUSTER_LABELS[c]
        for c in CLUSTER_ORDER
    ]


def apply_cluster_layout(
    fig
):

    fig.update_layout(
        legend_title_text=(
            "Cluster"
        ),
        hoverlabel=dict(
            namelength=-1
        ),
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.title(
    f"{PROJECT_ACRONYM}"
)

st.subheader(
    PROJECT_NAME
)

st.caption(
    "Interactive analysis of urban flash-flood observations, "
    "reported impacts, spatial patterns, and NOAA event profiles "
    "across five Florida study areas."
)


# ============================================================
# MAIN TABS
# ============================================================

tab_overview, tab_kmeans = st.tabs(
    [
        "Source Overview",
        "K-Means Clusters",
    ]
)


# ============================================================
# TAB 1:
# NOAA + GROUNDSOURCE OVERVIEW
# ============================================================

with tab_overview:

    st.header(
        "NOAA and Groundsource Overview"
    )

    st.markdown(
        """
        This section compares two different flood-record sources.

        **NOAA Storm Events** provides flash-flood timing,
        duration, and reported impacts.

        **Groundsource** provides an independent set of spatial
        flood observations that can be used as a corroborating
        flood-record source.
        """
    )


    # ========================================================
    # FILTERS
    # ========================================================

    filter_col1, filter_col2, filter_col3 = (
        st.columns(3)
    )


    source_values = sorted(
        mapping[
            "source"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    with filter_col1:

        selected_sources = (
            st.multiselect(
                "Source",
                source_values,
                default=source_values,
                key=(
                    "overview_source_filter"
                ),
            )
        )


    city_values = sorted(
        mapping[
            "city"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    with filter_col2:

        selected_cities = (
            st.multiselect(
                "Study Area",
                city_values,
                default=city_values,
                key=(
                    "overview_city_filter"
                ),
            )
        )


    min_year = int(
        mapping[
            "year"
        ]
        .min()
    )

    max_year = int(
        mapping[
            "year"
        ]
        .max()
    )


    with filter_col3:

        year_range = (
            st.slider(
                "Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(
                    min_year,
                    max_year,
                ),
                key=(
                    "overview_year_filter"
                ),
            )
        )


    overview_filtered = (
        mapping[
            mapping[
                "source"
            ]
            .isin(
                selected_sources
            )
            &
            mapping[
                "city"
            ]
            .isin(
                selected_cities
            )
            &
            mapping[
                "year"
            ]
            .between(
                year_range[0],
                year_range[1],
            )
        ]
        .copy()
    )


    # ========================================================
    # METRICS
    # ========================================================

    event_count = len(
        overview_filtered
    )


    years_selected = max(
        1,
        (
            year_range[1]
            - year_range[0]
            + 1
        ),
    )


    avg_per_year = (
        event_count
        / years_selected
    )


    noaa_overview = (
        overview_filtered[
            overview_filtered[
                "source"
            ]
            == "NOAA Storm Events"
        ]
    )


    total_damage = (
        noaa_overview[
            "noaa_total_damage_usd"
        ]
        .fillna(0)
        .sum()
    )


    mean_damage = (
        noaa_overview[
            "noaa_total_damage_usd"
        ]
        .fillna(0)
        .mean()

        if len(
            noaa_overview
        ) > 0

        else 0
    )


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    metric1.metric(
        "Events / Episodes",
        f"{event_count:,}",
    )


    metric2.metric(
        "Average Per Year",
        f"{avg_per_year:.1f}",
    )


    metric3.metric(
        "NOAA Reported Damage",
        f"${total_damage:,.0f}",
    )


    metric4.metric(
        "Mean NOAA Damage / Event",
        f"${mean_damage:,.0f}",
    )


    # ========================================================
    # SPATIAL MAP
    # ========================================================

    st.subheader(
        "Spatial Distribution of Recorded Flood Observations"
    )


    map_data = (
        overview_filtered
        .dropna(
            subset=[
                "latitude",
                "longitude",
            ]
        )
    )


    if not map_data.empty:

        flood_map = folium.Map(

            location=[
                map_data[
                    "latitude"
                ]
                .mean(),

                map_data[
                    "longitude"
                ]
                .mean(),
            ],

            zoom_start=6,

            tiles="OpenStreetMap",
        )


        for _, row in (
            map_data.iterrows()
        ):

            source = (
                row[
                    "source"
                ]
            )


            marker_color = (
                SOURCE_COLORS.get(
                    source,
                    "#7f7f7f",
                )
            )


            popup = (
                f"<b>Source:</b> {source}<br>"
                f"<b>Study Area:</b> "
                f"{row.get('city', 'N/A')}<br>"
                f"<b>Date:</b> "
                f"{row['event_date'].date() if pd.notna(row['event_date']) else 'N/A'}"
            )


            if (
                source
                == "NOAA Storm Events"
            ):

                if pd.notna(
                    row.get(
                        "duration_hours",
                        np.nan,
                    )
                ):

                    popup += (
                        f"<br><b>Duration:</b> "
                        f"{row['duration_hours']:.2f} hours"
                    )


                if pd.notna(
                    row.get(
                        "noaa_total_damage_usd",
                        np.nan,
                    )
                ):

                    popup += (
                        f"<br><b>Reported Damage:</b> "
                        f"${row['noaa_total_damage_usd']:,.0f}"
                    )


            else:

                if pd.notna(
                    row.get(
                        "groundsource_area_max_km2",
                        np.nan,
                    )
                ):

                    popup += (
                        f"<br><b>Flooded Area:</b> "
                        f"{row['groundsource_area_max_km2']:.2f} km²"
                    )


            folium.CircleMarker(

                location=[
                    row[
                        "latitude"
                    ],

                    row[
                        "longitude"
                    ],
                ],

                radius=5,

                color=marker_color,

                fill=True,

                fill_color=(
                    marker_color
                ),

                fill_opacity=0.7,

                popup=(
                    folium.Popup(
                        popup,
                        max_width=350,
                    )
                ),

            ).add_to(
                flood_map
            )


        st_folium(
            flood_map,
            width="stretch",
            height=550,
            key=(
                "overview_map"
            ),
        )


    else:

        st.info(
            "No mapped records match the selected filters."
        )


    # ========================================================
    # FLOOD ACTIVITY OVER TIME
    #
    # NEW:
    # NOAA and Groundsource shown separately in a 1x2 layout.
    # ========================================================

    st.subheader(
        "Flood Activity Over Time"
    )


    annual_source = (
        overview_filtered
        .groupby(
            [
                "year",
                "source",
            ]
        )
        .size()
        .reset_index(
            name="record_count"
        )
    )


    noaa_annual = (
        annual_source[
            annual_source[
                "source"
            ]
            == "NOAA Storm Events"
        ]
        .copy()
    )


    groundsource_annual = (
        annual_source[
            annual_source[
                "source"
            ]
            == "Groundsource"
        ]
        .copy()
    )


    noaa_time_col, gs_time_col = (
        st.columns(2)
    )


    # --------------------------------------------------------
    # NOAA PANEL
    # --------------------------------------------------------

    with noaa_time_col:

        st.markdown(
            "#### NOAA Flash-Flood Events"
        )


        if not noaa_annual.empty:

            fig_noaa_year = (
                px.line(

                    noaa_annual,

                    x="year",

                    y="record_count",

                    markers=True,

                    labels={
                        "year":
                            "Year",

                        "record_count":
                            "NOAA Events",
                    },
                )
            )


            fig_noaa_year.update_traces(
                line=dict(
                    color=(
                        SOURCE_COLORS[
                            "NOAA Storm Events"
                        ]
                    )
                ),
                marker=dict(
                    color=(
                        SOURCE_COLORS[
                            "NOAA Storm Events"
                        ]
                    )
                ),
            )


            fig_noaa_year.update_layout(

                showlegend=False,

                margin=dict(
                    l=35,
                    r=15,
                    t=20,
                    b=35,
                ),
            )


            fig_noaa_year.update_yaxes(
                rangemode="tozero"
            )


            st.plotly_chart(
                fig_noaa_year,
                width="stretch",
                key=(
                    "overview_noaa_annual_chart"
                ),
            )


        else:

            st.info(
                "No NOAA records match the selected filters."
            )


    # --------------------------------------------------------
    # GROUNDSOURCE PANEL
    # --------------------------------------------------------

    with gs_time_col:

        st.markdown(
            "#### Groundsource Episodes"
        )


        if not groundsource_annual.empty:

            fig_groundsource_year = (
                px.line(

                    groundsource_annual,

                    x="year",

                    y="record_count",

                    markers=True,

                    labels={
                        "year":
                            "Year",

                        "record_count":
                            "Groundsource Episodes",
                    },
                )
            )


            fig_groundsource_year.update_traces(
                line=dict(
                    color=(
                        SOURCE_COLORS[
                            "Groundsource"
                        ]
                    )
                ),
                marker=dict(
                    color=(
                        SOURCE_COLORS[
                            "Groundsource"
                        ]
                    )
                ),
            )


            fig_groundsource_year.update_layout(

                showlegend=False,

                margin=dict(
                    l=35,
                    r=15,
                    t=20,
                    b=35,
                ),
            )


            fig_groundsource_year.update_yaxes(
                rangemode="tozero"
            )


            st.plotly_chart(
                fig_groundsource_year,
                width="stretch",
                key=(
                    "overview_groundsource_annual_chart"
                ),
            )


        else:

            st.info(
                "No Groundsource records match the selected filters."
            )


    # ========================================================
    # RECORDS BY STUDY AREA
    # ========================================================

    st.subheader(
        "Records by Study Area"
    )


    by_city = (
        overview_filtered
        .groupby(
            [
                "city",
                "source",
            ]
        )
        .size()
        .reset_index(
            name="record_count"
        )
    )


    if not by_city.empty:

        fig_city = px.bar(

            by_city,

            x="city",

            y="record_count",

            color="source",

            barmode="group",

            color_discrete_map=(
                SOURCE_COLORS
            ),

            labels={
                "city":
                    "Study Area",

                "record_count":
                    "Number of Records",

                "source":
                    "Source",
            },
        )


        st.plotly_chart(
            fig_city,
            width="stretch",
            key=(
                "overview_city_chart"
            ),
        )


# ============================================================
# TAB 2:
# K-MEANS CLUSTERS
# ============================================================

with tab_kmeans:

    st.header(
        "NOAA K-Means Event Clusters"
    )


    st.markdown(
        """
        K-means grouped NOAA flash-flood events using three
        conceptual characteristics:

        - **seasonal timing**
        - **flood duration**
        - **NOAA-reported financial damage**

        Duration and damage were log-transformed. Seasonal timing
        was represented cyclically using sine and cosine values
        derived from day of year. All clustering variables were
        standardized before fitting the model.
        """
    )


    # ========================================================
    # MODEL SELECTION
    # ========================================================

    st.subheader(
        "Model Selection"
    )


    k4_row = (
        model_selection[
            model_selection[
                "k"
            ]
            == 4
        ]
    )


    k5_row = (
        model_selection[
            model_selection[
                "k"
            ]
            == 5
        ]
    )


    k4_silhouette = (

        float(
            k4_row[
                "silhouette_score"
            ]
            .iloc[0]
        )

        if not k4_row.empty

        else np.nan
    )


    k5_silhouette = (

        float(
            k5_row[
                "silhouette_score"
            ]
            .iloc[0]
        )

        if not k5_row.empty

        else np.nan
    )


    m1, m2, m3 = (
        st.columns(3)
    )


    m1.metric(
        "k=4 Silhouette",
        f"{k4_silhouette:.3f}",
    )


    m2.metric(
        "k=5 Silhouette",
        f"{k5_silhouette:.3f}",
    )


    m3.metric(
        "Primary Solution",
        "k = 5",
    )


    st.caption(
        "The five-cluster solution is used as the primary "
        "exploratory interpretation. Model-selection statistics "
        "are used alongside cluster size and interpretability."
    )


    # ========================================================
    # MODEL-SELECTION CURVES
    # ========================================================

    selection_col1, selection_col2 = (
        st.columns(2)
    )


    with selection_col1:

        st.markdown(
            "#### Elbow"
        )


        fig_elbow = px.line(

            model_selection,

            x="k",

            y="inertia",

            markers=True,

            labels={
                "k":
                    "Number of Clusters (k)",

                "inertia":
                    "Within-Cluster Sum of Squares",
            },
        )


        fig_elbow.update_xaxes(
            dtick=1
        )


        fig_elbow.update_layout(
            showlegend=False
        )


        st.plotly_chart(
            fig_elbow,
            width="stretch",
            key=(
                "kmeans_elbow_chart"
            ),
        )


    with selection_col2:

        st.markdown(
            "#### Silhouette"
        )


        fig_silhouette = px.line(

            model_selection,

            x="k",

            y="silhouette_score",

            markers=True,

            labels={
                "k":
                    "Number of Clusters (k)",

                "silhouette_score":
                    "Silhouette Score",
            },
        )


        fig_silhouette.update_xaxes(
            dtick=1
        )


        fig_silhouette.update_layout(
            showlegend=False
        )


        st.plotly_chart(
            fig_silhouette,
            width="stretch",
            key=(
                "kmeans_silhouette_chart"
            ),
        )


    # ========================================================
    # CLUSTER INTERPRETATION GUIDE
    # ========================================================

    with st.expander(
        "Cluster interpretation guide"
    ):

        cluster_guide = (
            pd.DataFrame(
                {
                    "Cluster": [
                        CLUSTER_LABELS[c]
                        for c
                        in CLUSTER_ORDER
                    ],

                    "Interpretation": [
                        PROFILE_DESCRIPTIONS[c]
                        for c
                        in CLUSTER_ORDER
                    ],
                }
            )
        )


        st.dataframe(
            cluster_guide,
            width="stretch",
            hide_index=True,
        )


        st.caption(
            "These descriptions were assigned after clustering. "
            "They were not predefined categories used to assign events."
        )


    # ========================================================
    # FILTERS
    # ========================================================

    filter1, filter2, filter3 = (
        st.columns(3)
    )


    with filter1:

        selected_clusters = (
            st.multiselect(
                "Cluster",
                cluster_category_order(),
                default=(
                    cluster_category_order()
                ),
                key=(
                    "kmeans_cluster_filter"
                ),
            )
        )


    kmeans_city_values = sorted(
        kmeans[
            "city"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    with filter2:

        selected_kmeans_cities = (
            st.multiselect(
                "Study Area",
                kmeans_city_values,
                default=(
                    kmeans_city_values
                ),
                key=(
                    "kmeans_city_filter"
                ),
            )
        )


    kmeans_min_year = int(
        kmeans[
            "year"
        ]
        .min()
    )


    kmeans_max_year = int(
        kmeans[
            "year"
        ]
        .max()
    )


    with filter3:

        kmeans_year_range = (
            st.slider(
                "K-Means Year Range",
                min_value=(
                    kmeans_min_year
                ),
                max_value=(
                    kmeans_max_year
                ),
                value=(
                    kmeans_min_year,
                    kmeans_max_year,
                ),
                key=(
                    "kmeans_year_filter"
                ),
            )
        )


    cluster_filtered = (
        kmeans[
            kmeans[
                "cluster_label"
            ]
            .isin(
                selected_clusters
            )
            &
            kmeans[
                "city"
            ]
            .isin(
                selected_kmeans_cities
            )
            &
            kmeans[
                "year"
            ]
            .between(
                kmeans_year_range[0],
                kmeans_year_range[1],
            )
        ]
        .copy()
    )


    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    event_n = len(
        cluster_filtered
    )


    median_duration = (

        cluster_filtered[
            "duration_hours"
        ]
        .median()

        if event_n > 0

        else np.nan
    )


    percent_damaging = (

        100
        * cluster_filtered[
            "damaging_event"
        ]
        .mean()

        if event_n > 0

        else np.nan
    )


    damaging_filtered = (
        cluster_filtered[
            cluster_filtered[
                "noaa_total_damage_usd"
            ]
            > 0
        ]
    )


    damaging_median = (

        damaging_filtered[
            "noaa_total_damage_usd"
        ]
        .median()

        if len(
            damaging_filtered
        ) > 0

        else np.nan
    )


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    metric1.metric(
        "NOAA Events",
        f"{event_n:,}",
    )


    metric2.metric(
        "Median Duration",
        (
            f"{median_duration:.2f} hr"
            if pd.notna(
                median_duration
            )
            else "N/A"
        ),
    )


    metric3.metric(
        "Events With Damage",
        (
            f"{percent_damaging:.1f}%"
            if pd.notna(
                percent_damaging
            )
            else "N/A"
        ),
    )


    metric4.metric(
        "Median Damage When Damaging",
        (
            f"${damaging_median:,.0f}"
            if pd.notna(
                damaging_median
            )
            else "N/A"
        ),
    )


    if cluster_filtered.empty:

        st.warning(
            "No NOAA cluster events match the selected filters."
        )


    else:

        # ====================================================
        # EVENT COUNT
        # ====================================================

        st.subheader(
            "NOAA Events by Cluster"
        )


        cluster_counts = (
            cluster_filtered
            .groupby(
                [
                    "cluster",
                    "cluster_label",
                ]
            )
            .size()
            .reset_index(
                name="event_count"
            )
        )


        fig_counts = px.bar(

            cluster_counts,

            x="cluster_label",

            y="event_count",

            color="cluster_label",

            color_discrete_map=(
                PLOTLY_CLUSTER_COLORS
            ),

            category_orders={
                "cluster_label":
                    cluster_category_order()
            },

            labels={
                "cluster_label":
                    "K-Means Cluster",

                "event_count":
                    "Number of NOAA Events",
            },
        )


        fig_counts.update_layout(
            showlegend=False
        )


        st.plotly_chart(
            fig_counts,
            width="stretch",
            key=(
                "kmeans_counts_chart"
            ),
        )


        # ====================================================
        # LOG DURATION VS LOG DAMAGE
        # ====================================================

        st.subheader(
            "Duration vs. Reported Damage"
        )


        fig_scatter = px.scatter(

            cluster_filtered,

            x="duration_log",

            y="damage_log",

            color="cluster_label",

            color_discrete_map=(
                PLOTLY_CLUSTER_COLORS
            ),

            category_orders={
                "cluster_label":
                    cluster_category_order()
            },

            hover_data={

                "cluster_label":
                    True,

                "city":
                    True,

                "event_date":
                    True,

                "duration_hours":
                    ":.2f",

                "noaa_total_damage_usd":
                    ":,.0f",

                "duration_log":
                    False,

                "damage_log":
                    False,
            },

            labels={

                "duration_log":
                    "log(1 + Flood Duration [hours])",

                "damage_log":
                    "log(1 + NOAA-Reported Damage [USD])",

                "cluster_label":
                    "Cluster",
            },
        )


        fig_scatter = (
            apply_cluster_layout(
                fig_scatter
            )
        )


        st.plotly_chart(
            fig_scatter,
            width="stretch",
            key=(
                "kmeans_scatter_chart"
            ),
        )


        # ====================================================
        # LOG-DURATION DISTRIBUTION
        #
        # NEW:
        # Uses duration_log rather than raw duration_hours.
        # ====================================================

        st.subheader(
            "Flood Duration Distribution"
        )


        fig_duration_box = px.box(

            cluster_filtered,

            x="cluster_label",

            y="duration_log",

            color="cluster_label",

            color_discrete_map=(
                PLOTLY_CLUSTER_COLORS
            ),

            category_orders={
                "cluster_label":
                    cluster_category_order()
            },

            points="outliers",

            labels={

                "cluster_label":
                    "K-Means Cluster",

                "duration_log":
                    "log(1 + Flood Duration [hours])",
            },
        )


        fig_duration_box.update_layout(
            showlegend=False
        )


        st.plotly_chart(
            fig_duration_box,
            width="stretch",
            key=(
                "kmeans_duration_box"
            ),
        )


        # ====================================================
        # DAMAGE DISTRIBUTION
        # ====================================================

        st.subheader(
            "Reported Damage Distribution"
        )


        fig_damage_box = px.box(

            cluster_filtered,

            x="cluster_label",

            y="damage_log",

            color="cluster_label",

            color_discrete_map=(
                PLOTLY_CLUSTER_COLORS
            ),

            category_orders={
                "cluster_label":
                    cluster_category_order()
            },

            points="outliers",

            labels={

                "cluster_label":
                    "K-Means Cluster",

                "damage_log":
                    "log(1 + NOAA-Reported Damage [USD])",
            },
        )


        fig_damage_box.update_layout(
            showlegend=False
        )


        st.plotly_chart(
            fig_damage_box,
            width="stretch",
            key=(
                "kmeans_damage_box"
            ),
        )


        # ====================================================
        # MONTHLY DISTRIBUTION
        # ====================================================

        st.subheader(
            "Monthly Distribution by Cluster"
        )


        monthly = (
            cluster_filtered
            .groupby(
                [
                    "month",
                    "cluster_label",
                ]
            )
            .size()
            .reset_index(
                name="event_count"
            )
        )


        full_month_cluster = (
            pd.MultiIndex
            .from_product(
                [
                    range(
                        1,
                        13
                    ),
                    selected_clusters,
                ],
                names=[
                    "month",
                    "cluster_label",
                ],
            )
            .to_frame(
                index=False
            )
        )


        monthly = (
            full_month_cluster
            .merge(
                monthly,
                on=[
                    "month",
                    "cluster_label",
                ],
                how="left",
            )
        )


        monthly[
            "event_count"
        ] = (
            monthly[
                "event_count"
            ]
            .fillna(0)
        )


        fig_month = px.line(

            monthly,

            x="month",

            y="event_count",

            color="cluster_label",

            markers=True,

            color_discrete_map=(
                PLOTLY_CLUSTER_COLORS
            ),

            category_orders={
                "cluster_label":
                    cluster_category_order()
            },

            labels={
                "month":
                    "Month",

                "event_count":
                    "Number of NOAA Events",

                "cluster_label":
                    "Cluster",
            },
        )


        fig_month.update_xaxes(
            tickmode="linear",
            dtick=1,
        )


        fig_month = (
            apply_cluster_layout(
                fig_month
            )
        )


        st.plotly_chart(
            fig_month,
            width="stretch",
            key=(
                "kmeans_month_chart"
            ),
        )


        # ====================================================
        # PERCENT DAMAGING
        # ====================================================

        st.subheader(
            "Percentage of Events With Reported Damage"
        )


        percent_damage = (
            cluster_filtered
            .groupby(
                [
                    "cluster",
                    "cluster_label",
                ]
            )[
                "damaging_event"
            ]
            .mean()
            .mul(100)
            .reset_index(
                name="percent_damaging"
            )
        )


        fig_percent = px.bar(

            percent_damage,

            x="cluster_label",

            y="percent_damaging",

            color="cluster_label",

            color_discrete_map=(
                PLOTLY_CLUSTER_COLORS
            ),

            category_orders={
                "cluster_label":
                    cluster_category_order()
            },

            labels={

                "cluster_label":
                    "K-Means Cluster",

                "percent_damaging":
                    "Events With Reported Damage (%)",
            },
        )


        fig_percent.update_layout(
            showlegend=False
        )


        fig_percent.update_yaxes(
            range=[
                0,
                100,
            ]
        )


        st.plotly_chart(
            fig_percent,
            width="stretch",
            key=(
                "kmeans_percent_chart"
            ),
        )


        # ====================================================
        # MEDIAN DURATION
        # ====================================================

        st.subheader(
            "Median Flood Duration"
        )


        median_duration_cluster = (
            cluster_filtered
            .groupby(
                [
                    "cluster",
                    "cluster_label",
                ]
            )[
                "duration_hours"
            ]
            .median()
            .reset_index(
                name="median_duration"
            )
        )


        fig_median_duration = px.bar(

            median_duration_cluster,

            x="cluster_label",

            y="median_duration",

            color="cluster_label",

            color_discrete_map=(
                PLOTLY_CLUSTER_COLORS
            ),

            category_orders={
                "cluster_label":
                    cluster_category_order()
            },

            labels={

                "cluster_label":
                    "K-Means Cluster",

                "median_duration":
                    "Median Duration (hours)",
            },
        )


        fig_median_duration.update_layout(
            showlegend=False
        )


        st.plotly_chart(
            fig_median_duration,
            width="stretch",
            key=(
                "kmeans_median_duration_chart"
            ),
        )


        # ====================================================
        # MEDIAN DAMAGE
        # ====================================================

        st.subheader(
            "Median NOAA-Reported Damage"
        )


        median_damage_cluster = (
            cluster_filtered
            .groupby(
                [
                    "cluster",
                    "cluster_label",
                ]
            )[
                "noaa_total_damage_usd"
            ]
            .median()
            .reset_index(
                name="median_damage"
            )
        )


        fig_median_damage = px.bar(

            median_damage_cluster,

            x="cluster_label",

            y="median_damage",

            color="cluster_label",

            color_discrete_map=(
                PLOTLY_CLUSTER_COLORS
            ),

            category_orders={
                "cluster_label":
                    cluster_category_order()
            },

            labels={

                "cluster_label":
                    "K-Means Cluster",

                "median_damage":
                    "Median NOAA-Reported Damage (USD)",
            },
        )


        fig_median_damage.update_layout(
            showlegend=False
        )


        st.plotly_chart(
            fig_median_damage,
            width="stretch",
            key=(
                "kmeans_median_damage_chart"
            ),
        )


        # ====================================================
        # ANNUAL STACKED BAR
        # ====================================================

        st.subheader(
            "Annual Frequency of NOAA Event Clusters"
        )


        annual_cluster = (
            cluster_filtered
            .groupby(
                [
                    "year",
                    "cluster_label",
                ]
            )
            .size()
            .reset_index(
                name="event_count"
            )
        )


        full_year_cluster = (
            pd.MultiIndex
            .from_product(
                [
                    range(
                        kmeans_year_range[0],
                        kmeans_year_range[1]
                        + 1,
                    ),
                    selected_clusters,
                ],
                names=[
                    "year",
                    "cluster_label",
                ],
            )
            .to_frame(
                index=False
            )
        )


        annual_cluster = (
            full_year_cluster
            .merge(
                annual_cluster,
                on=[
                    "year",
                    "cluster_label",
                ],
                how="left",
            )
        )


        annual_cluster[
            "event_count"
        ] = (
            annual_cluster[
                "event_count"
            ]
            .fillna(0)
        )


        fig_annual = px.bar(

            annual_cluster,

            x="year",

            y="event_count",

            color="cluster_label",

            barmode="stack",

            color_discrete_map=(
                PLOTLY_CLUSTER_COLORS
            ),

            category_orders={
                "cluster_label":
                    cluster_category_order()
            },

            labels={
                "year":
                    "Year",

                "event_count":
                    "Number of NOAA Events",

                "cluster_label":
                    "Cluster",
            },
        )


        fig_annual = (
            apply_cluster_layout(
                fig_annual
            )
        )


        st.plotly_chart(
            fig_annual,
            width="stretch",
            key=(
                "kmeans_annual_stacked_chart"
            ),
        )


        # ====================================================
        # DAMAGE AMONG DAMAGING EVENTS
        # ====================================================

        st.subheader(
            "Median Damage Among Damaging Events"
        )


        damaging_only = (
            cluster_filtered[
                cluster_filtered[
                    "noaa_total_damage_usd"
                ]
                > 0
            ]
        )


        damaging_median_cluster = (
            damaging_only
            .groupby(
                [
                    "cluster",
                    "cluster_label",
                ]
            )[
                "noaa_total_damage_usd"
            ]
            .median()
            .reset_index(
                name=(
                    "median_damaging_damage"
                )
            )
        )


        if not (
            damaging_median_cluster.empty
        ):

            fig_damage_only = (
                px.bar(

                    damaging_median_cluster,

                    x="cluster_label",

                    y=(
                        "median_damaging_damage"
                    ),

                    color="cluster_label",

                    color_discrete_map=(
                        PLOTLY_CLUSTER_COLORS
                    ),

                    category_orders={
                        "cluster_label":
                            cluster_category_order()
                    },

                    labels={

                        "cluster_label":
                            "K-Means Cluster",

                        "median_damaging_damage":
                            "Median Damage Among Damaging Events (USD)",
                    },
                )
            )


            fig_damage_only.update_layout(
                showlegend=False
            )


            st.plotly_chart(
                fig_damage_only,
                width="stretch",
                key=(
                    "kmeans_damage_only_chart"
                ),
            )


        # ====================================================
        # 3D CLUSTER VIEW
        # ====================================================

        st.subheader(
            "Interactive 3D Cluster View"
        )


        st.caption(
            "Conceptually, the clustering compares timing, "
            "duration, and reported damage. This visualization "
            "uses log-transformed duration, log-transformed damage, "
            "and day of year."
        )


        fig_3d = px.scatter_3d(

            cluster_filtered,

            x="duration_log",

            y="damage_log",

            z="day_of_year",

            color="cluster_label",

            color_discrete_map=(
                PLOTLY_CLUSTER_COLORS
            ),

            category_orders={
                "cluster_label":
                    cluster_category_order()
            },

            hover_data={

                "cluster_label":
                    True,

                "city":
                    True,

                "event_date":
                    True,

                "duration_hours":
                    ":.2f",

                "noaa_total_damage_usd":
                    ":,.0f",

                "duration_log":
                    False,

                "damage_log":
                    False,

                "day_of_year":
                    True,
            },

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
        )


        fig_3d.update_layout(
            legend_title_text=(
                "Cluster"
            )
        )


        st.plotly_chart(
            fig_3d,
            width="stretch",
            key=(
                "kmeans_3d_chart"
            ),
        )


        # ====================================================
        # SPATIAL CLUSTER MAP
        # ====================================================

        st.subheader(
            "Spatial Distribution of NOAA Clusters"
        )


        cluster_map_data = (
            cluster_filtered
            .dropna(
                subset=[
                    "latitude",
                    "longitude",
                ]
            )
        )


        if not (
            cluster_map_data.empty
        ):

            cluster_map = folium.Map(

                location=[
                    cluster_map_data[
                        "latitude"
                    ]
                    .mean(),

                    cluster_map_data[
                        "longitude"
                    ]
                    .mean(),
                ],

                zoom_start=6,

                tiles="OpenStreetMap",
            )


            for _, row in (
                cluster_map_data
                .iterrows()
            ):

                cluster_id = int(
                    row[
                        "cluster"
                    ]
                )


                marker_color = (
                    CLUSTER_COLORS[
                        cluster_id
                    ]
                )


                popup = (
                    f"<b>{CLUSTER_LABELS[cluster_id]}</b><br>"
                    f"<b>Study Area:</b> "
                    f"{row.get('city', 'N/A')}<br>"
                    f"<b>Date:</b> "
                    f"{row['event_date'].date() if pd.notna(row['event_date']) else 'N/A'}<br>"
                    f"<b>Duration:</b> "
                    f"{row['duration_hours']:.2f} hours<br>"
                    f"<b>Reported Damage:</b> "
                    f"${row['noaa_total_damage_usd']:,.0f}"
                )


                folium.CircleMarker(

                    location=[
                        row[
                            "latitude"
                        ],

                        row[
                            "longitude"
                        ],
                    ],

                    radius=6,

                    color=(
                        marker_color
                    ),

                    fill=True,

                    fill_color=(
                        marker_color
                    ),

                    fill_opacity=0.75,

                    popup=(
                        folium.Popup(
                            popup,
                            max_width=350,
                        )
                    ),

                ).add_to(
                    cluster_map
                )


            st_folium(
                cluster_map,
                width="stretch",
                height=550,
                key=(
                    "kmeans_cluster_map"
                ),
            )


        # ====================================================
        # RAW DATA TABLE
        # ====================================================

        with st.expander(
            "View filtered NOAA cluster records"
        ):

            table_columns = [
                "cluster_label",
                "profile_description",
                "city",
                "event_date",
                "duration_hours",
                "duration_log",
                "noaa_total_damage_usd",
                "damage_log",
                "month",
                "year",
            ]


            available_columns = [
                col
                for col
                in table_columns
                if col
                in cluster_filtered.columns
            ]


            raw_table = (
                cluster_filtered[
                    available_columns
                ]
                .sort_values(
                    [
                        "cluster_label",
                        "event_date",
                    ]
                )
            )


            st.dataframe(
                raw_table,
                width="stretch",
                hide_index=True,
            )