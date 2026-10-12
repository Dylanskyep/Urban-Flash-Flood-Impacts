# Urban-Flash-Flood-Impacts

Researching the impacts of flash floods across five urban Florida study areas: Miami, Tampa, Orlando, Gainesville, and Jacksonville.

## Overview

This project compares NOAA Storm Events records with Groundsource flood observations to study flash-flood frequency, duration, spatial correspondence, and NOAA-reported financial impacts.

NOAA is treated as the primary source for official flash-flood characteristics and impacts, while Groundsource serves as an independent observational source used to corroborate likely corresponding flood episodes.

The main relationships examined are:

- flood duration vs. reported financial damage;
- annual flood frequency vs. annual reported damage;
- damaging vs. non-damaging events;
- NOAA-Groundsource match coverage; and
- sensitivity to stricter versus broader matching criteria.

## Study Areas

| Study Area | County |
|---|---|
| Miami | Miami-Dade County |
| Tampa | Hillsborough County |
| Orlando | Orange County |
| Gainesville | Alachua County |
| Jacksonville | Duval County |

Groundsource observations are spatially filtered using official county polygons. City names are used as study-area labels, while the corresponding county is the geographic filtering unit.

## Research Questions

1. How do NOAA and Groundsource differ in recorded flood frequency and timing?
2. What proportion of NOAA flash-flood events can be corroborated by Groundsource?
3. How does match coverage change between High-confidence and High + Medium criteria?
4. How is flood duration related to NOAA-reported financial damage?
5. Do years with more matched floods also have greater total damage?
6. Are damaging events generally longer than events with no reported damage?
7. How do these relationships differ across the five study areas?

## Data Sources

### NOAA Storm Events

NOAA Storm Events is the primary source for official flash-flood events and impacts.

The analysis retains records where:

- `EVENT_TYPE = Flash Flood`
- `CZ_TYPE = C`

NOAA fields used include:

- event begin and end time;
- coordinates;
- property damage;
- crop damage;
- total financial damage;
- deaths; and
- injuries.

NOAA impacts remain NOAA-reported attributes throughout the analysis.

### Groundsource

Groundsource provides flood observations containing:

- start and end dates;
- geometry;
- flooded area; and
- event identifiers.

Nearby Groundsource records are grouped into candidate flood episodes before matching to NOAA.

## Processing Workflow

The final analysis uses two Python scripts.

### 1. Build and Harmonize Event Records

`/scripts/01_build_harmonized_events.py`

This script:

- loads NOAA and Groundsource data;
- filters NOAA to Flash Flood events;
- filters Groundsource using county polygons;
- calculates event duration;
- preserves NOAA impact fields;
- groups Groundsource records into candidate episodes;
- restricts data to overlapping study years; and
- creates harmonized analysis files.

Main matching input:

`14_harmonized_overlap_events.csv`

### 2. Match NOAA and Groundsource Events

`/scripts/02_match_noaa_groundsource.py`

A candidate match must:

- occur in the same study area;
- be within 10 km; and
- have overlapping intervals or a temporal gap of no more than 24 hours.

Global one-to-one optimization is used so that a record cannot be assigned to multiple counterparts.

The match score is a heuristic ranking score, not a probability.

## Match Confidence

### High Confidence

A match is High confidence when:

- distance ≤ 5 km; and
- intervals overlap or temporal gap ≤ 12 hours.

### Medium Confidence

A match is Medium confidence when it satisfies the broader accepted limits:

- distance ≤ 10 km; and
- temporal gap ≤ 24 hours;

but does not meet the stricter High-confidence definition.

High-confidence matches are used for the primary impact analysis. High + Medium matches are used for sensitivity analysis.

## Main Analytical Relationships

### NOAA vs. Groundsource Event Frequency

Annual and monthly counts are compared to examine differences in recorded flood activity and dataset coverage.

### NOAA-Groundsource Corroboration

The analysis compares:

- High-confidence match percentage; and
- High + Medium match percentage.

This shows the tradeoff between stricter match certainty and broader coverage.

### Flood Duration vs. Financial Damage

High-confidence NOAA events are used to compare duration with NOAA-reported financial damage through:

- duration-damage plots;
- median damage by duration category;
- damaging vs. non-damaging events; and
- duration distributions by damage status.

### Annual Flood Frequency vs. Annual Damage

Annual matched-event counts are compared with total annual NOAA-reported damage to test whether more active flood years also produce greater losses.

## Current Findings

The current analysis shows that:

- NOAA and Groundsource record substantially different numbers of flood observations.
- High + Medium criteria produce greater match coverage than High confidence alone.
- High-confidence matches provide a smaller, more conservative impact-analysis sample.
- Match coverage varies across study areas.
- Miami contributes much of the reported financial damage in the High-confidence sample.
- Damage is strongly right-skewed, with a small number of large-loss events dominating totals.
- Longer events can produce greater damage, but duration alone does not show a strong consistent relationship.
- More matched events in a year do not necessarily mean greater total annual damage.
- No direct or indirect deaths or injuries were reported among the current High-confidence matched NOAA events.

These findings describe associations and should not be interpreted as causal relationships.

## Figures

Figures are organized by analytical purpose.

### `figures/temporal/`

- Annual Matched Flash-Flood Frequency vs NOAA-Reported Financial Damage
- Annual NOAA-Reported Financial Damage for High-Confidence Matched Flood Events
- Number of High-Confidence Matched NOAA Flood Events by Year

### `figures/matching/`

- High vs Medium Matches by County
- Percentage of NOAA Flash-Flood Events Corroborated by Groundsource by Match Confidence

### `figures/impacts/`

- High-Confidence Matched Flood Events With and Without NOAA-Reported Financial Damage
- Median NOAA-Reported Damage by Flood Duration Category
- Median NOAA-Reported Property Damage by County for High-Confidence Matched Events
- NOAA Flood Duration for High-Confidence Events With and Without Reported Financial Damage
- NOAA Flood Duration vs. Reported Financial Damage for High-Confidence Matched Events
- NOAA-Reported Flood Damage by County for High-Confidence Matches
- Number of High-Confidence Matched NOAA Events by Flood Duration Category

### `figures/sensitivity/`

- NOAA Flood Duration vs Reported Damage for High- and Medium-Confidence Matched Events
- NOAA Flood Duration vs. Reported Damage for High- and Medium-Confidence Matches

## Repository Structure

```text
Urban-Flash-Flood-Impacts/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 01_build_harmonized_events.py
│   └── 02_match_noaa_groundsource.py
├── data/
│   ├── README.md
│   └── processed/
├── figures/
│   ├── temporal/
│   ├── matching/
│   ├── impacts/
│   └── sensitivity/
├── dashboard/
└── archive/
    └── older_scripts/
