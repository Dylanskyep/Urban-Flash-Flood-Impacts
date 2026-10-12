# Data 

This folder contains the processed datasets used in the Urban Flash-Flood Impacts project.

The repository keeps the main harmonized, matched, and graph-ready files rather than every intermediate output produced during development.

## Data Sources

### NOAA Storm Events

NOAA provides the official flash-flood event records used for:

- event timing;
- coordinates;
- duration;
- property damage;
- crop damage;
- deaths; and
- injuries.

Only NOAA records classified as Flash Flood events with county designation are retained.

### Groundsource

Groundsource provides independent flood observations containing:

- start and end dates;
- geometry;
- flooded area; and
- event identifiers.

Nearby Groundsource observations are grouped into candidate flood episodes before matching to NOAA.

## Processed Files

### `14_harmonized_overlap_events.csv`

Main harmonized dataset used as input to the NOAA-Groundsource matcher.

Contains NOAA flash-flood events and Groundsource candidate flood episodes during overlapping study years.

Used for:

- event matching;
- source comparisons;
- duration analysis; and
- future spatial analysis.

### `24_matched_event_pairs_enriched.csv`

Contains all accepted NOAA-Groundsource matched pairs.

Includes both:

- High-confidence matches;
- Medium-confidence matches.

Useful for:

- match sensitivity analysis;
- spatial distance;
- temporal gap;
- paired NOAA and Groundsource event attributes; and
- future matched-event maps.

### `25_groundsource_with_noaa_impacts.csv`

Contains the Groundsource candidate flood episodes with matching information added.

For matched Groundsource episodes, NOAA-reported impact attributes are linked to the likely corresponding episode.

Typical added fields include:

- whether the Groundsource episode matched NOAA;
- match ID;
- match confidence;
- match score;
- temporal gap;
- match distance;
- NOAA property damage;
- NOAA crop damage;
- NOAA total damage;
- NOAA deaths; and
- NOAA injuries.

NOAA impact values in this file remain NOAA-reported attributes linked to likely corresponding Groundsource episodes. They are not Groundsource-reported impacts.

### `26_noaa_event_match_status.csv`

Contains NOAA flash-flood events with their matching status.

Used to identify:

- whether a NOAA event has usable coordinates;
- whether it matched a Groundsource episode;
- match ID;
- match confidence;
- match score;
- temporal gap; and
- match distance.

This file is useful for evaluating matched versus unmatched NOAA events and calculating match rates.

### `27_match_summary_by_county.csv`

Summarizes matching results by study area.

Includes:

- Groundsource episode count;
- NOAA event count;
- NOAA events with usable coordinates;
- matched pairs;
- High-confidence matches;
- Medium-confidence matches;
- unmatched events; and
- overall NOAA match rate.

### `28_high_confidence_matched_events.csv`

Primary event-level dataset for the main impact analysis.

Contains only High-confidence matched NOAA-Groundsource pairs.

Used for:

- flood duration vs. damage;
- county damage comparisons;
- event-level impact analysis; and
- High-confidence temporal summaries.

### `32_noaa_damage_by_duration_bin.csv`

Summarizes NOAA-reported damage by flood-duration category.

Duration categories:

- `<1 hour`
- `1–3 hours`
- `3–6 hours`
- `6–12 hours`
- `>12 hours`

Includes:

- event count;
- events with non-missing damage;
- mean damage;
- median damage;
- total damage; and
- maximum damage.

### `33_match_confidence_by_county.csv`

Contains the number of High- and Medium-confidence matches by study area.

Used for the High vs Medium match-count figure.

### `34_noaa_damage_by_year_high_confidence.csv`

Annual summary of High-confidence matched NOAA events.

Includes:

- year;
- event count;
- total NOAA-reported damage;
- mean NOAA-reported damage; and
- median NOAA-reported damage.

Used for:

- annual event counts;
- annual damage trends; and
- annual frequency vs. damage analysis.

### `35_match_rate_by_confidence.csv`

Compares NOAA corroboration rates using:

- High confidence only;
- High + Medium confidence.

The denominator is NOAA events with usable coordinates because spatial matching requires valid NOAA coordinates.

### `36_damage_status_by_county_high_confidence.csv`

Counts High-confidence matched events by study area and damage status.

Damage categories:

- `Reported damage`
- `No reported damage`

Events with missing NOAA damage values are excluded rather than treated as zero.

### `37_duration_by_damage_status_high_confidence.csv`

Event-level High-confidence data used to compare flood duration between:

- events with reported financial damage; and
- events with no reported financial damage.

Used for duration-distribution and box-plot analysis.

## Match Confidence

### High Confidence

A matched pair is High confidence when:

- distance ≤ 5 km; and
- event intervals overlap or temporal gap ≤ 12 hours.

### Medium Confidence

A matched pair is Medium confidence when it satisfies:

- distance ≤ 10 km; and
- temporal gap ≤ 24 hours;

but does not meet the stricter High-confidence criteria.

High-confidence matches are used for the main impact analysis.

High + Medium matches are used for sensitivity analysis.

## Data Interpretation

### NOAA-Reported Impacts

Financial damage, deaths, and injuries remain NOAA-reported attributes.

They should not be described as Groundsource-reported impacts.

### Missing vs. Zero Damage

A missing damage value is not the same as zero reported damage.

Missing values are kept separate from explicit zero-damage records where possible.

### Study-Area Labels

The `city` field is used as a study-area label.

The corresponding county is the geographic filtering unit:

| Study Area | County |
|---|---|
| Miami | Miami-Dade |
| Tampa | Hillsborough |
| Orlando | Orange |
| Gainesville | Alachua |
| Jacksonville | Duval |

## Regenerating the Data

Processed files are generated by:

```bash
python scripts/01_build_harmonized_events.py
python scripts/02_match_noaa_groundsource.py
