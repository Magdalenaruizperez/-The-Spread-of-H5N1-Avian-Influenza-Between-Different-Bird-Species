#  Avian Influenza GISAID Pipeline

A bioinformatics pipeline for analysing Avian Influenza (H5N1) outbreak data sourced from [GISAID](https://www.gisaid.org/). The pipeline cleans and standardises raw isolate data, then branches into geospatial visualisation, migration analysis, and network-based epidemiology.

---

## 📁 Repository Structure

```
├── Gisaid.py                            # Entry point — data ingestion & cleaning
├── data_quality_check.py                # Independent data quality audit
├── choropleth.py                        # Choropleth world map of outbreaks
├── heatmaps.py                          # Host species heatmap per country/year
├── migration_summary.py                 # Classifies hosts as migratory/non-migratory
├── migration_analysis.py                # Migration trend visualisations
├── migration_route_plotter.py           # Arrow-based migration route map
├── host_location.py                     # Host–country table with migratory status
├── network.py                           # Bipartite species–country network analysis
├── countries_manual_correction_mapping.txt   # Country name correction dictionary
├── species_mapping.txt                  # Species name standardisation dictionary
├── migration_paths_dictionary.txt       # Known migration routes per species
└── world_shapefile/
    └── ne_110m_admin_0_countries.shp    # Natural Earth world shapefile
```

---

##  Pipeline Execution Order

The pipeline has a single entry point and then splits into parallel tracks. Follow the order below:

### Step 1 — Data Ingestion & Cleaning
**Script:** `Gisaid.py`  
**Input:** `gisaid_epiflu_isolates_full_period.csv` (raw GISAID download-available in this repository)  
**Outputs:**
- `Animal_period.csv` — host species counts per year
- `host_appearances.csv` — host counts per country per year
- `flock_size_FP_outbreak.csv` — flock size per location and collection date

This script standardises `Location` fields to continent/country format, applies country name corrections, and maps raw species names to standardised common names.

---

### Step 2 — Data Quality Audit *(independent, run anytime)*
**Script:** `data_quality_check.py`  
**Input:** `gisaid_epiflu_isolates_full_period.csv`  
**Outputs:**
- `Country_Reported_Values.csv`
- `Summary_Country_Submissions.csv`

Audits which countries submitted `Animal_Specimen_Source`, `Animal_Health_Status`, and `Domestic_Status` fields. Does not feed into downstream scripts.

---

### Step 3 — Visualisation (run after Step 1)

**Choropleth map**  
**Script:** `choropleth.py`  
**Input:** `flock_size_FP_outbreak.csv`, `ne_110m_admin_0_countries.shp`  
**Output:** Choropleth PNG of outbreak flock size by country

**Heatmap**  
**Script:** `heatmaps.py`  
**Input:** `host_appearances.csv`  
**Output:** Heatmap PNG of host species per country for a selected year (edit `selected_year` in script)

---

### Step 4 — Migration Summary (run after Step 1)
**Script:** `migration_summary.py`  
**Input:** `Animal_period.csv`  
**Outputs:**
- `migratory_vs_nonmigratory_summary.csv`
- `migratory_detailed.csv`
- `nonmigratory_detailed.csv`
- `Animals_period_with_migration.csv`
- `species_per_year_list.csv`

Classifies each host species as migratory or non-migratory based on a predefined list, then calculates yearly outbreak counts and percentages for each group.

---

### Step 5 — Migration Analysis (run after Step 4)
**Script:** `migration_analysis.py`  
**Inputs:** `migratory_vs_nonmigratory_summary.csv`, `migratory_detailed.csv`, `nonmigratory_detailed.csv`  
**Outputs:** Bar plots, trend lines, species activity heatmaps, and cumulative sum plots comparing migratory vs non-migratory outbreak contributions over time.

---

### Step 6 — Host–Location Table (run after Step 1)
**Script:** `host_location.py`  
**Input:** `gisaid_epiflu_isolates_full_period.csv` (re-cleans raw data independently)  
**Output:** `host_location.csv` — one row per host–country pair with migratory status

Also produces inline analyses: species geographic spread bar charts, country species richness charts, and a species similarity heatmap.

---

### Step 7 — Network Analysis (run after Step 6)
**Script:** `network.py`  
**Input:** `host_location.csv`  
**Outputs:**
- `species_centrality_metrics.csv`
- `countries_centrality_metrics.csv`
- `all_nodes_centrality_metrics.csv`
- `node_p_values.csv`
- `mann_whitney_results.csv`
- `group_summary_stats.csv`
- `communities_report.txt`

Builds a bipartite species–country network. Runs Louvain community detection, calculates degree and betweenness centrality, performs permutation-based statistical validation, and compares migratory vs non-migratory species using a Mann-Whitney U test.

---

### Step 8 — Migration Route Map *(standalone, run anytime)*
**Script:** `migration_route_plotter.py`  
**Input:** `ne_110m_admin_0_countries.shp`  
**Output:** Arrow-based world map of migration routes per species

Uses hardcoded coordinates from `migration_paths_dictionary.txt`. Does not depend on any CSV outputs from other scripts.

---

## 🔄 Pipeline Overview

```
gisaid_epiflu_isolates_full_period.csv
│
├── Gisaid.py
│   ├── Animal_period.csv ──────────────► migration_summary.py
│   │                                           └── migration_analysis.py
│   ├── host_appearances.csv ───────────► heatmaps.py
│   └── flock_size_FP_outbreak.csv ─────► choropleth.py
│
├── data_quality_check.py  (standalone QC)
│
├── host_location.py  (re-cleans raw CSV)
│   └── host_location.csv ──────────────► network.py
│
└── migration_route_plotter.py  (standalone, no CSV needed)
```

---

## 📦 Dependencies

```bash
pip install pandas geopandas matplotlib seaborn contextily mapclassify networkx scipy tqdm
```

> **Note:** A world shapefile (`ne_110m_admin_0_countries.shp`) from [Natural Earth](https://www.naturalearthdata.com/) is required. Place it in a `world_shapefile/` directory at the root of the project.

---

## 📂 Input Data

Raw data is sourced from GISAID's EpiFlu database. The expected file is:

```
gisaid_epiflu_isolates_full_period.csv
```

This file is included in this repository. Alternatively, download it directly from [GISAID](https://www.gisaid.org/) (requires a free registered account).

---

## ⚠️ Known Issues

- `Gisaid.py` and `host_location.py` both independently re-clean the raw GISAID CSV using the same corrections and species mapping. If you update `countries_manual_correction_mapping.txt` or `species_mapping.txt`, **you must update both scripts** to keep them in sync.
- `migration_analysis.py` contains a section that references a variable `df` that must be loaded manually (either `migratory_detailed.csv` or `nonmigratory_detailed.csv`) — see inline comments in the script.
- File paths in several scripts are hardcoded to a local Windows directory. Update these before running.

---

## 📊 Outputs Summary

| Script | Key Output Files |
|---|---|
| `Gisaid.py` | `Animal_period.csv`, `host_appearances.csv`, `flock_size_FP_outbreak.csv` |
| `data_quality_check.py` | `Summary_Country_Submissions.csv` |
| `choropleth.py` | Choropleth PNG |
| `heatmaps.py` | Heatmap PNG |
| `migration_summary.py` | `migratory_vs_nonmigratory_summary.csv`, `migratory_detailed.csv` |
| `migration_analysis.py` | Bar plots, trend lines, heatmap PNGs |
| `host_location.py` | `host_location.csv`, bar chart PNGs |
| `network.py` | Centrality CSVs, `communities_report.txt`, `mann_whitney_results.csv` |
| `migration_route_plotter.py` | Migration route map PNG |
