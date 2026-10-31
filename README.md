# Registration-number disclosure on short-term rental platforms in Europe

Analysis code and derived data for the manuscript *Mandated Registration Number Disclosure
Erodes When Its Legal Basis Is Annulled in European Short-Term Rental Markets*.

Regulation (EU) 2024/1028 applies from 20 May 2026 and requires short-term rental platforms to
collect and display a registration number for each listing where a registration procedure
exists. Using quarterly public snapshots of listings (Inside Airbnb) in 64 markets in 23
countries, the study measures the share of listings that display a registration number before
and after the application date, follows individual listings across snapshots, and compares
markets where the duty was extended, where its legal basis was annulled (Spain), where it was
already in force, and where no registration procedure operates.


This repository contains the derived data and the analysis code for the study. It does not
contain the listing-level source files, which are not redistributed here: the snapshots are published by Inside Airbnb under a Creative Commons
Attribution 4.0 licence and can be downloaded from the addresses recorded in
`data/snapshots.csv` and `data/download_plan.csv`. The checksums of the files actually used
are in `data/download_log.csv`, so the inputs can be reproduced exactly.

## What is deliberately not here

The public registration field can contain personal data, including, in some markets, a host
name or a home address. No value of that field is reproduced anywhere in this repository, in the
article or in the supplementary materials. Only derived variables are retained: the assigned
category, an indicator for a number being present, a token count, an indicator for a line
break in the original field, and a length band. The listing-level analysis file used for the
model in Section 4.4 is also withheld, because it carries listing coordinates.

## Data files

| File | Content |
|---|---|
| `results.json` | Every number reported in the article and in the supplementary materials, as key-value pairs |
| `city_snapshot_summary.csv` | Market x snapshot: listing counts and the composition of the registration field |
| `city_period_panel.csv` | The same, with each snapshot placed relative to the application date |
| `city_snapshot_by_roomtype.csv` | Market x snapshot x room type |
| `transitions_v2.csv` | Market x window x subgroup x origin state: counts of each destination state |
| `transitions_final.csv` | Transition shares used for Figure 3 |
| `panel_transitions.csv`, `panel_transitions_clean.csv`, `panel_transitions_grouped.csv` | Earlier transition aggregates retained for completeness |
| `coverage_diagnostics.csv` | Listing counts per market and snapshot, and the March 2026 coverage ratio |
| `pretrend_controls.csv` | The three control markets with snapshots before December 2025 |
| `ml_results.json` | Cross-validation results and permutation importance for the model in Section 4.4 |
| `table1_sample_and_rates.csv`, `table2_did.csv`, `table3_coverage.csv` | The article tables as data |
| `snapshots.csv`, `city_eligibility.csv`, `download_plan.csv`, `download_log.csv` | The snapshot catalogue, the selection of markets, and the download record with checksums |
| `robots_txt_20261030.txt` | The robots file of the source site as retrieved on the collection date |

## Code

Run in the order of the file names. `s1_lic.py` holds the classification of the registration
field and is imported by the others; it never returns a field value.

| File | Step |
|---|---|
| `s1_01_build_catalog.py` | Build the snapshot catalogue and select markets |
| `s1_02_make_dl_plan.py`, `s1_03_download.py`, `s1_03b_download_pretrend.py` | Download the snapshots, one at a time with a delay between requests |
| `s1_lic.py` | Classify the registration field |
| `s1_04_summarise.py` | Market x snapshot aggregates |
| `s1_05_panel.py`, `s1_12_transitions_final.py`, `s1_16_transitions_v2.py` | Follow individual listings across snapshots |
| `s1_06_make_ml_sample.py` | Build the listing-level analysis file for the model |
| `s1_08_privacy_audit.py` | Check mechanically that no field value appears in any output |
| `s1_09_estimate.py` | Market-level estimation, wild cluster bootstrap, randomization inference |
| `s1_11_coverage_robust.py` | Estimates that exclude the incomplete March 2026 wave |
| `s1_13_tabpfn.py` | The model in Section 4.4 |
| `s1_14_assemble.py`, `s1_17_result_keys.py` | Collect every reported number into `results.json` |
| `s1_19_pretrend.py` | The control markets with earlier snapshots |
| `s1_20_within_eu.py` | Estimates against European markets where the duty remained in force |
| `s1_10_figs.py`, `s1_15_tables.py`, `s1_21_supp.py` | Figures, tables and supplementary materials |

## Figures and tables to files

| Item | Produced by | From |
|---|---|---|
| Figure 1 | `s1_10_figs.py` | `results.json`, `city_period_panel.csv` |
| Figure 2 | `s1_10_figs.py` | `results.json` |
| Figure 3 | `s1_10_figs.py` | `transitions_final.csv` |
| Figure 4 | `s1_10_figs.py` | `ml_results.json` |
| Figure S1 | `s1_21_supp.py` | `city_period_panel.csv` |
| Table 1 | Primary legal and government sources, listed in Table S1 | — |
| Tables 2 to 4 | `s1_15_tables.py`, `s1_17_result_keys.py` | `results.json`, `transitions_v2.csv` |
| Tables S2 to S6 | `s1_21_supp.py` | `coverage_diagnostics.csv`, `results.json`, `ml_results.json` |

## Environment

Python 3.11 with pandas, numpy, scikit-learn, matplotlib and statsmodels. The model in
Section 4.4 is a tabular foundation model, version 9.2.0, run locally
on its published weights on a single GPU; no network call and no credential is used at any
point. Random seeds are fixed in the scripts.

## Licence and attribution

The listing snapshots are published by Inside Airbnb under Creative Commons Attribution 4.0.
Cite that source when reusing the derived files in `data/`.

## Licence

The code is released under the MIT licence (see `LICENSE`). The derived files in `data/` are
derived from Inside Airbnb data published under Creative Commons Attribution 4.0 and are
shared under the same licence; please attribute Inside Airbnb (https://insideairbnb.com) when
reusing them.
