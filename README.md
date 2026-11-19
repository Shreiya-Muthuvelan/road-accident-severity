# Road Accident Severity Prediction

Predicting UK road collision severity (Severe/Fatal vs. Slight, and Fatal vs. not)
from pre-crash circumstances, using the DfT STATS19 dataset (2020–2025).

## Repo structure

```
road-accident-severity/
├── data/
│   ├── collision-last-5-years.csv      # raw STATS19 collision table, 2020-2025
│   ├── vehicle-last-5-years.csv        # raw STATS19 vehicle table, 2020-2025
│   ├── casualty-last-5-years.csv       # raw STATS19 casualty table, 2020-2025
│   ├── collisions.parquet              # output of src/build_dataset.py
│   └── collisions_clean.parquet        # output of src/clean_dataset.py
├── notebooks/
│   └── diagnostic_eda.ipynb            
├── reports/                            # EDA figures + model result CSVs are stored here
├── src/
│   ├── build_dataset.py                # used to build the one row per collision parquet
│   └── clean_dataset.py                # missing values, outliers, encoding prep
├── train/
│   └── train_logreg_imbalance.py       # Logistic Regression x 6 imbalance strategies
├── eda.py                              # deeper EDA visuals from the cleaned data
├── requirements.txt
└── readme.md
```

## Pipeline

Each stage reads the previous stage's output. Commands below assume you `cd`
into the folder containing the script before running it (paths are relative).

**1. Build the dataset** (`src/`) — merges the three raw STATS19 tables into
one row per collision. Vehicle and casualty tables are aggregated up (never
row-joined) to avoid duplicating collisions. Builds both targets
(`target_severe_fatal`, `target_fatal_only`) from `collision_severity`, then
drops it and every other outcome-adjacent column to prevent leakage.

```bash
cd src
python build_dataset.py --years 2022 2023 2024 --out ../data/collisions.parquet
```

**2. Clean the dataset** (`src/`) — fixes missingness and outliers found
during diagnostic EDA: drops `local_authority_district` (100% unpopulated at
source), drops the handful of rows missing lat/long, fills STATS19's
"unknown (self-reported)" categoricals with an explicit `Missing` category
rather than imputing them away, caps implausible ages before median-imputing
(with a `_was_missing` indicator per field), fixes rows left with `min > max`
by independent imputation, and replaces high-cardinality road-number IDs with
binary `is_classified_road` flags.

```bash
cd src
python clean_dataset.py --in ../data/collisions.parquet --out ../data/collisions_clean.parquet
```

**3. Deeper EDA** (root) — generates target-rate-by-category charts,
numeric-by-target boxplots, temporal and geospatial patterns, a correlation
heatmap, and missingness-as-signal checks, saved to `reports/`.

```bash
python eda.py --in data/collisions_clean.parquet --out-dir reports/figures
```

**4. Train** (`train/`) — Logistic Regression baseline evaluated under all
six imbalance-handling strategies (S0–S5), with repeated stratified 5-fold
CV (3 repeats) and a validation-tuned decision threshold alongside the
default 0.5.

```bash
cd train
python train_logreg_imbalance.py --in ../data/collisions_clean.parquet \
    --target target_severe_fatal --out ../reports/logreg_results.csv
```
Add `--quick` for a fast 3-fold/1-repeat debug run before committing to the
full protocol.

## Key decisions / known leakage exclusions

- **Unit of analysis is one row per collision.** Vehicle and casualty data
  are aggregated to collision level before merging.
- **Dropped as leakage:** `casualty_severity`, all `*_adjusted_severity_*`
  fields, `collision_injury_based`, `number_of_casualties`, and the
  aggregated `n_casualties_listed` — all are outcome-adjacent rather than
  pre-crash information.
- **Dropped as 2024-only / no history in earlier years:**
  `junction_detail`, `pedestrian_crossing`, `carriageway_hazards`,
  `enhanced_severity_collision`, `local_authority_highway_current`.
- **2020 excluded** from modeling years (lockdown traffic is atypical);
  2025 excluded pending confirmation the DfT release is final, not
  provisional.
- **Collinear driver-age features** (`min`/`max`/`mean_driver_age`) are
  collapsed to `mean_driver_age` + engineered `driver_age_range` for the
  Logistic Regression model, to avoid unstable coefficients.
- **Raw `latitude`/`longitude` dropped for Logistic Regression** —
  `police_force` is the only remaining (coarse) location signal, since
  `local_authority_district` turned out to be 100% missing at source.
  Tree-based models may revisit coarse-grid lat/long later.

## Modeling results so far

**Logistic Regression, target = `target_severe_fatal`, Protocol A (5-fold
× 3 repeats):**

| Strategy | ROC-AUC | PR-AUC | Best F2 (tuned threshold) |
|---|---|---|---|
| S0 — no handling | 0.737 | 0.465 | 0.652 |
| S1 — class weighting | 0.737 | 0.463 | **0.653** |
| S2 — SMOTENC | 0.694 | 0.409 | 0.635 |
| S3 — SMOTENC + class weighting | 0.694 | 0.409 | 0.635 |
| S4 — ADASYN | 0.732 | 0.455 | 0.651 |
| S5 — S0 @ tuned threshold | 0.737 | 0.465 | 0.652 |

Class weighting matches or slightly beats the untrained baseline on every
metric; SMOTENC underperforms both despite being the more expensive method.
S2/S3 are numerically identical because SMOTENC was run at
`sampling_strategy=1.0` (full rebalancing), which makes `class_weight` a
no-op on an already-balanced training set — a result worth noting rather
than a bug. Full writeup in `reports/`.

## Not yet done

- Protocol B (temporal train/test split across years)
- Random Forest / LightGBM on the same imbalance-strategy grid
- `target_fatal_only` results (currently only `target_severe_fatal` has
  been run)
- SMOTENC sensitivity run at `sampling_strategy=0.5` to properly test
  whether class weighting adds anything on top of partial resampling

## Setup

```bash
pip install -r requirements.txt
```
