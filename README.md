# Road Accident Severity Prediction

This folder contains the project's data preparation, exploratory analysis,
model training, evaluation, and model-explanation code for UK STATS19 road
collisions. The modeling target is binary:

- `target_severe_fatal`: 1 for a severe or fatal collision, 0 for slight.
- `target_fatal_only`: 1 for a fatal collision, 0 otherwise.

The scripts default to `target_severe_fatal`. Pass `--target
target_fatal_only` to run a supported training script on the fatal-only
target.

## Folder contents

```text
FINAL_CODES/
├── data/
│   ├── collision-last-5-years.csv
│   ├── vehicle-last-5-years.csv
│   ├── casualty-last-5-years.csv
│   ├── collisions.parquet
│   └── collisions_clean.parquet
├── eda/
│   ├── diagnostic_eda.ipynb
│   └── eda.py
├── report/
│   ├── figures/
│   ├── final_results/
│   ├── shap/
│   └── tuning_of_sampling_techniques/
├── src/
│   ├── build_dataset.py
│   └── clean_dataset.py
├── train/
│   ├── train_logreg_imbalance.py
│   ├── tuning_of_sampling_techniques.py
│   ├── train_with_shap.py
│   ├── location_features.py
│   ├── explain_shap.py
│   └── analyze_false_positives_shap.py
└── requirements.txt
```

Run commands below from the `FINAL_CODES` directory unless stated otherwise.
Input and output paths are relative to the current directory.

## Setup

Use Python with the packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Some optional scripts also import LightGBM, XGBoost, or SHAP. Install those
packages in the same environment if you run a script/model that needs them.

## Data preparation

### 1. Build the collision-level table

`src/build_dataset.py` reads the three raw STATS19 tables, filters each to the
requested years, and joins vehicle and casualty summaries onto the collision
table. Vehicle/casualty records are aggregated by collision first, avoiding
row multiplication in the joins.

The build step:

- Creates both binary targets from `collision_severity`, then removes the
  original severity column.
- Adds time features (`hour`, `hour_sin`, `hour_cos`, and `month`).
- Converts configured numeric STATS19 missing/unknown codes to null values.
- Drops identifier, inconsistent-year, and outcome-adjacent fields listed in
  `COLLISION_DROP_COLS`.
- Checks that the merge retained one row per collision and that required
  columns are present.

Example (the year selection is explicit; choose years appropriate to your
dataset and analysis):

```bash
python src/build_dataset.py \
  --data-dir data \
  --years 2022 2023 2024 \
  --out data/collisions.parquet
```

### 2. Clean the built table

`src/clean_dataset.py` applies the configured cleanup steps:

- Drops the configured fully missing field and rows with missing latitude or
  longitude.
- Replaces missing categorical values with the explicit `"Missing"` label.
- Nulls age values outside configured plausibility bounds and adds
  `_was_missing` indicators.
- Swaps min/max age pairs when they are inverted.
- Adds a missingness flag for `share_male_drivers`.
- Replaces road-number identifiers with `is_classified_road` and
  `is_classified_second_road`.

It does **not** impute numeric values. The model scripts impute numeric
predictors inside their training folds, so the imputer is fitted on training
data rather than on the full dataset.

```bash
python src/clean_dataset.py \
  --in data/collisions.parquet \
  --out data/collisions_clean.parquet
```

### 3. Run exploratory analysis (optional)

`eda/eda.py` reads the cleaned parquet and creates descriptive plots, including
target distributions, rates by category, numeric comparisons, time patterns,
location patterns, correlations, and missingness checks. These plots describe
the dataset; they are not held-out model evaluation.

```bash
python eda/eda.py \
  --in data/collisions_clean.parquet \
  --out-dir report/figures
```

The diagnostic notebook is at `eda/diagnostic_eda.ipynb`.

## Feature engineering and leakage notes

`train/train_logreg_imbalance.py` owns the shared `engineer_features()`
function and the categorical/numeric model feature lists. It adds features
such as `driver_age_range`, missing-value flags, `single_vehicle_collision`,
and speed/lighting interactions. The other training scripts reuse this
function.

`train/location_features.py` provides optional location features for the
holdout/CV model pipelines: raw latitude/longitude and smoothed severe/fatal
rates for fine and coarse coordinate grid cells. Cell rates are fitted on
training data only; training rows use inner out-of-fold encodings, and unseen
cells use the training prior.

The standard Logistic Regression feature lists do not include raw coordinates
or the location target encodings. The holdout model scripts do include
coordinates/location encodings.

The build script excludes target-adjacent fields such as the original
`collision_severity`, adjusted severity fields, `collision_injury_based`,
`number_of_casualties`, and the aggregated casualty count. Casualty-derived
features may be present in the cleaned table, but the current model feature
allowlist does not include them. Review feature timing carefully if the
intended use is prediction before a collision or before casualty information
is known.

## Training and evaluation

### Logistic Regression imbalance comparison

Run the repeated stratified CV experiment:

```bash
python train/train_logreg_imbalance.py \
  --in data/collisions_clean.parquet \
  --target target_severe_fatal \
  --out report/logreg_results.csv
```

Defaults are 5 folds × 3 repeats. Within each outer fold, 25% of the outer
training partition is used for validation and threshold selection; the
remaining fit portion is used for preprocessing, optional resampling, and
model fitting. The held-out outer fold is used for scoring. The script reports
a default 0.5 threshold and a validation-tuned threshold.

Strategies are:

- **S0:** no class rebalancing.
- **S1:** class weighting.
- **S2:** SMOTENC oversampling.
- **S3:** SMOTENC plus class weighting.
- **S4:** ADASYN oversampling.
- **S5:** S0 scored at its validation-tuned threshold (an output alias).

Use `--quick` for 3 folds × 1 repeat while debugging; quick results are not
the full reported protocol. Threshold tuning defaults to F2; use
`--threshold-mode recall_target --target-recall 0.8` to choose a threshold
according to the requested recall.

### Holdout model comparison and sampling-strength tuning

`train/train_with_shap.py` runs an 80/20 stratified holdout evaluation with
development-set CV. Despite its filename, its command-line entry point does
not calculate SHAP values: it compares models and strategies, selects a
configuration by CV balanced accuracy (then F1, recall, and lower threshold),
refits on the full development split, and evaluates the holdout once.

```bash
python train/train_with_shap.py \
  --model all \
  --strategies S0 S1 S2 S3 S4 \
  --out report/final_results/FINAL_all_model_results.csv
```

Models can be `logreg`, `rf`, `lightgbm`, `xgboost`, or `all`. The default
threshold list is 0.50 through 0.70 in 0.05 increments. The default output
base path writes:

- `<stem>.csv`: per-fold development CV metrics.
- `<stem>_cv_summary.csv`: mean CV metrics by model, strategy, and threshold.
- `<stem>_holdout.csv`: one final holdout result per model, using the
  configuration selected by CV.

`train/tuning_of_sampling_techniques.py` follows the same holdout/CV pattern
but also sweeps the balancing strength for S1-S4. Its default parameter grids
are defined in `PARAM_GRID`; use it to compare degrees of class weighting or
oversampling, rather than only the fixed S0-S4 settings.

```bash
python train/tuning_of_sampling_techniques.py \
  --model all \
  --strategies S0 S1 S2 S3 S4 \
  --out report/tuning_of_sampling_techniques/tuning_results.csv
```

Both holdout scripts support `--quick` (3 CV folds for debugging). Do not use
quick-mode metrics as final evaluation.

### Reading the metrics

The positive class is severe/fatal (or fatal-only when that target is
selected). Precision, recall, F1, accuracy, and balanced accuracy are
calculated at a chosen threshold. ROC-AUC and PR-AUC use probabilities and
are threshold-independent. With an imbalanced target, accuracy alone can be
misleading; consider balanced accuracy and positive-class recall/precision
alongside the confusion counts.

Thresholds and strategies are selected using development/validation data,
not the final holdout. Use the holdout result once for final reporting; do not
choose a model based on the holdout and then present that same score as an
unbiased final estimate.

## SHAP and false-positive analysis

`train/explain_shap.py` creates global and individual SHAP explanations for a
tree model, including grouped feature importance, a beeswarm, a dependence
plot, and a waterfall plot. It imports `fit_model` from
`train_combination_2.py`, which is **not included in this folder**. That
dependency must be supplied before this script can run.

`train/analyze_false_positives_shap.py` calculates holdout confusion-group
counts, false-positive rates by motorcycle involvement, and SHAP summaries
for a sample of each outcome group. It also imports
`fit_model`/`NUM_ALL` from the missing `train_combination_2.py`, so it has the
same dependency requirement.

## Existing reports

The `report/` folder contains saved project outputs, including EDA figures,
model result CSVs, sampling-tuning results, and SHAP analyses. These are
artifacts from previous runs; rerun the documented commands to regenerate
them from the current input data and code.
