# Road Accident Severity Prediction Using Machine Learning

This project predicts the recorded severity of police-reported road collisions in Great Britain using STATS19 data. Each row represents one collision. The main target, `target_severe_fatal`, is **0 for Slight** and **1 for Serious or Fatal**. The data pipeline also produces `target_fatal_only` for a separate binary task.

The final experiment compares Logistic Regression, Random Forest, LightGBM, and XGBoost. It evaluates an unbalanced baseline, class weighting, SMOTENC, SMOTENC with class weighting, and ADASYN.

## Dataset

The UK Department for Transport publishes STATS19 collision, vehicle, and casualty records. This project uses **2022, 2023, and 2024**. The bundled raw extracts cover 2021–2025, and the build command filters each table to the three project years. The repository tracks both the raw extracts and the processed Parquet files. Use the bundled extracts for the closest reproduction of the saved run, since the Department for Transport can revise its rolling downloads. [Department for Transport road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data)

If the CSVs are missing from your copy, download the three five-year extracts and save them under `data/` with these exact names. Check that all three cover 2022–2024 before building.

| Table | Local path | Official download |
| --- | --- | --- |
| Collisions | `data/collision-last-5-years.csv` | [Collisions CSV](https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-last-5-years.csv) |
| Vehicles | `data/vehicle-last-5-years.csv` | [Vehicles CSV](https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-vehicle-last-5-years.csv) |
| Casualties | `data/casualty-last-5-years.csv` | [Casualties CSV](https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-casualty-last-5-years.csv) |

Steps 1 and 2 below regenerate both Parquet files from the CSVs.

## Project structure

```text
.
|-- README.md
|-- requirements.txt
|-- architecture_diagram.png
|-- data/
|   |-- collision-last-5-years.csv
|   |-- vehicle-last-5-years.csv
|   |-- casualty-last-5-years.csv
|   |-- collisions.parquet
|   `-- collisions_clean.parquet
|-- src/
|   |-- build_dataset.py
|   `-- clean_dataset.py
|-- eda/
|   |-- eda.py
|   `-- diagnostic_eda.ipynb
|-- train/
|   |-- train_logreg_imbalance.py
|   |-- location_features.py
|   |-- train_with_shap.py
|   |-- tuning_of_sampling_techniques.py
|   |-- explain_shap.py
|   `-- analyze_false_positives_shap.py
`-- report/
    |-- figures/
    |-- final_results/
    |-- tuning_of_sampling_techniques/
    `-- shap/
        `-- false_positive_analysis/
```

| Script | Purpose |
| --- | --- |
| `src/build_dataset.py` | Filters the raw tables by year, creates the targets, and aggregates vehicle and casualty records before joining them to collisions. |
| `src/clean_dataset.py` | Cleans category values, flags age and other missingness, and creates road classification flags. |
| `eda/eda.py` | Generates descriptive figures from the cleaned data. The diagnostic notebook explores the earlier collision-level table. |
| `train/train_logreg_imbalance.py` | Defines shared features and runs a separate repeated-CV Logistic Regression experiment. |
| `train/location_features.py` | Implements smoothed, out-of-fold location target encoding for the holdout experiments. |
| `train/train_with_shap.py` | Runs the four-model development CV and final holdout comparison. |
| `train/tuning_of_sampling_techniques.py` | Runs a separate development CV and holdout experiment that varies the strength of class weighting and oversampling. |
| `train/explain_shap.py` and `train/analyze_false_positives_shap.py` | Generate model explanations and analyze false positives when their shared training module is available. |

## Installation

Python **3.14** was available for verification. The repository does not specify a supported Python version range or lock the environment used for the saved training runs.

Run commands from the repository root. Create and activate a virtual environment.

**Windows PowerShell**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS or Linux**

```bash
python3.14 -m venv .venv
source .venv/bin/activate
```

Install the repository requirements and the additional packages used by the four-model comparison.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install imbalanced-learn scipy lightgbm xgboost
python -m pip check
```

The workflow uses pandas and NumPy for data processing, PyArrow for Parquet, scikit-learn for preprocessing and evaluation, imbalanced-learn for resampling, SciPy for sparse matrices, LightGBM and XGBoost for boosted trees, and Matplotlib for plots. The requirements file also lists Fastparquet and seaborn. The package that provides the `imblearn` import is `imbalanced-learn`. The optional diagnostic notebook uses seaborn and requires a notebook environment such as JupyterLab.

## Run the project

### 1. Build the collision-level table

Place the CSVs in `data/` as listed above, then run:

```bash
python src/build_dataset.py --data-dir data --years 2022 2023 2024 --out data/collisions.parquet
```

This creates `data/collisions.parquet` with one row per collision. Vehicle and casualty records are summarized by collision before joining. The script derives both targets, adds time features, and removes the original severity label and selected outcome-adjacent fields.

### 2. Clean and prepare the dataset

```bash
python src/clean_dataset.py --in data/collisions.parquet --out data/collisions_clean.parquet
```

This creates `data/collisions_clean.parquet`. Cleaning removes rows with missing coordinates, marks unknown categories, handles implausible or inverted age values, and derives road classification flags. The bundled cleaned table contains **311,155 collisions**, including **74,954 Serious/Fatal**. Numeric missing values are imputed during training.

### 3. Generate exploratory figures

```bash
python eda/eda.py --in data/collisions_clean.parquet --out-dir report/figures
```

This optional step writes numbered PNGs to `report/figures/`, including class distribution, feature comparisons, time patterns, geographic severity rates, correlations, and missingness. To use the diagnostic notebook, open it with `eda/` as its working directory so its relative Parquet path resolves.

### 4. Run the final four-model comparison

```bash
python train/train_with_shap.py --in data/collisions_clean.parquet --target target_severe_fatal --model all --strategies S0 S1 S2 S3 S4 --thresholds 0.45 0.50 0.55 0.60 --n-splits 5 --out report/final_results/FINAL_all_model_results.csv
```

This command uses the strategies, thresholds, and fold count recorded in the saved results. Feature engineering and location encoding run as part of training. The script sets aside a stratified 20% holdout and runs five-fold stratified CV on the remaining 80%. For each model it selects the strategy and threshold by mean CV balanced accuracy, breaking ties by F1, recall, and then the lower threshold. It refits each selected configuration on the entire development partition and scores the shared holdout once.

| Strategy | Treatment in the final comparison |
| --- | --- |
| S0 | No class rebalancing |
| S1 | Balanced class weights, implemented through sample weights for XGBoost |
| S2 | SMOTENC with a minority-to-majority ratio of 0.5 |
| S3 | SMOTENC with a ratio of 0.5 plus balanced weighting |
| S4 | ADASYN on the encoded training features |

### 5. Run the optional comparison experiments

To reproduce the saved sampling-strength grid, run:

```bash
python train/tuning_of_sampling_techniques.py --in data/collisions_clean.parquet --target target_severe_fatal --model all --strategies S1 S2 S3 S4 --thresholds 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 --n-splits 5 --out report/tuning_of_sampling_techniques/tuning_of_sampling_technqiues_results.csv
```

The output stem preserves the spelling in the repository. The script uses the same development/holdout procedure and searches the class weights and sampling ratios defined in its `PARAM_GRID`. It writes a per-fold CSV, a CV summary, and a holdout CSV in the specified directory.

To run the standalone Logistic Regression experiment, run:

```bash
python train/train_logreg_imbalance.py --in data/collisions_clean.parquet --target target_severe_fatal --n-splits 5 --n-repeats 3 --out report/logreg_results.csv
```

This experiment writes `report/logreg_results.csv` and `report/logreg_results_summary.csv`. It uses repeated stratified CV and reserves 25% of each outer training partition to tune a threshold by F2. It compares that threshold with 0.5. Strategy S5 is the output label for the tuned S0 result. The outer test folds supply the evaluation scores.

## Expected outputs

```text
data/
|-- collisions.parquet
`-- collisions_clean.parquet
report/
|-- figures/                         # Numbered exploratory PNGs
|-- final_results/
|   |-- FINAL_all_model_results.csv
|   |-- FINAL_all_model_results_cv_summary.csv
|   `-- FINAL_all_model_results_holdout.csv
|-- tuning_of_sampling_techniques/
|   |-- tuning_of_sampling_technqiues_results.csv
|   |-- tuning_of_sampling_technqiues_results_cv_summary.csv
|   `-- tuning_of_sampling_technqiues_results_holdout.csv
|-- logreg_results.csv               # Generated by the optional baseline
`-- logreg_results_summary.csv       # Generated by the optional baseline
```

The main CSV contains individual development-fold scores. The `_cv_summary.csv` averages scores for each candidate configuration and supports selection. The `_holdout.csv` contains one final score row per model on the same **62,231 collisions**, plus `tn`, `fp`, `fn`, and `tp` for constructing confusion matrices. Use this file for the final comparison. [Saved holdout results](report/final_results/FINAL_all_model_results_holdout.csv)

| Metric | Meaning |
| --- | --- |
| Accuracy | Share of all collisions classified correctly |
| Balanced accuracy | Mean of Slight recall and Serious/Fatal recall |
| Precision | Share of predicted Serious/Fatal collisions that were Serious/Fatal |
| Recall | Share of actual Serious/Fatal collisions correctly identified |
| F1-score | Harmonic mean of Serious/Fatal precision and recall |
| ROC-AUC | Area under the receiver operating characteristic curve, based on probability rankings |
| PR-AUC | The CSV field `pr_auc` stores average precision, calculated from the precision–recall trade-off across thresholds |

The repository also includes a saved [LightGBM confusion matrix image](report/figures/confusion_lgbm_from_results.png) and [SHAP plots and tables](report/shap/). The main training command generates metrics and confusion counts. The two explanation scripts require SHAP and import `train_combination_2.py`, which is missing from this checkout. The saved explanation outputs are available for inspection.

## Reproducibility

- The final experiment uses seed **42** for the stratified holdout, shuffled development CV, and final refit. Development folds use seeds **999 + fold number** for the seeded training steps.
- Shared feature engineering derives row-level features such as driver age range and speed/lighting interactions. The model feature allowlist excludes the target labels and casualty-derived predictors.
- Imputers, encoders, and scalers are fitted on the fitting partition. Oversampling runs on those training rows only. Validation and holdout data are transformed using the fitted preprocessing.
- The holdout models use raw coordinates and smoothed rates for 0.05° and 0.25° grid cells. Training rows receive rates from inner out-of-fold fits. Validation and holdout rows use statistics fitted on their training partition. Unseen cells receive that partition’s overall positive rate.
- Use the bundled inputs and the explicit settings in Step 4 to reproduce the reported comparison. Keep `--quick` for smoke tests. It reduces the fold count. Exact scores may vary with library versions, platform, and data revisions.

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| A raw CSV is missing | Check the three exact filenames and the `data/` directory. Use the official extracts if needed. |
| A script or Parquet path cannot be found | Run from the repository root. Run Steps 1 and 2 if the processed file is absent. |
| A Python package cannot be imported | Activate the virtual environment and rerun both installation commands. Use `python -m pip check` to identify incompatible dependencies. |
| Dependency installation fails | Check the interpreter version and read the failing package’s installer message. On Windows, use the environment’s `.venv\Scripts\python.exe` directly if activation is blocked. |
| An output directory is missing or unwritable | The scripts create their output directories. Check the explicit output argument and ensure you have write access to its parent directory. |
| The final results differ from the saved table | Compare the input snapshot, five-fold setting, strategy list, and explicit threshold grid. Keep the final comparison and sampling-strength outputs in their respective folders. |
