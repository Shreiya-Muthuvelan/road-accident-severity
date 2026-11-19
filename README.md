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
├── data/                                     # Raw and processed STATS19 data
│   ├── collision-last-5-years.csv            # Raw STATS19 collision records
│   ├── vehicle-last-5-years.csv              # Raw vehicle records
│   ├── casualty-last-5-years.csv             # Raw casualty records
│   ├── collisions.parquet                    # Merged collision-level dataset
│   └── collisions_clean.parquet              # Cleaned dataset used for training
│
├── src/                                      # Dataset preparation
│   ├── build_dataset.py                      # Builds one row per collision and creates the targets
│   └── clean_dataset.py                      # Cleans values and adds missingness and road flags
│
├── eda/                                      # Exploratory data analysis
│   ├── eda.py                                # Generates exploratory figures
│   └── diagnostic_eda.ipynb                  # Interactive exploration of the collision table
│
├── train/                                    # Model training and evaluation
│   ├── train_logreg_imbalance.py             # Shared features and repeated-CV Logistic Regression
│   ├── location_features.py                  # Out-of-fold location target encoding
│   ├── train_with_shap.py                    # Four-model CV and final holdout evaluation
│   ├── tuning_of_sampling_techniques.py      # Compares class weights and sampling ratios
│   ├── explain_shap.py                       # Generates SHAP explanations
│   └── analyze_false_positives_shap.py       # Analyzes false positives using SHAP
│
├── report/                                   # Saved results and figures
│   ├── figures/                              # Exploratory plots and a saved confusion matrix
│   ├── final_results/                        # Four-model CV and holdout results
│   ├── tuning_of_sampling_techniques/        # Sampling-strength experiment results
│   └── shap/                                 # Saved SHAP plots and tables
│       └── false_positive_analysis/          # Saved false-positive plots and tables
│
├── architecture_diagram.png                  # Project workflow diagram
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project overview and run commands
```

## Run the project

Run the commands below from the repository root.

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

## Final results

These results are from the final holdout evaluation on 62,231 collisions. [Holdout results CSV](report/final_results/FINAL_all_model_results_holdout.csv)

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 64.03% | 63.58% | 35.89% | 62.70% | 45.65% | 0.6878 |
| Random Forest | **68.28%** | 63.71% | **38.80%** | 54.89% | 45.46% | 0.6985 |
| LightGBM | 64.00% | 64.25% | 36.18% | **64.73%** | 46.42% | 0.7026 |
| XGBoost | 64.63% | **64.32%** | 36.56% | 63.71% | **46.46%** | **0.7038** |

## Reproducibility

- The final experiment uses seed **42** for the stratified holdout, shuffled development CV, and final refit. Development folds use seeds **999 + fold number** for the seeded training steps.
- Shared feature engineering derives row-level features such as driver age range and speed/lighting interactions. The model feature allowlist excludes the target labels and casualty-derived predictors.
- Imputers, encoders, and scalers are fitted on the fitting partition. Oversampling runs on those training rows only. Validation and holdout data are transformed using the fitted preprocessing.
- The holdout models use raw coordinates and smoothed rates for 0.05° and 0.25° grid cells. Training rows receive rates from inner out-of-fold fits. Validation and holdout rows use statistics fitted on their training partition. Unseen cells receive that partition’s overall positive rate.
- Use the bundled inputs and the explicit settings in Step 4 to reproduce the reported comparison. Keep `--quick` for smoke tests. It reduces the fold count. Exact scores may vary with library versions, platform, and data revisions.
