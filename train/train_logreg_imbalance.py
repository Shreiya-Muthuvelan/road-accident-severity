"""
Trains Logistic Regression under all 6 imbalance-handling strategies from
the proposal (S0-S5), evaluated with repeated stratified
k-fold CV (Protocol A).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from imblearn.over_sampling import ADASYN, SMOTENC

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Feature configuration
# Columns dropped entirely for this model (identifiers, redundant-with-
# engineered-feature raw columns, and — per the driver-age/lat-long
# discussion — collinear raw ages and raw coordinates).
DROP_COLS = [
    "collision_index",
    "collision_year",   # kept aside for temporal split later, not fed to CV model
    "hour",              # superseded by hour_sin / hour_cos
    "latitude",
    "longitude",         # no separate location feature currently retained for LR;
                          # police_force is the coarse location proxy that remains
    "min_driver_age",
    "max_driver_age",    # replaced by mean_driver_age + driver_age_range below
]

CATEGORICAL_COLS = [
    "police_force",
    "day_of_week",
    "first_road_class",
    "road_type",
    "speed_limit",
    "junction_detail_historic",
    "junction_control",
    "second_road_class",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "special_conditions_at_site",
    "carriageway_hazards_historic",
    "urban_or_rural_area",
    "did_police_officer_attend_scene_of_accident",
    "trunk_road_flag",
    "month",
]

NUMERIC_COLS = [
    "n_vehicles_listed",
    "has_motorcycle",
    "has_hgv",
    "mean_driver_age",
    "driver_age_range",   # engineered: max_driver_age - min_driver_age
    "share_male_drivers",
    "max_vehicle_age",
    "any_skidded",
    "any_left_carriageway",
    "has_pedestrian",
    "has_cyclist",
    "has_child_casualty",
    "min_casualty_age",
    "max_casualty_age",
    "is_classified_road",
    "is_classified_second_road",
    "hour_sin",
    "hour_cos",
    "min_driver_age_was_missing",
    "max_driver_age_was_missing",
    "mean_driver_age_was_missing",
    "max_vehicle_age_was_missing",
    "min_casualty_age_was_missing",
    "max_casualty_age_was_missing",
    "share_male_drivers_was_missing",
]

STRATEGIES = ["S0", "S1", "S2", "S3", "S4"]  



# Feature engineering

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["driver_age_range"] = df["max_driver_age"] - df["min_driver_age"]
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Categorical columns may be a mix of int codes and the "Missing"
    # string from clean_dataset.py — cast to str uniformly so OrdinalEncoder
    # / OneHotEncoder see one consistent dtype per column.
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    missing_cols = set(CATEGORICAL_COLS + NUMERIC_COLS) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Expected columns not found in dataframe: {missing_cols}")

    return df


# Strategy-specific fit

def fit_strategy(strategy: str, X_fit: pd.DataFrame, y_fit: pd.Series, seed: int) -> dict:
    n_cat = len(CATEGORICAL_COLS)
    n_num = len(NUMERIC_COLS)

    if strategy == "S4":
        pre = ColumnTransformer(
            [
                ("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
                ("num_scale", StandardScaler(), NUMERIC_COLS),
            ]
        )
        X_fit_enc = pre.fit_transform(X_fit)
        n_ohe = sum(len(c) for c in pre.named_transformers_["cat_ohe"].categories_)
        sampler = ADASYN(random_state=seed)
        X_res, y_res = sampler.fit_resample(X_fit_enc, y_fit)
        X_res[:, :n_ohe] = np.clip(np.round(X_res[:, :n_ohe]), 0, 1)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_res, y_res)
        return {"kind": "onehot", "pre": pre, "model": clf}

  
    pre_ord = ColumnTransformer(
        [
            ("cat_ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_COLS),
            ("num_pass", "passthrough", NUMERIC_COLS),
        ]
    )
    X_fit_ord = pre_ord.fit_transform(X_fit)
    if strategy == "S0":
        X_res, y_res, class_weight = X_fit_ord, y_fit, None
    elif strategy == "S1":
        X_res, y_res, class_weight = X_fit_ord, y_fit, "balanced"
    elif strategy in ("S2", "S3"):
        sampler = SMOTENC(categorical_features=list(range(n_cat)), random_state=seed)
        X_res, y_res = sampler.fit_resample(X_fit_ord, y_fit)
        class_weight = "balanced" if strategy == "S3" else None
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    post = ColumnTransformer(
        [
            ("cat_ohe", OneHotEncoder(handle_unknown="ignore"), list(range(n_cat))),
            ("num_scale", StandardScaler(), list(range(n_cat, n_cat + n_num))),
        ]
    )
    X_res_enc = post.fit_transform(X_res)

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight=class_weight)
    clf.fit(X_res_enc, y_res)
    return {"kind": "ordinal", "pre_ord": pre_ord, "post": post, "model": clf}


def predict_proba(fitted: dict, X: pd.DataFrame) -> np.ndarray:
    if fitted["kind"] == "onehot":
        X_enc = fitted["pre"].transform(X)
    else:
        X_ord = fitted["pre_ord"].transform(X)
        X_enc = fitted["post"].transform(X_ord)
    return fitted["model"].predict_proba(X_enc)[:, 1]


# Threshold tuning + metrics

def tune_threshold(y_val: np.ndarray, probs_val: np.ndarray, mode: str, target_recall: float) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    precisions, recalls = precisions[:-1], recalls[:-1]  # align with thresholds

    if mode == "f2":
        f2 = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-12)
        if not np.isfinite(f2).any():
            return 0.5
        return float(thresholds[np.nanargmax(f2)])

    if mode == "recall_target":
        valid = recalls >= target_recall
        if not valid.any():
            log.warning("No threshold reaches target recall=%.2f on this validation slice; falling back to 0.5", target_recall)
            return 0.5
        candidate_thresholds = thresholds[valid]
        candidate_precisions = precisions[valid]
        return float(candidate_thresholds[np.argmax(candidate_precisions)])

    raise ValueError(f"Unknown threshold mode: {mode}")


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "f2": fbeta_score(y_true, preds, beta=2, zero_division=0),
        # threshold-independent ranking metrics, included on every row for convenience
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
    }


# Main CV loop
def run_cv(
    df: pd.DataFrame,
    target: str,
    n_splits: int,
    n_repeats: int,
    threshold_mode: str,
    target_recall: float,
) -> pd.DataFrame:
    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[target].astype(int)

    log.info(
        "Running Protocol A: %d-fold x %d repeats (%d outer fits) on target=%s (positive rate=%.4f)",
        n_splits, n_repeats, n_splits * n_repeats, target, y.mean(),
    )

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    results = []

    for fold_i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        seed = 1000 + fold_i
        X_train_full, y_train_full = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Stratified 25% validation slice from the training fold, used only
        # for threshold selection. Resampling touches the remaining 75%
        # ("fit" portion) only — never the val or test data.
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=seed
        )

        for strategy in STRATEGIES:
            log.info("Fold %d/%d — strategy %s", fold_i + 1, n_splits * n_repeats, strategy)
            fitted = fit_strategy(strategy, X_fit, y_fit, seed)

            val_probs = predict_proba(fitted, X_val)
            test_probs = predict_proba(fitted, X_test)
            tuned_threshold = tune_threshold(y_val.values, val_probs, threshold_mode, target_recall)

            row_default = compute_metrics(y_test.values, test_probs, 0.5)
            row_default.update(strategy=strategy, threshold_type="default_0.5", fold=fold_i)
            results.append(row_default)

            row_tuned = compute_metrics(y_test.values, test_probs, tuned_threshold)
            row_tuned.update(strategy=strategy, threshold_type="validation_tuned", fold=fold_i)
            results.append(row_tuned)

            if strategy == "S0":
                row_s5 = dict(row_tuned)
                row_s5["strategy"] = "S5"
                results.append(row_s5)

    return pd.DataFrame(results)


def summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["precision", "recall", "f1", "f2", "roc_auc", "pr_auc"]
    summary = results_df.groupby(["strategy", "threshold_type"])[metrics].agg(["mean", "std"])
    return summary


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_in_path = repo_root / "data" / "collisions_clean.parquet"
    default_out_path = repo_root / "reports" / "logreg_results.csv"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=str(default_in_path))
    parser.add_argument("--target", default="target_severe_fatal", choices=["target_severe_fatal", "target_fatal_only"])
    parser.add_argument("--out", default=str(default_out_path))
    parser.add_argument("--n-splits", type=int, default=5, help="Proposal default: 5 (fallback: 3 if too slow)")
    parser.add_argument("--n-repeats", type=int, default=3, help="Proposal default: 3 (fallback: 2 if too slow)")
    parser.add_argument("--threshold-mode", default="f2", choices=["f2", "recall_target"])
    parser.add_argument("--target-recall", type=float, default=0.8)
    parser.add_argument("--quick", action="store_true", help="3-fold x 1 repeat, for fast debugging — not the reported protocol")
    args = parser.parse_args()

    if args.quick:
        args.n_splits, args.n_repeats = 3, 1
        log.warning("--quick set: using 3-fold x 1 repeat. Re-run without --quick for reportable results.")

    df = pd.read_parquet(args.in_path)
    log.info("Loaded %s: %s", args.in_path, df.shape)

    df = engineer_features(df)

    results_df = run_cv(df, args.target, args.n_splits, args.n_repeats, args.threshold_mode, args.target_recall)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    log.info("Wrote per-fold results to %s", out_path)

    summary = summarize(results_df)
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path)
    log.info("Wrote summary (mean/std per strategy x threshold_type) to %s", summary_path)
    print("\n" + summary.to_string())


if __name__ == "__main__":
    main()