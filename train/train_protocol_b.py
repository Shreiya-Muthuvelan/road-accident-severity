"""Protocol B: temporal split evaluation for XGBoost only.

Design
------
* Train years = all years except the last (default), test year = the last year.
  Override with --train-years / --test-year.
* Strategy + threshold are selected on a TEMPORAL validation slice: the most
  recent training year is held out as validation, the earlier train years are
  used to fit. (If there is only one training year, a stratified 80/20 split of
  it is used instead.)
* The chosen (strategy, threshold) is refit on ALL training years and scored
  ONCE on the test year. The test year is never used for selection.
* Same engineered features, location encoder and fit_model() as Protocol A.

Place in train/ beside train_with_shap.py. Run from repo root:
    python train/train_protocol_b.py --strategies S0 S1 S2 S3 S4 \
        --out reports/protocol_b_xgboost.csv
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from train_logreg_imbalance import CATEGORICAL_COLS, NUMERIC_COLS, engineer_features
from location_features import LOC_RAW
from train_with_shap import fit_model

LOG = logging.getLogger(__name__)
ALL_STRATEGIES = ("S0", "S1", "S2", "S3", "S4")


def compute_metrics(y_true, probs, threshold):
    probs = np.asarray(probs)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "f1_score": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def predict(clf, transform, X):
    return clf.predict_proba(transform(X))[:, list(clf.classes_).index(1)]


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="input_path", default=str(root / "data" / "collisions_clean.parquet"))
    p.add_argument("--out", default=str(root / "reports" / "protocol_b_xgboost.csv"))
    p.add_argument("--target", choices=["target_severe_fatal", "target_fatal_only"], default="target_severe_fatal")
    p.add_argument("--strategies", nargs="+", choices=ALL_STRATEGIES, default=["S0", "S1"])
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    p.add_argument("--train-years", nargs="+", type=int, default=None)
    p.add_argument("--test-year", type=int, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw = pd.read_parquet(args.input_path)
    df = engineer_features(raw)  # drops collision_year, lat/long
    df[LOC_RAW] = raw.loc[df.index, LOC_RAW]
    year = raw.loc[df.index, "collision_year"].astype(int)

    years = sorted(year.unique())
    test_year = args.test_year or years[-1]
    train_years = sorted(args.train_years) if args.train_years else [y for y in years if y < test_year]
    if not train_years or max(train_years) >= test_year:
        p.error(f"Train years {train_years} must all precede test year {test_year} (available: {years})")
    LOG.info("Train years %s | test year %d", train_years, test_year)

    X = df[CATEGORICAL_COLS + NUMERIC_COLS + LOC_RAW]
    y = df[args.target].astype(int)

    tr_mask = year.isin(train_years)
    X_train, y_train = X[tr_mask], y[tr_mask]
    X_test, y_test = X[year == test_year], y[year == test_year]
    LOG.info("Train rows %d (pos %.4f) | test rows %d (pos %.4f)",
             len(y_train), y_train.mean(), len(y_test), y_test.mean())

    # Temporal validation slice for selection
    if len(train_years) > 1:
        val_year = max(train_years)
        fit_mask = tr_mask & (year != val_year)
        X_fit, y_fit = X[fit_mask], y[fit_mask]
        X_val, y_val = X[year == val_year], y[year == val_year]
        LOG.info("Selection: fit on %s, validate on %d", [t for t in train_years if t != val_year], val_year)
    else:
        LOG.warning("Only one training year: using stratified 80/20 split for selection")
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.20, stratify=y_train, random_state=42)

    thresholds = sorted(set(args.thresholds))
    val_rows = []
    for strategy in args.strategies:
        LOG.info("Validation fit | xgboost | %s", strategy)
        clf, transform = fit_model("xgboost", strategy, X_fit, y_fit, seed=42)
        probs = predict(clf, transform, X_val)
        for t in thresholds:
            val_rows.append({"strategy": strategy, "threshold": t, **compute_metrics(y_val, probs, t)})
    val = pd.DataFrame(val_rows)

    sel = val.sort_values(["balanced_acc", "f1_score", "recall", "threshold"],
                          ascending=[False, False, False, True], kind="stable").iloc[0]
    strategy, threshold = sel["strategy"], float(sel["threshold"])
    LOG.info("Selected %s @ %.2f (val balanced acc %.4f)", strategy, threshold, sel["balanced_acc"])

    # Refit on all training years, score the test year once
    clf, transform = fit_model("xgboost", strategy, X_train, y_train, seed=42)
    probs = predict(clf, transform, X_test)
    rows = [
        {"model": "xgboost", "protocol": "B", "strategy": strategy, "threshold": threshold,
         "train_years": ",".join(map(str, train_years)), "test_year": test_year,
         "val_balanced_acc": sel["balanced_acc"], "note": "selected",
         **compute_metrics(y_test, probs, threshold)},
        {"model": "xgboost", "protocol": "B", "strategy": strategy, "threshold": 0.50,
         "train_years": ",".join(map(str, train_years)), "test_year": test_year,
         "val_balanced_acc": np.nan, "note": "reference @0.50",
         **compute_metrics(y_test, probs, 0.50)},
    ]
    result = pd.DataFrame(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    val.to_csv(out.with_name(out.stem + "_validation.csv"), index=False)

    print("\nTEMPORAL VALIDATION (selection uses these):")
    print(val.round(4).to_string(index=False))
    print(f"\nFINAL TEMPORAL TEST (year {test_year}, scored once):")
    print(result.round(4).to_string(index=False))
    print("\nCompare against Protocol A holdout (FINAL_all_model_results_holdout.csv): "
          "a drop in ROC/PR-AUC here indicates temporal drift / optimism in random splits.")


if __name__ == "__main__":
    main()
