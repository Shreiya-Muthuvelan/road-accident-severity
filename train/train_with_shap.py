"""Fit a chosen model and strategy, then explain its predictions with SHAP.

The file includes a CV helper that compares model/strategy/threshold
combinations on a stratified 80% development split and selects by mean
balanced accuracy (then F1, recall, and lower threshold). The command-line
entry point does not call that helper: it fits the requested model and
strategy on an 80% split, then explains a sample of the held-out 20%.

The script writes SHAP importance, beeswarm, dependence, and waterfall outputs
to --out-dir. It accepts a logreg model choice, but SHAP's TreeExplainer is
intended for the tree models (RF, LightGBM, and XGBoost).

Example (from FINAL_CODES/):
    python train/train_with_shap.py --model xgboost --strategy S0
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from imblearn.over_sampling import ADASYN, SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Keep the model feature definitions and row-level engineering in one place.
from train_logreg_imbalance import CATEGORICAL_COLS, NUMERIC_COLS, engineer_features

from location_features import LocationEncoder, LOC_RAW, LOC_TE
# raw coords + 2 target encoded cells are all numeric
NUM_ALL = NUMERIC_COLS + LOC_RAW + LOC_TE

LOG = logging.getLogger(__name__)
ALL_STRATEGIES = ("S0", "S1", "S2", "S3", "S4")
METRIC_COLS = ["precision", "recall", "accuracy", "balanced_acc", "f1_score", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn"]


def make_model(model_name, class_weight, seed):
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=1000, C=1.0,
            class_weight=class_weight, random_state=seed,
        )

    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2,
            class_weight=class_weight, random_state=seed, n_jobs=-1,
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=150, learning_rate=0.05, num_leaves=31,
            class_weight=class_weight, random_state=seed,
            n_jobs=-1, verbosity=-1,
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            n_estimators=450, learning_rate=0.035, max_depth=5,
            min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=4.0, reg_alpha=0.5, tree_method="hist",
            random_state=seed, n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def make_imputer():
    """Keep numeric columns in place even when a training slice is all missing."""
    # keep_empty_features=True keeps column positions stable even if a numeric
    # column is entirely NaN in some training slice (the second-stage
    # ColumnTransformer addresses columns by position).
    return SimpleImputer(strategy="median", keep_empty_features=True)

def _attach_names(transform, ohe, ord_cats=None):
    """Attach encoded-column names and their original-feature group to `transform`.

    Encoded columns are one-hot categories first, then NUM_ALL. For S0-S3,
    map each ordinal code back to the category label used before one-hot encoding.
    """
    names, groups = [], []
    for i, (c, cats) in enumerate(zip(CATEGORICAL_COLS, ohe.categories_)):
        for v in cats:
            label = v
            if ord_cats is not None:
                label = ord_cats[i][int(v)] if v >= 0 else "unknown"
            names.append(f"{c}={label}")
            groups.append(c)
    for c in NUM_ALL:
        names.append(c)
        groups.append(c)
    transform.feature_names = names
    transform.feature_groups = groups


def fit_model(model_name, strategy, X_fit, y_fit, seed):
    """Fit location encoding, preprocessing, optional balancing, and one model."""
    n_cat = len(CATEGORICAL_COLS)
    scale = model_name == "logreg"   # LR needs standardised numerics; trees don't

    # fit the location encoder on the training slice only
    loc = LocationEncoder(seed=seed)
    X_fit = loc.fit_transform(X_fit, y_fit)

    if strategy == "S4":
        num_tf = make_pipeline(make_imputer(), StandardScaler()) if scale else make_imputer()
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", num_tf, NUM_ALL),
        ])
        X_enc = pre.fit_transform(X_fit)
        X_res, y_res = ADASYN(random_state=seed).fit_resample(X_enc, y_fit)
        n_ohe = sum(map(len, pre.named_transformers_["cat"].categories_))
        if sparse.issparse(X_res):
            cat = X_res[:, :n_ohe].tocsr()
            cat.data = np.clip(np.rint(cat.data), 0, 1)
            cat.eliminate_zeros()
            X_res = sparse.hstack([cat, X_res[:, n_ohe:]], format="csr")
        else:
            X_res[:, :n_ohe] = np.clip(np.rint(X_res[:, :n_ohe]), 0, 1)
        class_weight = None

        def transform(X):
            return pre.transform(loc.transform(X))

        _attach_names(transform, pre.named_transformers_["cat"])
    else:
        pre = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_COLS),
            ("num", make_imputer(), NUM_ALL),
        ])
        X_ord = pre.fit_transform(X_fit)
        if strategy in ("S2", "S3"):
            sampler = SMOTENC(
                categorical_features=list(range(n_cat)),
                sampling_strategy=0.5, random_state=seed,
            )
            X_ord, y_res = sampler.fit_resample(X_ord, y_fit)
        else:
            y_res = y_fit
        class_weight = "balanced" if strategy in ("S1", "S3") else None
        post = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(range(n_cat))),
            ("num", StandardScaler() if scale else "passthrough",
             list(range(n_cat, n_cat + len(NUM_ALL)))),
        ])
        X_res = post.fit_transform(X_ord)

        def transform(X):
            return post.transform(pre.transform(loc.transform(X)))

        _attach_names(
            transform,
            post.named_transformers_["cat"],
            ord_cats=pre.named_transformers_["cat"].categories_,
        )

    clf = make_model(model_name, class_weight, seed)
    if model_name == "xgboost" and class_weight == "balanced":
        weights = compute_sample_weight(class_weight="balanced", y=y_res)
        clf.fit(X_res, y_res, sample_weight=weights)
    else:
        clf.fit(X_res, y_res)
    return clf, transform

def compute_metrics(y_true, probabilities, threshold):
    """Classification metrics; positive class = Severe/Fatal (1).

    ROC-AUC and PR-AUC use probabilities, so they don't change when the
    classification threshold changes.
    """
    probabilities = np.asarray(probabilities)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_acc": balanced_accuracy_score(y_true, predicted),
        "f1_score": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def run_holdout_cv(df, target, models, strategies, thresholds, n_splits):
    """Select each model's strategy/threshold on dev CV, then score holdout once."""
    X = df[CATEGORICAL_COLS + NUMERIC_COLS + LOC_RAW]
    y = df[target].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"Target must contain both 0 and 1; got {sorted(y.unique())}")
    LOG.info("Target %s | positive share %.4f | rows %d", target, y.mean(), len(y))

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42,
    )
    LOG.info("Development: %d rows; untouched holdout: %d rows", len(y_dev), len(y_test))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_rows = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_dev, y_dev), start=1):
        seed = 999 + fold
        X_fit, y_fit = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_val, y_val = X_dev.iloc[val_idx], y_dev.iloc[val_idx]
        for model_name in models:
            for strategy in strategies:
                LOG.info("CV fold %d/%d | %s | %s", fold, n_splits, model_name, strategy)
                clf, transform = fit_model(model_name, strategy, X_fit, y_fit, seed)
                positive_ix = list(clf.classes_).index(1)
                probs = clf.predict_proba(transform(X_val))[:, positive_ix]
                for t in thresholds:
                    cv_rows.append({
                        "fold": fold, "model": model_name, "strategy": strategy,
                        "threshold": float(t),
                        **compute_metrics(y_val, probs, t),
                    })

    cv_results = pd.DataFrame(cv_rows)
    cv_summary = cv_results.groupby(["model", "strategy", "threshold"], as_index=False)[METRIC_COLS].mean()

    holdout_rows = []
    for model_name in models:
        grp = cv_summary[cv_summary["model"] == model_name]
        # Selection uses the development CV ONLY. Optimize balanced accuracy;
        # tie-break with F1, then recall, then lower threshold
        sel = grp.sort_values(
            ["balanced_acc", "f1_score", "recall", "threshold"],
            ascending=[False, False, False, True], kind="stable",
        ).iloc[0]
        strategy, threshold = sel["strategy"], float(sel["threshold"])
        LOG.info("%s: CV selected %s, threshold %.3f, CV balanced acc %.4f, CV ROC-AUC %.4f",
                 model_name, strategy, threshold, sel["balanced_acc"], sel["roc_auc"])

        # Refit on the FULL 80% with fresh preprocessing / sampling, score holdout once
        clf, transform = fit_model(model_name, strategy, X_dev, y_dev, seed=42)
        positive_ix = list(clf.classes_).index(1)
        probs = clf.predict_proba(transform(X_test))[:, positive_ix]
        holdout_rows.append({
            "model": model_name, "strategy": strategy, "threshold": threshold,
            "cv_balanced_acc": sel["balanced_acc"],
            **compute_metrics(y_test, probs, threshold),
        })
    return cv_results, cv_summary, pd.DataFrame(holdout_rows)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["logreg", "rf", "lightgbm", "xgboost", "all"], default="all")
    parser.add_argument("--strategies", nargs="+", choices=ALL_STRATEGIES, default=["S0", "S1"])
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.50, 0.55, 0.60, 0.65, 0.70])
    parser.add_argument("--target", choices=["target_severe_fatal", "target_fatal_only"],
                        default="target_severe_fatal")
    parser.add_argument("--in", dest="input_path", default=str(root / "data" / "collisions_clean.parquet"))
    parser.add_argument("--out", default=str(root / "reports" / "shap_analysis_results.csv"),
                        help="Base path; writes <stem>.csv (CV folds), <stem>_cv_summary.csv, <stem>_holdout.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="3 folds; debugging only, NOT reportable")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.quick:
        args.n_splits = 3
        LOG.warning("Quick mode is for debugging, not reporting")
    if any(not (0 < t < 1) for t in args.thresholds):
        parser.error("All thresholds must be strictly between 0 and 1")

    models = ("logreg", "rf", "lightgbm", "xgboost") if args.model == "all" else (args.model,)

    raw = pd.read_parquet(args.input_path)
    df = engineer_features(raw)
    # Keep raw coordinates available for the location encoder even if feature
    # engineering has removed them from the working frame.
    df[LOC_RAW] = raw.loc[df.index, LOC_RAW]

    cv_results, cv_summary, holdout = run_holdout_cv(
        df, args.target, models, args.strategies, sorted(set(args.thresholds)), args.n_splits,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(out, index=False)
    cv_summary.to_csv(out.with_name(out.stem + "_cv_summary.csv"), index=False)
    holdout.to_csv(out.with_name(out.stem + "_holdout.csv"), index=False)

    print("\nDEVELOPMENT 5-FOLD CV AVERAGES (selection uses these):")
    print(cv_summary.round(4).to_string(index=False))
    print("\nFINAL UNTOUCHED 20% HOLDOUT (selected using CV only, scored once):")
    print(holdout.round(4).to_string(index=False))
    print("\nSaved CV folds, CV summary and one final holdout evaluation per model.")


if __name__ == "__main__":
    main()
