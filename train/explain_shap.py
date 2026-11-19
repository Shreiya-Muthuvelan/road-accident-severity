"""Explain a selected tree model on a sample of the held-out collision data.

The script fits the requested model and strategy on a stratified 80% development
split, then explains up to --n-samples rows from the 20% holdout. It does not
run CV to choose the model or strategy.

Outputs (in --out-dir):
  shap_importance_grouped.csv   mean |SHAP| per ORIGINAL feature (one-hot columns summed)
  shap_bar_grouped.png          bar chart of the above
  shap_beeswarm_encoded.png     beeswarm over the top encoded columns
  shap_dependence_<feat>.png    dependence plot for the top numeric feature
  shap_waterfall_top_risk.png   local explanation for the highest-risk sample

This file imports fit_model from train_with_shap.py, which must be
available in the train folder and attach transform.feature_names /
transform.feature_groups. Example (from FINAL_CODES/):
    python train/explain_shap.py --model xgboost --strategy S0 --n-samples 3000
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.model_selection import train_test_split

from train_logreg_imbalance import CATEGORICAL_COLS, NUMERIC_COLS, engineer_features
from location_features import LOC_RAW, LOC_TE
from train_with_shap import fit_model

LOG = logging.getLogger(__name__)


def to_dense(a):
    """Convert a sparse matrix to an array, leaving dense input as an array."""
    return a.toarray() if sparse.issparse(a) else np.asarray(a)


def positive_class_shap(explainer, X):
    """Return SHAP values for class 1 and its expected/base model output."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # older shap/LightGBM: [class0, class1]
        sv = sv[1]
    elif sv.ndim == 3:                # newer shap/RF: (n, f, classes)
        sv = sv[:, :, 1]
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    return sv, base


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["rf", "lightgbm", "xgboost"], default="xgboost")
    p.add_argument("--strategy", choices=["S0", "S1", "S2", "S3", "S4"], default="S0",
                   help="S0 recommended: explains the model on the real data distribution")
    p.add_argument("--target", choices=["target_severe_fatal", "target_fatal_only"], default="target_severe_fatal")
    p.add_argument("--in", dest="input_path", default=str(root / "data" / "collisions_clean.parquet"))
    p.add_argument("--out-dir", default=str(root / "reports" / "shap"))
    p.add_argument("--n-samples", type=int, default=3000, help="Holdout rows to explain (RF is slow; keep small)")
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Keep coordinates available for the location encoder, even if feature
    # engineering removed them from the working frame.
    raw = pd.read_parquet(args.input_path)
    df = engineer_features(raw)
    for c in LOC_RAW:
        if c not in df.columns:
            df[c] = raw.loc[df.index, c]

    X = df[CATEGORICAL_COLS + NUMERIC_COLS + LOC_RAW]
    y = df[args.target].astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    LOG.info("Fitting %s / %s on %d dev rows", args.model, args.strategy, len(y_dev))
    clf, transform = fit_model(args.model, args.strategy, X_dev, y_dev, seed=42)
    names, groups = transform.feature_names, np.array(transform.feature_groups)

    n = min(args.n_samples, len(X_test))
    X_s = X_test.sample(n=n, random_state=0)
    Xt = to_dense(transform(X_s)).astype(float)
    assert Xt.shape[1] == len(names), f"{Xt.shape[1]} columns vs {len(names)} names"

    LOG.info("Computing SHAP values for %d rows x %d features", *Xt.shape)
    explainer = shap.TreeExplainer(clf)
    sv, base = positive_class_shap(explainer, Xt)

    # Aggregate one-hot columns back to their original feature (SHAP is additive)
    uniq = list(dict.fromkeys(groups))
    grouped = np.column_stack([sv[:, groups == g].sum(axis=1) for g in uniq])
    imp = (pd.DataFrame({"feature": uniq, "mean_abs_shap": np.abs(grouped).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False))
    imp.to_csv(out / "shap_importance_grouped.csv", index=False)
    print(imp.head(args.top_k).round(4).to_string(index=False))

    top = imp.head(args.top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top) + 1.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4c72b0")
    ax.set_xlabel("mean |SHAP| (model output units, grouped by original feature)")
    ax.set_title(f"{args.model} / {args.strategy}: global feature importance")
    fig.tight_layout()
    fig.savefig(out / "shap_bar_grouped.png", dpi=150)
    plt.close(fig)

    # SHAP Beeswarm Plot
    plt.figure(figsize=(16, 10))

    shap.summary_plot(
        sv,
        Xt,
        feature_names=names,
        max_display=args.top_k,
        show=False,
        plot_size=None
    )

    fig = plt.gcf()
    ax = plt.gca()

    # Increase horizontal space for the actual graph
    fig.set_size_inches(16, 10)

    # Move feature labels further to the left
    ax.tick_params(axis="y", labelsize=10, pad=8)

    # Improve readability
    ax.tick_params(axis="x", labelsize=10)

    # Adjust margins
    fig.subplots_adjust(
        left=0.36,
        right=0.88,
        top=0.95,
        bottom=0.12
    )

    # Save high-resolution image
    plt.savefig(
        out / "shap_beeswarm_encoded.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.3
    )

    plt.close(fig)

    num_cols = NUMERIC_COLS + LOC_RAW + LOC_TE
    top_num = next((f for f in imp["feature"] if f in num_cols), None)
    if top_num is not None:
        shap.dependence_plot(names.index(top_num), sv, Xt, feature_names=names, show=False)
        plt.tight_layout()
        plt.savefig(out / f"shap_dependence_{top_num}.png", dpi=150, bbox_inches="tight")
        plt.close()

    i = int(np.argmax(sv.sum(axis=1)))
    exp = shap.Explanation(values=sv[i], base_values=base, data=Xt[i], feature_names=names)
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.savefig(out / "shap_waterfall_top_risk.png", dpi=150, bbox_inches="tight")
    plt.close()

    LOG.info("Wrote SHAP outputs to %s", out)


if __name__ == "__main__":
    main()
