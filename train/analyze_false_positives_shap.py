"""Inspect XGBoost false positives and their SHAP contributions.

Place in train/ and run from the repository root:
 python train/analyze_false_positives_shap.py --strategy S1 --threshold 0.50

The script fits the requested strategy on a stratified 80% development split
and scores the untouched 20% holdout at a fixed threshold. It uses the full
holdout for confusion counts and motorcycle-group rates, then calculates SHAP
on a reproducible sample from each non-empty confusion-matrix group.
It imports fit_model from train_combination_2.py, which must be available in
the train folder.
"""
import argparse
import inspect
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from train_combination_2 import fit_model, NUM_ALL
from train_logreg_imbalance import CATEGORICAL_COLS, NUMERIC_COLS, engineer_features
from location_features import LOC_RAW

COLORS = {'FP': '#d95f70', 'TP': '#397ca5', 'TN': '#799b83', 'FN': '#c19a54'}


def encoded_feature_names(transform, n_features):
    """Recover fitted one-hot names from fit_model's transform closure.

    Also supports a newer version that exposes transform.feature_names.
    """
    if hasattr(transform, 'feature_names'):
        names = list(transform.feature_names)
    else:
        captured = inspect.getclosurevars(transform).nonlocals
        transformer = captured.get('post', captured.get('pre'))
        if transformer is None:
            raise RuntimeError('Unable to locate fitted pre/post transformer.')
        onehot = transformer.named_transformers_['cat']
        cat_names = list(onehot.get_feature_names_out(CATEGORICAL_COLS))
        names = cat_names + list(NUM_ALL)
    if len(names) != n_features:
        raise RuntimeError(f'Feature-name mismatch: {len(names)} labels, {n_features} columns')
    return names


def positive_shap(explainer, matrix):
    """Return SHAP values for the positive (severe/fatal) class."""
    values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[1]
    elif np.asarray(values).ndim == 3:
        values = values[:, :, 1]
    return np.asarray(values)


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--in', dest='input_path', default=str(root / 'data' / 'collisions_clean.parquet'))
    p.add_argument('--out-dir', default=str(root / 'reports' / 'shap' / 'false_positive_analysis'))
    p.add_argument('--strategy', choices=['S0', 'S1', 'S2', 'S3', 'S4'], default='S1')
    p.add_argument('--threshold', type=float, default=0.50)
    p.add_argument('--n-per-group', type=int, default=1000,
                   help='Max samples from each confusion group for SHAP (counts use full holdout)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if not 0 < args.threshold < 1 or args.n_per_group < 1:
        p.error('Threshold must be between 0 and 1; n-per-group must be positive.')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(args.input_path)
    df = engineer_features(raw)
    for c in LOC_RAW:
        if c not in df.columns:
            df[c] = raw.loc[df.index, c]
    X = df[CATEGORICAL_COLS + NUMERIC_COLS + LOC_RAW]
    y = df['target_severe_fatal'].astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    print(f'Training XGBoost, strategy={args.strategy} on development split...')
    clf, transform = fit_model('xgboost', args.strategy, X_dev, y_dev, seed=42)
    X_enc = transform(X_test)
    proba = clf.predict_proba(X_enc)[:, list(clf.classes_).index(1)]
    pred = (proba >= args.threshold).astype(int)
    actual = y_test.to_numpy()
    group = np.select(
        [(actual == 0) & (pred == 1), (actual == 1) & (pred == 1),
         (actual == 0) & (pred == 0), (actual == 1) & (pred == 0)],
        ['FP', 'TP', 'TN', 'FN'], default='?')
    counts = pd.Series(group).value_counts().reindex(['TN', 'FP', 'FN', 'TP'], fill_value=0)
    print('\nHoldout counts (all rows):')
    print(counts.to_string())
    print('\nConfusion matrix (rows actual 0/1, columns predicted 0/1):')
    print(confusion_matrix(actual, pred, labels=[0, 1]))

    # Group-level denominators matter: P(false positive | actual slight, motorcycle status).
    # This reports false-positive RATE, not merely motorcycle prevalence among FPs.
    status = X_test['has_motorcycle'].to_numpy()
    status_bool = pd.Series(status).astype('Int64').eq(1).fillna(False).to_numpy(dtype=bool)
    rates = []
    for motorcycle in [0, 1]:
        mask = status_bool == bool(motorcycle)
        n_slight = int(np.sum(mask & (actual == 0)))
        fp = int(np.sum(mask & (group == 'FP')))
        n_severe = int(np.sum(mask & (actual == 1)))
        tp = int(np.sum(mask & (group == 'TP')))
        rates.append({'has_motorcycle': motorcycle, 'holdout_n': int(mask.sum()),
                      'actual_slight_n': n_slight, 'false_positives': fp,
                      'false_positive_rate': fp / n_slight if n_slight else np.nan,
                      'actual_severe_fatal_n': n_severe, 'true_positives': tp,
                      'recall': tp / n_severe if n_severe else np.nan})
    rates_df = pd.DataFrame(rates)
    rates_df.to_csv(out / 'fp_rates_by_motorcycle.csv', index=False)
    print('\nFalse-positive rates by motorcycle involvement (whole holdout):')
    print(rates_df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    rng = np.random.default_rng(args.seed)
    selected = []
    for label in ['FP', 'TP', 'TN', 'FN']:
        inds = np.flatnonzero(group == label)
        if len(inds):
            chosen = rng.choice(inds, size=min(args.n_per_group, len(inds)), replace=False)
            selected.extend(chosen.tolist())
    selected = np.asarray(selected, dtype=int)
    if not len(selected):
        raise RuntimeError('No holdout examples selected.')
    sampled_enc = X_enc[selected]
    if sparse.issparse(sampled_enc):
        sampled_enc = sampled_enc.toarray()
    sampled_enc = np.asarray(sampled_enc, dtype=float)
    names = encoded_feature_names(transform, sampled_enc.shape[1])
    if 'has_motorcycle' not in names:
        raise RuntimeError('has_motorcycle is not a standalone encoded feature; inspect encoded names.')
    col = names.index('has_motorcycle')

    print(f'\nCalculating SHAP for {len(selected)} sampled rows...')
    explainer = shap.TreeExplainer(clf)
    sv = positive_shap(explainer, sampled_enc)
    if sv.shape != sampled_enc.shape:
        raise RuntimeError(f'SHAP shape {sv.shape} differs from encoded X shape {sampled_enc.shape}')

    sample_details = pd.DataFrame({
        'holdout_position': selected,
        'original_index': X_test.index.to_numpy()[selected],
        'group': group[selected], 'actual': actual[selected], 'predicted': pred[selected],
        'predicted_probability': proba[selected],
        'has_motorcycle': status[selected],
        'has_motorcycle_SHAP': sv[:, col],
    })
    sample_details.to_csv(out / 'sampled_predictions_motorcycle_shap.csv', index=False)

    summary = (sample_details.groupby(['group', 'has_motorcycle'], dropna=False)
               .agg(sample_n=('has_motorcycle_SHAP', 'size'),
                    mean_SHAP=('has_motorcycle_SHAP', 'mean'),
                    mean_absolute_SHAP=('has_motorcycle_SHAP', lambda z: np.abs(z).mean()),
                    fraction_positive_SHAP=('has_motorcycle_SHAP', lambda z: (z > 0).mean()),
                    mean_predicted_probability=('predicted_probability', 'mean'))
               .reset_index())
    summary.to_csv(out / 'motorcycle_shap_by_outcome.csv', index=False)
    print('\nSHAP summary by prediction outcome and motorcycle status (sample):')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    group_rows = []
    for label in ['FP', 'TP', 'TN', 'FN']:
        mask = group[selected] == label
        if not mask.any():
            continue
        vals = sv[mask]
        mean_abs = np.abs(vals).mean(axis=0)
        mean_signed = vals.mean(axis=0)
        for name, a, s in zip(names, mean_abs, mean_signed):
            group_rows.append({'group': label, 'feature': name,
                               'mean_abs_SHAP': a, 'mean_signed_SHAP': s,
                               'sample_n': int(mask.sum())})
    importance = pd.DataFrame(group_rows)
    importance.to_csv(out / 'encoded_shap_importance_by_outcome.csv', index=False)

    fp_imp = importance[importance['group'] == 'FP'].nlargest(15, 'mean_abs_SHAP').iloc[::-1]
    if not fp_imp.empty:
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(fp_imp['feature'], fp_imp['mean_abs_SHAP'], color=COLORS['FP'])
        ax.set_xlabel('Mean absolute SHAP on false positives (raw XGBoost output)')
        ax.set_title('Top 15 features contributing to false-positive predictions')
        fig.subplots_adjust(left=0.38, right=0.97, bottom=0.12, top=0.91)
        fig.savefig(out / 'fp_top_features.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    groups_to_plot = [g for g in ['FP', 'TP', 'TN', 'FN'] if np.any(group[selected] == g)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    vals = [sample_details.loc[sample_details['group'] == g, 'has_motorcycle_SHAP'].to_numpy()
            for g in groups_to_plot]
    b = ax.boxplot(vals, tick_labels=groups_to_plot, patch_artist=True, showfliers=False)
    for patch, g in zip(b['boxes'], groups_to_plot):
        patch.set_facecolor(COLORS[g]); patch.set_alpha(0.7)
    ax.axhline(0, color='#555555', linewidth=1)
    ax.set_ylabel('SHAP value: has_motorcycle (raw XGBoost output)')
    ax.set_title('Motorcycle feature contribution by prediction outcome')
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / 'motorcycle_shap_by_outcome.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['No motorcycle', 'Motorcycle'], 100 * rates_df['false_positive_rate'],
           color=['#397ca5', '#d95f70'])
    ax.set_ylabel('False-positive rate among actual slight accidents (%)')
    ax.set_title('False-positive rate by motorcycle involvement (full holdout)')
    ax.set_ylim(0, max(1, 100 * rates_df['false_positive_rate'].max() * 1.2))
    for i, r in rates_df.iterrows():
        ax.text(i, 100 * r['false_positive_rate'],
                f"{100 * r['false_positive_rate']:.1f}%\n({int(r['false_positives'])}/{int(r['actual_slight_n'])})",
                ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'fp_rate_by_motorcycle.png', dpi=300)
    plt.close(fig)

    print(f'\nSaved analysis to {out}')
    print('NOTE: SHAP describes associations in the fitted model, not causality.')
    print('NOTE: Use the CV-selected strategy/threshold for comparable performance.')


if __name__ == '__main__':
    main()
