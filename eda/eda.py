"""
Deeper EDA on the cleaned collision dataset. Generates modeling-relevant
visuals (target-rate-by-category, not just raw distributions) and saves
them as PNGs for the report.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TARGET = "target_severe_fatal" 

CATEGORICAL_FOR_RATE = [
    "light_conditions",
    "weather_conditions",
    "road_type",
    "urban_or_rural_area",
    "junction_control",
    "speed_limit",
    "has_motorcycle",
    "has_hgv",
    "has_pedestrian",
    "has_cyclist",
    "has_child_casualty",
    "any_skidded",
    "any_left_carriageway",
    "is_classified_road",
]

NUMERIC_FOR_BOXPLOT = [
    "min_driver_age",
    "max_driver_age",
    "mean_driver_age",
    "max_vehicle_age",
    "share_male_drivers",
    "min_casualty_age",
    "max_casualty_age",
]

NUMERIC_FOR_CORR = [
    "min_driver_age",
    "max_driver_age",
    "mean_driver_age",
    "max_vehicle_age",
    "min_casualty_age",
    "max_casualty_age",
    "share_male_drivers",
    "n_vehicles_listed",
    "hour",
    "hour_sin",
    "hour_cos",
    "month",
]

HAS_FLAGS = ["has_motorcycle", "has_hgv", "has_pedestrian", "has_cyclist", "has_child_casualty"]


def savefig(fig, out_dir: Path, name: str):
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


# Target distribution
def plot_target_distribution(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ["target_severe_fatal", "target_fatal_only"]):
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=["#4c72b0", "#c44e52"])
        ax.set_title(f"{col} distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v}\n({v/len(df):.1%})", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    savefig(fig, out_dir, "01_target_distribution")


# Severity rate by category (the most modeling-relevant chart)

def plot_rate_by_category(df: pd.DataFrame, out_dir: Path):
    baseline = df[TARGET].mean()
    for col in CATEGORICAL_FOR_RATE:
        if col not in df.columns:
            log.warning("Skipping %s: not in dataframe", col)
            continue
        grouped = df.groupby(col)[TARGET].agg(["mean", "count"]).sort_values("mean")
        # drop tiny categories that make the chart noisy / unreliable
        grouped = grouped[grouped["count"] >= 30]
        if grouped.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(grouped))))
        ax.barh(grouped.index.astype(str), grouped["mean"], color="#55a868")
        ax.axvline(baseline, color="black", linestyle="--", linewidth=1, label=f"overall rate={baseline:.2f}")
        ax.set_xlabel(f"{TARGET} rate")
        ax.set_title(f"{TARGET} rate by {col}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        savefig(fig, out_dir, f"02_rate_by_{col}")



#Numeric distributions split by target

def plot_numeric_by_target(df: pd.DataFrame, out_dir: Path):
    for col in NUMERIC_FOR_BOXPLOT:
        if col not in df.columns:
            log.warning("Skipping %s: not in dataframe", col)
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        data = [df.loc[df[TARGET] == 0, col].dropna(), df.loc[df[TARGET] == 1, col].dropna()]
        labels = ["0 (slight)", "1 (severe/fatal)"]
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)  # matplotlib >= 3.9
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)  # matplotlib < 3.9
        ax.set_title(f"{col} by {TARGET}")
        ax.set_ylabel(col)
        fig.tight_layout()
        savefig(fig, out_dir, f"03_boxplot_{col}_by_target")

#Temporal patterns
def plot_temporal_patterns(df: pd.DataFrame, out_dir: Path):
    # by hour
    if "hour" in df.columns:
        rate_by_hour = df.groupby("hour")[TARGET].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(rate_by_hour.index, rate_by_hour.values, marker="o", color="#c44e52")
        ax.axhline(df[TARGET].mean(), color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("hour of day")
        ax.set_ylabel(f"{TARGET} rate")
        ax.set_title(f"{TARGET} rate by hour of day")
        ax.set_xticks(range(0, 24))
        fig.tight_layout()
        savefig(fig, out_dir, "04_rate_by_hour")

    # by month
    if "month" in df.columns:
        rate_by_month = df.groupby("month")[TARGET].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(rate_by_month.index.astype(str), rate_by_month.values, color="#4c72b0")
        ax.axhline(df[TARGET].mean(), color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("month")
        ax.set_ylabel(f"{TARGET} rate")
        ax.set_title(f"{TARGET} rate by month")
        fig.tight_layout()
        savefig(fig, out_dir, "04_rate_by_month")

    # by day_of_week
    if "day_of_week" in df.columns:
        rate_by_dow = df.groupby("day_of_week")[TARGET].mean().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(rate_by_dow.index.astype(str), rate_by_dow.values, color="#8172b2")
        ax.axhline(df[TARGET].mean(), color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("day_of_week (STATS19 coding)")
        ax.set_ylabel(f"{TARGET} rate")
        ax.set_title(f"{TARGET} rate by day of week")
        fig.tight_layout()
        savefig(fig, out_dir, "04_rate_by_day_of_week")


#Geospatial

def plot_geospatial(df: pd.DataFrame, out_dir: Path):
    if not {"latitude", "longitude"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(7, 8))
    hb = ax.hexbin(
        df["longitude"], df["latitude"], C=df[TARGET],
        reduce_C_function=np.mean, gridsize=60, cmap="RdYlGn_r", mincnt=5,
    )
    fig.colorbar(hb, ax=ax, label=f"{TARGET} rate")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"{TARGET} rate by location (hexbin, min 5 collisions/cell)")
    fig.tight_layout()
    savefig(fig, out_dir, "05_geospatial_rate")


# Correlation heatmap

def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path):
    cols = [c for c in NUMERIC_FOR_CORR if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation among numeric features")
    fig.tight_layout()
    savefig(fig, out_dir, "06_correlation_heatmap")

# Missingness as a signal

def plot_missingness_signal(df: pd.DataFrame, out_dir: Path):
    flag_cols = [c for c in df.columns if c.endswith("_was_missing")]
    if not flag_cols:
        log.warning("No _was_missing flag columns found — skipping missingness-signal plot")
        return
    
    baseline = df[TARGET].mean()
    rates = {}
    for col in flag_cols:
        rates[col] = df.groupby(col)[TARGET].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(flag_cols))
    width = 0.35
    present_rates = [rates[c].get(0, np.nan) for c in flag_cols]
    missing_rates = [rates[c].get(1, np.nan) for c in flag_cols]
    ax.bar(x - width / 2, present_rates, width, label="value present", color="#4c72b0")
    ax.bar(x + width / 2, missing_rates, width, label="was missing", color="#c44e52")
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1, label=f"overall rate={baseline:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_was_missing", "") for c in flag_cols], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"{TARGET} rate")
    ax.set_title(f"{TARGET} rate: present vs originally-missing values")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, out_dir, "07_missingness_signal")


# Feature engineering sanity checks

def plot_engineering_sanity_checks(df: pd.DataFrame, out_dir: Path):
    present = [c for c in HAS_FLAGS if c in df.columns]
    if not present:
        return
    rates = df[present].mean().sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(rates.index, rates.values, color="#dd8452")
    for i, v in enumerate(rates.values):
        ax.text(v, i, f" {v:.1%}", va="center", fontsize=8)
    ax.set_xlabel("proportion of collisions = True")
    ax.set_title("Engineered involvement flags: sanity check on prevalence")
    fig.tight_layout()
    savefig(fig, out_dir, "08_engineered_flag_prevalence")

    if "has_cyclist" in df.columns and df["has_cyclist"].mean() == 0:
        log.warning(
            "has_cyclist is 0%% across ALL rows — CASUALTY_TYPE_CYCLIST_CODE in "
            "build_dataset.py is very likely wrong. Verify against the STATS19 "
            "casualty_type lookup before trusting this feature."
        )



# Main

def run_eda(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_target_distribution(df, out_dir)
    plot_rate_by_category(df, out_dir)
    plot_numeric_by_target(df, out_dir)
    plot_temporal_patterns(df, out_dir)
    plot_geospatial(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
    plot_missingness_signal(df, out_dir)
    plot_engineering_sanity_checks(df, out_dir)
    log.info("EDA complete. Figures written to %s", out_dir)


def main():
    repo_root = Path(__file__).resolve().parent
    default_in_path = repo_root / "data" / "collisions_clean.parquet"
    default_out_dir = repo_root / "reports" / "figures"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=str(default_in_path))
    parser.add_argument("--out-dir", dest="out_dir", default=str(default_out_dir))
    args = parser.parse_args()

    df = pd.read_parquet(args.in_path)
    log.info("Loaded %s: %s", args.in_path, df.shape)

    run_eda(df, Path(args.out_dir))


if __name__ == "__main__":
    main()