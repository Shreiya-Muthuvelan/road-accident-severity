"""
clean_dataset.py
-----------------
Cleans the collision-level dataset produced by build_dataset.py, based on
the diagnostic EDA pass (df.shape, isnull().sum(), value_counts, boxplots).

Usage:
    python clean_dataset.py --in ../data/collisions.parquet \
        --out ../data/collisions_clean.parquet

What this does, and why (see report paragraph / team notes for full detail):
  1. Drops `local_authority_district` entirely — confirmed 100% == -1 in the
     raw source file, i.e. genuinely unpopulated at the data source, not a
     pipeline bug. Not imputable.
  2. Drops the ~34 rows missing `longitude`/`latitude` — coordinates can't
     be meaningfully imputed without placing a crash somewhere it didn't
     happen, and the row loss is negligible.
  3. Low-missingness categoricals (<3%) get an explicit "Missing" category
     rather than mode imputation, for consistency with (4) below.
  4. Moderate/high-missingness categoricals (5-45%) — these come from
     STATS19 fields that used -1/9/99 for "unknown (self reported)", i.e.
     "unknown" is itself a real recorded state, not noise. Filled with an
     explicit "Missing" category, never imputed to the mode.
  5. Age-related numeric fields: implausible outliers (drivers 100+,
     vehicles 100+ years old) are treated as data-entry errors and nulled
     BEFORE imputation, then median-imputed with a companion `_was_missing`
     binary flag added per column, since missingness here is unlikely to
     be random (e.g. correlates with hit-and-run / self-report gaps).
  6. `first_road_number` / `second_road_number` are dropped (high-
     cardinality IDs, not ordinal magnitudes) in favour of a binary
     `is_classified_road` / `is_classified_second_road` flag derived from
     whether the number is 0 (unclassified, per STATS19 codebook) or not.

NOTE: ideally the age-outlier capping in step 5 would happen on the raw
per-vehicle/per-casualty rows in build_dataset.py, before min/max/mean
aggregation — an outlier polluting a `max_driver_age` aggregate is exactly
the bug this step is patching after the fact. Left here for now since
that's where the data currently lives; consider moving upstream later.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — adjust thresholds here if EDA on a new data pull changes them
# ---------------------------------------------------------------------------
FULLY_MISSING_COLS = ["local_authority_district"]

COORD_COLS = ["longitude", "latitude"]

LOW_MISSING_CATEGORICALS = [
    "speed_limit",
    "light_conditions",
    "urban_or_rural_area",
    "road_surface_conditions",
]

MODERATE_HIGH_MISSING_CATEGORICALS = [
    "road_type",
    "weather_conditions",
    "junction_detail_historic",
    "junction_control",
    "second_road_class",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic",
    "special_conditions_at_site",
    "carriageway_hazards_historic",
    "trunk_road_flag",
]

MISSING_CATEGORY_LABEL = "Missing"

# Plausibility bounds — values outside these are treated as data-entry
# errors and nulled before imputation. VERIFY against boxplots each time
# the raw pull changes; these are eyeballed from the current EDA, not a
# hard STATS19 rule.
DRIVER_AGE_BOUNDS = (15, 100)     # UK minimum driving age ~16/17; 100 as upper cutoff
VEHICLE_AGE_MAX = 40              # vehicles reported 40-120 yrs old treated as errors
CASUALTY_AGE_BOUNDS = (0, 100)

AGE_COLUMNS_TO_CLEAN = {
    "min_driver_age": DRIVER_AGE_BOUNDS,
    "max_driver_age": DRIVER_AGE_BOUNDS,
    "mean_driver_age": DRIVER_AGE_BOUNDS,
    "max_vehicle_age": (0, VEHICLE_AGE_MAX),
    "min_casualty_age": CASUALTY_AGE_BOUNDS,
    "max_casualty_age": CASUALTY_AGE_BOUNDS,
}

# min/max column pairs that can become inverted when each side is
# median-imputed independently (e.g. min_driver_age missing but
# max_driver_age present, or vice versa, on the same row).
INVERTED_MIN_MAX_PAIRS = [
    ("min_driver_age", "max_driver_age"),
    ("min_casualty_age", "max_casualty_age"),
]

# share_male_drivers is a proportion (0-1), not an age — impute separately
PROPORTION_COLUMNS = ["share_male_drivers"]

ROAD_NUMBER_COLS = {
    "first_road_number": "is_classified_road",
    "second_road_number": "is_classified_second_road",
}


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def drop_fully_missing(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in FULLY_MISSING_COLS if c in df.columns]
    if present:
        null_fracs = df[present].isnull().mean()
        for col, frac in null_fracs.items():
            if frac < 0.99:
                log.warning(
                    "%s is only %.1f%% null, not ~100%% as expected — "
                    "double check before dropping.", col, frac * 100
                )
        df = df.drop(columns=present)
        log.info("Dropped fully-missing columns: %s", present)
    return df


def drop_missing_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[c for c in COORD_COLS if c in df.columns])
    dropped = before - len(df)
    log.info("Dropped %d rows with missing lat/long (%.3f%% of data)", dropped, 100 * dropped / before)
    return df


def fill_categoricals_with_missing_label(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        n_missing = df[col].isnull().sum()
        df[col] = df[col].astype("string")
        df[col] = df[col].where(df[col].notna(), MISSING_CATEGORY_LABEL)
        log.info("%s: filled %d missing values with '%s' category", col, n_missing, MISSING_CATEGORY_LABEL)
    return df


def clean_age_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (low, high) in AGE_COLUMNS_TO_CLEAN.items():
        if col not in df.columns:
            continue
        implausible = ~df[col].between(low, high) & df[col].notnull()
        n_implausible = implausible.sum()
        if n_implausible:
            log.info("%s: nulling %d implausible values outside [%s, %s]", col, n_implausible, low, high)
            df.loc[implausible, col] = np.nan

        n_missing = df[col].isnull().sum()
        flag_col = f"{col}_was_missing"
        df[flag_col] = df[col].isnull().astype(int)

        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        log.info(
            "%s: imputed %d missing/invalid values with median=%.1f, added flag '%s'",
            col, n_missing, median_val, flag_col,
        )
    return df


def fix_inverted_min_max(df: pd.DataFrame) -> pd.DataFrame:
    """Swap min/max pairs left inverted by independent median imputation.

    E.g. a row missing only min_driver_age gets it filled with the global
    median (32), which can end up above an actually-recorded max_driver_age
    (e.g. 28) on the same row. Swapping is a reasonable fix since both
    values came from the same underlying (partially missing) fact — we
    just don't know which imputed number was meant to be the min vs max.
    """
    df = df.copy()
    for min_col, max_col in INVERTED_MIN_MAX_PAIRS:
        if min_col not in df.columns or max_col not in df.columns:
            continue
        inverted = df[min_col] > df[max_col]
        n = inverted.sum()
        if n:
            tmp = df.loc[inverted, min_col].copy()
            df.loc[inverted, min_col] = df.loc[inverted, max_col]
            df.loc[inverted, max_col] = tmp
            log.info(
                "%s/%s: swapped %d rows where %s > %s (independent imputation artifact)",
                min_col, max_col, n, min_col, max_col,
            )
    return df


def clean_proportion_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in PROPORTION_COLUMNS:
        if col not in df.columns:
            continue
        n_missing = df[col].isnull().sum()
        flag_col = f"{col}_was_missing"
        df[flag_col] = df[col].isnull().astype(int)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        log.info("%s: imputed %d missing values with median=%.3f, added flag '%s'", col, n_missing, median_val, flag_col)
    return df


def simplify_road_numbers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for raw_col, flag_col in ROAD_NUMBER_COLS.items():
        if raw_col not in df.columns:
            continue
        # 0 = unclassified road per STATS19 codebook; NaN treated as unclassified too
        df[flag_col] = (df[raw_col].fillna(0) != 0).astype(int)
        df = df.drop(columns=[raw_col])
        log.info("Replaced %s with binary %s", raw_col, flag_col)
    return df


def fill_low_missing_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return fill_categoricals_with_missing_label(df, LOW_MISSING_CATEGORICALS)


def fill_moderate_high_missing_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return fill_categoricals_with_missing_label(df, MODERATE_HIGH_MISSING_CATEGORICALS)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)

    df = drop_fully_missing(df)
    df = drop_missing_coordinates(df)
    df = fill_low_missing_categoricals(df)
    df = fill_moderate_high_missing_categoricals(df)
    df = clean_age_fields(df)
    df = fix_inverted_min_max(df)
    df = clean_proportion_fields(df)
    df = simplify_road_numbers(df)

    remaining_nulls = df.isnull().sum()
    still_missing = remaining_nulls[remaining_nulls > 0]
    if not still_missing.empty:
        log.warning("Columns still containing nulls after cleaning:\n%s", still_missing)
    else:
        log.info("No remaining nulls after cleaning.")

    log.info("Rows: %d -> %d (dropped %d)", n_start, len(df), n_start - len(df))
    log.info("Final cleaned shape: %s", df.shape)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, help="Path to parquet produced by build_dataset.py")
    parser.add_argument("--out", dest="out_path", required=True, help="Output path for cleaned parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.in_path)
    log.info("Loaded %s: %s", args.in_path, df.shape)

    cleaned = clean_dataset(df)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out_path, index=False)
    log.info("Wrote cleaned dataset to %s", out_path)


if __name__ == "__main__":
    main()