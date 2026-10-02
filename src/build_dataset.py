"""
Merges UK STATS19 collision / vehicle / casualty CSVs into a single
one-row-per-collision parquet file, ready for preprocess.py.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


RAW_FILENAMES = {
    "collision": "collision-last-5-years.csv",
    "vehicle": "vehicle-last-5-years.csv",
    "casualty": "casualty-last-5-years.csv",
}


YEAR_COLUMN = "collision_year"

# 1. Missing-code configuration
# Default: -1 always means "missing / out of range" everywhere it appears.
DEFAULT_MISSING_CODES = [-1]

# Columns that ALSO use 9, 99, or another code for "unknown (self reported)".
EXTRA_MISSING_CODES = {
    # collision table
    "speed_limit": [99],
    "junction_detail_historic": [99],
    "junction_control": [9],
    "second_road_class": [9],
    "pedestrian_crossing_human_control_historic": [9],
    "pedestrian_crossing_physical_facilities_historic": [9],
    "weather_conditions": [9],
    "road_surface_conditions": [9],
    "special_conditions_at_site": [9],
    "carriageway_hazards_historic": [9],
    "road_type": [9],
    # vehicle table
    "towing_and_articulation": [9],
    "vehicle_manoeuvre_historic": [99],
    "vehicle_direction_from": [9],
    "vehicle_direction_to": [9],
    "vehicle_location_restricted_lane_historic": [99],
    "junction_location": [9],
    "skidding_and_overturning": [9],
    "hit_object_in_carriageway": [99],
    "vehicle_leaving_carriageway": [9],
    "hit_object_off_carriageway": [99],
    "first_point_of_impact": [9],
    "vehicle_left_hand_drive": [9],
    "sex_of_driver": [3],  # "3: Not known" here, not 9
    # casualty table
    "sex_of_casualty": [9],
    "car_passenger": [9],
    "bus_or_coach_passenger": [9],
    "casualty_class": [],  # no unknown code documented
}

CASUALTY_TYPE_CYCLIST_CODE = 1

MOTORCYCLE_VEHICLE_TYPES = {2, 3, 4, 5, 23, 97, 103, 104, 105, 106}
HGV_VEHICLE_TYPES = {20, 21, 98, 113}

CHILD_AGE_CUTOFF = 18


def apply_missing_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Replace documented missing/unknown codes with NaN, column by column."""
    df = df.copy()
    for col in df.columns:
        codes = list(DEFAULT_MISSING_CODES) + EXTRA_MISSING_CODES.get(col, [])
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(codes, np.nan)
    return df


# Loading raw tables 

def load_table(data_dir: str, table: str, years: list[int]) -> pd.DataFrame:
    path = Path(data_dir) / RAW_FILENAMES[table]
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {table} file not found: {path}. "
            f"Check --data-dir, or adjust RAW_FILENAMES if your filenames differ."
        )
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded %s: %d total rows (all years)", path.name, len(df))

    if YEAR_COLUMN not in df.columns:
        raise KeyError(
            f"{path.name} has no '{YEAR_COLUMN}' column to filter years by. "
            f"Found columns: {list(df.columns)[:10]}..."
        )

    df = df[df[YEAR_COLUMN].isin(years)].reset_index(drop=True)
    log.info("Filtered %s to years %s: %d rows", table, years, len(df))
    if df.empty:
        raise ValueError(f"No rows left in {table} after filtering to years {years} — check the year values present.")
    return df


# 
# Vehicle table -> per-collision aggregates
def aggregate_vehicles(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    df = apply_missing_codes(vehicle_df)

    df["is_motorcycle"] = df["vehicle_type"].isin(MOTORCYCLE_VEHICLE_TYPES)
    df["is_hgv"] = df["vehicle_type"].isin(HGV_VEHICLE_TYPES)

    agg = df.groupby("collision_index").agg(
        n_vehicles_listed=("vehicle_reference", "nunique"),
        has_motorcycle=("is_motorcycle", "any"),
        has_hgv=("is_hgv", "any"),
        min_driver_age=("age_of_driver", "min"),
        max_driver_age=("age_of_driver", "max"),
        mean_driver_age=("age_of_driver", "mean"),
        share_male_drivers=("sex_of_driver", lambda s: (s == 1).mean() if s.notna().any() else np.nan),
        max_vehicle_age=("age_of_vehicle", "max"),
        any_skidded=("skidding_and_overturning", lambda s: (s.fillna(0) > 0).any()),
        any_left_carriageway=("vehicle_leaving_carriageway", lambda s: (s.fillna(0) > 0).any()),
    ).reset_index()

    for col in ["has_motorcycle", "has_hgv", "any_skidded", "any_left_carriageway"]:
        agg[col] = agg[col].astype(int)

    log.info("Aggregated vehicles: %d collisions", len(agg))
    return agg


#Casualty table -> per-collision aggregates
def aggregate_casualties(casualty_df: pd.DataFrame) -> pd.DataFrame:
    df = apply_missing_codes(casualty_df)

    df["is_pedestrian"] = df["casualty_class"] == 3
    df["is_cyclist"] = df["casualty_type"] == CASUALTY_TYPE_CYCLIST_CODE
    df["is_child"] = df["age_of_casualty"] < CHILD_AGE_CUTOFF

    agg = df.groupby("collision_index").agg(
        n_casualties_listed=("casualty_reference", "nunique"),
        has_pedestrian=("is_pedestrian", "any"),
        has_cyclist=("is_cyclist", "any"),
        has_child_casualty=("is_child", "any"),
        min_casualty_age=("age_of_casualty", "min"),
        max_casualty_age=("age_of_casualty", "max"),
    ).reset_index()

    for col in ["has_pedestrian", "has_cyclist", "has_child_casualty"]:
        agg[col] = agg[col].astype(int)

    log.info("Aggregated casualties: %d collisions", len(agg))
    return agg


# Collision table -> target + cleaned feature set
# Columns dropped as: identifiers, redundant geo, 2024-only new fields,
# or leakage (target-adjacent). See build_dataset design notes above.
COLLISION_DROP_COLS = [
    "collision_ref_no",
    "location_easting_osgr",
    "location_northing_osgr",
    "local_authority_ons_district",
    "local_authority_highway",
    "local_authority_highway_current",
    "junction_detail",  # 2024-only, keep junction_detail_historic
    "pedestrian_crossing",  # 2024-only
    "carriageway_hazards",  # keep carriageway_hazards_historic instead
    "enhanced_severity_collision",  # 2023+ only
    "lsoa_of_accident_location",
    # leakage / target-adjacent
    "collision_injury_based",
    "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight",
    "number_of_casualties",  # optional sensitivity-run only, excluded from main grid
]


def build_collision_base(collision_df: pd.DataFrame) -> pd.DataFrame:
    df = apply_missing_codes(collision_df)

    df["target_severe_fatal"] = df["collision_severity"].isin([1, 2]).astype(int)
    df["target_fatal_only"] = (df["collision_severity"] == 1).astype(int)

    dt = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce", dayfirst=True)
    df["hour"] = dt.dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month"] = dt.dt.month
    # day_of_week already exists as a column; keep as categorical

    df = df.drop(columns=["date", "time", "collision_severity"], errors="ignore")
    df = df.drop(columns=COLLISION_DROP_COLS, errors="ignore")

    log.info("Collision base: %d rows, %d columns", *df.shape)
    return df


# Merge + schema checks

EXPECTED_MIN_COLUMNS = {
    "collision_index",
    "collision_year",
    "target_severe_fatal",
    "target_fatal_only",
    "speed_limit",
    "road_type",
    "has_motorcycle",
    "has_pedestrian",
}


def build_dataset(data_dir: str, years: list[int]) -> pd.DataFrame:
    collision_raw = load_table(data_dir, "collision", years)
    vehicle_raw = load_table(data_dir, "vehicle", years)
    casualty_raw = load_table(data_dir, "casualty", years)

    collision_base = build_collision_base(collision_raw)
    vehicle_agg = aggregate_vehicles(vehicle_raw)
    casualty_agg = aggregate_casualties(casualty_raw)

    n_before = len(collision_base)
    merged = collision_base.merge(vehicle_agg, on="collision_index", how="left")
    merged = merged.merge(casualty_agg, on="collision_index", how="left")
    assert len(merged) == n_before, (
        f"Row count changed on merge ({n_before} -> {len(merged)}): "
        f"a join key is duplicated somewhere — check vehicle/casualty aggregation."
    )

    # n_casualties_listed is derived from the casualty table and carries the
    # same leakage risk as collision.number_of_casualties (both correlate
    # mechanically with severity rather than being purely pre-crash
    # information) — drop for consistency with that earlier decision.
    merged = merged.drop(columns=["n_casualties_listed"], errors="ignore")

    missing_expected = EXPECTED_MIN_COLUMNS - set(merged.columns)
    assert not missing_expected, f"Missing expected columns after build: {missing_expected}"

    forbidden = {
        "casualty_severity",
        "collision_severity",
        "casualty_adjusted_severity_serious",
        "casualty_adjusted_severity_slight",
        "collision_adjusted_severity_serious",
        "collision_adjusted_severity_slight",
    }
    leaked = forbidden & set(merged.columns)
    assert not leaked, f"Leakage columns present in output: {leaked}"

    log.info("Final dataset: %d rows, %d columns", *merged.shape)
    log.info(
        "Class balance (severe/fatal=1): %.4f",
        merged["target_severe_fatal"].mean(),
    )
    log.info(
        "Class balance (fatal=1): %.4f",
        merged["target_fatal_only"].mean(),
    )
    return merged


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_data_dir = repo_root / "data"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(default_data_dir),
        help=f"Directory containing the three *-last-5-years.csv files (default: {default_data_dir})",
    )
    parser.add_argument("--years", nargs="+", type=int, required=True, help="e.g. --years 2022 2023 2024")
    parser.add_argument("--out", required=True, help="Output parquet path")
    args = parser.parse_args()

    if 2020 in args.years:
        log.warning(
            "2020 included in --years: lockdown traffic makes this year atypical. "
            "Recommended to exclude unless you have a specific reason to keep it."
        )

    dataset = build_dataset(args.data_dir, args.years)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)
    log.info("Wrote dataset to %s", out_path)


if __name__ == "__main__":
    main()