commands

 python -m src.build_dataset --years 2022 2023 2024 --out ../data/collisions.parquet

 python src/clean_dataset.py --in data/collisions.parquet --out data/collisions_clean.parquet

 python eda.py --in data/collisions_clean.parquet --out-dir reports/figures