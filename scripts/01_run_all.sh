#!/usr/bin/env bash
set -e
python scripts/download_dataset.py
python scripts/cache_features.py
python scripts/build_img_features.py
python scripts/build_season_tables.py
python scripts/train_models.py
python scripts/eval.py
