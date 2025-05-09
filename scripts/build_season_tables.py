# scripts/build_season_tables.py
import polars as pl
import numpy as np
from fashioncast.constants import DATA_ROOT

INFILE = DATA_ROOT / "processed" / "img_features_clean.parquet"
OUT_CO = DATA_ROOT / "processed" / "colour_trend.parquet"
OUT_SH = DATA_ROOT / "processed" / "shape_trend.parquet"
OUT_VEC = DATA_ROOT / "processed" / "season_clip_mean.parquet"

print("→ reading image-level table …")
df = pl.read_parquet(INFILE)

# --------------------------------------------------------------------
# 1.  Colour-share table: one 12-bin histogram per season
# --------------------------------------------------------------------
colour_tbl = (
    df.group_by("season_code")
    .agg(pl.col("hsv_hist"))
    .with_columns(
        pl.col("hsv_hist").map_elements(
            lambda L: np.mean(
                np.stack(L, dtype=float), axis=0
            ).tolist(),  # ← stack then mean
            return_dtype=pl.List(pl.Float64),
        )
    )
    .sort("season_code")
)
colour_tbl.write_parquet(OUT_CO)
print(f"✔ colour_trend → {OUT_CO}")

# --------------------------------------------------------------------
# 2.  Shape-label counts per season
# --------------------------------------------------------------------
shape_tbl = (
    df.group_by("season_code")
    .agg(pl.col("shape_label").value_counts())
    .sort("season_code")
)
shape_tbl.write_parquet(OUT_SH)
print(f"✔ shape_trend → {OUT_SH}")

# --------------------------------------------------------------------
# 3.  Season-level mean CLIP vector  (optional, useful for ML models)
# --------------------------------------------------------------------
clip_tbl = (
    df.group_by("season_code")
    .agg(pl.col("clip_vec"))
    .with_columns(
        pl.col("clip_vec").map_elements(
            lambda L: np.mean(
                np.stack(L, dtype=float), axis=0
            ).tolist(),  # ← stack then mean
            return_dtype=pl.List(pl.Float64),
        )
    )
    .sort("season_code")
)
clip_tbl.write_parquet(OUT_VEC)
print(f"✔ season_clip_mean → {OUT_VEC}")
