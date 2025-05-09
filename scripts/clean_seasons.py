"""
Clean the free-text designer/season label into

• season_code   (SSYYYY / FWYYYY)
• collection_type  (menswear / ready to wear / couture / null)

Creates: data/processed/img_features_clean.parquet
"""

import polars as pl
from fashioncast.constants import DATA_ROOT, RAW_MANIFEST
from fashioncast.season_code import canonical_season

IMG_FILE = DATA_ROOT / "processed" / "img_features.parquet"
OUT_FILE = DATA_ROOT / "processed" / "img_features_clean.parquet"

print("→ reading image-level features …")
img = pl.read_parquet(IMG_FILE)

print("→ reading raw manifest (original labels) …")
manifest = pl.read_parquet(RAW_MANIFEST).select(
    "img_path", pl.col("season_code").alias("raw_label")
)

# --------------------------------------------------------------------
# Join raw label back onto features via img_path
# --------------------------------------------------------------------
df = img.join(manifest, on="img_path", how="left")


# Safe wrappers ------------------------------------------------------
def safe_code(label: str):
    ps = canonical_season(label)
    return ps.season_code if ps else None


def safe_type(label: str):
    ps = canonical_season(label)
    return ps.collection_type if ps else None


raw = pl.col("raw_label")

df = df.with_columns(
    raw.map_elements(safe_code, return_dtype=pl.Utf8).alias("season_code"),
    raw.map_elements(safe_type, return_dtype=pl.Utf8).alias("collection_type"),
)

df.write_parquet(OUT_FILE)
print(f"✔ clean file saved: {OUT_FILE}  ({df.shape[0]:,} rows)")
