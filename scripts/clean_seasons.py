import polars as pl
from fashioncast.constants import DATA_ROOT
from fashioncast.season_code import canonical_season

INFILE = DATA_ROOT / "processed" / "img_features.parquet"
OUTFILE = DATA_ROOT / "processed" / "img_features_clean.parquet"

print("→ reading image-level table …")
df = pl.read_parquet(INFILE)


# ------- safe wrappers so we don't crash on None -------------------
def safe_code(label: str):
    ps = canonical_season(label)
    return ps.season_code if ps else None


def safe_type(label: str):
    ps = canonical_season(label)
    return ps.collection_type if ps else None


df = df.with_columns(
    pl.col("season_code")
    .map_elements(safe_code, return_dtype=pl.Utf8)
    .alias("season_code"),
    pl.col("season_code")  # original text still in the row
    .map_elements(safe_type, return_dtype=pl.Utf8)
    .alias("collection_type"),
)

df.write_parquet(OUTFILE)
print(f"✔ clean file saved: {OUTFILE}  ({df.shape[0]:,} rows)")
