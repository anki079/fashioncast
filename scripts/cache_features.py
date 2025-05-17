import polars as pl
from tqdm import tqdm
from fashioncast.colour import process_one as colour_one
from fashioncast.clip_labels import clip_label


manifest = pl.read_parquet("data/raw/manifest.parquet")

for row in tqdm(manifest.iter_rows(named=True)):
    colour_one(row["img_path"])
    clip_label(row["img_path"])
