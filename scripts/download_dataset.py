"""
saves every image to
data/raw/vogue/<SEASON_CODE>/<DESIGNER>/<index>.jpg
and writes data/raw/manifest.parquet with four columns:
  season_code | designer | img_idx | img_path
skip files that already exist.
"""

from datasets import load_dataset
from pathlib import Path
import polars as pl
from tqdm import tqdm
import re


# "alexander mcqueen" -> "alexander_mcqueen"
def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# parse the label string
def parse_label(label_str: str):
    # "designer,fall 1996 ready to wear" : designer | season | year | type
    parts = label_str.split(",")
    designer = parts[0].strip()

    # default values
    season = "unknown"
    year = "0000"

    m = re.search(
        r"(spring|fall|pre\s+fall|pre\s+spring|resort)\s+(\d{4})", label_str, flags=re.I
    )
    if m:
        season = m.group(1).lower()  # spring / fall / resort / pre fall etc
        year = m.group(2)

    return designer, season, year


# map (season + year) to season_code like "SS1996" or "FW2005"
def make_season_code(season: str, year: str):
    table = {
        "spring": "SS",
        "fall": "FW",
        "pre fall": "PF",
        "pre spring": "PS",
        "resort": "RS",
    }
    prefix = table.get(season, "UK")  # UK = unknown
    return f"{prefix}{year}"


# folders
DATA_DIR = Path("data/raw/vogue")
MANIFEST_PATH = Path("data/raw/manifest.parquet")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# load dataset
ds = load_dataset(
    "tonyassi/vogue-runway-top15-512px-nobg",
    split="train",
    streaming=False,
)

label_names = ds.features["label"].names  # length 1677

records = []
for idx, item in enumerate(tqdm(ds, desc="Saving images")):
    label_str = label_names[item["label"]]
    designer, season, year = parse_label(label_str)
    season_code = make_season_code(season, year)

    out_dir = DATA_DIR / season_code / slugify(designer)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_name = f"{idx:06d}.jpg"  # unique index within whole dataset
    img_path = out_dir / img_name

    if not img_path.exists():
        item["image"].save(img_path)

    records.append(
        {
            "season_code": season_code,
            "designer": designer,
            "img_idx": idx,
            "img_path": str(img_path),
        }
    )

# save manifest
pl.DataFrame(records).write_parquet(MANIFEST_PATH)
print(f" Finished: {len(records):,} images manifest saved to {MANIFEST_PATH}")
