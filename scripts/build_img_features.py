import polars as pl
import numpy as np
import torch
from pathlib import Path
from fashioncast.constants import CACHE_ROOT, RAW_MANIFEST

print("Loading manifest...")
df = pl.read_parquet(RAW_MANIFEST)  # ~87 k rows

# build python lists for the two new columns (no Polars inside loop)
hsv_hist_list = []
clip_vec_list = []
shape_label_list = []

for p in df["img_path"]:
    stem = Path(p).stem

    # colour
    hsv = np.load(CACHE_ROOT / "colour" / f"{stem}.npy")
    hsv_hist_list.append(hsv.tolist())  # 12 floats

    # clip
    d = torch.load(CACHE_ROOT / "clip" / f"{stem}.pt", map_location="cpu")
    vec = d["vec"].detach().cpu().squeeze(0).half().numpy().tolist()
    clip_vec_list.append(vec)
    shape_label_list.append(int(d["label_idx"]))


# attach columns
df = df.with_columns(
    [
        pl.Series("hsv_hist", hsv_hist_list),
        pl.Series("clip_vec", clip_vec_list),
        pl.Series("shape_label", shape_label_list, dtype=pl.Int32),
    ]
)

out_path = "data/processed/img_features.parquet"
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
df.write_parquet(out_path)
print(f"Saved {out_path} with {df.shape[0]:,} rows")
