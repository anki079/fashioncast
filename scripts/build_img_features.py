# scripts/build_img_features.py  (bullet-proof version)
import polars as pl
import numpy as np
import torch
from pathlib import Path
from fashioncast.constants import CACHE_ROOT, RAW_MANIFEST

print("-> loading manifest …")
df = pl.read_parquet(RAW_MANIFEST)  # ~87 k rows

# --------------------------------------------------------------------
# 1.  Build Python lists for the two new columns (no Polars inside loop)
# --------------------------------------------------------------------
hsv_hist_list = []
clip_vec_list = []
shape_label_list = []

for p in df["img_path"]:
    stem = Path(p).stem

    # colour ----------------------------------------------------------
    hsv = np.load(CACHE_ROOT / "colour" / f"{stem}.npy")
    hsv_hist_list.append(hsv.tolist())  # 12 floats

    # clip ------------------------------------------------------------
    d = torch.load(CACHE_ROOT / "clip" / f"{stem}.pt", map_location="cpu")
    vec = d["vec"].detach().cpu().half().numpy().tolist()  # 512 floats
    clip_vec_list.append(vec)
    shape_label_list.append(int(d["label_idx"]))

# --------------------------------------------------------------------
# 2.  Attach the columns in one go
# --------------------------------------------------------------------
df = df.with_columns(
    [
        pl.Series("hsv_hist", hsv_hist_list),  # let Polars infer
        pl.Series("clip_vec", clip_vec_list),  # ← no dtype arg
        pl.Series("shape_label", shape_label_list, dtype=pl.Int32),
    ]
)

# --------------------------------------------------------------------
# 3.  Save
# --------------------------------------------------------------------
out_path = "data/processed/img_features.parquet"
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
df.write_parquet(out_path)
print(f"✅  Saved {out_path} with {df.shape[0]:,} rows")
