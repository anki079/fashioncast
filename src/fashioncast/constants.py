from pathlib import Path

DATA_ROOT = Path("data")
RAW_MANIFEST = DATA_ROOT / "raw" / "manifest.parquet"
CACHE_ROOT = DATA_ROOT / "cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
