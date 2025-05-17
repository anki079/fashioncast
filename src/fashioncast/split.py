from datetime import datetime
import polars as pl

# 68-row cutoffs: 1991-2024
TRAIN_END = datetime(2020, 10, 1)  # FW20   -> training <= this date (60 rows)
VAL_START = datetime(2021, 4, 1)  # SS21
VAL_END = datetime(2021, 10, 1)  # FW21   -> validation = SS21 & FW21 (2 rows)
# test = all seasons >= SS22 (6 rows)  [SS22, FW22, SS23, FW23, SS24, FW24]

"""
split a polars table with SS / FW season_code into (train, val, test) acc to the boundaries above
"""


def proposal_split(df: pl.DataFrame, date_col: str = "season_code"):

    def _to_date(code: str) -> datetime:
        year = int(code[2:])
        month = 4 if code.startswith("SS") else 10
        return datetime(year, month, 1)

    dt = df.with_columns(pl.col(date_col).map_elements(_to_date).alias("_dt"))
    train = dt.filter(pl.col("_dt") <= TRAIN_END).drop("_dt")
    val = dt.filter((pl.col("_dt") >= VAL_START) & (pl.col("_dt") <= VAL_END)).drop(
        "_dt"
    )
    test = dt.filter(pl.col("_dt") > VAL_END).drop("_dt")
    return train, val, test
