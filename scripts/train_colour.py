import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from prophet import Prophet
from sklearn.multioutput import MultiOutputRegressor

from fashioncast.constants import DATA_ROOT
from fashioncast.split import proposal_split


COLOUR_TBL = DATA_ROOT / "processed" / "colour_trend.parquet"
OUT_DIR = Path("models/colour")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading colour table...")
tbl = pl.read_parquet(COLOUR_TBL)
train, val, test = proposal_split(tbl)


# helpers
def code_to_date(code: str) -> str:
    # SSYYYY -> YYYY-04-01 ,  FWYYYY -> YYYY-10-01 (string for prophet)
    year = int(code[2:])
    month = 4 if code.startswith("SS") else 10
    return f"{year}-{month:02d}-01"


def lag_xy(df: pl.DataFrame, lags=(1, 2)):
    # return (X, y) numeric arrays suitable for multi-output lightgbm
    mat = np.vstack(df["hsv_hist"])  # (n_seasons, 12)
    X, y = [], []
    for t in range(max(lags), len(mat)):
        X.append(np.hstack([mat[t - lag] for lag in lags]))  # 24-col row
        y.append(mat[t])  # 12-col target
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


# baselines
y_test = np.vstack(test["hsv_hist"])

persist = np.vstack(train["hsv_hist"])[-len(test) - 1 : -1]  # shift by 1
mae_persist = np.abs(persist - y_test).mean()

# prophet per hue bin
prophet_pred = []
for i in range(12):
    y_series = train["hsv_hist"].map_elements(
        lambda v, idx=i: float(v[idx]), return_dtype=pl.Float64
    )
    dfp = pl.DataFrame(
        {
            "ds": train["season_code"].map_elements(code_to_date, return_dtype=pl.Utf8),
            "y": y_series,
        }
    ).to_pandas()

    m = Prophet(yearly_seasonality=True)
    m.fit(dfp)

    future = m.make_future_dataframe(periods=len(test), freq="6MS")
    yhat = m.predict(future).tail(len(test))["yhat"].to_numpy()
    prophet_pred.append(yhat)

    pickle.dump(m, open(OUT_DIR / f"prophet_bin{i}.pkl", "wb"))

mae_prophet = np.abs(np.stack(prophet_pred, axis=1) - y_test).mean()


# lightgbm multi-output with lagged features
X_train, y_train = lag_xy(train)
X_val, y_val = lag_xy(pl.concat([train, val]))
X_test, _ = lag_xy(pl.concat([train, val, test]))

base_gbm = lgb.LGBMRegressor(
    objective="regression_l2",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
)
gbm = MultiOutputRegressor(base_gbm)
gbm.fit(X_train, y_train)  # train all 12 bins

pred_test = gbm.predict(X_test)[-len(test) :]  # keep last 6 rows
mae_gbm = np.abs(pred_test - y_test).mean()


pickle.dump(gbm, open(OUT_DIR / "lgb_colour.pkl", "wb"))
json.dump(
    dict(
        mae_persist=float(mae_persist),
        mae_prophet=float(mae_prophet),
        mae_gbm=float(mae_gbm),
    ),
    open(OUT_DIR / "metrics.json", "w"),
    indent=2,
)

print(
    f"Colour MAE | persistence {mae_persist:.3f} "
    f"| Prophet {mae_prophet:.3f} | LightGBM {mae_gbm:.3f}"
)
print("mean hue share train:", np.vstack(train["hsv_hist"]).mean())
print("mean absolute error :", mae_gbm)
