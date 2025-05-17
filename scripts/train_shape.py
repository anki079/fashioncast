import json
import pickle
import numpy as np
import polars as pl
import lightgbm as lgb

from sklearn.metrics import f1_score, accuracy_score

from pathlib import Path
from fashioncast.split import proposal_split
from fashioncast.constants import DATA_ROOT

SHAPE_TBL = DATA_ROOT / "processed" / "shape_trend.parquet"
OUT_DIR = Path("models/shape")
OUT_DIR.mkdir(parents=True, exist_ok=True)

tbl = pl.read_parquet(SHAPE_TBL)
train, val, test = proposal_split(tbl)


# def counts_vec(col, k=6):
#     return col.map_elements(
#         lambda lst: [dict(lst).get(i, 0) for i in range(k)],
#         return_dtype=pl.List(pl.UInt32),
#     )


# Convert a List[struct{'shape_label': int, 'counts': int}] into a fixed len python list [c0, c1,..., c5].
def counts_vec(col, k=6):
    def to_vec(lst):
        vec = [0] * k
        if lst is None:
            return vec
        for s in lst:
            # pull out the two fields, with fallbacks
            lbl_raw = s.get("shape_label", s.get("label", None))
            cnt_raw = s.get("counts", s.get("count", None))
            if lbl_raw is None or cnt_raw is None:
                continue
            lbl = int(lbl_raw)
            cnt = int(cnt_raw)
            if 0 <= lbl < k:
                vec[lbl] = cnt
        return vec

    return col.map_elements(to_vec, return_dtype=pl.List(pl.UInt32))


train = train.with_columns(counts_vec(pl.col("shape_label")).alias("counts"))
val = val.with_columns(counts_vec(pl.col("shape_label")).alias("counts"))
test = test.with_columns(counts_vec(pl.col("shape_label")).alias("counts"))


def lag_xy(df, lags=(1, 2)):
    arr = np.vstack(df["counts"])
    X, y = [], []
    for t in range(max(lags), len(arr)):
        X.append(np.hstack([arr[t - lag] for lag in lags]))
        y.append(arr[t].argmax())  # majority label
    return np.vstack(X), np.array(y)


X_train, y_train = lag_xy(train)
X_val, y_val = lag_xy(pl.concat([train, val]))
X_test, y_test = lag_xy(pl.concat([train, val, test]))
X_test, y_test = X_test[-len(test) :], y_test[-len(test) :]

clf = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=6,
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
)

clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
pred = clf.predict(X_test)
f1 = f1_score(y_test, pred, average="macro")

all_labels = list(range(6))  # 0-5 even if absent
f1_all = f1_score(y_test, pred, average="macro", labels=all_labels, zero_division=0)
acc = accuracy_score(y_test, pred)

np.save(OUT_DIR / "y_test.npy", np.array(y_test))
np.save(OUT_DIR / "y_pred.npy", np.array(pred))

pickle.dump(clf, open(OUT_DIR / "lgb_shape.pkl", "wb"))
json.dump(
    dict(
        macro_f1_present=f1,
        macro_f1_all=f1_all,
        accuracy=acc,
    ),
    open(OUT_DIR / "metrics.json", "w"),
    indent=2,
)

print(
    f"Shape LightGBM  macro-F1(present) {f1:.3f}  | "
    f"macro-F1(all) {f1_all:.3f}  | accuracy {acc:.3f}"
)
