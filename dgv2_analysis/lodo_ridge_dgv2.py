#!/usr/bin/env python3
"""Leave-One-Subset-Out Ridge 回帰 (DeepGaitV2 特徴量のみ)。

ryu の LODO_CASIA.py を DeepGaitV2-only 特徴量に差し替えたもの。
- 入力: dgv2_analysis/metrics_dgv2.json
- 特徴量: log_people + DeepGaitV2 系5指標のみ (Inception/DINO/生画素は使わない)
- 目的変数: acc_nm, acc_bg, acc_cl, acc_all
- bg1-cl1 (id=7) は学習完了後、13/13 全サブセットで実行する
- 評価: MAE, RMSE, Spearman, Ridge 係数平均

実行: docker exec opengait_container python /app/OpenGait/dgv2_analysis/lodo_ridge_dgv2.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "metrics_dgv2.json")

TARGETS = ["acc_nm", "acc_bg", "acc_cl", "acc_all"]

FEATURE_COLS = [
    "log_people",
    "DeepGaitV2_MSD",
    "DeepGaitV2_1NN",
    "DeepGaitV2_kNN",
    "DeepGaitV2_FID",
    "FID_train_test_DeepGaitV2",
]
DATASET_COL = "dataset_id"


def load_df():
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = DATASET_COL
    df.reset_index(inplace=True)
    df["log_people"] = np.log(df["Number of people"])
    # DeepGaitV2 特徴が欠測 (bg1-cl1) の行は落とす
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    dropped = before - len(df)
    for col in FEATURE_COLS + TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"loaded {len(df)} subsets (dropped {dropped} with missing DeepGaitV2 feats)")
    return df


def run_lodo(df, target_col):
    y_true, y_pred, coefs = [], [], []
    for test_id in df[DATASET_COL].unique():
        tr = df[df[DATASET_COL] != test_id]
        te = df[df[DATASET_COL] == test_id]
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        model.fit(tr[FEATURE_COLS].values, tr[target_col].values)
        y_pred.append(model.predict(te[FEATURE_COLS].values)[0])
        y_true.append(te[target_col].values[0])
        coefs.append(model.named_steps["ridge"].coef_)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)
    coef_mean = np.mean(coefs, axis=0)

    print(f"\n===== Target = {target_col} =====")
    print(f"MAE={mae:.4f}  RMSE={rmse:.4f}  Spearman={rho:.4f}")
    coef_df = pd.DataFrame({"feature": FEATURE_COLS, "coef_mean": coef_mean}) \
        .sort_values("coef_mean", key=np.abs, ascending=False)
    print(coef_df.to_string(index=False))
    return {"mae": mae, "rmse": rmse, "spearman": rho,
            "coef": {f: float(c) for f, c in zip(FEATURE_COLS, coef_mean)}}


def main():
    df = load_df()
    out = {}
    for tgt in TARGETS:
        out[tgt] = run_lodo(df, tgt)
    with open(os.path.join(HERE, "result_ridge_dgv2.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {os.path.join(HERE, 'result_ridge_dgv2.json')}")


if __name__ == "__main__":
    main()
