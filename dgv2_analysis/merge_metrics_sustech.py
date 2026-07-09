#!/usr/bin/env python3
"""ryu の metrics.json (13データセット, id 1..13) に DeepGaitV2 特徴量列(SUSTech1K版・
共通モデルで抽出したもの)を追加して metrics_dgv2_sustech.json を作る。

merge_metrics.py(per-subset自前学習版)との違いは、DGV2_COLSの参照元が
dgv2_features_sustech/dgv2_metrics_sustech.json である点のみ。
acc_nm/acc_bg/acc_cl/acc_all(target)は per-subset版と全く同じ metrics_ryu.json を使う
(認識器はサブセットごと学習のまま変更しない。変えるのは特徴抽出器だけ)。

id ↔ サブセット対応 (論文 Table 1 の並び):
   1=default, 2=nm2, 3=bg2, 4=cl2, 5=nm1-bg1, 6=nm1-cl1, 7=bg1-cl1,
   8=000-180, 9=000-090, 10=090-180, 11=nm1-bg1-cl1, 12=nm2-bg2-cl2, 13=nm6

実行: docker exec opengait_container python /app/OpenGait/dgv2_analysis/merge_metrics_sustech.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

ID2SUBSET = {
    "1": "default",
    "2": "nm2",
    "3": "bg2",
    "4": "cl2",
    "5": "nm1-bg1",
    "6": "nm1-cl1",
    "7": "bg1-cl1",
    "8": "000-180",
    "9": "000-090",
    "10": "090-180",
    "11": "nm1-bg1-cl1",
    "12": "nm2-bg2-cl2",
    "13": "nm6",
}

DGV2_COLS = [
    "DeepGaitV2_MSD",
    "DeepGaitV2_1NN",
    "DeepGaitV2_kNN",
    "DeepGaitV2_FID",
    "FID_train_test_DeepGaitV2",
]


def main():
    with open(os.path.join(HERE, "metrics_ryu.json")) as f:
        base = json.load(f)
    with open(os.path.join(REPO, "dgv2_features_sustech", "dgv2_metrics_sustech.json")) as f:
        dgv2 = json.load(f)

    merged = {}
    for did, row in base.items():
        subset = ID2SUBSET[did]
        row = dict(row)
        row["subset_name"] = subset
        src = dgv2.get(subset)
        for col in DGV2_COLS:
            row[col] = (src[col] if src else None)
        merged[did] = row
        status = "ok" if src else "-- no DeepGaitV2 model (null) --"
        print(f"id={did:>2} {subset:<12} {status}")

    out = os.path.join(HERE, "metrics_dgv2_sustech.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
