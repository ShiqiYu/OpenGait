#!/usr/bin/env python3
"""ryu の metrics.json (13データセット, id 1..13) に DeepGaitV2 特徴量列を追加して
metrics_dgv2.json を作る。

id ↔ サブセット対応 (論文 Table 1 の並び。acc の nm/bg/cl 単純平均が
論文 Table 1 の認証精度と一致することで検証済み):
   1=default, 2=nm2, 3=bg2, 4=cl2, 5=nm1-bg1, 6=nm1-cl1, 7=bg1-cl1,
   8=000-180, 9=000-090, 10=090-180, 11=nm1-bg1-cl1, 12=nm2-bg2-cl2, 13=nm6

bg1-cl1 (id=7) は当初学習済みモデルが無く欠測扱いだったが、
DeepGaitV2-bg1-cl1-60000.pt の学習完了後に抽出し直し、13/13 全サブセットが揃っている。

実行: docker exec opengait_container python /app/OpenGait/dgv2_analysis/merge_metrics.py
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
    with open(os.path.join(REPO, "dgv2_features", "dgv2_metrics.json")) as f:
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

    out = os.path.join(HERE, "metrics_dgv2.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
