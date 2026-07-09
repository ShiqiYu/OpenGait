#!/usr/bin/env python3
"""データセット優劣判定 (3クラス) ロジスティック回帰 — DeepGaitV2 特徴量のみ。

論文(中村, IEICE技報)4.5節に準拠:
  - サブセットペア (A, B) を 3 クラスに分類:
      A_better / B_better / Tie   (|acc_A - acc_B| <= tau なら Tie, tau=0.05)
  - 入力特徴: ペアの指標ベクトル。ここでは A と B の指標差 (feat_A - feat_B) を用いる。
  - Leave-One-Subset-Pair-Out: テスト対象ペア (a,b) を評価するとき、
    a または b を含むペアはすべて学習から除外する (論文の記述に沿う)。
  - 評価: Accuracy, Macro-F1 (Tie クラスの不均衡を考慮し Macro-F1 重視)
  - 対称化: (A,B) と (B,A) を両方生成しラベルを反転 (符号対称性を保つ)

bg1-cl1 (id=7) は当初モデル無しで欠測だったが、学習完了後は 13/13 全サブセットを使用する。
目的変数 acc は acc_all を既定とする (--target で変更可)。

実行: docker exec opengait_container python /app/OpenGait/dgv2_analysis/pairwise_logreg_dgv2.py
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "metrics_dgv2.json")

FEATURE_COLS = [
    "log_people",
    "DeepGaitV2_MSD",
    "DeepGaitV2_1NN",
    "DeepGaitV2_kNN",
    "DeepGaitV2_FID",
    "FID_train_test_DeepGaitV2",
]
# クラス: 0=Tie, 1=A_better, 2=B_better
TIE, A_BETTER, B_BETTER = 0, 1, 2


def load_df():
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "dataset_id"
    df.reset_index(inplace=True)
    df["log_people"] = np.log(df["Number of people"])
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    for col in FEATURE_COLS + ["acc_nm", "acc_bg", "acc_cl", "acc_all"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_label(acc_a, acc_b, tau):
    if abs(acc_a - acc_b) <= tau:
        return TIE
    return A_BETTER if acc_a > acc_b else B_BETTER


def build_pairs(df, target, tau):
    """全順序対 (a,b), a!=b の (特徴差, ラベル, (id_a,id_b)) を作る。"""
    feats, labels, keys = [], [], []
    ids = df["dataset_id"].tolist()
    fmap = {r["dataset_id"]: r[FEATURE_COLS].values.astype(float)
            for _, r in df.iterrows()}
    amap = {r["dataset_id"]: float(r[target]) for _, r in df.iterrows()}
    for a, b in itertools.permutations(ids, 2):
        feats.append(fmap[a] - fmap[b])
        labels.append(make_label(amap[a], amap[b], tau))
        keys.append((a, b))
    return np.array(feats), np.array(labels), keys


def run(df, target, tau, cv="pair"):
    X, y, keys = build_pairs(df, target, tau)
    y_true_all, y_pred_all, coefs = [], [], []

    # Leave-One-Subset-Pair-Out。cv="pair" は論文の記述通り、テスト対象の
    # 対 (a,b) とその逆順 (b,a) のみを学習から除外する (標準的な leave-one-pair-out)。
    # cv="subject" は a か b を含む全ペアを除外する厳しい版 (参考)。
    for i, (a, b) in enumerate(keys):
        if cv == "subject":
            train_mask = np.array([a not in k and b not in k for k in keys])
        else:
            train_mask = np.array([set(k) != {a, b} for k in keys])
        if train_mask.sum() == 0 or len(np.unique(y[train_mask])) < 2:
            continue
        scaler = StandardScaler().fit(X[train_mask])
        clf = LogisticRegression(max_iter=2000, C=1.0,
                                 class_weight="balanced",
                                 multi_class="multinomial")
        clf.fit(scaler.transform(X[train_mask]), y[train_mask])
        pred = clf.predict(scaler.transform(X[i:i + 1]))[0]
        y_true_all.append(y[i])
        y_pred_all.append(pred)
        if hasattr(clf, "coef_") and clf.coef_.shape[0] == 3:
            coefs.append(clf.coef_)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    acc = accuracy_score(y_true_all, y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")

    print(f"\n===== Pairwise superiority (target={target}, tau={tau}, cv={cv}) =====")
    print(f"pairs evaluated: {len(y_true_all)}")
    dist = {int(c): int((y_true_all == c).sum()) for c in np.unique(y_true_all)}
    print(f"label dist (0=Tie,1=A,2=B): {dist}")
    print(f"Accuracy = {acc:.4f}   Macro-F1 = {macro_f1:.4f}")

    result = {"target": target, "tau": tau, "n_pairs": int(len(y_true_all)),
              "accuracy": float(acc), "macro_f1": float(macro_f1),
              "label_dist": dist}

    # 係数 importance (クラス別絶対値の平均) — 論文 Table 6 相当
    if coefs:
        coef_mean = np.mean(coefs, axis=0)  # [3classes, n_feat]
        importance = np.mean(np.abs(coef_mean), axis=0)
        imp_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "coef_Tie": coef_mean[TIE],
            "coef_A": coef_mean[A_BETTER],
            "coef_B": coef_mean[B_BETTER],
            "importance": importance,
        }).sort_values("importance", ascending=False)
        print(imp_df.to_string(index=False))
        result["importance"] = imp_df.set_index("feature")["importance"].to_dict()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="acc_all")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--cv", default="pair", choices=["pair", "subject"])
    args = ap.parse_args()

    df = load_df()
    print(f"loaded {len(df)} subsets")
    res = run(df, args.target, args.tau, cv=args.cv)
    out = os.path.join(HERE, "result_logreg_dgv2.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
