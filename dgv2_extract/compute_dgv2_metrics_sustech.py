#!/usr/bin/env python3
"""DeepGaitV2 埋め込み (dgv2_features_sustech/{subset}/{train,test}.npz) から
データ特性指標を計算する(SUSTech1K版・共通モデル抽出)。

compute_dgv2_metrics.py との違いは FEAT_DIR / 出力ファイル名のみ。
埋め込みは「CASIA-Bを一切学習していない共通の DeepGaitV2(SUSTech1K版)」で
抽出したもの (dgv2_extract/gen_extract_configs_sustech.py 参照)。
train-on-train(自前学習モデルで自分の学習データを暗記した状態で測る問題)を
解消した版として、per-subset版(compute_dgv2_metrics.py の結果)と比較する。

論文(中村, IEICE技報)の4指標を DeepGaitV2 特徴空間で再現する:
  1. DeepGaitV2_MSD  : 平均クラス内分散 (論文 eq.4:
                       (1/M) Σ_s (1/N_s) Σ_i ||f_i - f̄_s||²)
  2. DeepGaitV2_1NN  : 平均類似クラス間距離 (k=1)
  3. DeepGaitV2_kNN  : 平均類似クラス間距離 (k=5)
  4. DeepGaitV2_FID  : 平均クラス間距離 (被験者ペアFIDの全ペア平均)
  5. FID_train_test_DeepGaitV2 : Train-Test データ間分布距離

埋め込みは [n, 256, 16] を 4096 次元に flatten して使う。

FID の Tr((Σ1Σ2)^{1/2}) は Gram トリックで厳密に計算する:
  Σi = Ai Aiᵀ (Ai = centered/√(n-1)) のとき
  Tr((Σ1Σ2)^{1/2}) = Σ σ_j(A1ᵀA2)  (特異値の和 = 核ノルム)
  → d×d の sqrtm を避け、n1×n2 の SVD で済む。

実行はコンテナ内 (numpy が必要):
  docker exec opengait_container python /app/OpenGait/dgv2_extract/compute_dgv2_metrics_sustech.py

出力: /app/OpenGait/dgv2_features_sustech/dgv2_metrics_sustech.json
"""
import json
import os
import time

import numpy as np

FEAT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "dgv2_features_sustech")
KNN_K = 5

SUBSETS = [
    "default", "nm2", "bg2", "cl2", "nm1-bg1", "nm1-cl1",
    "000-180", "000-090", "090-180", "nm1-bg1-cl1", "nm2-bg2-cl2", "nm6",
    "bg1-cl1",
]


def load_flat(path):
    d = np.load(path)
    emb = d["embeddings"].astype(np.float64)  # [n, 256, 16]
    emb = emb.reshape(emb.shape[0], -1)       # [n, 4096]
    return emb, d["labels"], d["types"], d["views"]


def compute_msd(emb, labels):
    """平均クラス内分散 (論文 eq.4)。"""
    total = 0.0
    subjects = np.unique(labels)
    for s in subjects:
        x = emb[labels == s]
        mean = x.mean(axis=0)
        total += np.mean(np.sum((x - mean) ** 2, axis=1))
    return float(total / len(subjects))


def compute_knn(emb, labels, k=KNN_K):
    """平均類似クラス間距離: 各サンプルから他クラスサンプルへの 1NN / kNN 平均距離。"""
    # 距離行列 (n×n) をブロックで計算してメモリを抑える
    n = emb.shape[0]
    sq = np.sum(emb ** 2, axis=1)
    one_nn, k_nn = [], []
    block = 512
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        d2 = sq[i0:i1, None] - 2.0 * emb[i0:i1] @ emb.T + sq[None, :]
        np.clip(d2, 0, None, out=d2)
        dist = np.sqrt(d2)
        for r in range(i1 - i0):
            mask = labels != labels[i0 + r]
            valid = dist[r][mask]
            one_nn.append(valid.min())
            k_nn.append(np.mean(np.partition(valid, k)[:k]))
    return float(np.mean(one_nn)), float(np.mean(k_nn))


def _center_scale(x):
    """A = centered/√(n-1): Σ = A Aᵀ (np.cov と同じ 1/(n-1) 規約)。"""
    n = x.shape[0]
    a = (x - x.mean(axis=0)) / np.sqrt(max(n - 1, 1))
    return a


def fid_between(x1, x2):
    """FID(x1, x2) を Gram トリックで厳密計算。x: [n, d]"""
    mu1, mu2 = x1.mean(axis=0), x2.mean(axis=0)
    a1, a2 = _center_scale(x1), _center_scale(x2)
    tr1 = float(np.sum(a1 ** 2))            # Tr Σ1
    tr2 = float(np.sum(a2 ** 2))            # Tr Σ2
    c = a1 @ a2.T                            # [n1, n2]
    sv = np.linalg.svd(c, compute_uv=False)  # 特異値 = sqrt(eig(CᵀC))
    tr_sqrt = float(np.sum(sv))              # Tr((Σ1Σ2)^{1/2})
    diff = float(np.sum((mu1 - mu2) ** 2))
    return diff + tr1 + tr2 - 2.0 * tr_sqrt


def compute_pair_fid(emb, labels):
    """被験者ペア FID の全ペア平均 (平均クラス間距離)。"""
    subjects = np.unique(labels)
    groups = {s: emb[labels == s] for s in subjects}
    fids = []
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            fids.append(fid_between(groups[subjects[i]], groups[subjects[j]]))
    return float(np.mean(fids))


def main():
    results = {}
    for subset in SUBSETS:
        train_npz = os.path.join(FEAT_DIR, subset, "train.npz")
        test_npz = os.path.join(FEAT_DIR, subset, "test.npz")
        if not (os.path.exists(train_npz) and os.path.exists(test_npz)):
            print(f"[skip] {subset}: npz not found")
            continue
        t0 = time.time()
        tr_emb, tr_lab, _, _ = load_flat(train_npz)
        te_emb, _, _, _ = load_flat(test_npz)

        msd = compute_msd(tr_emb, tr_lab)
        one_nn, k_nn = compute_knn(tr_emb, tr_lab)
        pair_fid = compute_pair_fid(tr_emb, tr_lab)
        tt_fid = fid_between(tr_emb, te_emb)

        results[subset] = {
            "DeepGaitV2_MSD": msd,
            "DeepGaitV2_1NN": one_nn,
            "DeepGaitV2_kNN": k_nn,
            "DeepGaitV2_FID": pair_fid,
            "FID_train_test_DeepGaitV2": tt_fid,
            "n_train_seqs": int(tr_emb.shape[0]),
            "n_test_seqs": int(te_emb.shape[0]),
        }
        print(f"[done] {subset}: MSD={msd:.4f} 1NN={one_nn:.4f} kNN={k_nn:.4f} "
              f"pairFID={pair_fid:.4f} ttFID={tt_fid:.4f} ({time.time()-t0:.1f}s)")

    out = os.path.join(FEAT_DIR, "dgv2_metrics_sustech.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
