# DeepGaitV2 特徴量による学習データ有効性評価 (0705)

中村さんの論文 (IEICE技報「深層特徴空間におけるデータ特性に基づく学習データの有効性評価」)
の枠組みを、**特徴抽出器を Inception/DINO から DeepGaitV2 に置き換えて**再現したもの。

## 方針
- 各サブセットの埋め込みは **そのサブセット自身の学習済み DeepGaitV2** で抽出する
  (Inception/DINO のような単一共通モデルではない)。
- `bg1-cl1` (id=7) は当初 (0705時点) 学習済みモデルが存在せず DeepGaitV2 特徴が欠測 → 12/13サブセットで実行。
  その後 `DeepGaitV2-bg1-cl1-60000.pt` の学習が完了したため抽出をやり直し、**現在は 13/13 全サブセットで実行済み** (0706)。

## パイプラインと成果物

```
OpenGait/
├── opengait/evaluation/evaluator.py       # dump_features() を追加 (埋め込みをnpz保存)
│
├── dgv2_extract/                          # ① 埋め込み抽出
│   ├── gen_extract_configs.py             #   26 config + partition JSON を生成
│   ├── run_extract_all.sh                 #   13サブセット×{train,test}=26抽出を順次実行
│   └── compute_dgv2_metrics.py            # ② 埋め込み→MSD/1NN/kNN/FID/train-test FID
│
├── configs/deepgaitv2/extract/            #   生成された抽出用 config (26 yaml)
├── datasets/CASIA-B/extract/              #   extract-train75.json / extract-test49.json
│
├── dgv2_features/                         #   中間成果
│   ├── {subset}/train.npz                 #   学習データの埋め込み [n,256,16]+labels/types/views
│   ├── {subset}/test.npz                  #   共通テスト49人の埋め込み
│   └── dgv2_metrics.json                  #   13サブセットの5指標
│
└── dgv2_analysis/                         # ③④⑤ 回帰
    ├── metrics_ryu.json                   #   ryu の元 metrics.json のコピー (13データセット)
    ├── merge_metrics.py                   # ③ DeepGaitV2列を metrics に追加
    ├── metrics_dgv2.json                  #   マージ結果 (13/13 すべて DeepGaitV2 列あり)
    ├── lodo_ridge_dgv2.py                 # ④ Leave-One-Subset-Out Ridge 回帰
    ├── result_ridge_dgv2.json
    ├── pairwise_logreg_dgv2.py            # ⑤ 3クラス優劣判定 ロジスティック回帰
    └── result_logreg_dgv2.json
```

## 実行手順 (すべてコンテナ内)

```bash
# ① 抽出 (ホストで)
python3 dgv2_extract/gen_extract_configs.py
bash    dgv2_extract/run_extract_all.sh          # docker exec で 24 回 test phase を回す

# ②③④⑤ (numpy/sklearn が要るのでコンテナ内)
docker exec opengait_container python /app/OpenGait/dgv2_extract/compute_dgv2_metrics.py
docker exec opengait_container python /app/OpenGait/dgv2_analysis/merge_metrics.py
docker exec opengait_container python /app/OpenGait/dgv2_analysis/lodo_ridge_dgv2.py
docker exec opengait_container python /app/OpenGait/dgv2_analysis/pairwise_logreg_dgv2.py --cv pair
```

## id ↔ サブセット対応 (論文 Table 1 の Rank-1 と照合し13件全一致で確定)
1=default, 2=nm2, 3=bg2, 4=cl2, 5=nm1-bg1, 6=nm1-cl1, 7=bg1-cl1,
8=000-180, 9=000-090, 10=090-180, 11=nm1-bg1-cl1, 12=nm2-bg2-cl2, 13=nm6

## 指標計算の注意
- 埋め込み [n,256,16] は 4096 次元に flatten して使用。
- FID の `Tr((Σ1Σ2)^{1/2})` は d×d の sqrtm を避け、Gram トリック
  (`Σ σ_j(A1ᵀA2)` = 核ノルム) で厳密計算している (`fid_between`)。

## 結果 (target=acc_all)

### 13/13 (0706, bg1-cl1 学習完了後)
- Ridge (LODO): MAE=0.1453, RMSE=0.2376, Spearman=0.6978
- Logistic (3クラス優劣, τ=0.05, leave-one-pair-out, 156ペア): Accuracy=0.7179, Macro-F1=0.7024
- 参考: 論文 (Inception/DINO, 13サブセット) Ridge Spearman=0.8352 / 優劣 Acc=0.7949, Macro-F1=0.7296
  → DeepGaitV2-only は依然やや劣るが、12/13時点より論文値との差は縮まった。

### 12/13 (0705, bg1-cl1 欠測時点。参考として保持)
- Ridge (LODO): MAE=0.1616, RMSE=0.2582, Spearman=0.5594
- Logistic (3クラス優劣, τ=0.05, leave-one-pair-out, 132ペア): Accuracy=0.6667, Macro-F1=0.6435

### 12/13 → 13/13 での変化
- サブセットが1つ増えただけで Spearman が 0.56→0.70、優劣判定 Accuracy が 0.667→0.718 に改善。
  LODO/leave-one-pair-out は学習データ数に敏感なため、13サンプルという少なさが12/13結果の
  不安定要因の一つだったことを示唆する。
- 両モデルとも一貫して `FID_train_test_DeepGaitV2` (train-test 分布距離) が最重要特徴のまま
  → 「Inception 指標が最も影響力を持つ」という論文の結論と、少なくとも
  「train-test 分布距離が最終精度を最もよく説明する」という定性的な傾向は整合している。
