#!/bin/bash
# DeepGaitV2 埋め込み抽出: 13サブセット × {train, test} を順次実行する。
# 出力: /home/kera/OpenGait/dgv2_features/{subset}/{train|test}.npz
# ログ: /home/kera/OpenGait/dgv2_extract/logs/{subset}-{split}.log
# 既に npz が存在する組はスキップする（再実行安全）。

set -u
REPO=/home/kera/OpenGait
LOGDIR=$REPO/dgv2_extract/logs
mkdir -p "$LOGDIR"

SUBSETS=(default nm2 bg2 cl2 nm1-bg1 nm1-cl1 000-180 000-090 090-180 nm1-bg1-cl1 nm2-bg2-cl2 nm6 bg1-cl1)

for subset in "${SUBSETS[@]}"; do
  for split in train test; do
    npz=$REPO/dgv2_features/$subset/$split.npz
    if [ -f "$npz" ]; then
      echo "[skip] $subset/$split (exists)"
      continue
    fi
    cfg=./configs/deepgaitv2/extract/extract-$subset-$split.yaml
    log=$LOGDIR/$subset-$split.log
    echo "[run ] $subset/$split -> $log ($(date +%H:%M:%S))"
    docker exec opengait_container bash -c \
      "cd /app/OpenGait && CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29511 opengait/main.py --cfgs $cfg --phase test" \
      > "$log" 2>&1
    if [ -f "$npz" ]; then
      echo "[ ok ] $subset/$split"
    else
      echo "[FAIL] $subset/$split (see $log)"
    fi
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
