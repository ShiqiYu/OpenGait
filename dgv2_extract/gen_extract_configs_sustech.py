#!/usr/bin/env python3
"""DeepGaitV2 埋め込み抽出用の config YAML を生成する(SUSTech1K版・共通モデル)。

gen_extract_configs.py との違い:
  - 特徴抽出に使うチェックポイントを「サブセットごとの自前学習モデル」ではなく、
    CASIA-Bを一切学習していない共通の SUSTech1K 版 DeepGaitV2 に固定する
    (train-on-train / 13モデルに分裂した特徴空間の問題を解消するため)。
  - restore_hint は文字列でチェックポイントの絶対パスを直接渡す
    (base_model.resume_ckpt: isinstance(restore_hint, str) の分岐)。
  - チェックポイントは SeparateBNNecks.class_num=250 (SUSTech1Kの学習ID数)で
    学習されているため、config側も class_num=250 にして strict load を通す。
    抽出される埋め込み (embed_1 = FCsの出力) は BNNecks/class_num に依存しないので、
    75 と 250 の違いは埋め込みの値そのものには影響しない。
  - dump_path は既存の dgv2_features/ とは別ディレクトリ dgv2_features_sustech/ にして、
    per-subset版の結果を上書きせず並存させる。

partition JSON (extract-train75.json / extract-test49.json) は
gen_extract_configs.py で既に生成済みのものをそのまま使う。

生成物:
  configs/deepgaitv2/extract_sustech/extract-{subset}-{train|test}.yaml
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /home/kera/OpenGait

# コンテナ内では /home/kera/OpenGait が /app/OpenGait にマウントされ、
# 実行は "cd /app/OpenGait" した状態で行われる (run_extract_all.sh と同様)。
# dataset_root 等と同じく相対パス("./...")で書き、コンテナ内カレントディレクトリ基準で解決させる。
SUSTECH_CKPT = "./output/SUSTech1K/DeepGaitV2/DeepGaitV2/checkpoints/DeepGaitV2_30_DA-50000.pt"
SUSTECH_CKPT_HOST = os.path.join(
    REPO, "output", "SUSTech1K", "DeepGaitV2", "DeepGaitV2",
    "checkpoints", "DeepGaitV2_30_DA-50000.pt")

# subset_key, pkl_dir (train側 dataset_root に使うディレクトリ名)
SUBSETS = [
    ("default",     "20par-default"),
    ("nm2",         "nm1-2"),
    ("bg2",         "bg1-2"),
    ("cl2",         "cl1-2"),
    ("nm1-bg1",     "nm1-bg1"),
    ("nm1-cl1",     "nm1-cl1"),
    ("000-180",     "000-180"),
    ("000-090",     "000-090"),
    ("090-180",     "090-180"),
    ("nm1-bg1-cl1", "nm1-bg1-cl1"),
    ("nm2-bg2-cl2", "nm1-2bg1-2cl1-2"),
    ("nm6",         "nm6"),
    ("bg1-cl1",     "bg1-cl1"),
]

CONFIG_TEMPLATE = """\
data_cfg:
  dataset_name: CASIA-B
  dataset_root: {dataset_root}
  dataset_partition: {dataset_partition}
  num_workers: 1
  remove_no_gallery: false
  test_dataset_name: CASIA-B

evaluator_cfg:
  enable_float16: true
  restore_ckpt_strict: true
  restore_hint: {ckpt_path}
  save_name: DeepGaitV2-sustech1k-common
  eval_func: dump_features
  dump_path: {dump_path}
  sampler:
    batch_shuffle: false
    batch_size: 2
    sample_type: all_ordered
    frames_all_limit: 720
  metric: euc
  transform:
    - type: BaseSilCuttingTransform

loss_cfg:
  - loss_term_weight: 1.0
    margin: 0.2
    type: TripletLoss
    log_prefix: triplet
  - loss_term_weight: 1.0
    scale: 16
    type: CrossEntropyLoss
    log_prefix: softmax
    log_accuracy: true

model_cfg:
  model: DeepGaitV2
  Backbone:
    mode: p3d
    in_channels: 1
    layers:
      - 1
      - 1
      - 1
      - 1
    channels:
      - 64
      - 128
      - 256
      - 512
  SeparateBNNecks:
    class_num: 250

optimizer_cfg:
  lr: 0.1
  momentum: 0.9
  solver: SGD
  weight_decay: 0.0005

scheduler_cfg:
  gamma: 0.1
  milestones:
    - 20000
    - 40000
    - 50000
  scheduler: MultiStepLR

trainer_cfg:
  enable_float16: true
  fix_BN: false
  log_iter: 100
  with_test: false
  restore_ckpt_strict: true
  restore_hint: 0
  save_iter: 30000
  save_name: DeepGaitV2-sustech1k-common
  sync_BN: true
  total_iter: 60000
  sampler:
    batch_shuffle: true
    batch_size:
      - 8
      - 16
    frames_num_fixed: 30
    frames_skip_num: 4
    sample_type: fixed_ordered
    type: TripletSampler
  transform:
    - type: Compose
      trf_cfg:
        - type: RandomPerspective
          prob: 0.0
        - type: BaseSilCuttingTransform
        - type: RandomHorizontalFlip
          prob: 0.0
        - type: RandomRotate
          prob: 0.0
"""


def main():
    cfg_dir = os.path.join(REPO, "configs", "deepgaitv2", "extract_sustech")
    os.makedirs(cfg_dir, exist_ok=True)

    if not os.path.exists(SUSTECH_CKPT_HOST):
        raise FileNotFoundError(f"checkpoint not found: {SUSTECH_CKPT_HOST}")

    for subset, pkl_dir in SUBSETS:
        for split in ("train", "test"):
            if split == "train":
                dataset_root = f"./CASIA-B/{pkl_dir}"
                partition = "./datasets/CASIA-B/extract/extract-train75.json"
            else:
                dataset_root = "./CASIA-B/GaitDatasetB-silh"
                partition = "./datasets/CASIA-B/extract/extract-test49.json"
            cfg = CONFIG_TEMPLATE.format(
                dataset_root=dataset_root,
                dataset_partition=partition,
                ckpt_path=SUSTECH_CKPT,
                dump_path=f"./dgv2_features_sustech/{subset}/{split}.npz",
            )
            path = os.path.join(cfg_dir, f"extract-{subset}-{split}.yaml")
            with open(path, "w") as f:
                f.write(cfg)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
