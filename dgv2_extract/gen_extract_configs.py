#!/usr/bin/env python3
"""DeepGaitV2 埋め込み抽出用の partition JSON / config YAML を生成する。

各サブセットについて2つの config を作る:
  - train 側: dataset_root = そのサブセットの pkl ディレクトリ
              (partition の TEST_SET=001..075 のうちディレクトリに実在する被験者だけが
               OpenGait 側で自動的に使われる)
  - test 側:  dataset_root = GaitDatasetB-silh (フルデータ)、TEST_SET=076..124
              → 全サブセット共通のテストデータをそのサブセットのモデルで埋め込む

生成物:
  datasets/CASIA-B/extract/extract-train75.json
  datasets/CASIA-B/extract/extract-test49.json
  configs/deepgaitv2/extract/extract-{subset}-{train|test}.yaml
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /home/kera/OpenGait

# subset_key, pkl_dir, checkpoint save_name, class_num
SUBSETS = [
    ("default",     "20par-default",   "DeepGaitV2-ryu-default",   74),
    ("nm2",         "nm1-2",           "DeepGaitV2-ryu-nm2",       75),
    ("bg2",         "bg1-2",           "DeepGaitV2-ryu-bg2",       75),
    ("cl2",         "cl1-2",           "DeepGaitV2-ryu-cl2",       75),
    ("nm1-bg1",     "nm1-bg1",         "DeepGaitV2-ryu-nm1-bg1",   75),
    ("nm1-cl1",     "nm1-cl1",         "DeepGaitV2-ryu-nm1-cl1",   75),
    ("000-180",     "000-180",         "DeepGaitV2-000-180",       75),
    ("000-090",     "000-090",         "DeepGaitV2-ryu-000-090",   75),
    ("090-180",     "090-180",         "DeepGaitV2-090-180",       75),
    ("nm1-bg1-cl1", "nm1-bg1-cl1",     "DeepGaitV2-nm1-bg1-cl1",   75),
    ("nm2-bg2-cl2", "nm1-2bg1-2cl1-2", "DeepGaitV2-ryu-nm2bg2cl2", 75),
    ("nm6",         "nm6",             "DeepGaitV2-nm6",           75),
    ("bg1-cl1",     "bg1-cl1",         "DeepGaitV2-bg1-cl1",       75),
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
  restore_hint: 60000
  save_name: {save_name}
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
    class_num: {class_num}

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
  save_name: {save_name}
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
    part_dir = os.path.join(REPO, "datasets", "CASIA-B", "extract")
    cfg_dir = os.path.join(REPO, "configs", "deepgaitv2", "extract")
    os.makedirs(part_dir, exist_ok=True)
    os.makedirs(cfg_dir, exist_ok=True)

    # --- partition JSONs ---
    train75 = {"TRAIN_SET": [], "TEST_SET": [f"{i:03d}" for i in range(1, 76)]}
    test49 = {"TRAIN_SET": [], "TEST_SET": [f"{i:03d}" for i in range(76, 125)]}
    with open(os.path.join(part_dir, "extract-train75.json"), "w") as f:
        json.dump(train75, f, indent=4)
    with open(os.path.join(part_dir, "extract-test49.json"), "w") as f:
        json.dump(test49, f, indent=4)
    print(f"wrote {part_dir}/extract-train75.json (TEST_SET=001..075)")
    print(f"wrote {part_dir}/extract-test49.json (TEST_SET=076..124)")

    # --- configs ---
    for subset, pkl_dir, save_name, class_num in SUBSETS:
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
                save_name=save_name,
                dump_path=f"./dgv2_features/{subset}/{split}.npz",
                class_num=class_num,
            )
            path = os.path.join(cfg_dir, f"extract-{subset}-{split}.yaml")
            with open(path, "w") as f:
                f.write(cfg)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
