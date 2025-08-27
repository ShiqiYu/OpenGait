# Tutorial for [Scoliosis1K](https://zhouzi180.github.io/Scoliosis1K)
## Download the Scoliosis1K Dataset

You can download the dataset from the [official website](https://zhouzi180.github.io/Scoliosis1K).
The dataset is provided as four compressed files:

* `Scoliosis1K-sil-raw.zip`
* `Scoliosis1K-sil-pkl.zip`
* `Scoliosis1K-pose-raw.zip`
* `Scoliosis1K-pose-pkl.zip`

We recommend using the provided pickle (`.pkl`) files for convenience.
Decompress them with the following commands:

```bash
unzip -P <password> Scoliosis1K-sil-pkl.zip
unzip -P <password> Scoliosis1K-pose-pkl.zip
```

> **Note**: The \<password\> can be obtained by signing the [release agreement](https://zhouzi180.github.io/Scoliosis1K/static/resources/Scoliosis1k_release_agreement.pdf) and sending it to **[12331257@mail.sustech.edu.cn](mailto:12331257@mail.sustech.edu.cn)**.

### Dataset Structure

After decompression, you will get the following structure:

```
├── Scoliosis1K-sil-pkl
│   ├── 00000                     # Identity
│   │   ├── Positive              # Class
│   │   │   ├── 000_180           # View
│   │   │   └── 000_180.pkl       # Estimated Silhouette (PP-HumanSeg v2)
│
├── Scoliosis1K-pose-pkl
│   ├── 00000                     # Identity
│   │   ├── Positive              # Class
│   │   │   ├── 000_180           # View
│   │   │   └── 000_180.pkl       # Estimated 2D Pose (ViTPose)
```

### Processing from RAW Dataset (optional)

If you prefer, you can process the raw dataset into `.pkl` format.

```bash
# For silhouette raw data
python datasets/pretreatment.py --input_path=<path_to_raw_silhouettes> -output_path=<output_path>

# For pose raw data
python datasets/pretreatment.py --input_path=<path_to_raw_pose> -output_path=<output_path> --pose --dataset=OUMVLP
```
---

## Training and Testing

Before training or testing, modify the `dataset_root` field in
`configs/sconet/sconet_scoliosis1k.yaml`.

Then run the following commands:

```bash
# Training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
opengait/main.py --cfgs configs/sconet/sconet_scoliosis1k.yaml --phase train --log_to_file

# Testing
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
opengait/main.py --cfgs configs/sconet/sconet_scoliosis1k.yaml --phase test --log_to_file
```

---

## Pose-to-Heatmap Conversion

*From our paper: **Pose as Clinical Prior: Learning Dual Representations for Scoliosis Screening (MICCAI 2025)***

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
datasets/pretreatment_heatmap.py \
  --pose_data_path=<path_to_pose_pkl> \
  --save_root=<output_path> \
  --dataset_name=OUMVLP
```