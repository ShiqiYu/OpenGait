# SkeletonGait: Gait Recognition Using Skeleton Maps

This [paper](https://arxiv.org/abs/2311.13444) has been accepted by AAAI 2023.

## Step 1: Generating Heatmap
Leveraging the power of Distributed Data Parallel (DDP), we've streamlined the heatmap generation process. Below is the script to initiate the generation:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
--nproc_per_node=4 \
datasets/pretreatment_heatmap.py \
--pose_data_path=<your pose .pkl files path> \
--save_root=<your_path> \
--dataset_name=<dataset_name>
```

Parameter Guide:
- `--pose_data_path`: Specifies the directory containing the pose data files (`.pkl`, ID-Level). This is **required**.
- `--save_root`: Designates the root directory for storing the generated heatmap files (`.pkl`, ID-Level). This is **required**.
- `--dataset_name`: The name of the dataset undergoing preprocessing. This is required.
- `--ext_name`: An **optional** suffix for the 'save_root' directory to facilitate identification. Defaults to an empty string.
- `--heatmap_cfg_path`: Path to the configuration file of the heatmap generator. The default setting is `configs/skeletongait/pretreatment_heatmap.yaml`. 

**Optional**

## Step 2: Creating Symbolic Links for Heatmap and Silhouette Data

The script to symlink heatmaps and silouettes is as follows:

```
python datasets/ln_sil_heatmap.py \
--heatmap_data_path=<path_to_your_heatmap_folder> \
--silhouette_data_path=<path_to_your_silhouette_folder> \
--output_path=<path_to_your_output_folder>
```

Parameter Guide:
- `--heatmap_data_path`: The **absolute** path to your heatmap data. This is **required**.
- `--silhouette_data_path`: The **absolute** path to your silhouette data. This is **required**.
- `--output_path`: Designates the directory for linked output data. This is **required**.
- `--dataset_pkl_ext_name`: An **optional** parameter to specify the extension for `.pkl` silhouette files. Defaults to `.pkl`. 

## Step3: Training SkeletonGait or SkeletonGait++

The script to SkeletonGait is as follows:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
 python -m torch.distributed.launch \
 --nproc_per_node=4 opengait/main.py \
 --cfgs ./configs/skeletongait/skeletongait_Gait3D.yaml \
 --phase train --log_to_file
```

The script to SkeletonGait++ is as follows:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
 python -m torch.distributed.launch \
 --nproc_per_node=4 opengait/main.py \
 --cfgs ./configs/skeletongait/skeletongait++_Gait3D.yaml \
 --phase train --log_to_file
```
