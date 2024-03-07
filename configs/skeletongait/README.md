# SkeletonGait: Gait Recognition Using Skeleton Maps

This [paper](https://arxiv.org/abs/2311.13444) has been accepted by AAAI 2024.

## Generating Heatmap and Training Steps

### Step 1: Generating Heatmap
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

Note: If your pose data follows the COCO 18 format (for instance, OU-MVLP pose data or data extracted using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) in COCO format), ensure to set `transfer_to_coco17` to True in the configuration file `configs/skeletongait/pretreatment_heatmap.yaml`.


**Optional**

### Step 2: Creating Symbolic Links for Heatmap and Silhouette Data

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
- `--dataset_pkl_ext_name`: An **optional** parameter to specify the extension for `.pkl` silhouette files. Defaults to `.pkl`. CCPG is `aligned-sils.pkl`, SUSTech-1K is `Camera-Sils_aligned.pkl`, and other is `.pkl`.

### Step3: Training SkeletonGait or SkeletonGait++

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

## Performance for SkeletonGait and SkeletonGait++

### SkeletonGait
| Datasets            | `Rank1` | Configuration                                |
|---------------------|---------|----------------------------------------------|
| CCPG                | CL: 52.4, UP: 65.4, DN: 72.8, BG: 80.9        | [skeletongait_CCPG.yaml](./skeletongait_CCPG.yaml) | 
| OU-MVLP (AlphaPose) |   TODO                                             | [skeletongait_OUMVLP.yaml](./skeletongait_OUMVLP.yaml) |
| SUSTech-1K          | Normal: 54.2, Bag: 51.7, Clothing: 21.34, Carrying: 51.59, Umberalla: 44.5, Uniform: 53.37, Occlusion: 67.07, Night: 44.15, Overall: 51.46 | [skeletongait_SUSTech1K.yaml](./skeletongait_SUSTech1K.yaml) |
| Gait3D              | 38.1                                         | [skeletongait_Gait3D.yaml](./skeletongait_Gait3D.yaml) |
| GREW                | TODO                                               | [skeletongait_GREW.yaml](./skeletongait_GREW.yaml) |

### SkeletonGait++
| Datasets            | `Rank1` | Configuration                                   |
|---------------------|---------|-------------------------------------------------|
| CCPG                | CL: 90.1, UP: 95.0, DN: 92.9, BG: 97.0          | [skeletongait++_CCPG.yaml](./skeletongait++_CCPG.yaml) |
| SUSTech-1K          | Normal: 85.09, Bag: 82.90, Clothing: 46.53, Carrying: 81.88, Umberalla: 80.76, Uniform: 82.50, Occlusion: 86.16, Night: 47.48, Overall: 81.33 | [skeletongait++_SUSTech1K.yaml](./skeletongait++_SUSTech1K.yaml) |
| Gait3D              | 77.40                                          | [skeletongait++_Gait3D.yaml](./skeletongait++_Gait3D.yaml) |
| GREW                | 87.04                                          | [skeletongait++_GREW.yaml](./skeletongait++_GREW.yaml) |


