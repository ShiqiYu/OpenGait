# CCGR-Mini Benchmark: Cross-Covariate Gait Recognition (Mini)

This page includes intructions to preprocess the CCGR-Mini dataset.

Please refer to the author's original [repo](https://github.com/ShinanZou/CCGR/tree/CCGR-Benchmark) to **download** and extract the dataset.

## Data pretreatment

Run following command to pretreat the dataset into the format required by OpenGait:
```bash
python datasets/CCGR-MINI/organize_ccgr.py \
         --sil_par_pose_path 'CCGR_MINI_SIL_PAR_POSE/CCGR-MINI' \ # Path to subdirectories containing silhouettes, parsing, and poses
         --rgb_path 'CCGR_MINI_RGB_CUT/CCGR-MINI-RGB-V1' \        # Path to subdirectories containing raw RGB files
         --output_path 'CCGR-Mini/CCGR-rgb-silh-par-pose-pkl'
```

**Note**: For RGB sequences, this script automatically handles preprocessing by resizing frames and removing the original padding.

#### Additional options:
- `--copy-mode`: Defines how the data is handled. By default (copy), a full copy of the silhouette, parsing, and pose `pkl` files will be made. Alternatively, use the `symlink` option to create symbolic links instead.
- `--num-workers`: Allows you to specify the number of parallel workers to speed up preprocessing.

Pretreated dataset will follow this directory structure:

```
    DATASET_ROOT/
        subject id (1, 2, ..., 1000)/
            seq type (AS1, ASBGCLBXCV1, ASBX1, ...)/
                    view (180_a.avi, 67_5_2.avi, ...)/
                        000-{view}ratios_HW-{sub-seq}.pkl  # Height and Width of every RGB frame
                        001-{view}rgb_f-{sub-seq}.pkl       # Pretreated RGB sequence [128 x 128]
                        002-{view}sil-{sub-seq}.pkl         # Pretreated silhouette sequence
                        003-{view}par-{sub-seq}.pkl         # Pretreated parsing sequence
                        004-{view}pose-{sub-seq}.pkl        # Pretreated pose coordinate sequence
                ......
            ......
        ......
```

## Training & Test

### BigGait model:

Update the `dataset_root` path in `configs/biggait/BigGait_CCGR-Mini.yaml`, and then run:

`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/biggait/BigGait_CCGR-Mini.yaml --phase train`


## Acknowledgements

This dataset was collected by the [Zou et. al.](https://github.com/ShinanZou/CCGR/tree/CCGR-Benchmark) Portions of the code from their original repository regarding dataset configuration and evaluation have been integrated and adapted for this repository.
