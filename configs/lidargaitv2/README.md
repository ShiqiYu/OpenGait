# LidarGait++: Learning Local Features and Size Awareness from LiDAR Point Clouds for 3D Gait Recognition

This [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_LidarGait_Learning_Local_Features_and_Size_Awareness_from_LiDAR_Point_CVPR_2025_paper.pdf) has been accepted by CVPR 2025.



## Prepare dataset
**SUSTech1K**: 
- Step 1. Apply for [SUSTech1K](https://lidargait.github.io/).

**FreeGait** (Optional): 

- Step 1. Download [FreeGait](https://drive.google.com/drive/folders/1I9zOCmqUuBUcOmvO1cgZtUC6uSfmAq7h) first. 

- Then rearrange the folder structure like SUSTech1K/CASIA-B to fit OpenGait framework.
    ```
        python datasets/FreeGait/rearrange_freegait.py --input_path yout_freegait_path
    ```

## Train
To train on SUSTech1K, run
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/lidargaitv2/lidargaitv2_sustech1k.yaml --phase train
```
or train on FreeGait, run
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/lidargaitv2/lidargaitv2_freegait.yaml --phase train
```

## Citation

```bibtex
@inproceedings{shen2023lidargait,
  title={Lidargait: Benchmarking 3d gait recognition with point clouds},
  author={Shen, Chuanfu and Fan, Chao and Wu, Wei and Wang, Rui and Huang, George Q and Yu, Shiqi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1054--1063},
  year={2023}
}

@inproceedings{shen2025lidargait++,
  title={LidarGait++: Learning Local Features and Size Awareness from LiDAR Point Clouds for 3D Gait Recognition},
  author={Shen, Chuanfu and Wang, Rui and Duan, Lixin and Yu, Shiqi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={6627--6636},
  year={2025}
}
```
