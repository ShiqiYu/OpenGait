# Gait3D
This is the pre-processing instructions for the Gait3D dataset. The original dataset can be found [here](https://gait3d.github.io/). The original dataset is not publicly available. You need to request access to the dataset in order to download it. This README explains how to extract the original dataset and convert it to a format suitable for OpenGait.
## Data Preparation
https://github.com/Gait3D/Gait3D-Benchmark#data-preparation
## Data Pretreatment
```python
python datasets/pretreatment.py --input_path 'Gait3D/2D_Silhouettes' --output_path 'Gait3D-sils-64-64-pkl'
python datasets/Gait3D/pretreatment_smpl.py --input_path 'Gait3D/3D_SMPLs' --output_path 'Gait3D-smpls-pkl'

(optional) python datasets/pretreatment.py --input_path 'Gait3D/2D_Silhouettes' --img_size 128 --output_path 'Gait3D-sils-128-128-pkl'

python datasets/Gait3D/merge_two_modality.py --sils_path 'Gait3D-sils-64-64-pkl' --smpls_path 'Gait3D-smpls-pkl' --output_path 'Gait3D-merged-pkl' --link 'hard'
```

## Train
### Baseline model:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/baseline/baseline_Gait3D.yaml --phase train`
### SMPLGait model:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/smplgait/smplgait.yaml --phase train`

## Citation
If you use this dataset in your research, please cite the following paper:
```
@inproceedings{zheng2022gait3d,
title={Gait Recognition in the Wild with Dense 3D Representations and A Benchmark},
author={Jinkai Zheng, Xinchen Liu, Wu Liu, Lingxiao He, Chenggang Yan, Tao Mei},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2022}
}
```
If you think the re-implementation of OpenGait is useful, please cite the following paper:
```
@misc{fan2022opengait,
      title={OpenGait: Revisiting Gait Recognition Toward Better Practicality}, 
      author={Chao Fan and Junhao Liang and Chuanfu Shen and Saihui Hou and Yongzhen Huang and Shiqi Yu},
      year={2022},
      eprint={2211.06597},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Acknowledgements
This dataset was collected by the [Zheng at. al.](https://gait3d.github.io/). The pre-processing instructions are modified from (https://github.com/Gait3D/Gait3D-Benchmark).
