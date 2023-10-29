# Gait3D-Parsing
This is the pre-processing instructions for the Gait3D-Parsing dataset. The original dataset can be found [here](https://gait3d.github.io/gait3d-parsing-hp/). The original dataset is not publicly available. You need to request access to the dataset in order to download it. This README explains how to extract the original dataset and convert it to a format suitable for OpenGait.
## Data Preparation
https://github.com/Gait3D/Gait3D-Benchmark#data-preparation
## Data Pretreatment
```python
python datasets/Gait3D-Parsing/pretreatment_gps.py -i 'Gait3D/2D_Parsings' -o 'Gait3D-pars-64-64-pkl' -r 64 -p
```

## Train
### ParsingGait model:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 opengait/main.py --cfgs ./configs/parsinggait/parsinggait_gait3d_parsing.yaml --phase train`

## Citation
If you use this dataset in your research, please cite the following paper:
```
@inproceedings{zheng2023parsinggait,
  title={Parsing is All You Need for Accurate Gait Recognition in the Wild},
  author={Jinkai Zheng, Xinchen Liu, Shuai Wang, Lihao Wang, Chenggang Yan, Wu Liu},
  booktitle={ACM International Conference on Multimedia (ACM MM)},
  year={2023}
}

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
