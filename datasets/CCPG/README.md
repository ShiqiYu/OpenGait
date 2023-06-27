# The CCPG Benchmark

A Cloth-Changing Benchmark for Person re-identification and Gait Recognition (CCPG).

The original dataset can be found [here](https://github.com/BNU-IVC/CCPG). The original dataset is not publicly available. You need to request access to the dataset in order to download it.
## Data Pretreatment
```python
python datasets/CCPG/organize_ccpg.py --sil_path 'CCPG/CCPG_D_MASK_FACE_SHOE' --rgb_path 'CCPG/CCPG_G_SIL' --output_path 'CCPG/CCPG-end2end-pkl'
```

## Train
### GatiBase model:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_ccpg.yaml --phase train`


## Citation
If you use this dataset in your research, please cite the following paper:
```
@InProceedings{Li_2023_CVPR,
    author    = {Li, Weijia and Hou, Saihui and Zhang, Chunjie and Cao, Chunshui and Liu, Xu and Huang, Yongzhen and Zhao, Yao},
    title     = {An In-Depth Exploration of Person Re-Identification and Gait Recognition in Cloth-Changing Conditions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {13824-13833}
}
```
