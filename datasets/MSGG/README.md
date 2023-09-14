# Datasets for MSGG
MSGG needs to convert the pose keypoints format of coco17 into the input format of Pyramid keypoints.

## Data Pretreatment
```python
python datasets/MSGG/pyramid_keypoints_msgg.py --input_path Path_of_pose_pkl --output_path Path_of_pose_pyramid_pkl
```

## Citation
```
@article{peng2023learning,
  title={Learning rich features for gait recognition by integrating skeletons and silhouettes},
  author={Peng, Yunjie and Ma, Kang and Zhang, Yang and He, Zhiqiang},
  journal={Multimedia Tools and Applications},
  pages={1--22},
  year={2023},
  publisher={Springer}
}
```