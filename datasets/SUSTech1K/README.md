# Tutorial for [SUSTech1K](https://lidargait.github.io)

## Download the SUSTech1K dataset
Download the dataset from the [link](https://lidargait.github.io).
decompress these two file by following command:
```shell
unzip -P password SUSTech1K-pkl.zip   | xargs -n1 tar xzvf
```
password should be obtained by signing [agreement](https://lidargait.github.io/static/resources/SUSTech1KAgreement.pdf) and sending to email (shencf2019@mail.sustech.edu.cn)

Then you will get SUSTech1K formatted as:
```
SUSTech1K-Released-pkl
├── 0000                            # Identity
│   ├── 00-nm                       # sequence_number - sequence_covariates
│   │   ├── 000                     # viewpoint_angle
│   │   │   ├── 00-000-LiDAR-PCDs.pkl                 # (10Hz) Point Clouds 
│   │   │   ├── 01-000-LiDAR-PCDs_depths.pkl          # (10Hz) Projected Depths from Point Clouds
│   │   │   ├── 02-000-LiDAR-PCDs_sils.pkl            # (10Hz) Projected Silhouettes from Point Clouds
│   │   │   ├── 03-000-Camera-Pose.pkl                # (30Hz) Estimated Skeleton using ViTPose
│   │   │   ├── 04-000-Camera-Ratios-HW.pkl           # (30Hz) (H,W) of Camera Images
│   │   │   ├── 05-000-Camera-RGB_raw.pkl             # (30Hz) Raw Camera images (frames, 64, 64, 3) (if you want larger resolution, you can process SUSTech1K-Released-RAW by yourself using pretreatment_SUSTech1K.py 
│   │   │   ├── 06-000-Camera-Sils_aligned.pkl        # (30Hz) Aligned silhouettes
│   │   │   ├── 07-000-Camera-Sils_raw.pkl            # (30Hz) Estimated silhouettes without alignment
│   │   │   ├── 08-sync-000-LiDAR-PCDs.pkl            # (10Hz synchronized to Camera) Point Clouds, 
│   │   │   ├── 09-sync-000-LiDAR-PCDs_depths.pkl     # (10Hz synchronized to Camera) Projected Depths from Point Clouds
│   │   │   ├── 10-sync-000-LiDAR-PCDs_sils.pkl       # (10Hz synchronized to Camera) Projected Silhouettes from Point Clouds
│   │   │   ├── 11-sync-000-Camera-Pose.pkl           # (10Hz synchronized to LiDAR) Estimated Skeleton using ViTPose
│   │   │   ├── 12-sync-000-Camera-Ratios-HW.pkl      # (10Hz synchronized to LiDAR) (H,W) of Camera Images
│   │   │   ├── 13-sync-000-Camera-RGB_raw.pkl        # (10Hz synchronized to LiDAR) Raw Camera images (frames, 64, 64, 3)
│   │   │   ├── 14-sync-000-Camera-Sils_aligned.pkl   # (10Hz synchronized to LiDAR) Aligned silhouettes
│   │   │   └── 15-sync-000-Camera-Sils_raw.pkl       # (10Hz synchronized to LiDAR) Estimated silhouettes without alignment
                ......
            ......
        ......
    ......
```

## Train the dataset
Modify the `dataset_root` in `configs/lidargait/lidargait_sustech1k.yaml`, and then run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs configs/lidargait/lidargait_sustech1k.yaml --phase train
```


## Process from RAW dataset

### Preprocess the dataset (Optional)
Download the raw dataset from the [official link](https://lidargait.github.io). You will get two compressed files, i.e. `DATASET_DOWNLOAD.md5`, `SUSTeck1K-RAW.zip`, and `SUSTeck1K-pkl.zip`.
We recommend using our provided pickle files for convenience, or process raw dataset into pickle by this command:
```shell
python datasets/SUSTech1K/pretreatment_SUSTech1K.py -i SUSTech1K-Released-2023 -o SUSTech1K-pkl -n 8
```

### Projecting PointCloud into Depth image (Optional)
You can use our processed depth images, or you can process via the command:
```shell
python datasets/SUSTech1K/point2depth.py -i SUSTech1K-Released-2023/ -o SUSTech1K-Released-2023/ -n 8
```
We recommend using our provided depth images for convenience.

