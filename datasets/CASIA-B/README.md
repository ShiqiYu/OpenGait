# CASIA-B
Download URL: http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip
- Original
    ```
    CASIA-B
        001 (subject)
            bg-01 (type)
                000 (view)
                    001-bg-01-000-001.png (frame)
                    001-bg-01-000-002.png (frame)
                    ......
                ......
            ......
        ......
    ```
- Run `python datasets/pretreatment.py --input_path CASIA-B --output_path CASIA-B-pkl`
- Processed
    ```
    CASIA-B-pkl
        001 (subject)
            bg-01 (type)
                    000 (view)
                        000.pkl (contains all frames)
                ......
            ......
        ......
    ```
    
# CASIA-B\*
## Introduction
CASIA-B\* is a re-segmented version of CASIA-B processed by Liang et al. The extra import of CASIA-B* owes to the background subtraction algorithm that CASIA-B uses for generating the silhouette data tends to produce much noise and is outdated for real-world applications nowadays. We use the up-to-date pretreatment strategy to re-segment the raw videos, i.e., the deep pedestrian track and segmentation algorithms. As a result, CASIA-B\* consists of the cropped RGB images, binary silhouettes, the height-width ratio of the obtained bounding boxes and the aligned silhouettes. Please refer to [GaitEdge](../../configs/gaitedge/README.md) for more details. If you need this sub-set, please apply with the instruction mentioned in [http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp]. In the Email Subject, please mark the specific dataset you need, i.e., Dataset B*.

## Data structure
```
casiab-128-end2end/
    001 (subject)
        bg-01 (type)
                000 (view)
                    000-aligned-sils.pkl (aligned sils, nx64x44)
                    000-ratios.pkl (aspect ratio of bounding boxes, n)
                    000-rgbs.pkl (cropped RGB images, nx3x128x128)
                    000-sils.pkl (binary silhouettes, nx128x128)
            ......
        ......
    ......
```

## How to use
By default, it loads all file directory information like other datasets before training starts. If you need to use some of these data separately, such as `aligned-sils`, then you can use the `data_in_use` parameter in `data_cfg` lexicographically, *i.e.* `data_in_use: [true, false, false, false]`.
