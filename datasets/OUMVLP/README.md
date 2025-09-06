# OUMVLP
Step1: Download URL: http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html

Step2: Unzip the dataset, you will get a structure directory like:
```
python datasets/OUMVLP/extractor.py --input_path Path_of_OUMVLP-base --output_path Path_of_OUMVLP-raw --password Given_Password
```  

- Original
    ```
    OUMVLP-raw
        Silhouette_000-00 (view-sequence)
            00001 (subject)
                0001.png (frame)
                0002.png (frame)
                ......
            00002
                0001.png (frame)
                0002.png (frame)
                ......
            ......
        Silhouette_000-01
            00001
                0001.png (frame)
                0002.png (frame)
                ......
            00002
                0001.png (frame)
                0002.png (frame)
                ......
            ......
        Silhouette_015-00
            ......
        Silhouette_015-01
            ......
        ......
    ```
Step3-1 : To rearrange directory of OUMVLP dataset(for silhouette), turning to id-type-view structure, Run 
```
python datasets/OUMVLP/rearrange_OUMVLP.py --input_path Path_of_OUMVLP-raw --output_path Path_of_OUMVLP-silu-rearranged
```  
Step3-2 : To rearrange directory of OUMVLP dataset(for pose), turning to id-type-view structure, Run 
```
python datasets/OUMVLP/rearrange_OUMVLP_pose.py --input_path Path_of_OUMVLP-pose --output_path Path_of_OUMVLP-pose-rearranged
```  

Step4-1: Transforming images to pickle file, run 
```
python datasets/pretreatment.py --input_path Path_of_OUMVLP-silu-rearranged --output_path Path_of_OUMVLP-pkl
```
Step4-2: Transforming pose txts to pickle file, run 

> [!IMPORTANT]
> Before extracting pose pkls, **you need to possess the pose selection index files** ([Why](https://github.com/ShiqiYu/OpenGait/pull/280)). Here are two ways to get it:
> 1. `Approach 1`: Directly download it:
>    - Open [Download Link](https://drive.google.com/drive/folders/1gkXdrVtNuGbU5wd8lWoPfAo_qYpokm52?usp=sharing), choose `AlphaPose` or `OpenPose` version
>    - Find a suitable location to unzip it, like `<somewhere>/OUMVLP/Pose/match_idx`. 
>    - Move the zip file into the `match_idx` dir and unzip it there. 
>    - You will finally get the index root: `<somewhere>/OUMVLP/Pose/match_idx/AlphaPose`   
>      *(Here we take `AlphaPose` version as an example, this path is what we call `Path_of_OUMVLP-pose-index` below)*
> 
> 2. `Approach 2`: Run the following command to generate it by yourself (**rearranged silhouette dataset is needed**):    
> 
>    ```bash
>    python datasets/OUMVLP/pose_index_extractor.py \
>    -p Path_of_OUMVLP-pose-rearranged \
>    -s Path_of_OUMVLP-silu-rearranged \
>    -o Path_of_OUMVLP-pose-index
>    ```

```bash
python datasets/pretreatment.py \
--input_path Path_of_OUMVLP-pose-rearranged \
--output_path Path_of_OUMVLP-pose-pkl \
--pose \
--dataset OUMVLP \
--oumvlp_index_dir Path_of_OUMVLP-pose-index
```

gernerate the 17 Number of Pose Points Format from 18 Number of Pose Points
```
python datasets/OUMVLP/rearrange_OUMVLP_pose.py --input_path Path_of_OUMVLP-pose18 --output_path Path_of_OUMVLP-pose17
```

- Processed
    ```
    OUMVLP-pkl
        00001 (subject)
            00 (sequence)
                000 (view)
                    000.pkl (contains all frames)
                015 (view)
                    015.pkl (contains all frames)
                ...
            01 (sequence)
                000 (view)
                    000.pkl (contains all frames)
                015 (view)
                    015.pkl (contains all frames)
                ......
        00002 (subject)
            ......
        ......
    ```
