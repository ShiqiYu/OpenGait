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
python datasets/OUMVLP/rearrange_OUMVLP.py --input_path Path_of_OUMVLP-raw --output_path Path_of_OUMVLP-rearranged
```  
Step3-2 : To rearrange directory of OUMVLP dataset(for pose), turning to id-type-view structure, Run 
```
python datasets/OUMVLP/rearrange_OUMVLP_pose.py --input_path Path_of_OUMVLP-pose --output_path Path_of_OUMVLP-pose-rearranged
```  

Step4-1: Transforming images to pickle file, run 
```
python datasets/pretreatment.py --input_path Path_of_OUMVLP-rearranged --output_path Path_of_OUMVLP-pkl
```
Step4-2: Transforming pose txts to pickle file, run 
```
python datasets/pretreatment.py --input_path Path_of_GREW-pose-rearranged --output_path Path_of_GREW-pose-pkl --pose --dataset GREW
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
