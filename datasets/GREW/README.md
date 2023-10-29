# GREW Tutorial
<!-- ![](http://hid2022.iapr-tc4.org/wp-content/uploads/sites/7/2022/03/%E5%9B%BE%E7%89%871-2.png) -->
This is for [GREW-Benchmark](https://github.com/GREW-Benchmark/GREW-Benchmark). We report our result of 48% using the baseline model. In order for participants to better start the first step, we provide a tutorial on how to use OpenGait for GREW.

## Preprocess the dataset
Download the raw dataset from the [official link](https://www.grew-benchmark.org/download.html). You will get three compressed files, i.e. `train.zip`, `test.zip` and `distractor.zip`.

Step 1: Unzip train and test:
```shell
unzip -P password train.zip (password is the obtained password)
tar -xzvf train.tgz
cd train
ls *.tgz | xargs -n1 tar xzvf
```

```shell
unzip -P password test.zip (password is the obtained password)
tar -xzvf test.tgz
cd test & cd gallery
ls *.tgz | xargs -n1 tar xzvf
cd .. & cd probe
ls *.tgz | xargs -n1 tar xzvf
```

After unpacking these compressed files, run this command:

Step2-1 : To rearrange directory of GREW dataset(for silhouette), turning to id-type-view structure, Run 
```
python datasets/GREW/rearrange_GREW.py --input_path Path_of_GREW-raw --output_path Path_of_GREW-rearranged
```  
Step2-2 : To rearrange directory of GREW dataset(for pose), turning to id-type-view structure, Run 
```
python datasets/GREW/rearrange_GREW_pose.py --input_path Path_of_GREW-pose --output_path Path_of_GREW-pose-rearranged
```  

Step3-1: Transforming images to pickle file, run 
```
python datasets/pretreatment.py --input_path Path_of_GREW-rearranged --output_path Path_of_GREW-pkl --dataset GREW
```
Step3-2: Transforming pose txts to pickle file, run 
```
python datasets/pretreatment.py --input_path Path_of_GREW-pose-rearranged --output_path Path_of_GREW-pose-pkl --pose --dataset GREW
```

Then you will see the structure like:

- Processed
    ```
    GREW-pkl
    ├── 00001train (subject in training set)
        ├── 00
            ├── 4XPn5Z28
                ├── 4XPn5Z28.pkl
            ├──5TXe8svE
                ├── 5TXe8svE.pkl
                ......
    ├── 00001 (subject in testing set)
        ├── 01
            ├── 79XJefi8
                ├── 79XJefi8.pkl
        ├── 02
            ├── t16VLaQf
                ├── t16VLaQf.pkl
    ├── probe
        ├── etaGVnWf
            ├── etaGVnWf.pkl
        ├── eT1EXpgZ
            ├── eT1EXpgZ.pkl
        ...
    ...
    ```

## Train the dataset
Modify the `dataset_root` in `./config/baseline/baseline_GREW.yaml`, and then run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./config/baseline/baseline_GREW.yaml --phase train
```

## Get the submission file
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./config/baseline/baseline_GREW.yaml --phase test
```
The result will be generated in your working directory, you must rename and compress it as the requirements before submitting.

## Evaluation locally
While the original grew treat both seq_01 and seq_02 as gallery, but there is no ground truth for probe. Therefore, it is nessesary to upload the submission file on grew competitation. We seperate test set to: seq_01 as gallery, seq_02 as probe. Then you can modify `eval_func` in the `./config/baseline/baseline_GREW.yaml` to `evaluate_real_scene`, you can obtain result localy like setting of OUMVLP. 
