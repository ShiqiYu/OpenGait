**Note:**
This code is only used for **academic purposes**, people cannot use this code for anything that might be considered commercial use.

# OpenGait

OpenGait is a flexible and extensible gait recognition project provided by the [Shiqi Yu Group](https://faculty.sustech.edu.cn/yusq/) and supported in part by [WATRIX.AI](http://www.watrix.ai). Just the pre-beta version is released now, and more documentations as well as the reproduced methods will be offered as soon as possible.

**Highlighted features:**
- **Multiple Models Support**: We reproduced several SOTA methods, and reached the same or even better performance. 
- **DDP Support**: The officially recommended [`Distributed Data Parallel (DDP)`](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) mode is used during the training and testing phases.
- **AMP Support**: The [`Auto Mixed Precision (AMP)`](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html?highlight=amp) option is available.
- **Nice log**: We use [`tensorboard`](https://pytorch.org/docs/stable/tensorboard.html) and `logging` to log everything, which looks pretty.


# Model Zoo

|                                                                                          Model                                                                                          |     NM     |     BG     |     CL     | Configuration                                                                                | Input Size | Inference Time |    Model Size    |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :--------: | :--------: | :------------------------------------------------------------------------------------------- | :--------: | :------------: | :--------------: |
|                                                                                        Baseline                                                                                         |    96.3    |    92.2    |    77.6    | [baseline.yaml](config/baseline.yaml)                                                        |   64x44    |      12s       |      3.78M       |
|                                                                [GaitSet(AAAI2019)](https://arxiv.org/pdf/1811.06186.pdf)                                                                | 95.8(95.0) | 90.0(87.2) | 75.4(70.4) | [gaitset.yaml](config/gaitset.yaml)                                                          |   64x44    |      11s       |      2.59M       |
|                                                   [GaitPart(CVPR2020)](http://home.ustc.edu.cn/~saihui/papers/cvpr2020_gaitpart.pdf)                                                    | 96.1(96.2) | 90.7(91.5) | 78.7(78.7) | [gaitpart.yaml](config/gaitpart.yaml)                                                        |   64x44    |      22s       |      1.20M       |
|                                                        [GLN*(ECCV2020)](http://home.ustc.edu.cn/~saihui/papers/eccv2020_gln.pdf)                                                        | 96.4(95.6) | 93.1(92.0) | 81.0(77.2) | [gln_phase1.yaml](config/gln/gln_phase1.yaml), [gln_phase2.yaml](config/gln/gln_phase2.yaml) |   128x88   |      14s       | 9.46M / 15.6214M |
| [GaitGL(ICCV2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Gait_Recognition_via_Effective_Global-Local_Feature_Representation_and_Local_Temporal_ICCV_2021_paper.pdf) | 97.4(97.4) | 94.5(94.5) | 83.8(83.6) | [gaitgl.yaml](config/gaitgl.yaml)                                                            |   64x44    |      31s       |      3.10M       |

The results in the parentheses are mentioned in the papers


**Note**:
- All the models were tested on [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp) (Rank@1, excluding identical-view cases).
- The shown result of GLN is implemented without compact block. 
- Only 2 RTX6000 are used during the inference phase.
- The results on [OUMVLP](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html) will be released soon.
It's inference process just cost about 90 secs(Baseline & 8 RTX6000).


# Get Started
## Installation
1. clone this repo.
    ```
    git clone https://github.com/ShiqiYu/OpenGait.git
    ```
2. Install dependenices:
    - pytorch >= 1.6
    - torchvision
    - pyyaml
    - tensorboard
    - opencv-python
    - tqdm
    
    Install dependenices by [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):
    ```
    conda install tqdm pyyaml tensorboard opencv
    conda install pytorch==1.6.0 torchvision -c pytorch
    ```    
    Or, Install dependenices by pip:
    ```
    pip install tqdm pyyaml tensorboard opencv-python
    pip install torch==1.6.0 torchvision==0.7.0
    ```
## Prepare dataset
See [prepare dataset](doc/prepare_dataset.md).

## Get pretrained model
- Option 1:
    ```
    python misc/download_pretrained_model.py
    ```
- Option 2: Go to the [release page](https://github.com/ShiqiYu/OpenGait/releases/), then download the model file and uncompress it to `output`.

## Train
Train a model by
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/baseline.yaml --phase train
```
- `python -m torch.distributed.launch` Our implementation uses DistributedDataParallel.
- `--nproc_per_node` The number of gpu to use, it must equal the length of `CUDA_VISIBLE_DEVICES`.
- `--cfgs` The path of config file.
- `--phase` Specified as `train`.
- `--iter` You can specify a number of iterations or use `restore_hint` in the configuration file and resume training from there.
- `--log_to_file` If specified, log will be written on disk simultaneously. 

You can run commands in [train.sh](train.sh) for training different models.

## Test
Use trained model to evaluate by
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/baseline.yaml --phase test
```
- `--phase` Specified as `test`.
- `--iter` You can specify a number of iterations or or use `restore_hint` in the configuration file and restore model from there.

**Tip**: Other arguments are the same as train phase.

You can run commands in [test.sh](test.sh) for testing different models.
## Customize
If you want customize your own model, see [here](doc/how_to_create_your_model.md).

# Warning
- Some models may not be compatible with `AMP`, you can disable it by setting `enable_float16` **False**.
- In `DDP` mode, zombie processes may occur when the program terminates abnormally. You can use this command `kill $(ps aux | grep main.py | grep -v grep | awk '{print $2}')` to clear them. 
- We implemented the functionality of testing while training, but it slightly affected the results. None of our published models use this functionality. You can disable it by setting `with_test` **False**.

# Authors:
**Open Gait Team (OGT)**
- [Chao Fan (樊超)](https://faculty.sustech.edu.cn/?p=128578&tagid=yusq&cat=2&iscss=1&snapid=1&orderby=date)
- [Chuanfu Shen (沈川福)](https://faculty.sustech.edu.cn/?p=95396&tagid=yusq&cat=2&iscss=1&snapid=1&orderby=date)
- [Junhao Liang (梁峻豪)](https://faculty.sustech.edu.cn/?p=95401&tagid=yusq&cat=2&iscss=1&snapid=1&orderby=date)

# Acknowledgement
- GLN: [Saihui Hou (侯赛辉)](http://home.ustc.edu.cn/~saihui/index_english.html)
- GaitGL: Beibei Lin (林贝贝)
