<img src="./assets/logo2.png" width = "320" height = "110" alt="logo" />

<div align="center"><img src="./assets/nm.gif" width = "100" height = "100" alt="nm" /><img src="./assets/bg.gif" width = "100" height = "100" alt="bg" /><img src="./assets/cl.gif" width = "100" height = "100" alt="cl" /></div>

------------------------------------------

OpenGait is a flexible and extensible gait recognition project provided by the [Shiqi Yu Group](https://faculty.sustech.edu.cn/yusq/) and supported in part by [WATRIX.AI](http://www.watrix.ai).

## What's New
- **[Feb 2023]** [HID 2023 competition](https://hid2023.iapr-tc4.org/) is open, welcome to participate. Additionally, tutorial for the competition has been updated in [datasets/HID/](./datasets/HID).
- [Dec 2022] Dataset [Gait3D](https://github.com/Gait3D/Gait3D-Benchmark) is supported in [datasets/Gait3D](./datasets/Gait3D).
- [Mar 2022] Dataset [GREW](https://www.grew-benchmark.org) is supported in [datasets/GREW](./datasets/GREW).


## Our Publications
- [CVPR 2023] LidarGait: Benchmarking 3D Gait Recognition with Point Clouds, [*Paper*](https://arxiv.org/pdf/2211.10598), [*Dataset and Code(Coming Soon)*](https://lidargait.github.io).
- [CVPR 2023] OpenGait: Revisiting Gait Recognition Toward Better Practicality, [*Paper*](https://arxiv.org/pdf/2211.06597.pdf), [*Code*](configs/gaitbase).
- [ECCV 2022] GaitEdge: Beyond Plain End-to-end Gait Recognition for Better Practicality, [*Paper*](https://arxiv.org/pdf/2203.03972), [*Code*](configs/gaitedge/README.md).

## Highlighted features
- **Mutiple Dataset supported**: OpenGait supports four popular gait datasets: [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp), [OUMVLP](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html), [HID](http://hid2022.iapr-tc4.org/), and [GREW](https://www.grew-benchmark.org).
- **Multiple Models Support**: We reproduced several SOTA methods, and reached the same or even the better performance. 
- **DDP Support**: The officially recommended [`Distributed Data Parallel (DDP)`](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) mode is used during both the training and testing phases.
- **AMP Support**: The [`Auto Mixed Precision (AMP)`](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html?highlight=amp) option is available.
- **Nice log**: We use [`tensorboard`](https://pytorch.org/docs/stable/tensorboard.html) and `logging` to log everything, which looks pretty.

## Getting Started


Please see [0.get_started.md](docs/0.get_started.md). We also provide the following tutorials for your reference:
- [Prepare dataset](docs/2.prepare_dataset.md)
- [Detailed configuration](docs/3.detailed_config.md)
- [Customize model](docs/4.how_to_create_your_model.md)
- [Advanced usages](docs/5.advanced_usages.md) 

## Model Zoo
Results and models are available in the [model zoo](docs/1.model_zoo.md).


## Authors:
**Open Gait Team (OGT)**
- [Chao Fan (樊超)](https://chaofan996.github.io), 12131100@mail.sustech.edu.cn
- [Chuanfu Shen (沈川福)](https://faculty.sustech.edu.cn/?p=95396&tagid=yusq&cat=2&iscss=1&snapid=1&orderby=date), 11950016@mail.sustech.edu.cn
- [Junhao Liang (梁峻豪)](https://faculty.sustech.edu.cn/?p=95401&tagid=yusq&cat=2&iscss=1&snapid=1&orderby=date), 12132342@mail.sustech.edu.cn

## Acknowledgement
- GLN: [Saihui Hou (侯赛辉)](http://home.ustc.edu.cn/~saihui/index_english.html)
- GaitGL: [Beibei Lin (林贝贝)](https://scholar.google.com/citations?user=KyvHam4AAAAJ&hl=en&oi=ao)
- GREW: [GREW TEAM](https://www.grew-benchmark.org)

## Citation
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

**Note:**
This code is only used for **academic purposes**, people cannot use this code for anything that might be considered commercial use.
