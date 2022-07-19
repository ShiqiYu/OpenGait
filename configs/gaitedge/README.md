# GaitEdge: Beyond Plain End-to-end Gait Recognition for Better Practicality

This [paper](https://arxiv.org/abs/2203.03972) has been accepted by ECCV 2022.

## Abstract
Gait is one of the most promising biometrics to identify individuals at a long distance. Although most previous methods have focused on recognizing the silhouettes, several end-to-end methods that extract gait features directly from RGB images perform better. However, we demonstrate that these end-to-end methods may inevitably suffer from the gait-irrelevant noises, i.e., low-level texture and colorful information. Experimentally, we design the **cross-domain** evaluation to support this view. In this work, we propose a novel end-to-end framework named **GaitEdge** which can effectively block gait-irrelevant information and release end-to-end training potential Specifically, GaitEdge synthesizes the output of the pedestrian segmentation network and then feeds it to the subsequent recognition network, where the synthetic silhouettes consist of trainable edges of bodies and fixed interiors to limit the information that the recognition network receives. Besides, **GaitAlign** for aligning silhouettes is embedded into the GaitEdge without losing differentiability. Experimental results on CASIA-B and our newly built TTG-200 indicate that GaitEdge significantly outperforms the previous methods and provides a more practical end-to-end paradigm.

![img](../../assets/gaitedge.png)
## CASIA-B*
Since the silhouettes of CASIA-B were obtained by the outdated background subtraction, there exists much noise caused by the background and clothes of subjects. Hence, we re-annotate the
silhouettes of CASIA-B and denote it as CASIA-B*. Refer to [here](../../datasets/CASIA-B*/README.md) for more details.
## Performance
|    Model   |  NM  |  BG  |  CL  | TTG-200 (cross-domain) |                  Configuration                 |
|:----------:|:----:|:----:|:----:|:----------------------:|:----------------------------------------------:|
|   GaitGL   | 94.0 | 89.6 | 81.0 |          53.2          |      [phase1_rec.yaml](./phase1_rec.yaml)      |
| GaitGL-E2E | 99.1 | 98.2 | 89.1 |          45.6          |      [phase2_e2e.yaml](./phase2_e2e.yaml)      |
|  GaitEdge  | 98.0 | 96.3 | 88.0 |          53.9          | [phase2_gaitedge.yaml](./phase2_gaitedge.yaml) |

***The results here are higher than those in the paper because we use a different optimization strategy. But this does not affect the conclusion of the paper.***