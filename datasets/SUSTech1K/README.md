# Tutorial for [SUSTech1K](https://lidargait.github.io)

## Download the SUSTech1K dataset
Download the dataset from the [link](https://lidargait.github.io).
decompress these two file by following command:
```shell
unzip -P password SUSTech1K-pkl.zip   | xargs -n1 tar xzvf
```
password should be obtained by signing [agreement](https://lidargait.github.io/static/resources/SUSTech1KAgreement.pdf) and sending to email (shencf2019@mail.sustech.edu.cn)

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

