# Tutorial for [Scoliosis1K](https://zhouzi180.github.io/Scoliosis1K)

## Download the Scoliosis1K dataset
Download the dataset from the [link](https://zhouzi180.github.io/Scoliosis1K).
decompress these two file by following command:
```shell
unzip -P password Scoliosis1K-pkl.zip   | xargs -n1 tar xzvf
```
password should be obtained by signing [agreement](https://zhouzi180.github.io/Scoliosis1K/static/resources/Scoliosis1KAgreement.pdf) and sending to email (12331257@mail.sustech.edu.cn)

Then you will get Scoliosis1K formatted as:
```
    DATASET_ROOT/
        00000 (subject)/
            positive (category)/
                    000-180 (view)/
                        000.pkl (contains all frames)
        ......
```
## Train the dataset
Modify the `dataset_root` in `configs/sconet/sconet_scoliosis1k.yaml`, and then run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs configs/sconet/sconet_scoliosis1k.yaml --phase train
```


## Process from RAW dataset

### Preprocess the dataset (Optional)
Download the raw dataset from the [official link](https://zhouzi180.github.io/Scoliosis1K). You will get two compressed files, i.e. `Scoliosis1K-raw.zip`, and `Scoliosis1K-pkl.zip`.
We recommend using our provided pickle files for convenience, or process raw dataset into pickle by this command:
```shell
python datasets/pretreatment.py --input_path Scoliosis1K_raw --output_path Scoliosis1K-pkl
```
