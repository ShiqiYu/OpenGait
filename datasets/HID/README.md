# HID Tutorial
![](http://hid2022.iapr-tc4.org/wp-content/uploads/sites/7/2022/03/%E5%9B%BE%E7%89%871-2.png)
This is the official support for competition of [Human Identification at a Distance (HID)](http://hid2022.iapr-tc4.org/). We report our result of 68.7% using the baseline model and 80.0% with re-ranking. In order for participants to better start the first step, we provide a tutorial on how to use OpenGait for HID.

## Preprocess the dataset
Download the raw dataset from the [official link](http://hid2022.iapr-tc4.org/). You will get three compressed files, i.e. `train.tar`, `HID2022_test_gallery.zip` and `HID2022_test_probe.zip`.
After unpacking these three files, run this command:
```shell
python datasets/HID/pretreatment_HID.py --input_train_path="train" --input_gallery_path="HID2022_test_gallery" --input_probe_path="HID2022_test_probe" --output_path="HID-128-pkl" 
```

## Train the dataset
Modify the `dataset_root` in `configs/baseline/baseline_hid.yaml`, and then run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs configs/baseline/baseline_hid.yaml --phase train
```
You can also download the [trained model](https://github.com/ShiqiYu/OpenGait/releases/download/v1.1/pretrained_hid_model.zip) and place it in `output` after unzipping.

## Get the submission file
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs configs/baseline/baseline_hid.yaml --phase test
```
The result will be generated in your working directory.

## Submit the result
Follow the steps in the [official submission guide](https://codalab.lisn.upsaclay.fr/competitions/2542#participate), you need rename the file to `submission.csv` and compress it to a zip file. Finally, you can upload the zip file to the [official submission link](https://codalab.lisn.upsaclay.fr/competitions/2542#participate-submit_results).