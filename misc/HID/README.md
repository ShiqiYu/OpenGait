# HID Tutorial
This is the official suppor for competition of Human Identification at a Distance (HID). We report our result is 68.7% using the baseline model. In order for participants to better start the first step, we provide a tutorial on how to use OpenGait for HID.

## Preprocess the dataset
Download the raw dataset from the [official link](http://hid2022.iapr-tc4.org/). You will get three compressed files, i.e. `train.tar`, `HID2022_test_gallery.zip` and `HID2022_test_probe.zip`.
After unpacking these three files, run this command:
```shell
python misc/HID/pretreatment_HID.py --input_train_path="train" --input_gallery_path="HID2022_test_gallery" --input_probe_path="HID2022_test_probe" --output_path="HID-128-pkl" 
```

## Train the dataset
Modify the `dataset_root` in `./misc/HID/baseline_hid.yaml`, and then run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./misc/HID/baseline_hid.yaml --phase train
```
You can also download the [trained model](https://github.com/ShiqiYu/OpenGait/releases/download/v1.1/pretrained_hid_model.pt) and place it in `output` after unzipping.

## Get the submission file.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./misc/HID/baseline_hid.yaml --phase test
```
The result will be generated in your working directory, you must rename and compress it as the requirements before submitting.
