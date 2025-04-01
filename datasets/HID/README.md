# Human Identification at a Distance (HID) Competition
![](http://hid2022.iapr-tc4.org/wp-content/uploads/sites/7/2022/03/%E5%9B%BE%E7%89%871-2.png)
This is the official support for [Human Identification at a Distance (HID)](https://hid2025.iapr-tc4.org/) competition. We provide the baseline code for this competition.

## Tutorial for HID 2025
For HID 2025, we will not provide a training set.  In this competition, you can use any dataset, such as CASIA-B, OUMVLP, CASIA-E, and/or their own dataset, to train your model. In this tutorial, we will use the model trained on previous HID competition training set as the baseline model.

### Download the test set
Download the test gallery and probe from the [link](https://hid.iapr-tc4.org/).
You should decompress these two file by following command:
```
mkdir hid_2025
tar -zxvf gallery.tar.gz
mv gallery/* hid_2025/
rm gallery -rf
# For Phase 1
tar -zxvf probe_phase1.tar.gz -C hid_2025
mv hid_2025/probe_phase1 hid_2025/probe
# For Phase 2
tar -zxvf probe_phase2.tar.gz -C hid_2025
mv hid_2025/probe_phase2 hid_2025/probe

```

### Download the pretrained model
Download the [pretrained model](https://github.com/ShiqiYu/OpenGait/releases/download/v1.1/pretrained_hid_model.zip) from the official website and place it in `output` after unzipping.
```
wget https://github.com/ShiqiYu/OpenGait/releases/download/v1.1/pretrained_hid_model.zip
unzip pretrained_hid_model.zip -d output/
```

## Generate the result
Modify the `dataset_root` in `configs/gaitbase/gaitbase_hid.yaml`, and then run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs configs/gaitbase/gaitbase_hid.yaml --phase test
```
The result will be generated in `HID_result/current_time.csv`.

## Submit the result
Rename the csv file to `submission.csv`, then zip it and upload to [official submission link](https://codalab.lisn.upsaclay.fr/competitions/21845).

---

## (Deprecated) Tutorial for HID 2022 
 We report our result of 68.7% using the baseline model and 80.0% with re-ranking. In order for participants to better start the first step, we provide a tutorial on how to use OpenGait for HID.

### Preprocess the dataset
Download the raw dataset from the [official link](http://hid2022.iapr-tc4.org/). You will get three compressed files, i.e. `train.tar`, `HID2022_test_gallery.zip` and `HID2022_test_probe.zip`.
After unpacking these three files, run this command:
```shell
python datasets/HID/pretreatment_HID.py --input_train_path="train" --input_gallery_path="HID2022_test_gallery" --input_probe_path="HID2022_test_probe" --output_path="HID-128-pkl" 
```

### Train the dataset
Modify the `dataset_root` in `configs/baseline/baseline_hid.yaml`, and then run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs configs/baseline/baseline_hid.yaml --phase train
```
If you trained a model, place it in `output` after unzipping.

### Get the submission file
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs configs/baseline/baseline_hid.yaml --phase test
```
The result will be generated in your working directory.

### Submit the result
Follow the steps in the [official submission guide](https://codalab.lisn.upsaclay.fr/competitions/2542#participate), you need rename the file to `submission.csv` and compress it to a zip file. Finally, you can upload the zip file to the [official submission link](https://codalab.lisn.upsaclay.fr/competitions/2542#participate-submit_results).
