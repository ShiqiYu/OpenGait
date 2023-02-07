# Configuration item

### data_cfg
* Data configuration
>
>  * Args
>     * dataset_name: Only support `CASIA-B` and `OUMVLP` now.
>     * dataset_root: The path of storing your dataset.
>     * num_workers: The number of workers to collect data.
>     * dataset_partition: The path of storing your dataset partition file. It splits the dataset to two parts, including train set and test set.
>     * cache: If `True`, load all data to memory during buiding dataset.
>     * test_dataset_name: The name of test dataset. 
----

### loss_cfg
* Loss function
>  * Args
>     * type: Loss function type, support `TripletLoss` and `CrossEntropyLoss`.
>     * loss_term_weight: loss weight.
>     * log_prefix: the prefix of loss log.

----
### optimizer_cfg
* Optimizer
>  * Args
>     * solver: Optimizer type, example: `SGD`, `Adam`.
>     * **others**: Please refer to `torch.optim`.


### scheduler_cfg
* Learning rate scheduler
>  * Args
>     * scheduler : Learning rate scheduler, example: `MultiStepLR`.
>     * **others** : Please refer to `torch.optim.lr_scheduler`.
----
### model_cfg
* Model to be trained
>  * Args
>     * model : Model type, please refer to [Model Library](../opengait/modeling/models) for the supported values.
>     * **others** : Please refer to the [Training Configuration File of Corresponding Model](../configs).
----
### evaluator_cfg
* Evaluator configuration
>  * Args
>     * enable_float16: If `True`, enable the auto mixed precision mode.
>     * restore_ckpt_strict: If `True`, check whether the checkpoint is the same as the defined model.
>     * restore_hint: `int` value indicates the iteration number of restored checkpoint; `str` value indicates the path to restored checkpoint.
>     * save_name: The name of the experiment.
>     * eval_func: The function name of evaluation. For `CASIA-B`, choose `identification`.
>     * sampler:
>       - type: The name of sampler. Choose `InferenceSampler`.
>       - sample_type: In general, we use `all_ordered` to input all frames by its natural order, which makes sure the tests are consistent.
>       - batch_size: `int` values.
>       - **others**: Please refer to [data.sampler](../opengait/data/sampler.py) and [data.collate_fn](../opengait/data/collate_fn.py)
>     * transform: Support `BaseSilCuttingTransform`, `BaseSilTransform`. The difference between them is `BaseSilCuttingTransform` cut out the black pixels on both sides horizontally.
>     * metric: `euc` or `cos`, generally, `euc` performs better.

----
### trainer_cfg
* Trainer configuration
>  * Args
>     * restore_hint: `int` value indicates the iteration number of restored checkpoint; `str` value indicates the path to restored checkpoint. The option is often used to finetune on new dataset or restore the interrupted training process.
>     * fix_BN: If `True`, we fix the weight of all `BatchNorm` layers.
>     * log_iter: Log the information per `log_iter` iterations.
>     * save_iter: Save the checkpoint per `save_iter` iterations.
>     * with_test: If `True`, we test the model every `save_iter` iterations. A bit of performance impact.(*Disable in Default*)
>     * optimizer_reset: If `True` and `restore_hint!=0`, reset the optimizer while restoring the model.
>     * scheduler_reset: If `True` and `restore_hint!=0`, reset the scheduler while restoring the model.
>     * sync_BN: If `True`, applies Batch Normalization synchronously.
>     * total_iter: The total training iterations, `int` values.
>     * sampler:
>       - type: The name of sampler. Choose `TripletSampler`.
>       - sample_type: `[all, fixed, unfixed]` indicates the number of frames used to test, while `[unordered, ordered]` means whether input sequence by its natural order. Example: `fixed_unordered` means selecting fixed number of frames randomly.
>       - batch_size: *[P,K]* where `P` denotes the subjects in training batch while the `K` represents the sequences every subject owns. **Example**:
>         - 8
>         - 16
>       - **others**: Please refer to [data.sampler](../opengait/data/sampler.py) and [data.collate_fn](../opengait/data/collate_fn.py).
>     * **others**: Please refer to `evaluator_cfg`.
---
**Note**: 
- All the config items will be merged into [default.yaml](../configs/default.yaml), and the current config is preferable.
- The output directory, which includes the log, checkpoint and summary files, is depended on the defined `dataset_name`, `model` and `save_name` settings, like `output/${dataset_name}/${model}/${save_name}`.
# Example

```yaml
data_cfg:
  dataset_name: CASIA-B
  dataset_root:  your_path
  dataset_partition: ./datasets/CASIA-B/CASIA-B.json
  num_workers: 1
  remove_no_gallery: false # Remove probe if no gallery for it
  test_dataset_name: CASIA-B

evaluator_cfg:
  enable_float16: true
  restore_ckpt_strict: true
  restore_hint: 60000
  save_name: Baseline
  eval_func: evaluate_indoor_dataset
  sampler:
    batch_shuffle: false
    batch_size: 16
    sample_type: all_ordered # all indicates whole sequence used to test, while ordered means input sequence by its natural order; Other options:   fixed_unordered
    frames_all_limit: 720 # limit the number of sampled frames to prevent out of memory
  metric: euc # cos
  transform:
    - type: BaseSilCuttingTransform
      img_w: 64

loss_cfg:
  - loss_term_weight: 1.0
    margin: 0.2
    type: TripletLoss
    log_prefix: triplet
  - loss_term_weight: 0.1
    scale: 16
    type: CrossEntropyLoss
    log_prefix: softmax
    log_accuracy: true

model_cfg:
  model: Baseline
  backbone_cfg:
    in_channels: 1
    layers_cfg: # Layers configuration for automatically model construction
      - BC-64
      - BC-64
      - M
      - BC-128
      - BC-128
      - M
      - BC-256
      - BC-256
      # - M
      # - BC-512
      # - BC-512
    type: Plain
  SeparateFCs:
    in_channels: 256
    out_channels: 256
    parts_num: 31
  SeparateBNNecks:
    class_num: 74
    in_channels: 256
    parts_num: 31
  bin_num:
    - 16
    - 8
    - 4
    - 2
    - 1

optimizer_cfg:
  lr: 0.1
  momentum: 0.9
  solver: SGD
  weight_decay: 0.0005

scheduler_cfg:
  gamma: 0.1
  milestones: # Learning Rate Reduction at each milestones
    - 20000
    - 40000
  scheduler: MultiStepLR
trainer_cfg:
  enable_float16: true # half_percesion float for memory reduction and speedup
  fix_layers: false
  log_iter: 100
  restore_ckpt_strict: true
  restore_hint: 0
  save_iter: 10000
  save_name: Baseline
  sync_BN: true
  total_iter: 60000
  sampler:
    batch_shuffle: true
    batch_size:
      - 8 # TripletSampler, batch_size[0] indicates Number of Identity
      - 16 #                 batch_size[1] indicates Samples sequqnce for each Identity
    frames_num_fixed: 30 # fixed frames number for training
    frames_num_max: 50 # max frames number for unfixed training
    frames_num_min: 25 # min frames number for unfixed traing
    sample_type: fixed_unordered # fixed control input frames number, unordered for controlling order of input tensor; Other options: unfixed_ordered or all_ordered
    type: TripletSampler
  transform:
    - type: BaseSilCuttingTransform
      img_w: 64

```
