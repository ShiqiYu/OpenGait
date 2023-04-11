# CASIA-E
Application URL: https://www.scidb.cn/en/detail?dataSetId=57be0e918db743279baf44a38d013a06
## Data processing
- Original
    ```
    test615-1014.zip
    train001-500.zip
    val501-614.zip
    ```
- Run `python datasets/CASIA-E/extractor.py --input_path CASIA-E/ --output_path CASIA-E-processed/ -n 8 -s 64`. \
  `n` is number of workers. `s` is the target image size.
- Processed
    ```
    CASIA-E-processed
        forTrain # raw images
            001 (subject)
                H (height)
                    scene1 (scene)
                        bg (walking condition)
                            000 (view)
                                1 (sequence number)
                                    xxx.jpg (images)
                                    ......
                                ......
                            ......
                        ......
                    ......
                ......
            ......

        opengait # pickle file
            001 (subject)
                H_scene1_bg_1 (type)
                        000 (view)
                            000.pkl (contains all frames)
                        ......
                ......
            ......
    ```

## Setting
Compared with the settings in the original paper, we only used 200 people for training, and the rest were used as the test set, and the division of gallery and probe is more practical and difficult.
For specific experimental settings, please refer to [gaitbase_casiae.yaml](../../configs/gaitbase/gaitbase_casiae.yaml).
For the specific division of the probe and gallery, please refer to [evaluator.py](../../opengait/evaluation/evaluator.py).
