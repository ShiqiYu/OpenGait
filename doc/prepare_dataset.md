# Prepare dataset
Suppose you have downloaded the original dataset, we need to preprocess the data and save it as pickle file. Remember to set your path to the root of processed dataset in [config/*.yaml](config/).

## Preprocess
**CASIA-B**

- Original
    ```
    CASIA-B
        001 (subject)
            bg-01 (type)
                000 (view)
                    001-bg-01-000-001.png (frame)
                    001-bg-01-000-002.png (frame)
                    ......
                ......
            ......
        ......
    ```
- Run `python misc/pretreatment.py --input_path CASIA-B --output_path CASIA-B-pkl`
- Processed
    ```
    CASIA-B-pkl
        001 (subject)
            bg-01 (type)
                    000 (view)
                        000.pkl (contains all frames)
                ......
            ......
        ......
    ```

## Split dataset
You can use the partition file in [misc/partitions](misc/partitions/) directly, or you can create yours. Remember to set your path to the partition file in [config/*.yaml](config/).