# CASIA-B
Download URL: http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip
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
- Run `python datasets/pretreatment.py --input_path CASIA-B --output_path CASIA-B-pkl`
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