# Datasets
OpenGait officially supports a few gait datasets. In order to use them, you need to download them and use the code provided here to pre-process them to the format required by OpenGait.

## Pre-process
In general, we read the original image provided by the dataset and save a sequence as a pickle file to speed up the training IO. 

The expected dataset structure is as follows:
```
    DATASET_ROOT/
        001 (subject)/
            bg-01 (type)/
                    000 (view)/
                        000.pkl (contains all frames)
                ......
            ......
        ......
```

The specific preprocessing steps are described inside each dataset folder.

## Split dataset
For each dataset, we split the dataset into training and testing sets. The training set is used to train the model, and the testing set is used to evaluate the model.

You can use the partition file in [dataset folder](CASIA-B/CASIA-B.json) directly, or you can create yours. Remember to set your path to the partition file in [config/*.yaml](../config/).
