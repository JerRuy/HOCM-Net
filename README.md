
# README

## Overview

This project utilizes a 3D U-Net model for medical image processing. The provided script, `main.sh`, sets up the necessary parameters and runs the Python script `main.py` with those parameters.

## Requirements

This library requires python 2.7+
Ensure you have the following Python packages installed. You can install them using `pip`:
- opencv-python
- scipy
- tensorflow
- numpy

## Parameters
The script main.sh allows you to configure various parameters for the training and testing phases. Below is a description of the parameters and their default values:

* --phase: Specifies the phase of operation. train_step1 is used for training using image data only, while train_step2 and train_step3 is used for training using both the image data and the predicted data. Options are train_step1, train_step2, train_step3 and test. Default is train_step1.
* --batch_size: The batch size for training. Default is 1.
* --inputI_size: The size of the input images. Default is 96.
* --inputI_chn: The number of channels in the input images. Default is 1.
* --outputI_size: The size of the output images. Default is 96.
* --output_chn: The number of channels in the output images. Default is 9.
* --pred_filter: The filter used to pred data. Default is 0, 1, 2, 3, 4, 5, 6, 7, 8.
* --rename_map: The map used to rename labels. Default is 0, 1, 2, 3, 4, 5, 6, 7, 8.
* --resize_r: The resize ratio for the images. Default is 0.9.
* --traindata_dir: The directory where the training data is stored. Default is ../../../HCM_GD_resize/original.
* --chkpoint_dir: The directory where the train_Step1 model checkpoints will be saved. Default is ../outcome/model/checkpoint.
* --chkpoint_dir2: The directory where the train_Step2 model checkpoints will be saved. Default is ../outcome/model/checkpoint2.
* --chkpoint_dir3: The directory where the train_Step3 model checkpoints will be saved. Default is ../outcome/model/checkpoint3.
* --learning_rate: The learning rate for the optimizer. Default is 0.001.
* --beta1: The beta1 parameter for the Adam optimizer. Default is 0.5.
* --epoch: The number of epochs for training. Default is 54000.
* --model_name: The name of the saved model. Default is ds_ft_hybrid_4ct.model.
* --save_intval: The interval (in epochs) at which to save the model checkpoints. Default is 2000.
* --testdata_dir: The directory where the test data is stored. Default is ../../../HCM_GD_resize/test/image.
* --labeling_dir: The directory where the labeling results will be saved. Default is ../result-test.
* --testlabel_dir: The directory where the test labels are stored. Default is ../../../HCM_GD_resize/test/label.
* --predlabel_dir: The directory where the pred labels are stored. Default is ../../../HCM_GD_resize/test/label.
* --ovlp_ita: The overlap iteration parameter. Default is 4.
* 
## Running the Script

To run the script with default parameters:
```
python main.py
```
To override the default parameters, you can pass them as command-line arguments. For example:
```
python  main.py 
  --phase "train_step1" \
  --batch_size "1" \
  --inputI_size "96" \
  --inputI_chn "1" \
  --outputI_size "96" \
  --output_chn "9" \
  --pred_filter "1,2,3" \
  --rename_map "0, 1, 2, 3, 4, 5, 6, 7, 8" \
  --resize_r "0.9" \
  --traindata_dir "../../../HCM_GD_resize/original" \
  --chkpoint_dir "../outcome/model/checkpoint" \
  --chkpoint_dir2 "../outcome/model/checkpoint2" \
  --chkpoint_dir3 "../outcome/model/checkpoint3" \
  --learning_rate "0.001" \
  --beta1 "0.5" \
  --epoch "54000" \
  --model_name "ds_ft_hybrid_4ct.model" \
  --save_intval "2000" \
  --testdata_dir "../../../HCM_GD_resize/test1/image" \
  --labeling_dir "../result-test" \
  --testlabel_dir "../../../HCM_GD_resize/test1/label" \
  --predlabel_dir "../../../HCM_GD_resize/train/pred_label" \
  --ovlp_ita "4"
```

## Directory Structure

Ensure that the directory structure is set up correctly as follows:
project/
├── main.sh
├── main.py
├── model.py
├── ops.py
├── seg_eval.py
├── utils.py
├── result/
└── outcome/
    └── model/
        └── checkpoint/
        └── checkpoint2/
        └── checkpoint3/
-   `result/`: Directory output label data.
-   `outcome/model/checkpoint/`: Directory where model checkpoints will be saved.

## Additional Information

-   `main.sh`  is a Bash script that sets up and runs the Python script with the specified parameters.
-   `main.py`  is the main Python script that trains or tests the 3D U-Net model based on the provided parameters.

Ensure all directories and paths are correctly set up in  `main.sh`  before running the script.