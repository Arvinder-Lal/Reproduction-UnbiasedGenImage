## Real-Time Deepfake Detection in the Real-World
This project incorporates the official implementation of Real-Time Deepfake Detection in the Real-World by Bar Cavia, Eliahu Horwitz, Tal Reiss, and Yedid Hoshen.

Their work introduced LaDeDa, a patch-based deepfake detector, and Tiny-LaDeDa, a lightweight distilled version capable of efficient real-time performance. These models serve as the basis for the real-time module included in this project.


## Setup
virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training
To start a training LaDeDa on the raw GenImage dataset:
```bash
python3 train.py --name LaDeDa --dataroot {PATH_TO_DATASET} --checkpoints_dir {./NAME_OF_CHECKPOINT} --batch_size 32 --lr 0.0002 \
--delr_freq 10 --base_path {BASE_PATH} --csv_data_path {CSV_PATH} --class-map {CLASS_MAP} --balance_train_classes --generator {train_set} \
--dataset classic --continue_train --epoch latest
```

To start a training only using JPEG(qf=96) images:
```bash
python3 train.py --name LaDeDa --dataroot {PATH_TO_DATASET} --checkpoints_dir {./NAME_OF_CHECKPOINT} ---batch_size 32 --lr 0.0002 \
--delr_freq 10 --base_path {BASE_PATH} --csv_data_path {CSV_PATH} --class-map {PATH_TO_CLASS_MAP} --balance_train_classes \
--generator {train_set} --dataset jpeg96 --jpeg_qf 96 --continue_train --epoch latest
```

To start a training only using JPEG(qf=96) images and including size constrain: 
```bash
python3 train.py --name LaDeDa --dataroot {PATH_TO_DATASET} --checkpoints_dir {./NAME_OF_CHECKPOINT} ---batch_size 32 --lr 0.0002 \
--delr_freq 10 --base_path {BASE_PATH} --csv_data_path {CSV_PATH} --class-map {PATH_TO_CLASS_MAP} --balance_train_classes \
--generator {train_set} --dataset size_constrained --min_size {lower_bound} --max_size {upper_bound} --cropsize {lower_bound} --jpeg_qf 96 \
--continue_train --epoch latest
```

NOTE:
--continue_train --epoch latest is optional
--base_path is the path to the GenImage download and is prepended to the paths in the CSV file

## Evaluation
Example of Cross-Generator-Validation for LaDeDa: 
```bash
python3 test.py --dataroot {DATAROOT} --checkpoints_dir {PATH_TO_CHECKPOINT.pth} --model LaDeDa \
--base_path {BASE_PATH} --csv_data_path {CSV_PATH} --class-map {PATH_TO_CLASS_MAP} --is_validating --generator {eval_set} \
--dataset classic --result_path {filename} --jpeg_qf {qf} --resize {size} --cropsize {size}
```

NOTE: 
--jpeg_qf, --resize, --cropsize are optional
--compress_natural if you want to compress natural images as well
(see args for more possible experiments)

## Acknowledgment
The training pipeline used by Real-Time Deepfake Detection in the Real-World is similar to [NPR](https://github.com/peterwang512/CNNDetection), [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect) and [CNNDetection](https://github.com/peterwang512/CNNDetection).


